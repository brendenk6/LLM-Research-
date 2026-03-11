"""Negative Generator for ACT-V.

Produces corrupted (negative) text samples for training the Verifier.
Each corruption type operates directly on token IDs, introducing specific
types of errors that the Verifier must learn to detect.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch


# Default corruption strategies.
DEFAULT_CORRUPTION_TYPES: List[str] = [
    "entity_swap",
    "negation_insertion",
    "number_perturbation",
    "fact_substitution",
    "temporal_shift",
]

# Heuristic token-ID ranges (works with typical BPE tokenisers like
# SentencePiece / tiktoken where digits cluster together).  These are
# *approximate* and serve as a reasonable default; callers can override
# by subclassing or providing custom ranges.
_NEGATION_TOKENS: List[int] = [451, 694, 1790, 3782]  # "not", "n't", "never", "no"
_TEMPORAL_TOKENS: Dict[int, int] = {
    # Swap temporal markers: yesterday<->tomorrow, before<->after, etc.
    8091: 10643,   # yesterday -> tomorrow
    10643: 8091,   # tomorrow -> yesterday
    1434: 1156,    # before -> after
    1156: 1434,    # after -> before
    4940: 5765,    # past -> future
    5765: 4940,    # future -> past
}


class NegativeGenerator:
    """Generates corrupted text for Verifier training.

    Each corruption method takes a 1-D tensor of token IDs and returns:
      - ``corrupted_ids``: the corrupted sequence.
      - ``labels``: a float tensor of the same length where 1.0 marks
        clean tokens and 0.0 marks corrupted positions.
      - ``locations``: list of integer indices that were modified.
    """

    def __init__(
        self,
        corruption_types: Optional[List[str]] = None,
        vocab_size: int = 32000,
        negation_tokens: Optional[List[int]] = None,
        temporal_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """Initialise NegativeGenerator.

        Args:
            corruption_types: List of corruption strategy names to use.
                Defaults to :data:`DEFAULT_CORRUPTION_TYPES`.
            vocab_size: Size of the token vocabulary (for random sampling).
            negation_tokens: Token IDs representing negation words.
            temporal_map: Mapping of temporal token swaps.
        """
        self.corruption_types = corruption_types or list(DEFAULT_CORRUPTION_TYPES)
        self.vocab_size = vocab_size
        self.negation_tokens = negation_tokens or list(_NEGATION_TOKENS)
        self.temporal_map = temporal_map or dict(_TEMPORAL_TOKENS)

        # Dispatch table
        self._dispatch: Dict[str, callable] = {
            "entity_swap": self.entity_swap,
            "negation_insertion": self.negation_insertion,
            "number_perturbation": self.number_perturbation,
            "fact_substitution": self.fact_substitution,
            "temporal_shift": self.temporal_shift,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def corrupt(
        self,
        input_ids: torch.Tensor,
        corruption_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Apply a corruption to a single sequence of token IDs.

        Args:
            input_ids: 1-D ``LongTensor`` of token IDs (S,).
            corruption_type: Specific corruption to apply.  If *None*, one
                is chosen uniformly at random from the configured types.

        Returns:
            Tuple of (corrupted_ids, labels, locations):
                - ``corrupted_ids``: (S',) corrupted token IDs (length may
                  differ for insertion-based corruptions).
                - ``labels``: (S',) float tensor with 1.0 for clean and
                  0.0 for corrupted positions.
                - ``locations``: list of modified indices.
        """
        if corruption_type is None:
            corruption_type = random.choice(self.corruption_types)
        fn = self._dispatch.get(corruption_type)
        if fn is None:
            raise ValueError(
                f"Unknown corruption type '{corruption_type}'. "
                f"Available: {list(self._dispatch.keys())}"
            )
        return fn(input_ids)

    def batch_corrupt(
        self,
        input_ids_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt a batch of sequences.

        Each sequence in the batch receives a randomly chosen corruption.
        Sequences are padded to the length of the longest corrupted result.

        Args:
            input_ids_batch: (B, S) ``LongTensor`` of token IDs.

        Returns:
            Tuple of (corrupted_batch, labels_batch):
                - ``corrupted_batch``: (B, S') padded corrupted IDs.
                - ``labels_batch``: (B, S') padded labels (pad positions = 1.0).
        """
        batch_corrupted: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for i in range(input_ids_batch.size(0)):
            corrupted, labels, _ = self.corrupt(input_ids_batch[i])
            batch_corrupted.append(corrupted)
            batch_labels.append(labels)

        # Pad to max length in the batch
        max_len = max(t.size(0) for t in batch_corrupted)
        device = input_ids_batch.device

        padded_ids = torch.zeros(len(batch_corrupted), max_len, dtype=torch.long, device=device)
        padded_labels = torch.ones(len(batch_corrupted), max_len, dtype=torch.float, device=device)

        for i, (ids, lbl) in enumerate(zip(batch_corrupted, batch_labels)):
            length = ids.size(0)
            padded_ids[i, :length] = ids
            padded_labels[i, :length] = lbl

        return padded_ids, padded_labels

    # ------------------------------------------------------------------
    # Corruption strategies
    # ------------------------------------------------------------------

    def entity_swap(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Swap entity tokens with random tokens from the same vocabulary range.

        Selects 5-15% of positions and replaces them with random tokens,
        simulating entity substitution errors.
        """
        ids = input_ids.clone()
        seq_len = ids.size(0)
        labels = torch.ones(seq_len, dtype=torch.float, device=ids.device)

        # Choose number of swaps: 5-15% of sequence
        num_swaps = max(1, int(seq_len * random.uniform(0.05, 0.15)))
        positions = random.sample(range(seq_len), min(num_swaps, seq_len))

        for pos in positions:
            # Replace with a random token, avoiding the original
            original = ids[pos].item()
            new_token = random.randint(0, self.vocab_size - 1)
            while new_token == original:
                new_token = random.randint(0, self.vocab_size - 1)
            ids[pos] = new_token
            labels[pos] = 0.0

        return ids, labels, positions

    def negation_insertion(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Insert or remove negation tokens to flip meaning.

        Scans for existing negation tokens and removes them, or inserts
        negation tokens at random positions if none are found.
        """
        ids_list = input_ids.tolist()
        labels_list = [1.0] * len(ids_list)
        locations: List[int] = []
        device = input_ids.device

        # Find existing negation tokens
        neg_positions = [
            i for i, tok in enumerate(ids_list) if tok in self.negation_tokens
        ]

        if neg_positions and random.random() < 0.5:
            # Remove existing negations (process in reverse to maintain indices)
            for pos in reversed(neg_positions):
                ids_list.pop(pos)
                labels_list.pop(pos)
                locations.append(pos)
        else:
            # Insert negation tokens at 1-3 random positions
            num_insertions = random.randint(1, min(3, max(1, len(ids_list) // 10)))
            insert_positions = sorted(
                random.sample(range(len(ids_list)), min(num_insertions, len(ids_list))),
                reverse=True,
            )
            for pos in insert_positions:
                neg_token = random.choice(self.negation_tokens)
                ids_list.insert(pos, neg_token)
                labels_list.insert(pos, 0.0)
                locations.append(pos)

        ids = torch.tensor(ids_list, dtype=torch.long, device=device)
        labels = torch.tensor(labels_list, dtype=torch.float, device=device)
        return ids, labels, locations

    def number_perturbation(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Perturb number tokens by 10-50%.

        Identifies tokens that could represent digits (heuristic: token IDs
        in common digit ranges) and replaces them with nearby values.
        """
        ids = input_ids.clone()
        seq_len = ids.size(0)
        labels = torch.ones(seq_len, dtype=torch.float, device=ids.device)
        locations: List[int] = []

        # Heuristic: perturb random positions (simulating number changes)
        # In a real tokeniser, digit tokens would be identified by decoding.
        # Here we select 3-8% of positions as "number-like" and perturb them.
        num_perturb = max(1, int(seq_len * random.uniform(0.03, 0.08)))
        positions = random.sample(range(seq_len), min(num_perturb, seq_len))

        for pos in positions:
            original = ids[pos].item()
            # Perturb by 10-50% of the token ID range (clamped to vocab)
            perturbation = int(original * random.uniform(0.1, 0.5))
            if perturbation == 0:
                perturbation = random.randint(1, 10)
            if random.random() < 0.5:
                perturbation = -perturbation
            new_token = max(0, min(self.vocab_size - 1, original + perturbation))
            if new_token == original:
                new_token = (original + 1) % self.vocab_size
            ids[pos] = new_token
            labels[pos] = 0.0
            locations.append(pos)

        return ids, labels, locations

    def fact_substitution(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Substitute contiguous spans to simulate factual errors.

        Replaces 1-3 random spans of 2-5 tokens with random tokens,
        simulating the substitution of factual claims.
        """
        ids = input_ids.clone()
        seq_len = ids.size(0)
        labels = torch.ones(seq_len, dtype=torch.float, device=ids.device)
        locations: List[int] = []

        num_spans = random.randint(1, min(3, max(1, seq_len // 20)))

        for _ in range(num_spans):
            span_len = random.randint(2, min(5, seq_len))
            start = random.randint(0, max(0, seq_len - span_len))

            for offset in range(span_len):
                pos = start + offset
                if pos < seq_len:
                    ids[pos] = random.randint(0, self.vocab_size - 1)
                    labels[pos] = 0.0
                    locations.append(pos)

        return ids, labels, locations

    def temporal_shift(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Swap temporal markers to create time-inconsistency errors.

        Looks for known temporal tokens and swaps them with their
        opposites.  If no temporal tokens are found, falls back to
        entity_swap.
        """
        ids = input_ids.clone()
        seq_len = ids.size(0)
        labels = torch.ones(seq_len, dtype=torch.float, device=ids.device)
        locations: List[int] = []

        temporal_positions = [
            i for i in range(seq_len) if ids[i].item() in self.temporal_map
        ]

        if not temporal_positions:
            # Fallback: apply entity swap if no temporal tokens found
            return self.entity_swap(input_ids)

        for pos in temporal_positions:
            original = ids[pos].item()
            ids[pos] = self.temporal_map[original]
            labels[pos] = 0.0
            locations.append(pos)

        return ids, labels, locations
