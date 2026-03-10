"""
SemanticMemory (Level 3): Persistent key-value knowledge store.

Provides a large-scale key-value memory bank for storing and retrieving
factual knowledge.  Reads use k-nearest-neighbor lookup in a learned key
space with cosine similarity, returning a weighted sum of the top-k
matched values.  Writes add new entries during training; the store can be
frozen for inference.

Unlike WorkingMemory and EpisodicMemory, SemanticMemory does NOT inherit
from StatefulModule -- it uses standard nn.Parameter/buffer storage since
entries are managed explicitly (not via StateLink gradient flow).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticMemory(nn.Module):
    """Level-3 semantic memory: persistent key-value knowledge store."""

    def __init__(
        self,
        num_entries: int = 10000,
        d_key: int = 256,
        d_value: int = 768,
        d_model: int = 768,
        top_k: int = 5,
    ) -> None:
        """Initialise SemanticMemory.

        Args:
            num_entries: Maximum number of key-value entries.
            d_key: Dimension of key vectors.
            d_value: Dimension of value vectors.
            d_model: Dimension of model hidden states (for projections).
            top_k: Default number of nearest neighbours for retrieval.
        """
        super().__init__()
        self.num_entries = num_entries
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.default_top_k = top_k

        # ----- Key-value store --------------------------------------------
        # Keys and values stored as buffers (persistent, not trained by
        # backprop directly -- updated via explicit write/update calls).
        self.register_buffer("keys", torch.zeros(num_entries, d_key))
        self.register_buffer("values", torch.zeros(num_entries, d_value))

        # Track how many entries are actually populated
        self.register_buffer("num_written", torch.zeros(1, dtype=torch.long))

        # Track whether the store is frozen (no writes allowed)
        self.register_buffer("_frozen", torch.zeros(1, dtype=torch.bool))

        # ----- Projections ------------------------------------------------
        # Project hidden states to key space for querying
        self.query_proj = nn.Linear(d_model, d_key)

        # Project retrieved values back to model dimension
        self.value_proj = nn.Linear(d_value, d_model)

        # Output gate: blend retrieved info with original hidden
        self.output_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Initialise keys with small random values for better initial similarity spread
        nn.init.normal_(self.keys, std=0.02)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, query: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """Retrieve from memory using k-nearest-neighbor on keys.

        Args:
            query: (..., d_key) query vectors in key space.
            top_k: Number of nearest neighbours to retrieve.
                Defaults to ``self.default_top_k``.

        Returns:
            (..., d_value) weighted sum of top-k matched values.
        """
        if top_k is None:
            top_k = self.default_top_k

        n_written = self.num_written.item()
        if n_written == 0:
            # No entries yet -- return zeros
            out_shape = query.shape[:-1] + (self.d_value,)
            return torch.zeros(out_shape, device=query.device, dtype=query.dtype)

        # Only search among populated entries
        active_keys = self.keys[:n_written]    # (N, d_key)
        active_values = self.values[:n_written]  # (N, d_value)

        # Effective top_k
        k = min(top_k, n_written)

        # Normalise for cosine similarity
        query_norm = F.normalize(query, dim=-1)                # (..., d_key)
        keys_norm = F.normalize(active_keys, dim=-1)           # (N, d_key)

        # Flatten query for matmul, then reshape back
        orig_shape = query.shape[:-1]
        query_flat = query_norm.reshape(-1, self.d_key)        # (Q, d_key)

        # Cosine similarity: (Q, N)
        sim = torch.mm(query_flat, keys_norm.t())

        # Top-k selection
        topk_sim, topk_idx = sim.topk(k, dim=-1)              # (Q, k)

        # Softmax over top-k similarities for weighted retrieval
        topk_weights = F.softmax(topk_sim, dim=-1)            # (Q, k)

        # Gather top-k values: (Q, k, d_value)
        topk_values = active_values[topk_idx]

        # Weighted sum: (Q, d_value)
        retrieved = (topk_weights.unsqueeze(-1) * topk_values).sum(dim=1)

        # Reshape back to original query shape
        return retrieved.reshape(*orig_shape, self.d_value)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Add new entries to the store (training only).

        Args:
            key: (d_key,) or (W, d_key) key vector(s).
            value: (d_value,) or (W, d_value) corresponding value vector(s).

        Raises:
            RuntimeError: If the store is frozen.
        """
        if self._frozen.item():
            raise RuntimeError("SemanticMemory is frozen; writes are not allowed.")

        if key.dim() == 1:
            key = key.unsqueeze(0)
        if value.dim() == 1:
            value = value.unsqueeze(0)

        W = key.size(0)
        n_written = self.num_written.item()

        # Handle overflow: wrap around (FIFO eviction of oldest entries)
        for i in range(W):
            idx = (n_written + i) % self.num_entries
            self.keys[idx] = key[i].detach()
            self.values[idx] = value[i].detach()

        new_count = min(n_written + W, self.num_entries)
        self.num_written.fill_(new_count)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, key_idx: int, new_value: torch.Tensor) -> None:
        """Update the value at an existing entry.

        Args:
            key_idx: Index of the entry to update.
            new_value: (d_value,) new value vector.

        Raises:
            RuntimeError: If the store is frozen.
            IndexError: If ``key_idx`` is out of range.
        """
        if self._frozen.item():
            raise RuntimeError("SemanticMemory is frozen; updates are not allowed.")

        n_written = self.num_written.item()
        if key_idx < 0 or key_idx >= n_written:
            raise IndexError(
                f"key_idx {key_idx} out of range for {n_written} written entries."
            )

        self.values[key_idx] = new_value.detach()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the store: disallow further writes (for inference)."""
        self._frozen.fill_(True)

    def unfreeze(self) -> None:
        """Unfreeze the store: allow writes again."""
        self._frozen.fill_(False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden to key space, retrieve, blend with original.

        Args:
            hidden_states: (B, S, d_model) input hidden states.

        Returns:
            (B, S, d_model) memory-augmented hidden states.
        """
        # Project hidden states to key space
        query_keys = self.query_proj(hidden_states)  # (B, S, d_key)

        # Retrieve from memory
        retrieved_values = self.read(query_keys)  # (B, S, d_value)

        # Project retrieved values back to model dimension
        retrieved_proj = self.value_proj(retrieved_values)  # (B, S, d_model)

        # Gated fusion
        combined = torch.cat([hidden_states, retrieved_proj], dim=-1)
        gate = self.output_gate(combined)
        augmented = hidden_states + gate * retrieved_proj

        return self.layer_norm(augmented)

    # ------------------------------------------------------------------
    # Retrieval loss
    # ------------------------------------------------------------------

    def retrieval_loss(
        self,
        query: torch.Tensor,
        expected_value: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised retrieval accuracy loss.

        Measures how well the retrieval mechanism recovers the expected
        value given a query in key space.

        Args:
            query: (B, d_key) query vectors in key space.
            expected_value: (B, d_value) target value vectors.

        Returns:
            Scalar MSE loss between retrieved and expected values.
        """
        retrieved = self.read(query)  # (B, d_value)
        return F.mse_loss(retrieved, expected_value)
