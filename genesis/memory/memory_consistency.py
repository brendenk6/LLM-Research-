"""
MemoryConsistency: Auxiliary loss module for contradiction detection in memory
writes.

Prevents the memory system from storing contradictory information by computing
cosine similarity between new entries and existing memory, penalising writes
that are semantically close but not identical (i.e., likely contradictions).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryConsistency(nn.Module):
    """Detects and penalises contradictory memory writes.

    A new entry is considered *contradictory* if its cosine similarity to an
    existing entry exceeds ``threshold`` but the entries are not near-identical
    (similarity < 0.99).  The loss is the mean excess similarity above the
    threshold across all flagged pairs.
    """

    def __init__(self, d_model: int, threshold: float = 0.8) -> None:
        """Initialise MemoryConsistency.

        Args:
            d_model: Dimension of memory entry vectors.
            threshold: Cosine similarity above which two entries are
                considered potentially contradictory.
        """
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold

        # Learned projection to a comparison space (allows the model to
        # learn which dimensions matter for contradiction detection).
        self.proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def check_consistency(
        self,
        new_entry: torch.Tensor,
        existing_entries: torch.Tensor,
    ) -> Tuple[bool, float]:
        """Check whether *new_entry* contradicts any *existing_entries*.

        Args:
            new_entry: (d_model,) or (1, d_model) single entry to check.
            existing_entries: (N, d_model) bank of existing memory entries.

        Returns:
            Tuple of (has_contradiction, max_similarity_score).
        """
        if existing_entries.size(0) == 0:
            return False, 0.0

        new_entry = new_entry.detach().view(1, -1)
        existing_entries = existing_entries.detach()

        with torch.no_grad():
            new_proj = F.normalize(self.proj(new_entry), dim=-1)
            exist_proj = F.normalize(self.proj(existing_entries), dim=-1)
            # (1, N)
            sims = torch.mm(new_proj, exist_proj.t()).squeeze(0)
            max_sim = sims.max().item()

        # Contradiction: high similarity but not near-identical
        has_contradiction = (max_sim > self.threshold) and (max_sim < 0.99)
        return has_contradiction, max_sim

    def consistency_loss(
        self,
        memory_writes: torch.Tensor,
        existing_memory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a differentiable consistency loss.

        Penalises new writes that fall in the "contradiction zone" -- high
        cosine similarity to existing entries but not near-identical.

        Args:
            memory_writes: (W, d_model) batch of new entries to write.
            existing_memory: (N, d_model) current memory bank.

        Returns:
            Scalar loss tensor.
        """
        if existing_memory.size(0) == 0 or memory_writes.size(0) == 0:
            return torch.tensor(0.0, device=memory_writes.device, requires_grad=True)

        # Project into comparison space
        writes_proj = F.normalize(self.layer_norm(self.proj(memory_writes)), dim=-1)
        exist_proj = F.normalize(self.layer_norm(self.proj(existing_memory)), dim=-1)

        # Cosine similarity matrix: (W, N)
        sim_matrix = torch.mm(writes_proj, exist_proj.t())

        # Contradiction zone: above threshold but below near-identical
        contradiction_mask = (sim_matrix > self.threshold) & (sim_matrix < 0.99)

        if not contradiction_mask.any():
            return torch.tensor(0.0, device=memory_writes.device, requires_grad=True)

        # Penalty: excess similarity above threshold for contradictory pairs
        excess = (sim_matrix - self.threshold).clamp(min=0.0)
        penalty = excess * contradiction_mask.float()

        return penalty.sum() / contradiction_mask.float().sum().clamp(min=1.0)

    def forward(
        self,
        new_entries: torch.Tensor,
        memory_bank: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning scalar consistency loss.

        Args:
            new_entries: (W, d_model) new memory entries.
            memory_bank: (N, d_model) existing memory bank.

        Returns:
            Scalar loss tensor.
        """
        return self.consistency_loss(new_entries, memory_bank)
