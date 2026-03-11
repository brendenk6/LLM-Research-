"""Top-Down Conditioning for GENESIS.

Projects higher-tier representations (Tier 2 / Tier 3) back into the
Tier 1 token space so that planning and reasoning results can influence
the final token-level predictions.
"""

from __future__ import annotations


import torch
import torch.nn as nn


class TopDownConditioning(nn.Module):
    """Conditioning module that integrates higher-tier outputs into Tier 1.

    Uses a learned alpha scalar (initialised to 0 for training stability) to
    gradually blend in plan vectors from Tier 2 and refinements from Tier 3.
    """

    def __init__(
        self,
        d_model: int,
        tier2_dim: int,
        chunk_size: int = 32,
    ) -> None:
        """Initialise TopDownConditioning.

        Args:
            d_model: Tier 1 token dimension.
            tier2_dim: Tier 2 latent dimension (may differ from d_model).
            chunk_size: Tokens per chunk (must match TierGate / LatentPooling).
        """
        super().__init__()
        self.d_model = d_model
        self.tier2_dim = tier2_dim
        self.chunk_size = chunk_size

        # Projection from tier2 latent space to token space
        self.plan_proj = nn.Linear(tier2_dim, d_model, bias=False)

        # Learned blending scalar initialised to 0 for stability
        self.alpha = nn.Parameter(torch.zeros(1))

        # Projection for Tier 3 integration (same dim path)
        self.tier3_proj = nn.Linear(tier2_dim, d_model, bias=False)
        self.tier3_alpha = nn.Parameter(torch.zeros(1))

    def apply_cached_plan(
        self,
        tier1_out: torch.Tensor,
        plan_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a previously computed plan vector to Tier 1 output.

        The plan vector is broadcast across all tokens and added with a
        learned alpha scaling.

        Args:
            tier1_out: Tier 1 output of shape (B, S, d_model).
            plan_vector: Plan vector of shape (B, d_model) or (B, 1, d_model).

        Returns:
            Conditioned output of shape (B, S, d_model).
        """
        if plan_vector.dim() == 2:
            plan_vector = plan_vector.unsqueeze(1)  # (B, 1, d_model)

        # Alpha-scaled additive conditioning
        return tier1_out + torch.sigmoid(self.alpha) * plan_vector

    def integrate_tier3(
        self,
        tier1_out: torch.Tensor,
        tier3_out: torch.Tensor,
        tier2_mask: torch.Tensor,
        tier3_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate Tier 3 deliberative output back into Tier 1 token space.

        Tier 3 output is projected to token dimension and added to the
        corresponding token chunks that were escalated through Tier 2 and
        Tier 3.

        Args:
            tier1_out: Tier 1 output of shape (B, S, d_model).
            tier3_out: Tier 3 output of shape (B, N3, L, tier2_dim) where
                N3 is the number of chunks that reached Tier 3.
            tier2_mask: Boolean mask of shape (B, num_chunks) indicating which
                chunks were escalated to Tier 2.
            tier3_mask: Boolean mask of shape (B, num_tier2_chunks) indicating
                which Tier 2 chunks were further escalated to Tier 3.

        Returns:
            Conditioned output of shape (B, S, d_model).
        """
        B, S, D = tier1_out.shape

        if tier3_out is None or tier3_out.numel() == 0:
            return tier1_out

        # Project Tier 3 output to token dimension
        # tier3_out: (B, N3, L, tier2_dim) -> project last dim
        B3, N3, L, _ = tier3_out.shape
        tier3_proj = self.tier3_proj(tier3_out)  # (B, N3, L, d_model)

        # Mean-pool latent vectors per chunk to get chunk-level conditioning
        chunk_cond = tier3_proj.mean(dim=2)  # (B, N3, d_model)

        # Map Tier 3 chunks back to original chunk positions
        # First find Tier 2 chunk indices, then Tier 3 sub-indices
        output = tier1_out.clone()

        for b in range(B):
            # Get Tier 2 escalated chunk indices
            if tier2_mask.dtype == torch.bool:
                t2_indices = tier2_mask[b].nonzero(as_tuple=False).squeeze(-1)
            else:
                t2_indices = (tier2_mask[b] > 0.5).nonzero(as_tuple=False).squeeze(-1)

            # Get Tier 3 escalated chunk indices (relative to Tier 2)
            if tier3_mask.dtype == torch.bool:
                t3_relative = tier3_mask[b].nonzero(as_tuple=False).squeeze(-1)
            else:
                t3_relative = (tier3_mask[b] > 0.5).nonzero(as_tuple=False).squeeze(-1)

            if t3_relative.numel() == 0 or t2_indices.numel() == 0:
                continue

            # Map back to original chunk positions
            # Clamp to avoid index errors from mask size mismatches
            valid_t3 = t3_relative[t3_relative < t2_indices.size(0)]
            if valid_t3.numel() == 0:
                continue

            original_chunk_indices = t2_indices[valid_t3]

            for local_idx, chunk_idx in enumerate(original_chunk_indices):
                if local_idx >= N3:
                    break
                chunk_start = int(chunk_idx.item()) * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, S)
                if chunk_start >= S:
                    continue

                # Add conditioned signal to this chunk's tokens
                cond_vec = chunk_cond[b, local_idx]  # (d_model,)
                output[b, chunk_start:chunk_end] = (
                    output[b, chunk_start:chunk_end]
                    + torch.sigmoid(self.tier3_alpha) * cond_vec.unsqueeze(0)
                )

        return output

    def compute_plan_vector(
        self,
        tier2_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a global plan vector from Tier 2 output.

        Mean-pools all Tier 2 latent vectors and projects to token space.

        Args:
            tier2_out: Tier 2 output of shape (B, N, L, tier2_dim).

        Returns:
            Plan vector of shape (B, d_model).
        """
        # Global mean pool: (B, N, L, tier2_dim) -> (B, tier2_dim)
        plan = tier2_out.mean(dim=(1, 2))
        # Project to token space
        plan = self.plan_proj(plan)  # (B, d_model)
        return plan
