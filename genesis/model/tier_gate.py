"""TierGate for GENESIS.

Chunk-level gating mechanism that decides which chunks of tokens should be
escalated to higher-tier processing.  Operates on mean-pooled chunk
representations and produces a differentiable mask during training or a
hard threshold mask at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TierGate(nn.Module):
    """Gating network that selects which token chunks escalate to the next tier.

    During training the gate returns soft (sigmoid) probabilities so the
    decision is differentiable.  At inference time a hard threshold is applied
    to produce a boolean mask.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 32,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        """Initialise TierGate.

        Args:
            d_model: Dimension of input token representations.
            chunk_size: Number of tokens per chunk for mean pooling.
            threshold: Inference-time threshold for the sigmoid gate.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.eps = eps

        # MLP gate: d_model -> d_model//4 -> 1
        hidden = max(d_model // 4, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Store scores for load-balance loss computation
        self._last_scores: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute chunk-level gating scores and mask.

        Args:
            x: Token representations of shape (B, S, d_model).

        Returns:
            mask: Boolean mask of shape (B, num_chunks).  During training this
                is a soft float mask (sigmoid output); at inference it is a
                hard bool mask.
            scores: Raw sigmoid scores of shape (B, num_chunks).
        """
        B, S, D = x.shape

        # Pad sequence length to be divisible by chunk_size
        remainder = S % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            x = F.pad(x, (0, 0, 0, pad_len))  # pad sequence dim
            S = x.size(1)

        num_chunks = S // self.chunk_size

        # Reshape to (B, num_chunks, chunk_size, D) and mean-pool
        chunks = x.view(B, num_chunks, self.chunk_size, D)
        chunk_repr = chunks.mean(dim=2)  # (B, num_chunks, D)

        # Gate MLP: (B, num_chunks, D) -> (B, num_chunks, 1) -> (B, num_chunks)
        scores = torch.sigmoid(self.gate_mlp(chunk_repr).squeeze(-1))

        # Cache for load balance loss
        self._last_scores = scores

        if self.training:
            # Soft mask during training for differentiability
            mask = scores
        else:
            mask = (scores >= self.threshold)

        return mask, scores

    def load_balance_loss(self) -> torch.Tensor:
        """Compute auxiliary load-balancing loss.

        Encourages the gate to activate a moderate fraction of chunks rather
        than collapsing to all-on or all-off.  The loss is the variance of the
        per-batch mean activation rate, pushing it toward a balanced split.

        Returns:
            Scalar loss tensor.
        """
        if self._last_scores is None:
            return torch.tensor(0.0)

        scores = self._last_scores  # (B, num_chunks)
        # Mean activation per sample
        mean_activation = scores.mean(dim=-1)  # (B,)
        # We want diversity: penalise deviation from a target activation rate
        # Using variance across the chunk dimension encourages uniform routing
        chunk_load = scores.mean(dim=0)  # (num_chunks,)
        # Ideal: each chunk has equal probability of being selected
        # Loss: coefficient of variation across chunks
        loss = chunk_load.float().var() + (mean_activation.float().var())
        return loss
