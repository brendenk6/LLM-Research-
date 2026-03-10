"""RMSNorm layer for GENESIS.

RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
No bias, no mean subtraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467

    Unlike LayerNorm, RMSNorm does not subtract the mean (no re-centering)
    and only re-scales by the root mean square, which is cheaper and performs
    comparably in practice.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialise RMSNorm.

        Args:
            dim: Feature dimension to normalise over (last axis).
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalised tensor of the same shape.
        """
        # Cast to float32 for numerical stability, then cast back.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
