"""SwiGLU Feed-Forward Network for GENESIS.

FFN(x) = (x @ W1 * SiLU(x @ W_gate)) @ W2

Reference: https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU gated feed-forward network.

    Uses two parallel linear projections -- one gated through SiLU -- whose
    element-wise product is projected back down to d_model.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """Initialise SwiGLUFFN.

        Args:
            d_model: Input and output dimension.
            d_ff: Hidden dimension.  When *None* the default ``8/3 * d_model``
                  rounded up to the nearest multiple of 256 is used.
            dropout: Dropout applied after the gating multiplication.
            bias: Whether to use bias in linear projections.
        """
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # Round up to the nearest multiple of 256 for hardware efficiency.
            d_ff = ((d_ff + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Args:
            x: Input tensor of shape (B, S, d_model).

        Returns:
            Output tensor of shape (B, S, d_model).
        """
        # Gate path
        gate = F.silu(self.w_gate(x))
        # Value path
        value = self.w1(x)
        # Element-wise gating
        x = gate * value
        x = self.dropout(x)
        x = self.w2(x)
        return x
