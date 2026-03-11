"""Mixture-of-Experts Feed-Forward Network for GENESIS.

Wraps multiple SwiGLU FFN experts behind a learned top-k router.  Each
token is dispatched to its top-k experts and their outputs are combined
via the routing weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from genesis.model.ffn import SwiGLUFFN
from genesis.model.moe_router import ExpertRouter


class MoEFFN(nn.Module):
    """Mixture-of-Experts layer with SwiGLU expert networks.

    Each expert is an independent SwiGLU FFN.  A learned router dispatches
    each token to its top-k experts and the outputs are combined with the
    router weights.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        jitter_noise: float = 0.0,
    ) -> None:
        """Initialise MoEFFN.

        Args:
            d_model: Input and output dimension.
            d_ff: Hidden dimension per expert (defaults to SwiGLU heuristic).
            num_experts: Total number of expert FFN modules.
            top_k: Number of experts selected per token.
            capacity_factor: Multiplicative factor on expert capacity (used
                for potential token-dropping; currently informational).
            dropout: Dropout probability inside each expert FFN.
            jitter_noise: Router jitter noise for load balancing.
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = ExpertRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )

        self.experts = nn.ModuleList(
            [
                SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor of shape (B, S, d_model).

        Returns:
            Output tensor of shape (B, S, d_model).
        """
        B, S, D = x.shape

        # Flatten to (B*S, D) for routing
        x_flat = x.view(-1, D)

        # Route: get top-k expert weights and indices
        expert_weights, expert_indices, _ = self.router(x_flat)
        # expert_weights: (B*S, top_k)
        # expert_indices: (B*S, top_k)

        # Compute output by dispatching to selected experts
        # For efficiency, we batch tokens going to the same expert
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            indices_k = expert_indices[:, k]   # (B*S,)
            weights_k = expert_weights[:, k]   # (B*S,)

            for expert_idx in range(self.num_experts):
                # Find which tokens go to this expert for this k
                token_mask = (indices_k == expert_idx)
                if not token_mask.any():
                    continue

                expert_input = x_flat[token_mask]  # (num_tokens, D)
                expert_output = self.experts[expert_idx](expert_input)
                # Weight by routing score
                output[token_mask] += weights_k[token_mask].unsqueeze(-1) * expert_output

        return output.view(B, S, D)

    def load_balance_loss(self) -> torch.Tensor:
        """Return the router's load-balancing loss."""
        return self.router.load_balance_loss()
