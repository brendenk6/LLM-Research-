"""Expert Router for GENESIS Mixture-of-Experts layers.

Implements top-k expert routing with auxiliary load-balancing loss to ensure
even utilisation across experts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    """Top-k expert router with load-balancing loss.

    Given token representations, produces routing weights indicating which
    experts each token should be dispatched to, along with the corresponding
    soft combination weights.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
    ) -> None:
        """Initialise ExpertRouter.

        Args:
            d_model: Dimension of input representations.
            num_experts: Total number of experts to route across.
            top_k: Number of experts selected per token.
            jitter_noise: Multiplicative jitter noise for load balancing
                during training (0 = disabled).
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Cached values for load-balance loss
        self._last_router_probs: Optional[torch.Tensor] = None
        self._last_expert_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts.

        Args:
            x: Input tensor of shape (..., d_model).  Typically (B*S, d_model)
                after flattening batch and sequence dims.

        Returns:
            expert_weights: Softmax weights for selected experts, shape (..., top_k).
            expert_indices: Indices of selected experts, shape (..., top_k).
            router_logits: Raw router logits, shape (..., num_experts).
        """
        # Optional jitter noise during training for better load balancing
        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(x).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            x = x * noise

        # Router logits: (..., num_experts)
        router_logits = self.gate(x)

        # Full softmax for load-balance loss computation
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )

        # Normalise selected weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        top_k_weights = top_k_weights.type_as(x)

        # Cache for load-balance loss
        self._last_router_probs = router_probs.view(-1, self.num_experts)
        # Create one-hot expert mask for balance computation
        expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=-2)
        self._last_expert_mask = expert_mask.view(-1, self.num_experts).float()

        return top_k_weights, top_k_indices, router_logits

    def load_balance_loss(self) -> torch.Tensor:
        """Compute the auxiliary load-balancing loss.

        Uses the Switch Transformer formulation:
            loss = num_experts * sum_i(f_i * P_i)
        where f_i is the fraction of tokens routed to expert i and P_i is the
        mean router probability for expert i.

        Returns:
            Scalar loss tensor.
        """
        if self._last_router_probs is None or self._last_expert_mask is None:
            return torch.tensor(0.0)

        # f_i: fraction of tokens dispatched to each expert
        # Shape: (num_experts,)
        tokens_per_expert = self._last_expert_mask.float().mean(dim=0)

        # P_i: mean routing probability for each expert
        router_prob_per_expert = self._last_router_probs.float().mean(dim=0)

        # Switch Transformer load-balance loss
        loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()
        return loss
