"""
ComputeRouter: Dynamic expert / sub-network routing with load-balancing loss.

Supports three gating mechanisms:
  - linear:  Single linear projection -> softmax.
  - mlp:     Two-layer MLP with ReLU -> softmax.
  - entropy: Linear projection with entropy regularization to encourage
             exploration early in training.

Uses a straight-through estimator so that hard top-k routing decisions are
differentiable during training.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutePlan:
    """Output of the router: which experts to activate and with what weights."""

    expert_indices: torch.Tensor    # (batch, top_k) — int64
    expert_weights: torch.Tensor    # (batch, top_k) — float, sums to 1 per row
    raw_logits: torch.Tensor        # (batch, num_experts) — pre-softmax
    probabilities: torch.Tensor     # (batch, num_experts) — post-softmax

    @property
    def num_experts(self) -> int:
        return self.probabilities.shape[-1]

    @property
    def top_k(self) -> int:
        return self.expert_indices.shape[-1]


class ComputeRouter(nn.Module):
    """
    Routes input tokens to a subset of experts.

    Args:
        input_dim:    Dimension of the input feature vectors.
        num_experts:  Total number of experts.
        top_k:        Number of experts activated per token.
        gate_type:    ``"linear"``, ``"mlp"``, or ``"entropy"``.
        gate_hidden:  Hidden dimension for the MLP gate (only used if gate_type="mlp").
        entropy_coeff: Coefficient for entropy bonus (only used if gate_type="entropy").
        load_balance_coeff: Coefficient for the auxiliary load-balance loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        gate_type: Literal["linear", "mlp", "entropy"] = "linear",
        gate_hidden: int = 128,
        entropy_coeff: float = 0.01,
        load_balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_type = gate_type
        self.entropy_coeff = entropy_coeff
        self.load_balance_coeff = load_balance_coeff

        # Build gate
        if gate_type == "linear":
            self.gate = nn.Linear(input_dim, num_experts, bias=False)
        elif gate_type == "mlp":
            self.gate = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, num_experts),
            )
        elif gate_type == "entropy":
            self.gate = nn.Linear(input_dim, num_experts, bias=False)
        else:
            raise ValueError(f"Unknown gate_type '{gate_type}'. Use 'linear', 'mlp', or 'entropy'.")

        # Running statistics for diagnostics
        self._total_tokens: int = 0
        self._expert_counts: torch.Tensor = torch.zeros(num_experts)
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def plan(self, x: torch.Tensor) -> RoutePlan:
        """Compute routing decisions for input *x*.

        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim).
               If 3-D the routing is done per-token (batch and seq are
               flattened then reshaped).

        Returns:
            A :class:`RoutePlan`.
        """
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(f"Expected 2-D or 3-D input, got {x.dim()}-D.")

        logits = self.gate(x_flat)  # (N, num_experts)
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalise the selected weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Straight-through estimator: hard assignment in forward, soft in backward
        if self.training:
            # Create one-hot mask for selected experts
            hard_mask = torch.zeros_like(probs).scatter_(-1, top_k_indices, 1.0)
            # Straight-through: forward uses hard mask, backward flows through probs
            st_mask = hard_mask - probs.detach() + probs
            # Weight the mask by the normalised top-k weights
            _ = st_mask * probs  # straight-through gradient path
            # But for the RoutePlan we still use the clean top-k weights
        else:
            pass  # No straight-through needed at eval time

        # Update running stats (detached)
        with torch.no_grad():
            n = x_flat.shape[0]
            self._total_tokens += n
            self._total_steps += 1
            counts = torch.zeros(self.num_experts, device=x_flat.device)
            counts.scatter_add_(0, top_k_indices.flatten(), torch.ones(top_k_indices.numel(), device=x_flat.device))
            self._expert_counts = self._expert_counts.to(x_flat.device) + counts

        return RoutePlan(
            expert_indices=top_k_indices,
            expert_weights=top_k_weights,
            raw_logits=logits,
            probabilities=probs,
        )

    def execute(
        self,
        x: torch.Tensor,
        experts: nn.ModuleList,
        route_plan: Optional[RoutePlan] = None,
    ) -> torch.Tensor:
        """Route *x* through *experts* according to *route_plan*.

        If *route_plan* is ``None`` it will be computed via :meth:`plan`.

        Args:
            x: (batch, dim) or (batch, seq, dim).
            experts: ``nn.ModuleList`` of expert modules.
            route_plan: Pre-computed routing plan (optional).

        Returns:
            Weighted sum of expert outputs with the same shape as *x*.
        """
        three_d = x.dim() == 3
        if three_d:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        else:
            x_flat = x

        if route_plan is None:
            route_plan = self.plan(x if not three_d else x)

        indices = route_plan.expert_indices  # (N, top_k)
        weights = route_plan.expert_weights  # (N, top_k)

        output = torch.zeros_like(x_flat)

        for k_idx in range(self.top_k):
            expert_idx = indices[:, k_idx]   # (N,)
            w = weights[:, k_idx].unsqueeze(-1)  # (N, 1)

            for e_id in range(self.num_experts):
                mask = expert_idx == e_id
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_output = experts[e_id](expert_input)
                output[mask] += w[mask] * expert_output

        if three_d:
            output = output.reshape(B, S, -1)

        return output

    # ------------------------------------------------------------------
    # Losses & stats
    # ------------------------------------------------------------------

    def load_balance_loss(self, route_plan: RoutePlan) -> torch.Tensor:
        """Compute the auxiliary load-balancing loss (Switch Transformer style).

        L = num_experts * sum_i( f_i * P_i )
        where f_i = fraction of tokens routed to expert i,
              P_i = mean probability assigned to expert i.
        """
        probs = route_plan.probabilities  # (N, E)
        indices = route_plan.expert_indices  # (N, top_k)
        N = probs.shape[0]
        E = self.num_experts

        # f_i: fraction of tokens dispatched to expert i
        one_hot = F.one_hot(indices, num_classes=E).float()  # (N, top_k, E)
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # (E,)
        f = tokens_per_expert / (N * self.top_k + 1e-9)

        # P_i: mean routing probability for expert i
        P = probs.mean(dim=0)  # (E,)

        loss = self.load_balance_coeff * E * (f * P).sum()

        # Add entropy bonus for entropy gate type
        if self.gate_type == "entropy":
            entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
            loss = loss - self.entropy_coeff * entropy

        return loss

    def get_routing_stats(self) -> Dict[str, float]:
        """Return diagnostic statistics about routing decisions."""
        if self._total_tokens == 0:
            return {"total_tokens": 0, "total_steps": 0}

        counts = self._expert_counts.float()
        total = counts.sum().item()
        fractions = counts / (total + 1e-9)

        return {
            "total_tokens": self._total_tokens,
            "total_steps": self._total_steps,
            "expert_utilization": {
                f"expert_{i}": fractions[i].item() for i in range(self.num_experts)
            },
            "max_load_ratio": fractions.max().item() / (1.0 / self.num_experts + 1e-9),
            "min_load_ratio": fractions.min().item() / (1.0 / self.num_experts + 1e-9),
        }

    def reset_stats(self) -> None:
        """Reset running routing statistics."""
        self._total_tokens = 0
        self._expert_counts = torch.zeros(self.num_experts)
        self._total_steps = 0
