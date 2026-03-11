"""Inference-time cascade routing for HLRT.

At inference, the tier gates produce hard boolean masks.  The CascadeRouter
wraps this logic and tracks activation statistics so callers can monitor
how much compute each tier is consuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CascadeStats:
    """Tracks tier activation rates across generation steps."""

    tier1_tokens: int = 0
    tier2_tokens: int = 0
    tier3_tokens: int = 0
    total_tokens: int = 0

    def update(
        self,
        num_tokens: int,
        tier2_activated: bool,
        tier3_activated: bool,
    ) -> None:
        """Record activation for a generation step."""
        self.total_tokens += num_tokens
        self.tier1_tokens += num_tokens  # Tier 1 always runs
        if tier2_activated:
            self.tier2_tokens += num_tokens
        if tier3_activated:
            self.tier3_tokens += num_tokens

    @property
    def tier1_rate(self) -> float:
        return 1.0  # always active

    @property
    def tier2_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.tier2_tokens / self.total_tokens

    @property
    def tier3_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.tier3_tokens / self.total_tokens

    @property
    def compute_savings(self) -> float:
        """Estimated compute savings vs always running all tiers.

        Returns fraction of compute saved (0.0 = no savings, 1.0 = max savings).
        """
        if self.total_tokens == 0:
            return 0.0
        # Rough cost model: Tier1=1x, Tier2=3x, Tier3=5x
        max_cost = self.total_tokens * (1 + 3 + 5)
        actual_cost = (
            self.tier1_tokens * 1
            + self.tier2_tokens * 3
            + self.tier3_tokens * 5
        )
        return 1.0 - (actual_cost / max_cost)

    def reset(self) -> None:
        self.tier1_tokens = 0
        self.tier2_tokens = 0
        self.tier3_tokens = 0
        self.total_tokens = 0

    def summary(self) -> str:
        return (
            f"Cascade Stats: {self.total_tokens} tokens | "
            f"T1: 100% | T2: {self.tier2_rate:.1%} | T3: {self.tier3_rate:.1%} | "
            f"Savings: {self.compute_savings:.1%}"
        )


class CascadeRouter:
    """Manages tier escalation decisions at inference time.

    Wraps the HLRT's tier gates and provides:
    - Hard routing decisions (eval mode)
    - Activation statistics tracking
    - Optional threshold overrides for experimentation
    """

    def __init__(
        self,
        gate1_threshold: float | None = None,
        gate2_threshold: float | None = None,
    ) -> None:
        """Initialize CascadeRouter.

        Args:
            gate1_threshold: Override threshold for Gate 1->2.
                None uses the model's default.
            gate2_threshold: Override threshold for Gate 2->3.
                None uses the model's default.
        """
        self.gate1_threshold = gate1_threshold
        self.gate2_threshold = gate2_threshold
        self.stats = CascadeStats()

    def should_escalate(
        self,
        gate_scores: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, bool]:
        """Determine which chunks should escalate to the next tier.

        Args:
            gate_scores: (B, num_chunks) sigmoid gate scores.
            threshold: Escalation threshold.

        Returns:
            Tuple of (bool mask, whether any chunk was escalated).
        """
        mask = gate_scores >= threshold
        any_escalated = mask.any().item()
        return mask, any_escalated

    def route_gate1(
        self,
        gate_scores: torch.Tensor,
        model_threshold: float,
    ) -> tuple[torch.Tensor, bool]:
        """Route through Gate 1 (Tier 1 -> Tier 2).

        Args:
            gate_scores: (B, num_chunks) from TierGate.
            model_threshold: The model's configured threshold.

        Returns:
            (mask, any_escalated).
        """
        threshold = self.gate1_threshold or model_threshold
        return self.should_escalate(gate_scores, threshold)

    def route_gate2(
        self,
        gate_scores: torch.Tensor,
        model_threshold: float,
    ) -> tuple[torch.Tensor, bool]:
        """Route through Gate 2 (Tier 2 -> Tier 3).

        Args:
            gate_scores: (B, num_chunks) from TierGate.
            model_threshold: The model's configured threshold.

        Returns:
            (mask, any_escalated).
        """
        threshold = self.gate2_threshold or model_threshold
        return self.should_escalate(gate_scores, threshold)

    def record(
        self,
        num_tokens: int,
        tier2_activated: bool,
        tier3_activated: bool,
    ) -> None:
        """Record tier activation for statistics."""
        self.stats.update(num_tokens, tier2_activated, tier3_activated)
