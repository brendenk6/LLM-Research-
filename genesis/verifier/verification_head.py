"""Verification Head for ACT-V.

Multi-task classification heads that score text along four quality
dimensions: factual consistency, logical coherence, stylistic match,
and an overall weighted combination.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VerificationHead(nn.Module):
    """Multi-task verification scoring heads.

    Each head is a ``Linear(d_model, 1) + Sigmoid`` that maps the pooled
    verifier representation to a quality score in [0, 1].

    Heads:
        - ``factual_score``:   Factual consistency with source material.
        - ``logical_score``:   Logical coherence of the generated text.
        - ``stylistic_score``: Stylistic match with the target domain.
        - ``overall_score``:   Weighted combination of the above.

    The ``overall_score`` head is *learned* (not a fixed combination) so
    it can capture interactions that a simple weighted average cannot.
    """

    # Default weights used when combining individual scores into the
    # overall target during loss computation.
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "factual": 0.4,
        "logical": 0.35,
        "stylistic": 0.25,
    }

    def __init__(self, d_model: int = 512) -> None:
        """Initialise VerificationHead.

        Args:
            d_model: Dimension of the pooled input representation.
        """
        super().__init__()
        self.d_model = d_model

        # Individual scoring heads
        self.factual_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.logical_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.stylistic_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.overall_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for all head projections."""
        for module in [self.factual_head, self.logical_head,
                       self.stylistic_head, self.overall_head]:
            linear = module[0]
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, pooled_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute verification scores.

        Args:
            pooled_hidden: (B, d_model) pooled representation from the
                VerifierModel.

        Returns:
            Dict with keys ``factual_score``, ``logical_score``,
            ``stylistic_score``, ``overall_score`` each of shape (B, 1)
            with values in [0, 1].
        """
        factual = self.factual_head(pooled_hidden)      # (B, 1)
        logical = self.logical_head(pooled_hidden)      # (B, 1)
        stylistic = self.stylistic_head(pooled_hidden)  # (B, 1)
        overall = self.overall_head(pooled_hidden)      # (B, 1)

        return {
            "factual_score": factual,
            "logical_score": logical,
            "stylistic_score": stylistic,
            "overall_score": overall,
        }

    def compute_loss(
        self,
        scores: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-task BCE loss across all heads.

        Args:
            scores: Output of :meth:`forward`.  Each value has shape (B, 1).
            labels: Dict with matching keys, each a target tensor of shape
                (B, 1) with values in [0, 1].

                If ``overall_score`` is not present in *labels*, it is
                automatically computed as a weighted combination of the
                individual label scores using :attr:`DEFAULT_WEIGHTS`.

        Returns:
            Scalar loss tensor (sum of per-head BCE losses).
        """
        loss = torch.tensor(0.0, device=next(iter(scores.values())).device)

        # Per-head losses
        for key in ("factual_score", "logical_score", "stylistic_score"):
            if key in labels:
                loss = loss + F.binary_cross_entropy(
                    scores[key], labels[key].to(scores[key].dtype),
                )

        # Overall score: derive label from weighted combination if absent
        if "overall_score" in labels:
            overall_label = labels["overall_score"]
        else:
            overall_label = (
                self.DEFAULT_WEIGHTS["factual"] * labels.get(
                    "factual_score",
                    torch.zeros_like(scores["overall_score"]),
                )
                + self.DEFAULT_WEIGHTS["logical"] * labels.get(
                    "logical_score",
                    torch.zeros_like(scores["overall_score"]),
                )
                + self.DEFAULT_WEIGHTS["stylistic"] * labels.get(
                    "stylistic_score",
                    torch.zeros_like(scores["overall_score"]),
                )
            )

        loss = loss + F.binary_cross_entropy(
            scores["overall_score"],
            overall_label.to(scores["overall_score"].dtype),
        )

        return loss
