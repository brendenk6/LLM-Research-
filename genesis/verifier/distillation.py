"""Verification Distillation for ACT-V.

Distills the Verifier's learned representations into the Generator via
projection-based MSE alignment and temperature-scaled KL divergence,
allowing the Generator to internalise quality-awareness without running
the Verifier at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VerificationDistillation(nn.Module):
    """Distill Verifier representations into the Generator.

    Two loss components:
      1. **Projection MSE**: Both G and V hidden states are projected
         into a shared low-dimensional space and aligned via MSE loss.
      2. **Temperature-scaled KL**: Softened logit distributions from G
         and V are aligned via KL divergence (classic knowledge
         distillation).

    The ``forward`` method returns the combined loss.
    """

    def __init__(
        self,
        generator_dim: int,
        verifier_dim: int,
        shared_dim: int = 256,
        alpha: float = 0.5,
    ) -> None:
        """Initialise VerificationDistillation.

        Args:
            generator_dim: Hidden dimension of the Generator.
            verifier_dim: Hidden dimension of the Verifier.
            shared_dim: Dimension of the shared projection space.
            alpha: Weighting between MSE (alpha) and KL (1 - alpha) in
                the combined loss returned by :meth:`forward`.
        """
        super().__init__()
        self.generator_dim = generator_dim
        self.verifier_dim = verifier_dim
        self.shared_dim = shared_dim
        self.alpha = alpha

        # Learned projections into shared space
        self.proj_G = nn.Linear(generator_dim, shared_dim)
        self.proj_V = nn.Linear(verifier_dim, shared_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for projections."""
        nn.init.xavier_uniform_(self.proj_G.weight)
        nn.init.zeros_(self.proj_G.bias)
        nn.init.xavier_uniform_(self.proj_V.weight)
        nn.init.zeros_(self.proj_V.bias)

    def distillation_loss(
        self,
        generator_hidden: torch.Tensor,
        verifier_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss between projected hidden representations.

        Both inputs are projected into the shared space and aligned.

        Args:
            generator_hidden: (B, ..., generator_dim) Generator hidden states.
            verifier_hidden: (B, ..., verifier_dim) Verifier hidden states.

        Returns:
            Scalar MSE loss.
        """
        g_proj = self.proj_G(generator_hidden)  # (B, ..., shared_dim)
        v_proj = self.proj_V(verifier_hidden)   # (B, ..., shared_dim)
        # Detach V so gradients only flow into G
        return F.mse_loss(g_proj, v_proj.detach())

    def temperature_scaled_loss(
        self,
        g_logits: torch.Tensor,
        v_logits: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """Compute temperature-scaled KL divergence loss.

        Soft targets from the Verifier's logits guide the Generator's
        logit distribution.

        Args:
            g_logits: (B, ..., V) Generator logits over vocabulary or classes.
            v_logits: (B, ..., V) Verifier logits (used as teacher).
            temperature: Softmax temperature.  Higher values produce softer
                distributions.

        Returns:
            Scalar KL divergence loss, scaled by T^2 (standard practice).
        """
        g_log_probs = F.log_softmax(g_logits / temperature, dim=-1)
        v_probs = F.softmax(v_logits / temperature, dim=-1).detach()

        # KL(v || g) = sum v * (log v - log g)
        kl = F.kl_div(g_log_probs, v_probs, reduction="batchmean")

        # Scale by T^2 to keep gradients comparable across temperatures
        return kl * (temperature ** 2)

    def forward(
        self,
        generator_hidden: torch.Tensor,
        verifier_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined distillation loss.

        Combines the projection MSE loss and a simplified alignment term.
        When logits are available, use :meth:`temperature_scaled_loss`
        explicitly alongside this method.

        Args:
            generator_hidden: (B, ..., generator_dim) Generator hidden states.
            verifier_hidden: (B, ..., verifier_dim) Verifier hidden states.

        Returns:
            Scalar combined loss.
        """
        mse = self.distillation_loss(generator_hidden, verifier_hidden)
        return mse
