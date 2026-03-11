"""
ACT-V (Adversarial Co-Training with Verifier) Training Integration.

Coordinates the adversarial co-training loop between the Generator (G)
and Verifier (V). The Generator is trained on NTP loss augmented with
Verifier feedback, while the Verifier is trained to distinguish real
from corrupted text via binary cross-entropy. Periodic distillation
transfers Verifier knowledge into Generator representations.

Components:
- VerifierModel: Transformer encoder scoring text quality.
- VerificationHead: Multi-task heads (factual, logical, stylistic).
- NegativeGenerator: Produces corrupted samples for Verifier training.
- ReplayBuffer: Stores historical outputs for training stability.
- VerificationDistillation: Aligns G and V representations via MSE + KL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.core.training_orchestrator import (
    TrainingOrchestrator,
    ModelConfig,
)
from genesis.verifier.verifier_model import VerifierModel
from genesis.verifier.verification_head import VerificationHead
from genesis.verifier.negative_generator import NegativeGenerator
from genesis.verifier.replay_buffer import ReplayBuffer
from genesis.verifier.distillation import VerificationDistillation

logger = logging.getLogger(__name__)


@dataclass
class ACTVConfig:
    """Configuration for ACT-V adversarial co-training.

    Attributes:
        alpha: Weight of Verifier feedback in the Generator loss.
        distill_interval: Steps between distillation rounds.
        replay_max_size: Maximum replay buffer capacity.
        replay_min_size: Minimum entries before replay sampling begins.
        replay_batch_ratio: Fraction of batch drawn from replay buffer.
        distillation_shared_dim: Projection dimension for distillation.
        distillation_alpha: Interpolation between MSE and KL in distillation.
        verifier_lr_mult: LR multiplier for Verifier relative to Generator.
        gradient_accumulation_steps: Micro-batches per optimizer step.
        max_grad_norm: Gradient clipping threshold.
        device: Device string.
    """
    alpha: float = 0.1
    distill_interval: int = 1000
    replay_max_size: int = 10_000
    replay_min_size: int = 100
    replay_batch_ratio: float = 0.25
    distillation_shared_dim: int = 256
    distillation_alpha: float = 0.5
    verifier_lr_mult: float = 1.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cpu"


class ACTVTrainer:
    """Adversarial Co-Training with Verifier.

    Manages the training loop where:
      - The **Generator** is optimised on NTP + alpha * verifier_feedback.
      - The **Verifier** is trained on BCE over real vs. corrupted text.
      - Periodic **distillation** transfers V knowledge into G.

    Args:
        generator: Generator model (forward returns dict with ``"logits"``
            and optionally ``"hidden_states"``).
        verifier: VerifierModel backbone.
        neg_generator: NegativeGenerator for producing corrupted samples.
        replay_buffer: ReplayBuffer for historical Generator outputs.
        config: ACTVConfig with training hyperparameters.
    """

    def __init__(
        self,
        generator: nn.Module,
        verifier: VerifierModel,
        neg_generator: NegativeGenerator,
        replay_buffer: ReplayBuffer,
        config: ACTVConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Models
        self.generator = generator.to(self.device)
        self.verifier = verifier.to(self.device)
        self.neg_generator = neg_generator
        self.replay_buffer = replay_buffer

        # Verification head
        self.verification_head = VerificationHead(
            d_model=verifier.d_model,
        ).to(self.device)

        # Distillation module
        gen_dim = _get_hidden_dim(generator)
        self.distillation = VerificationDistillation(
            generator_dim=gen_dim,
            verifier_dim=verifier.d_model,
            shared_dim=config.distillation_shared_dim,
            alpha=config.distillation_alpha,
        ).to(self.device)

        # Optimizers (defaults; callers may override via config)
        self.generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=3e-4, weight_decay=0.01,
        )
        verifier_params = (
            list(self.verifier.parameters())
            + list(self.verification_head.parameters())
        )
        self.verifier_optimizer = torch.optim.AdamW(
            verifier_params,
            lr=3e-4 * config.verifier_lr_mult,
            weight_decay=0.01,
        )

        # Build orchestrator
        self.orchestrator = TrainingOrchestrator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device=self.device,
        )
        self.orchestrator.add_model(ModelConfig(
            name="generator",
            model=self.generator,
            optimizer=self.generator_optimizer,
            max_grad_norm=config.max_grad_norm,
        ))
        self.orchestrator.add_model(ModelConfig(
            name="verifier",
            model=self.verifier,
            optimizer=self.verifier_optimizer,
            max_grad_norm=config.max_grad_norm,
        ))

        self._step_count: int = 0

        g_params = sum(p.numel() for p in self.generator.parameters())
        v_params = sum(p.numel() for p in self.verifier.parameters())
        logger.info(
            "ACTVTrainer: G=%.2fM, V=%.2fM, alpha=%.3f",
            g_params / 1e6, v_params / 1e6, config.alpha,
        )

    # ------------------------------------------------------------------
    # Verifier training
    # ------------------------------------------------------------------

    def train_verifier_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Train Verifier on real (positive) + corrupted (negative) samples.

        Args:
            batch: Dict with ``"input_ids"`` (B, S).

        Returns:
            Dict with ``"pos_loss"``, ``"neg_loss"``, ``"replay_loss"``,
            ``"verifier_loss"``.
        """
        input_ids = batch["input_ids"].to(self.device)
        self.verifier.train()
        self.verification_head.train()

        # --- Positive examples ---
        pos_out = self.verifier(input_ids)
        pos_scores = self.verification_head(pos_out["pooled_output"])
        pos_labels = {
            k: torch.ones_like(v) for k, v in pos_scores.items()
            if k.endswith("_score")
        }
        pos_loss = self.verification_head.compute_loss(pos_scores, pos_labels)

        # --- Negative examples ---
        corrupted_ids, _ = self.neg_generator.batch_corrupt(input_ids)
        corrupted_ids = corrupted_ids.to(self.device)

        neg_out = self.verifier(corrupted_ids)
        neg_scores = self.verification_head(neg_out["pooled_output"])
        neg_labels = {
            k: torch.zeros_like(v) for k, v in neg_scores.items()
            if k.endswith("_score")
        }
        neg_loss = self.verification_head.compute_loss(neg_scores, neg_labels)

        # --- Replay buffer ---
        replay_loss = torch.tensor(0.0, device=self.device)
        if self.replay_buffer.is_ready(self.config.replay_min_size):
            n_replay = max(1, int(input_ids.size(0) * self.config.replay_batch_ratio))
            samples = self.replay_buffer.sample(n_replay)
            if samples:
                replay_ids = torch.stack(
                    [s["input_ids"] for s in samples]
                ).to(self.device)
                # Pad/truncate to match sequence length
                target_len = input_ids.size(1)
                if replay_ids.size(1) > target_len:
                    replay_ids = replay_ids[:, :target_len]
                elif replay_ids.size(1) < target_len:
                    pad = torch.zeros(
                        replay_ids.size(0), target_len - replay_ids.size(1),
                        dtype=replay_ids.dtype, device=self.device,
                    )
                    replay_ids = torch.cat([replay_ids, pad], dim=1)
                r_out = self.verifier(replay_ids)
                r_scores = self.verification_head(r_out["pooled_output"])
                r_labels = {
                    k: torch.stack([s["scores"][k] for s in samples]).to(self.device)
                    for k in ("factual_score", "logical_score", "stylistic_score")
                }
                replay_loss = self.verification_head.compute_loss(r_scores, r_labels)

        total = (pos_loss + neg_loss + replay_loss) / (
            3.0 if replay_loss.item() > 0 else 2.0
        )

        self.verifier_optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(
            list(self.verifier.parameters()) + list(self.verification_head.parameters()),
            self.config.max_grad_norm,
        )
        self.verifier_optimizer.step()

        return {
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "replay_loss": replay_loss.item(),
            "verifier_loss": total.item(),
        }

    # ------------------------------------------------------------------
    # Generator training
    # ------------------------------------------------------------------

    def train_generator_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Train Generator with NTP loss + Verifier feedback.

        Args:
            batch: Dict with ``"input_ids"`` (B, S) and ``"labels"`` (B, S).

        Returns:
            Dict with ``"ntp_loss"``, ``"feedback_loss"``,
            ``"generator_loss"``, and ``"v_scores"`` (dict).
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        self.generator.train()

        # NTP loss
        gen_out = self.generator(input_ids)
        logits = gen_out["logits"] if isinstance(gen_out, dict) else gen_out
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ntp_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Verifier feedback (no gradient through V)
        with torch.no_grad():
            v_out = self.verifier(input_ids)
            v_scores = self.verification_head(v_out["pooled_output"])
            overall = v_scores["overall_score"]  # (B, 1)
        feedback_loss = (1.0 - overall).mean()

        # Store individual sequences in replay buffer
        for i in range(input_ids.size(0)):
            seq_scores = {k: v[i] for k, v in v_scores.items()}
            self.replay_buffer.add(
                input_ids=input_ids[i],
                scores=seq_scores,
                metadata={"step": self._step_count},
            )

        total = ntp_loss + self.config.alpha * feedback_loss

        self.generator_optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(
            self.generator.parameters(), self.config.max_grad_norm,
        )
        self.generator_optimizer.step()

        mean_scores = {k: v.mean().item() for k, v in v_scores.items()}

        return {
            "ntp_loss": ntp_loss.item(),
            "feedback_loss": feedback_loss.item(),
            "generator_loss": total.item(),
            "v_scores": mean_scores,
        }

    # ------------------------------------------------------------------
    # Co-training step
    # ------------------------------------------------------------------

    def co_training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """One full ACT-V co-training iteration.

        Alternates between Verifier and Generator training, and
        periodically runs distillation.

        Args:
            batch: Dict with ``"input_ids"`` (B, S) and ``"labels"`` (B, S).

        Returns:
            Combined metrics from both V and G steps, plus distillation
            loss if applicable.
        """
        # Train Verifier
        v_metrics = self.train_verifier_step(batch)

        # Train Generator
        g_metrics = self.train_generator_step(batch)

        self._step_count += 1

        metrics: Dict[str, Any] = {
            **{f"v_{k}": v for k, v in v_metrics.items()},
            **{f"g_{k}": v for k, v in g_metrics.items()},
            "step": self._step_count,
        }

        # Periodic distillation
        if (
            self._step_count > 0
            and self._step_count % self.config.distill_interval == 0
        ):
            dist_loss = self._distill()
            if dist_loss is not None:
                metrics["distillation_loss"] = dist_loss
                logger.info(
                    "Distillation at step %d: loss=%.6f",
                    self._step_count, dist_loss,
                )

        return metrics

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    def _distill(self) -> Optional[float]:
        """Transfer Verifier knowledge into Generator via distillation.

        Returns:
            Distillation loss value, or None if replay buffer not ready.
        """
        if not self.replay_buffer.is_ready(self.config.replay_min_size):
            return None

        samples = self.replay_buffer.sample(
            min(32, self.replay_buffer.size),
        )
        if not samples:
            return None

        input_ids = torch.stack(
            [s["input_ids"] for s in samples]
        ).to(self.device)

        # Generator forward (with gradient)
        self.generator.train()
        gen_out = self.generator(input_ids)
        if isinstance(gen_out, dict) and "hidden_states" in gen_out:
            gen_hidden = gen_out["hidden_states"]
        elif isinstance(gen_out, dict):
            gen_hidden = gen_out["logits"]
        else:
            gen_hidden = gen_out

        # Verifier forward (frozen)
        with torch.no_grad():
            v_out = self.verifier(input_ids)
            v_hidden = v_out["hidden_states"]

        # Mean-pool to (B, dim)
        gen_pooled = gen_hidden.mean(dim=1) if gen_hidden.dim() == 3 else gen_hidden
        v_pooled = v_hidden.mean(dim=1) if v_hidden.dim() == 3 else v_hidden

        dist_loss = self.distillation(gen_pooled, v_pooled)

        self.generator_optimizer.zero_grad(set_to_none=True)
        dist_loss.backward()
        nn.utils.clip_grad_norm_(
            self.generator.parameters(), self.config.max_grad_norm,
        )
        self.generator_optimizer.step()

        return dist_loss.item()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save full ACT-V checkpoint via the orchestrator."""
        self.orchestrator.save_checkpoint(path)
        logger.info("ACT-V checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load ACT-V checkpoint via the orchestrator."""
        self.orchestrator.load_checkpoint(path)
        logger.info("ACT-V checkpoint loaded: %s", path)


# ======================================================================
# Helpers
# ======================================================================

def _get_hidden_dim(model: nn.Module) -> int:
    """Infer hidden dimension from a model via common attribute names."""
    for attr in ("d_model", "hidden_size", "config"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, int):
                return val
            if hasattr(val, "d_model"):
                return val.d_model
            if hasattr(val, "hidden_size"):
                return val.hidden_size
    for p in model.parameters():
        if p.dim() >= 2:
            return p.size(-1)
    return 512
