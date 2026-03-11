"""
Phase 2: RL Pretraining with GRPO (Group Relative Policy Optimization).

Builds on a Phase 1 checkpoint to train the model to produce useful
reasoning traces. Mixes standard NTP loss with GRPO policy gradient
at a configurable ratio (default 80% NTP + 20% RL) to maintain
language modelling quality while developing reasoning capabilities.

Reference: DeepSeek-R1 (2024) - https://arxiv.org/abs/2401.02954
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genesis.model.hlrt import HLRT
from genesis.training.grpo import GRPOTrainer, GRPOConfig
from olympus.core.training_orchestrator import (
    TrainingOrchestrator,
    ModelConfig,
)
from olympus.optim.muon_adamw_hybrid import MuonAdamWHybrid

logger = logging.getLogger(__name__)


# ======================================================================
# NTP loss (for the NTP component of the mixed objective)
# ======================================================================

def _ntp_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Compute standard next-token prediction loss.

    Args:
        model: Language model returning dict with ``"logits"`` and
            optional ``"aux_loss"``.
        input_ids: Token IDs of shape (B, S).

    Returns:
        Tuple of (scalar loss, perplexity float).
    """
    out = model(input_ids)
    logits = out["logits"]
    aux_loss = out.get("aux_loss", 0.0)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
    )

    if isinstance(aux_loss, torch.Tensor):
        loss = loss + aux_loss
    elif isinstance(aux_loss, (int, float)) and aux_loss > 0:
        loss = loss + aux_loss

    ppl = min(math.exp(loss.item()), 1e6) if loss.item() < 20 else 1e6
    return loss, ppl


# ======================================================================
# RLPretrainer
# ======================================================================

class RLPretrainer:
    """Phase 2: RL Pretraining mixing NTP with GRPO policy gradient.

    Loads a Phase 1 bootstrap checkpoint and continues training with a
    mixed objective:  ``(1 - rl_mix) * NTP_loss + rl_mix * GRPO_loss``.

    Args:
        model: The HLRT model to train.
        ref_model: Frozen copy of the model for KL penalty in GRPO.
        reward_fn: Callable ``(prompt, completion) -> float`` for GRPO.
        optimizer: MuonAdamWHybrid optimizer.
        config: Training configuration dict with keys:
            - rl_mix (float): RL loss weight, default 0.2.
            - gradient_accumulation_steps (int)
            - max_grad_norm (float)
            - grpo_group_size (int): Completions per prompt, default 8.
            - grpo_kl_coeff (float): KL penalty coefficient, default 0.1.
            - grpo_clip_range (float): PPO clip epsilon, default 0.2.
            - grpo_max_tokens (int): Max completion length, default 256.
            - log_interval (int)
            - save_interval (int)
            - checkpoint_dir (str)
            - device (str)
    """

    def __init__(
        self,
        model: HLRT,
        ref_model: HLRT,
        reward_fn: Callable[[str, str], float],
        optimizer: MuonAdamWHybrid,
        config: dict,
    ) -> None:
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.optimizer = optimizer
        self.reward_fn = reward_fn

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Mix ratio: 80% NTP + 20% RL by default
        self.rl_mix: float = config.get("rl_mix", 0.2)
        self.ntp_mix: float = 1.0 - self.rl_mix

        # Build GRPO trainer
        grpo_cfg = GRPOConfig(
            group_size=config.get("grpo_group_size", 8),
            kl_coeff=config.get("grpo_kl_coeff", 0.1),
            clip_range=config.get("grpo_clip_range", 0.2),
            max_completion_tokens=config.get("grpo_max_tokens", 256),
        )
        self.grpo = GRPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_fn=self.reward_fn,
            config=grpo_cfg,
        )

        # Build orchestrator for gradient accumulation
        grad_accum = config.get("gradient_accumulation_steps", 4)
        self.orchestrator = TrainingOrchestrator(
            gradient_accumulation_steps=grad_accum,
            device=self.device,
        )
        self.orchestrator.add_model(ModelConfig(
            name="hlrt",
            model=self.model,
            optimizer=self.optimizer,
            max_grad_norm=config.get("max_grad_norm", 1.0),
            enabled_phases=["rl_pretrain"],
        ))
        self.orchestrator.set_phase("rl_pretrain")

        # Reward statistics tracking
        self._reward_history: List[float] = []
        self._step_count: int = 0

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "RLPretrainer initialised: %.2fM params, rl_mix=%.2f, ntp_mix=%.2f",
            param_count / 1e6, self.rl_mix, self.ntp_mix,
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single mixed NTP + GRPO training step.

        Args:
            batch: Dict with ``"input_ids"`` (B, S) for NTP and
                ``"prompts"`` (list of str) + ``"tokenizer"`` for GRPO.

        Returns:
            Dict with ``"ntp_loss"``, ``"rl_loss"``, ``"total_loss"``,
            ``"perplexity"``, ``"avg_reward"``, ``"reward_std"``.
        """
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)

        # --- NTP component ---
        ntp_loss, perplexity = _ntp_loss(self.model, input_ids)

        # --- GRPO component ---
        grpo_metrics = self.grpo.step(batch)
        rl_loss_val = grpo_metrics.get("loss", 0.0)

        # --- Mixed loss ---
        # GRPO step already performed its own backward; we scale NTP
        # accordingly and let the orchestrator handle accumulation.
        mixed_loss = self.ntp_mix * ntp_loss
        scaled = mixed_loss / self.orchestrator.gradient_accumulation_steps
        scaled.backward()

        # Track reward statistics
        avg_reward = grpo_metrics.get("reward_mean", 0.0)
        self._reward_history.append(avg_reward)

        self._step_count += 1

        return {
            "ntp_loss": ntp_loss.item(),
            "rl_loss": rl_loss_val,
            "total_loss": self.ntp_mix * ntp_loss.item() + self.rl_mix * rl_loss_val,
            "perplexity": perplexity,
            "avg_reward": avg_reward,
            "reward_std": grpo_metrics.get("reward_std", 0.0),
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one full epoch.

        Args:
            dataloader: Yields dicts with ``"input_ids"``, ``"prompts"``,
                and ``"tokenizer"``.

        Returns:
            Dict with epoch-level aggregated metrics.
        """
        epoch_ntp = 0.0
        epoch_rl = 0.0
        epoch_reward = 0.0
        num_steps = 0
        log_interval = self.config.get("log_interval", 100)
        save_interval = self.config.get("save_interval", 5000)
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints/phase2")

        epoch_start = time.time()

        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_ntp += metrics["ntp_loss"]
            epoch_rl += metrics["rl_loss"]
            epoch_reward += metrics["avg_reward"]
            num_steps += 1

            global_step = self.orchestrator.global_step

            if num_steps % log_interval == 0:
                logger.info(
                    "Step %d | ntp=%.4f | rl=%.4f | ppl=%.2f | "
                    "reward=%.4f +/- %.4f",
                    global_step,
                    metrics["ntp_loss"],
                    metrics["rl_loss"],
                    metrics["perplexity"],
                    metrics["avg_reward"],
                    metrics["reward_std"],
                )

            if save_interval > 0 and global_step > 0 and global_step % save_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.orchestrator.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
                )

        n = max(num_steps, 1)
        avg_ntp = epoch_ntp / n
        avg_ppl = min(math.exp(avg_ntp), 1e6) if avg_ntp < 20 else 1e6
        elapsed = time.time() - epoch_start

        logger.info(
            "Phase 2 epoch done | ntp=%.4f | rl=%.4f | ppl=%.2f | "
            "reward=%.4f | %d steps in %.1fs",
            avg_ntp, epoch_rl / n, avg_ppl, epoch_reward / n,
            num_steps, elapsed,
        )

        return {
            "epoch_ntp_loss": avg_ntp,
            "epoch_rl_loss": epoch_rl / n,
            "epoch_perplexity": avg_ppl,
            "epoch_avg_reward": epoch_reward / n,
            "total_steps": num_steps,
        }
