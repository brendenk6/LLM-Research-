"""
Phase 1: Bootstrap Training (Next-Token Prediction).

Standard language model pretraining using cross-entropy loss on
next-token prediction. Uses the full GENESIS infrastructure:
- HLRT model with hierarchical tier routing
- MuonAdamWHybrid optimizer (Muon for attention, AdamW for embeddings)
- WSD learning rate schedule (warmup-stable-decay)
- TrainingOrchestrator for coordinated gradient accumulation
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genesis.model.hlrt import HLRT
from olympus.core.training_orchestrator import (
    TrainingOrchestrator,
    ModelConfig,
    ObjectiveConfig,
)
from olympus.core.training_context import TrainingContext
from olympus.data.tokenizer import TokenizerWrapper
from olympus.optim.muon_adamw_hybrid import MuonAdamWHybrid
from olympus.optim.schedulers import WSDScheduler

logger = logging.getLogger(__name__)


# ======================================================================
# NTP loss objective (for TrainingOrchestrator)
# ======================================================================

def ntp_objective(
    models: Dict[str, nn.Module],
    ctx: TrainingContext,
    batch: Any,
) -> torch.Tensor:
    """Next-token prediction cross-entropy loss.

    Shifts input_ids by one position: input[:, :-1] -> targets[:, 1:].
    Also adds the model's auxiliary loss (gate + MoE load balancing).

    Args:
        models: Dict with at least ``"hlrt"`` key.
        ctx: Current training context.
        batch: Dict with ``"input_ids"`` (B, S).

    Returns:
        Scalar loss tensor.
    """
    model = models["hlrt"]
    input_ids = batch["input_ids"]

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    out = model(input_ids)
    logits = out["logits"]  # (B, S, V)
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

    return loss


# ======================================================================
# BootstrapTrainer
# ======================================================================

class BootstrapTrainer:
    """Phase 1: Next-token prediction bootstrap training.

    Wraps the HLRT model, MuonAdamWHybrid optimizer, WSDScheduler,
    and TrainingOrchestrator into a cohesive training loop.

    Args:
        model: The HLRT model instance.
        optimizer: MuonAdamWHybrid optimizer.
        scheduler: WSD learning rate scheduler.
        tokenizer: TokenizerWrapper for vocabulary info and decoding.
        config: Training configuration dict with keys:
            - gradient_accumulation_steps (int)
            - max_grad_norm (float)
            - log_interval (int)
            - save_interval (int)
            - checkpoint_dir (str)
            - device (str)
    """

    def __init__(
        self,
        model: HLRT,
        optimizer: MuonAdamWHybrid,
        scheduler: WSDScheduler,
        tokenizer: TokenizerWrapper,
        config: dict,
    ) -> None:
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("HLRT model: %.2fM parameters", param_count / 1e6)

        # Build orchestrator for gradient accumulation coordination
        grad_accum = config.get("gradient_accumulation_steps", 4)
        self.orchestrator = TrainingOrchestrator(
            gradient_accumulation_steps=grad_accum,
            device=self.device,
        )
        self.orchestrator.add_model(ModelConfig(
            name="hlrt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            max_grad_norm=config.get("max_grad_norm", 1.0),
            enabled_phases=["bootstrap"],
        ))
        self.orchestrator.add_objective(ObjectiveConfig(
            name="ntp",
            compute_fn=ntp_objective,
            weight=1.0,
            enabled_phases=["bootstrap"],
        ))
        self.orchestrator.set_phase("bootstrap")

        self._tokens_seen: int = 0
        self._step_start_time: float = time.time()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training micro-step.

        Args:
            batch: Dict with ``"input_ids"`` (B, S).

        Returns:
            Dict with ``"loss"``, ``"perplexity"``, ``"lr"``, ``"tokens_per_sec"``.
        """
        step_start = time.time()

        loss_dict = self.orchestrator.step(batch)

        total_loss = loss_dict.get("total", loss_dict.get("ntp", 0.0))
        perplexity = min(math.exp(total_loss), 1e6) if total_loss < 20 else 1e6
        lr = self.scheduler.last_lr

        # Track tokens processed
        batch_tokens = batch["input_ids"].numel()
        self._tokens_seen += batch_tokens
        elapsed = time.time() - step_start
        tokens_per_sec = batch_tokens / max(elapsed, 1e-6)

        return {
            "loss": total_loss,
            "perplexity": perplexity,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one full epoch over the dataloader.

        Args:
            dataloader: Training data loader yielding dicts with ``"input_ids"``.

        Returns:
            Dict with epoch-level metrics: ``"epoch_loss"``, ``"epoch_perplexity"``,
            ``"avg_tokens_per_sec"``, ``"total_steps"``.
        """
        epoch_loss = 0.0
        epoch_tokens_per_sec = 0.0
        num_steps = 0
        log_interval = self.config.get("log_interval", 100)
        save_interval = self.config.get("save_interval", 5000)
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints/phase1")

        epoch_start = time.time()

        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_loss += metrics["loss"]
            epoch_tokens_per_sec += metrics["tokens_per_sec"]
            num_steps += 1

            global_step = self.orchestrator.global_step

            if num_steps % log_interval == 0:
                logger.info(
                    "Step %d | loss=%.4f | ppl=%.2f | lr=%.2e | %.0f tok/s",
                    global_step,
                    metrics["loss"],
                    metrics["perplexity"],
                    metrics["lr"],
                    metrics["tokens_per_sec"],
                )

            if save_interval > 0 and global_step > 0 and global_step % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
                )

        avg_loss = epoch_loss / max(num_steps, 1)
        avg_ppl = min(math.exp(avg_loss), 1e6) if avg_loss < 20 else 1e6
        elapsed = time.time() - epoch_start

        logger.info(
            "Epoch complete | avg_loss=%.4f | ppl=%.2f | %d steps in %.1fs",
            avg_loss, avg_ppl, num_steps, elapsed,
        )

        return {
            "epoch_loss": avg_loss,
            "epoch_perplexity": avg_ppl,
            "avg_tokens_per_sec": epoch_tokens_per_sec / max(num_steps, 1),
            "total_steps": num_steps,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint via the orchestrator.

        Args:
            path: File path for the checkpoint.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.orchestrator.save_checkpoint(path)
        logger.info("Checkpoint saved: %s (tokens_seen=%d)", path, self._tokens_seen)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint via the orchestrator.

        Args:
            path: File path to the checkpoint.
        """
        self.orchestrator.load_checkpoint(path)
        logger.info("Checkpoint loaded: %s (global_step=%d)", path, self.orchestrator.global_step)
