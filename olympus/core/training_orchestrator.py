"""
TrainingOrchestrator: Coordinates multi-model training with shared memory,
state linking, gradient accumulation, and phased objectives.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import os
import logging

import torch
import torch.nn as nn

from olympus.core.training_context import TrainingContext
from olympus.core.memory_bus import MemoryBus
from olympus.core.stateful_module import StatefulModule

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model managed by the orchestrator."""

    name: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any] = None
    max_grad_norm: float = 1.0
    enabled_phases: Optional[List[str]] = None  # None = all phases

    def is_active(self, phase: str) -> bool:
        if self.enabled_phases is None:
            return True
        return phase in self.enabled_phases


@dataclass
class ObjectiveConfig:
    """Configuration for a training objective (loss function).

    The ``compute_fn`` should have the signature::

        def compute(models: Dict[str, nn.Module], ctx: TrainingContext, batch: Any) -> torch.Tensor

    It receives a dict mapping model names to models, the current context,
    and the data batch, and should return a scalar loss tensor.
    """

    name: str
    compute_fn: Callable[..., torch.Tensor]
    weight: float = 1.0
    enabled_phases: Optional[List[str]] = None

    def is_active(self, phase: str) -> bool:
        if self.enabled_phases is None:
            return True
        return phase in self.enabled_phases


class TrainingOrchestrator:
    """
    Manages the full training loop for one or more models.

    Features:
      - Multi-model registration with per-model optimizers/schedulers.
      - Multi-objective loss with phase-dependent activation.
      - Gradient accumulation.
      - Automatic state linking for :class:`StatefulModule` instances.
      - Shared :class:`MemoryBus`.
      - Checkpoint save / load.

    Args:
        memory_bus:   Shared MemoryBus (created automatically if ``None``).
        gradient_accumulation_steps: Number of micro-batches before an
                                     optimizer step.
        device:       Default device for new tensors.
    """

    def __init__(
        self,
        memory_bus: Optional[MemoryBus] = None,
        gradient_accumulation_steps: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.memory_bus = memory_bus or MemoryBus()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device or torch.device("cpu")

        self._models: Dict[str, ModelConfig] = {}
        self._objectives: Dict[str, ObjectiveConfig] = {}
        self._phase: str = "bootstrap"
        self._global_step: int = 0
        self._epoch: int = 0
        self._prev_loss: float = float("inf")
        self._accumulation_step: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_model(self, config: ModelConfig) -> None:
        """Register a model with the orchestrator."""
        if config.name in self._models:
            raise ValueError(f"Model '{config.name}' is already registered.")
        self._models[config.name] = config
        logger.info("Registered model '%s' (%s params)",
                     config.name,
                     sum(p.numel() for p in config.model.parameters()))

    def add_objective(self, config: ObjectiveConfig) -> None:
        """Register a training objective."""
        if config.name in self._objectives:
            raise ValueError(f"Objective '{config.name}' is already registered.")
        self._objectives[config.name] = config
        logger.info("Registered objective '%s' (weight=%.4f)", config.name, config.weight)

    def set_phase(self, phase: str) -> None:
        """Switch training phase (e.g. 'bootstrap', 'rl_pretrain', 'flywheel', 'vision')."""
        logger.info("Phase transition: %s -> %s", self._phase, phase)
        self._phase = phase

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def step(self, batch: Any) -> Dict[str, float]:
        """Execute one micro-step of training.

        This handles:
          1. Building the :class:`TrainingContext`.
          2. Setting active models to train mode.
          3. Computing all active objectives (weighted sum).
          4. Backward pass (scaled for gradient accumulation).
          5. On accumulation boundary: gradient clipping, optimizer step,
             scheduler step, zero_grad.
          6. Memory bus step cleanup.

        Args:
            batch: Whatever the data loader yields.

        Returns:
            Dict mapping objective names to their (unscaled) loss values.
        """
        ctx = TrainingContext(
            global_step=self._global_step,
            epoch=self._epoch,
            phase=self._phase,
            prev_loss=self._prev_loss,
            is_training=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            current_accumulation_step=self._accumulation_step,
            memory_bus=self.memory_bus,
        )

        # Determine active models and objectives for this phase
        active_models = {
            name: cfg for name, cfg in self._models.items()
            if cfg.is_active(self._phase)
        }
        active_objectives = {
            name: cfg for name, cfg in self._objectives.items()
            if cfg.is_active(self._phase)
        }

        # Set training mode
        for cfg in active_models.values():
            cfg.model.train()

        # Build model dict for objective functions
        models_dict = {name: cfg.model for name, cfg in active_models.items()}

        # Compute losses
        loss_values: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for obj_name, obj_cfg in active_objectives.items():
            loss = obj_cfg.compute_fn(models_dict, ctx, batch)
            loss_values[obj_name] = loss.item()
            total_loss = total_loss + obj_cfg.weight * loss

        # Scale for gradient accumulation and backward
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        loss_values["total"] = total_loss.item()

        # Accumulation boundary: step optimizers
        self._accumulation_step += 1
        if self._accumulation_step >= self.gradient_accumulation_steps:
            self._accumulation_step = 0

            for cfg in active_models.values():
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(cfg.model.parameters(), cfg.max_grad_norm)
                cfg.optimizer.step()
                cfg.optimizer.zero_grad(set_to_none=True)
                if cfg.scheduler is not None:
                    cfg.scheduler.step()

            self._global_step += 1
            self._prev_loss = total_loss.item()

        # Memory bus end-of-step cleanup
        self.memory_bus.step()

        return loss_values

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch counter."""
        self._epoch = epoch

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def phase(self) -> str:
        return self._phase

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save a full checkpoint (all models, optimizers, schedulers, state)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "global_step": self._global_step,
            "epoch": self._epoch,
            "phase": self._phase,
            "prev_loss": self._prev_loss,
            "memory_bus": self.memory_bus.state_dict(),
            "models": {},
            "optimizers": {},
            "schedulers": {},
        }

        for name, cfg in self._models.items():
            if isinstance(cfg.model, StatefulModule):
                checkpoint["models"][name] = cfg.model.state_dict_with_state()
            else:
                checkpoint["models"][name] = cfg.model.state_dict()
            checkpoint["optimizers"][name] = cfg.optimizer.state_dict()
            if cfg.scheduler is not None and hasattr(cfg.scheduler, "state_dict"):
                checkpoint["schedulers"][name] = cfg.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s (step %d)", path, self._global_step)

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """Load a checkpoint and restore all state.

        Returns:
            The raw checkpoint dict (for custom extraction).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._global_step = checkpoint["global_step"]
        self._epoch = checkpoint["epoch"]
        self._phase = checkpoint["phase"]
        self._prev_loss = checkpoint["prev_loss"]
        self.memory_bus.load_state_dict(checkpoint["memory_bus"])

        for name, cfg in self._models.items():
            if name in checkpoint["models"]:
                if isinstance(cfg.model, StatefulModule):
                    cfg.model.load_state_dict_with_state(checkpoint["models"][name], strict=strict)
                else:
                    cfg.model.load_state_dict(checkpoint["models"][name], strict=strict)
            if name in checkpoint["optimizers"]:
                cfg.optimizer.load_state_dict(checkpoint["optimizers"][name])
            if name in checkpoint.get("schedulers", {}) and cfg.scheduler is not None:
                cfg.scheduler.load_state_dict(checkpoint["schedulers"][name])

        logger.info("Checkpoint loaded from %s (step %d)", path, self._global_step)
        return checkpoint
