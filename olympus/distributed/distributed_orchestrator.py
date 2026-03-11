"""
DistributedTrainingOrchestrator: FSDP-aware extension of TrainingOrchestrator.

Handles:
- FSDP model wrapping at registration time
- Gradient sync skip during accumulation (no_sync)
- Distributed checkpointing (rank 0 saves full state dict)
- Proper gradient clipping under FSDP
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from olympus.core.memory_bus import MemoryBus
from olympus.core.stateful_module import StatefulModule
from olympus.core.training_orchestrator import (
    TrainingOrchestrator,
    ModelConfig,
    ObjectiveConfig,
)
from olympus.distributed.fsdp_wrapper import FSDPConfig, wrap_model_fsdp

logger = logging.getLogger(__name__)


class DistributedTrainingOrchestrator(TrainingOrchestrator):
    """TrainingOrchestrator with FSDP multi-GPU support.

    Drop-in replacement that wraps models in FSDP at registration time
    and handles gradient synchronization during accumulation.

    Args:
        fsdp_config: FSDP configuration. If None, behaves like base class.
        local_rank: Local GPU rank (from torchrun).
        world_size: Total number of GPUs.
        memory_bus: Shared MemoryBus.
        gradient_accumulation_steps: Micro-batches before optimizer step.
        device: Default device.
    """

    def __init__(
        self,
        fsdp_config: Optional[FSDPConfig] = None,
        local_rank: int = 0,
        world_size: int = 1,
        memory_bus: Optional[MemoryBus] = None,
        gradient_accumulation_steps: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device(f"cuda:{local_rank}")
        super().__init__(
            memory_bus=memory_bus,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
        )
        self.fsdp_config = fsdp_config
        self.local_rank = local_rank
        self.world_size = world_size
        self._fsdp_enabled = fsdp_config is not None and world_size > 1

        if self._fsdp_enabled:
            logger.info(
                "Distributed orchestrator: rank %d/%d, FSDP enabled",
                local_rank, world_size,
            )
        else:
            logger.info("Distributed orchestrator: single GPU mode")

    def add_model(self, config: ModelConfig) -> None:
        """Register model, wrapping in FSDP if distributed."""
        if self._fsdp_enabled:
            # Move to device first, then wrap
            config.model = config.model.to(self.device)
            config.model = wrap_model_fsdp(
                config.model,
                self.fsdp_config,
                device_id=self.local_rank,
            )
            # Rebuild optimizer with FSDP-wrapped parameters
            # FSDP reshards parameters, so optimizer must be recreated
            old_opt = config.optimizer
            opt_cls = type(old_opt)
            # Extract non-default args from existing optimizer
            opt_defaults = {
                k: v for k, v in old_opt.defaults.items()
                if k != "params"
            }
            config.optimizer = opt_cls(config.model.parameters(), **opt_defaults)
            logger.info(
                "Rebuilt optimizer %s for FSDP-wrapped model '%s'",
                opt_cls.__name__, config.name,
            )
        super().add_model(config)

    def step(self, batch: Any) -> Dict[str, float]:
        """Execute one micro-step with FSDP gradient sync management.

        During gradient accumulation, FSDP's no_sync() context skips
        the all-reduce, saving communication overhead. On the accumulation
        boundary, gradients are synchronized and optimizer steps.
        """
        if not self._fsdp_enabled:
            return super().step(batch)

        # Determine if this is an accumulation step (skip sync) or boundary (sync)
        is_accumulation_step = (
            self._accumulation_step < self.gradient_accumulation_steps - 1
        )

        # Build context managers for each FSDP model
        active_models = {
            name: cfg for name, cfg in self._models.items()
            if cfg.is_active(self._phase)
        }

        # Use no_sync during accumulation to avoid premature all-reduce
        sync_contexts = {}
        for name, cfg in active_models.items():
            if isinstance(cfg.model, FSDP) and is_accumulation_step:
                sync_contexts[name] = cfg.model.no_sync()
            else:
                sync_contexts[name] = nullcontext()

        # Enter all no_sync contexts
        entered = {}
        for name, ctx in sync_contexts.items():
            entered[name] = ctx.__enter__()

        try:
            result = self._step_inner(batch, active_models)
        finally:
            # Exit all contexts
            for name, ctx in sync_contexts.items():
                ctx.__exit__(None, None, None)

        return result

    def _step_inner(
        self, batch: Any, active_models: Dict[str, ModelConfig]
    ) -> Dict[str, float]:
        """Core training step logic (shared with base class)."""
        from olympus.core.training_context import TrainingContext

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

        active_objectives = {
            name: cfg for name, cfg in self._objectives.items()
            if cfg.is_active(self._phase)
        }

        for cfg in active_models.values():
            cfg.model.train()

        models_dict = {name: cfg.model for name, cfg in active_models.items()}

        # Compute losses
        loss_values: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for obj_name, obj_cfg in active_objectives.items():
            loss = obj_cfg.compute_fn(models_dict, ctx, batch)
            loss_values[obj_name] = loss.item()
            total_loss = total_loss + obj_cfg.weight * loss

        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        loss_values["total"] = total_loss.item()

        # Accumulation boundary
        self._accumulation_step += 1
        if self._accumulation_step >= self.gradient_accumulation_steps:
            self._accumulation_step = 0

            for cfg in active_models.values():
                if cfg.max_grad_norm > 0:
                    if isinstance(cfg.model, FSDP):
                        cfg.model.clip_grad_norm_(cfg.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(
                            cfg.model.parameters(), cfg.max_grad_norm
                        )
                cfg.optimizer.step()
                cfg.optimizer.zero_grad(set_to_none=True)
                if cfg.scheduler is not None:
                    cfg.scheduler.step()

            self._global_step += 1
            self._prev_loss = total_loss.item()

        self.memory_bus.step()
        return loss_values

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint. Only rank 0 writes to disk."""
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
            "world_size": self.world_size,
        }

        for name, cfg in self._models.items():
            if isinstance(cfg.model, FSDP):
                # Gather full state dict from all ranks
                full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(cfg.model, StateDictType.FULL_STATE_DICT, full_cfg):
                    model_state = cfg.model.state_dict()
                    # Also grab StatefulModule extras if applicable
                    unwrapped = cfg.model.module if hasattr(cfg.model, "module") else cfg.model
                    if isinstance(unwrapped, StatefulModule):
                        model_state["_olympus_state_tensors"] = {
                            k: v.cpu() for k, v in unwrapped._state_tensors.items()
                        }
                        model_state["_olympus_state_meta"] = unwrapped._state_meta
                        model_state["_olympus_memory_tensors"] = {
                            k: v.cpu() for k, v in unwrapped._memory_tensors.items()
                        }
                        model_state["_olympus_memory_meta"] = unwrapped._memory_meta
            else:
                if isinstance(cfg.model, StatefulModule):
                    model_state = cfg.model.state_dict_with_state()
                else:
                    model_state = cfg.model.state_dict()

            checkpoint["models"][name] = model_state
            checkpoint["optimizers"][name] = cfg.optimizer.state_dict()
            if cfg.scheduler is not None and hasattr(cfg.scheduler, "state_dict"):
                checkpoint["schedulers"][name] = cfg.scheduler.state_dict()

        # Only rank 0 saves
        if self.local_rank == 0:
            torch.save(checkpoint, path)
            logger.info("Checkpoint saved to %s (step %d)", path, self._global_step)

        if self._fsdp_enabled:
            dist.barrier()

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """Load checkpoint, distributing state across FSDP ranks."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._global_step = checkpoint["global_step"]
        self._epoch = checkpoint["epoch"]
        self._phase = checkpoint["phase"]
        self._prev_loss = checkpoint["prev_loss"]
        self.memory_bus.load_state_dict(checkpoint["memory_bus"])

        for name, cfg in self._models.items():
            if name in checkpoint["models"]:
                model_state = checkpoint["models"][name]

                if isinstance(cfg.model, FSDP):
                    # Load full state dict into FSDP model
                    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                    with FSDP.state_dict_type(cfg.model, StateDictType.FULL_STATE_DICT, full_cfg):
                        # Separate Olympus state from model weights
                        state_extras = {}
                        for key in list(model_state.keys()):
                            if key.startswith("_olympus_"):
                                state_extras[key] = model_state.pop(key)

                        cfg.model.load_state_dict(model_state, strict=strict)

                        # Restore StatefulModule extras
                        unwrapped = cfg.model.module if hasattr(cfg.model, "module") else cfg.model
                        if isinstance(unwrapped, StatefulModule) and state_extras:
                            for sname, tensor in state_extras.get("_olympus_state_tensors", {}).items():
                                if sname in unwrapped._state_tensors:
                                    unwrapped.set_state(sname, tensor)
                            for mname, tensor in state_extras.get("_olympus_memory_tensors", {}).items():
                                if mname in unwrapped._memory_tensors:
                                    unwrapped.update_memory(mname, tensor)
                else:
                    if isinstance(cfg.model, StatefulModule):
                        cfg.model.load_state_dict_with_state(model_state, strict=strict)
                    else:
                        cfg.model.load_state_dict(model_state, strict=strict)

            if name in checkpoint["optimizers"]:
                cfg.optimizer.load_state_dict(checkpoint["optimizers"][name])
            if name in checkpoint.get("schedulers", {}) and cfg.scheduler is not None:
                cfg.scheduler.load_state_dict(checkpoint["schedulers"][name])

        logger.info("Checkpoint loaded from %s (step %d)", path, self._global_step)
        return checkpoint

    @property
    def is_main_rank(self) -> bool:
        """True if this is rank 0 (for logging, saving, etc.)."""
        return self.local_rank == 0
