"""
CheckpointManager: Periodic saving and loading of training state.
"""

import os
import glob
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages periodic checkpoint saving with rotation (keeps the N most recent).

    Args:
        save_dir:       Directory in which to store checkpoints.
        save_interval:  Save every *save_interval* steps.
        max_to_keep:    Maximum number of checkpoints to retain.  Oldest are
                        deleted when the limit is exceeded.  0 = keep all.
        prefix:         Filename prefix.
    """

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        max_to_keep: int = 5,
        prefix: str = "ckpt",
    ) -> None:
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.max_to_keep = max_to_keep
        self.prefix = prefix

        os.makedirs(save_dir, exist_ok=True)
        self._saved: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Save a checkpoint if *step* is on the save interval (or *force* is True).

        Args:
            step:      Current global step.
            model:     The model to save.
            optimizer: The optimizer to save.
            extra:     Any extra state to include (scheduler, metrics, etc.).
            force:     Save regardless of interval.

        Returns:
            The path to the saved checkpoint, or ``None`` if not saved.
        """
        if not force and step % self.save_interval != 0:
            return None

        path = os.path.join(self.save_dir, f"{self.prefix}_step{step:08d}.pt")

        state: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra:
            state.update(extra)

        # Handle StatefulModule
        from olympus.core.stateful_module import StatefulModule
        if isinstance(model, StatefulModule):
            state["model_state_dict"] = model.state_dict_with_state()

        torch.save(state, path)
        self._saved.append(path)
        logger.info("Checkpoint saved: %s", path)

        self._rotate()
        return path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint in *save_dir*.

        Args:
            model:        Model to load weights into.
            optimizer:    Optimizer to restore state (optional).
            map_location: Device mapping for ``torch.load``.

        Returns:
            The full checkpoint dict, or ``None`` if no checkpoint found.
        """
        pattern = os.path.join(self.save_dir, f"{self.prefix}_step*.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            logger.info("No checkpoints found in %s", self.save_dir)
            return None

        path = files[-1]
        return self.load(path, model, optimizer, map_location)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load a specific checkpoint.

        Args:
            path:         Path to the checkpoint file.
            model:        Model to load weights into.
            optimizer:    Optimizer to restore state (optional).
            map_location: Device mapping for ``torch.load``.

        Returns:
            The full checkpoint dict.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        from olympus.core.stateful_module import StatefulModule
        if isinstance(model, StatefulModule) and "__stateful_states__" in checkpoint.get("model_state_dict", {}):
            model.load_state_dict_with_state(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info("Checkpoint loaded: %s (step %d)", path, checkpoint.get("step", -1))
        return checkpoint

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of all checkpoint paths in save_dir."""
        pattern = os.path.join(self.save_dir, f"{self.prefix}_step*.pt")
        return sorted(glob.glob(pattern))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rotate(self) -> None:
        """Delete oldest checkpoints if we exceed max_to_keep."""
        if self.max_to_keep <= 0:
            return
        while len(self._saved) > self.max_to_keep:
            old_path = self._saved.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info("Rotated old checkpoint: %s", old_path)
