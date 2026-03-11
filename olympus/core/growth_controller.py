"""
GrowthController: Manages progressive model growth during training.

Supports scheduled (step-based) and adaptive (loss-plateau / metric-based)
triggers.  The actual tensor surgery operations (_widen, _deepen, _dense_to_moe)
are left as NotImplementedError stubs because they are architecture-specific;
subclasses should override them.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GrowthEvent:
    """Record of a single growth operation."""

    step: int
    event_type: str            # "widen", "deepen", "dense_to_moe"
    description: str = ""
    old_param_count: int = 0
    new_param_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GrowthController:
    """
    Orchestrates model growth (widening, deepening, MoE conversion).

    Args:
        model:       The model to grow.
        optimizer:   The current optimizer (will be replaced after growth).
        schedule:    List of ``(step, event_type, kwargs)`` tuples for
                     scheduled growth events.
        patience:    Number of steps to wait for improvement before adaptive
                     growth triggers.
        min_delta:   Minimum loss improvement to reset the patience counter.
        optimizer_factory:  Callable ``(params) -> Optimizer`` used after growth
                            to create a fresh optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedule: Optional[List[Tuple[int, str, Dict[str, Any]]]] = None,
        patience: int = 500,
        min_delta: float = 0.001,
        optimizer_factory: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.schedule = sorted(schedule or [], key=lambda x: x[0])
        self.patience = patience
        self.min_delta = min_delta
        self.optimizer_factory = optimizer_factory

        self._history: List[GrowthEvent] = []
        self._best_loss: float = float("inf")
        self._steps_without_improvement: int = 0
        self._completed_scheduled: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_grow(
        self,
        step: int,
        current_loss: float,
    ) -> Optional[GrowthEvent]:
        """Check if growth should happen at this step.

        Checks scheduled events first, then adaptive triggers.

        Args:
            step: Current global training step.
            current_loss: Most recent loss value.

        Returns:
            A :class:`GrowthEvent` if growth occurred, else ``None``.
        """
        # --- Scheduled growth ---
        for scheduled_step, event_type, kwargs in self.schedule:
            if scheduled_step == step and scheduled_step not in self._completed_scheduled:
                event = self._execute_growth(step, event_type, kwargs)
                self._completed_scheduled.add(scheduled_step)
                return event

        # --- Adaptive growth ---
        if current_loss < self._best_loss - self.min_delta:
            self._best_loss = current_loss
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if self._steps_without_improvement >= self.patience:
            event = self._execute_growth(step, "widen", {"factor": 1.5})
            self._steps_without_improvement = 0
            self._best_loss = current_loss
            return event

        return None

    @property
    def history(self) -> List[GrowthEvent]:
        """Return the history of all growth events."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Growth operations (to be overridden by subclasses)
    # ------------------------------------------------------------------

    def _widen(self, model: nn.Module, factor: float = 2.0, **kwargs) -> nn.Module:
        """Widen all layers by *factor* using Net2Net-style function-preserving transform.

        Must be overridden by architecture-specific subclass.
        """
        raise NotImplementedError(
            "_widen() is architecture-specific. Subclass GrowthController and "
            "implement this method for your model."
        )

    def _deepen(self, model: nn.Module, num_layers: int = 1, position: str = "end", **kwargs) -> nn.Module:
        """Insert *num_layers* identity-initialized layers.

        Must be overridden by architecture-specific subclass.
        """
        raise NotImplementedError(
            "_deepen() is architecture-specific. Subclass GrowthController and "
            "implement this method for your model."
        )

    def _dense_to_moe(self, model: nn.Module, num_experts: int = 4, top_k: int = 2, **kwargs) -> nn.Module:
        """Convert dense FFN layers to Mixture-of-Experts.

        Must be overridden by architecture-specific subclass.
        """
        raise NotImplementedError(
            "_dense_to_moe() is architecture-specific. Subclass GrowthController and "
            "implement this method for your model."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_growth(
        self,
        step: int,
        event_type: str,
        kwargs: Dict[str, Any],
    ) -> GrowthEvent:
        """Run a growth operation, rebuild optimizer, record event."""
        old_count = sum(p.numel() for p in self.model.parameters())

        dispatch = {
            "widen": self._widen,
            "deepen": self._deepen,
            "dense_to_moe": self._dense_to_moe,
        }

        if event_type not in dispatch:
            raise ValueError(f"Unknown growth event type: '{event_type}'. "
                             f"Must be one of {list(dispatch.keys())}.")

        try:
            self.model = dispatch[event_type](self.model, **kwargs)
        except NotImplementedError:
            raise

        new_count = sum(p.numel() for p in self.model.parameters())

        # Rebuild optimizer for the (potentially new) parameter set
        self.optimizer = self._migrate_optimizer()

        event = GrowthEvent(
            step=step,
            event_type=event_type,
            description=f"{event_type} at step {step} with {kwargs}",
            old_param_count=old_count,
            new_param_count=new_count,
            metadata=kwargs,
        )
        self._history.append(event)
        return event

    def _migrate_optimizer(self) -> torch.optim.Optimizer:
        """Create a fresh optimizer for the current model parameters.

        If an ``optimizer_factory`` was provided, use it.  Otherwise fall back
        to creating an Adam optimizer with the same LR as the first param
        group of the old optimizer.
        """
        if self.optimizer_factory is not None:
            return self.optimizer_factory(self.model.parameters())

        # Fallback: replicate basic settings from old optimizer
        old_defaults = self.optimizer.defaults.copy()
        lr = old_defaults.get("lr", 1e-4)
        weight_decay = old_defaults.get("weight_decay", 0.0)

        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
