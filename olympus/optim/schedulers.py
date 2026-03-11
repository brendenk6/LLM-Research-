"""
WSDScheduler: Warmup-Stable-Decay learning rate schedule with support for
post-growth warmup restarts.
"""

import math
from typing import Optional


class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) learning rate scheduler.

    Three phases:
      1. **Warmup** (0 to warmup_steps): Linear ramp from 0 to ``base_lr``.
      2. **Stable** (warmup_steps to decay_start): Constant ``base_lr``.
      3. **Decay** (decay_start to total_steps): Cosine decay to ``min_lr``.

    After a growth event (call :meth:`notify_growth`), a secondary warmup
    of ``post_growth_warmup_steps`` is applied on top of the main schedule.

    Args:
        optimizer:      The optimizer whose LR to control.
        base_lr:        Peak learning rate.
        min_lr:         Minimum learning rate at end of decay.
        warmup_steps:   Steps for initial warmup.
        total_steps:    Total training steps.
        decay_start:    Step at which cosine decay begins.  If ``None``,
                        defaults to ``total_steps - decay_steps``.
        decay_steps:    Length of the decay phase (used only if *decay_start*
                        is not given).  Default: 20% of total_steps.
        post_growth_warmup_steps: Steps for the post-growth mini warmup.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float = 3e-4,
        min_lr: float = 1e-5,
        warmup_steps: int = 2000,
        total_steps: int = 100_000,
        decay_start: Optional[int] = None,
        decay_steps: Optional[int] = None,
        post_growth_warmup_steps: int = 500,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.post_growth_warmup_steps = post_growth_warmup_steps

        if decay_start is not None:
            self.decay_start = decay_start
        elif decay_steps is not None:
            self.decay_start = total_steps - decay_steps
        else:
            self.decay_start = int(total_steps * 0.8)

        self._step_count: int = 0
        self._growth_step: Optional[int] = None  # step at which last growth occurred
        self._last_lr: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute the LR for *step* (defaults to internal counter)."""
        if step is None:
            step = self._step_count

        # Main schedule
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / max(self.warmup_steps, 1))
        elif step < self.decay_start:
            # Stable phase
            lr = self.base_lr
        else:
            # Cosine decay
            decay_total = self.total_steps - self.decay_start
            decay_progress = min((step - self.decay_start) / max(decay_total, 1), 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * decay_progress))

        # Post-growth mini-warmup overlay
        if self._growth_step is not None:
            steps_since_growth = step - self._growth_step
            if 0 <= steps_since_growth < self.post_growth_warmup_steps:
                warmup_fraction = steps_since_growth / max(self.post_growth_warmup_steps, 1)
                lr = lr * warmup_fraction

        return lr

    def step(self) -> None:
        """Advance the scheduler by one step and update optimizer LR."""
        lr = self.get_lr(self._step_count)
        self._last_lr = lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self._step_count += 1

    def notify_growth(self, step: Optional[int] = None) -> None:
        """Notify the scheduler that a growth event occurred.

        This triggers a post-growth warmup starting at *step*.

        Args:
            step: The step at which growth happened.  Defaults to current step.
        """
        self._growth_step = step if step is not None else self._step_count

    @property
    def last_lr(self) -> float:
        """Return the last computed learning rate."""
        return self._last_lr

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step_count": self._step_count,
            "growth_step": self._growth_step,
            "last_lr": self._last_lr,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "decay_start": self.decay_start,
            "post_growth_warmup_steps": self.post_growth_warmup_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]
        self._growth_step = state.get("growth_step")
        self._last_lr = state.get("last_lr", 0.0)
        # Allow overriding schedule params from checkpoint
        self.base_lr = state.get("base_lr", self.base_lr)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.total_steps = state.get("total_steps", self.total_steps)
        self.decay_start = state.get("decay_start", self.decay_start)
        self.post_growth_warmup_steps = state.get("post_growth_warmup_steps", self.post_growth_warmup_steps)
