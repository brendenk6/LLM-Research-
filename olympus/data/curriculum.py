"""
Training curriculum management for multi-phase training.

Manages data mixing ratios and phase transitions for the GENESIS
training pipeline (bootstrap -> RL pretrain -> flywheel -> vision).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Configuration for a single training phase.

    Args:
        name: Phase name (e.g. "bootstrap", "rl_pretrain").
        base_mix: Base data mixing ratios (source_name -> weight).
        min_steps: Minimum steps before a transition can occur.
        max_steps: Maximum steps; force transition after this.
        transition_metric: Metric name to monitor for auto-transition.
        transition_threshold: Threshold the metric must reach.
        transition_direction: ``"above"`` or ``"below"`` the threshold.
        mix_schedule: Optional list of ``(step, mix_dict)`` tuples for
                      time-varying mix ratios within the phase.
    """

    name: str
    base_mix: Dict[str, float] = field(default_factory=lambda: {"web": 1.0})
    min_steps: int = 0
    max_steps: int = 100_000
    transition_metric: str = "loss"
    transition_threshold: float = 2.0
    transition_direction: str = "below"  # "above" or "below"
    mix_schedule: List[Tuple[int, Dict[str, float]]] = field(default_factory=list)


# Default phase configurations for GENESIS
DEFAULT_PHASES: Dict[str, PhaseConfig] = {
    "bootstrap": PhaseConfig(
        name="bootstrap",
        base_mix={"web": 0.7, "books": 0.2, "code": 0.1},
        min_steps=5000,
        max_steps=200_000,
        transition_metric="loss",
        transition_threshold=2.5,
        transition_direction="below",
        mix_schedule=[
            (0, {"web": 0.7, "books": 0.2, "code": 0.1}),
            (50_000, {"web": 0.6, "books": 0.25, "code": 0.15}),
            (100_000, {"web": 0.5, "books": 0.3, "code": 0.2}),
        ],
    ),
    "rl_pretrain": PhaseConfig(
        name="rl_pretrain",
        base_mix={"web": 0.4, "reasoning": 0.4, "code": 0.2},
        min_steps=2000,
        max_steps=100_000,
        transition_metric="avg_reward",
        transition_threshold=0.5,
        transition_direction="above",
    ),
    "flywheel": PhaseConfig(
        name="flywheel",
        base_mix={"web": 0.3, "reasoning_traces": 0.4, "code": 0.15, "books": 0.15},
        min_steps=5000,
        max_steps=500_000,
        transition_metric="loss",
        transition_threshold=1.5,
        transition_direction="below",
    ),
    "vision": PhaseConfig(
        name="vision",
        base_mix={"web": 0.3, "image_text": 0.4, "code": 0.1, "reasoning_traces": 0.2},
        min_steps=5000,
        max_steps=200_000,
        transition_metric="loss",
        transition_threshold=1.8,
        transition_direction="below",
    ),
}

# Default phase ordering
DEFAULT_PHASE_ORDER: List[str] = ["bootstrap", "rl_pretrain", "flywheel", "vision"]


class Curriculum:
    """Training curriculum manager.

    Tracks the current phase, computes data mixing ratios (possibly
    time-varying), and decides when to transition to the next phase.

    Args:
        phases: Dict mapping phase name to :class:`PhaseConfig`.
                Defaults to the GENESIS default phases.
        phase_order: Ordered list of phase names.  Defaults to
                     ``["bootstrap", "rl_pretrain", "flywheel", "vision"]``.
    """

    def __init__(
        self,
        phases: Optional[Dict[str, PhaseConfig]] = None,
        phase_order: Optional[List[str]] = None,
    ) -> None:
        self.phases = phases or dict(DEFAULT_PHASES)
        self.phase_order = phase_order or list(DEFAULT_PHASE_ORDER)
        self._current_phase_idx: int = 0
        self._phase_start_step: int = 0
        self._transition_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> str:
        """Name of the current training phase."""
        if self._current_phase_idx < len(self.phase_order):
            return self.phase_order[self._current_phase_idx]
        return self.phase_order[-1]

    @property
    def current_phase_config(self) -> PhaseConfig:
        return self.phases[self.current_phase]

    @property
    def is_final_phase(self) -> bool:
        return self._current_phase_idx >= len(self.phase_order) - 1

    # ------------------------------------------------------------------
    # Mix ratios
    # ------------------------------------------------------------------

    def get_mix_ratios(
        self,
        phase: Optional[str] = None,
        step: int = 0,
    ) -> Dict[str, float]:
        """Compute data mixing ratios for the given phase and step.

        If the phase has a ``mix_schedule``, linearly interpolates between
        schedule checkpoints based on the step within the phase.

        Args:
            phase: Phase name.  Defaults to the current phase.
            step: Global training step.

        Returns:
            Dict mapping data source names to their mixing weights
            (normalised to sum to 1).
        """
        phase = phase or self.current_phase
        cfg = self.phases.get(phase)
        if cfg is None:
            logger.warning("Unknown phase %r, returning uniform mix", phase)
            return {"web": 1.0}

        phase_step = step - self._phase_start_step

        if not cfg.mix_schedule:
            return dict(cfg.base_mix)

        # Find the two surrounding schedule points
        schedule = sorted(cfg.mix_schedule, key=lambda x: x[0])

        # Before first checkpoint
        if phase_step <= schedule[0][0]:
            return dict(schedule[0][1])

        # After last checkpoint
        if phase_step >= schedule[-1][0]:
            return dict(schedule[-1][1])

        # Interpolate
        for i in range(len(schedule) - 1):
            s0, mix0 = schedule[i]
            s1, mix1 = schedule[i + 1]
            if s0 <= phase_step < s1:
                alpha = (phase_step - s0) / max(s1 - s0, 1)
                all_keys = set(mix0.keys()) | set(mix1.keys())
                interpolated = {}
                for k in all_keys:
                    v0 = mix0.get(k, 0.0)
                    v1 = mix1.get(k, 0.0)
                    interpolated[k] = v0 + alpha * (v1 - v0)
                # Normalise
                total = sum(interpolated.values())
                if total > 0:
                    interpolated = {k: v / total for k, v in interpolated.items()}
                return interpolated

        return dict(cfg.base_mix)

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def should_transition(
        self,
        metrics: Dict[str, float],
        step: int,
    ) -> bool:
        """Check whether the curriculum should transition to the next phase.

        Transition happens when:
        1. We are past ``min_steps`` within the phase AND the monitored
           metric has crossed its threshold, OR
        2. We have exceeded ``max_steps`` within the phase.

        Does NOT actually perform the transition; call :meth:`transition`
        to advance.

        Args:
            metrics: Current training metrics dict.
            step: Current global step.

        Returns:
            True if a transition should happen.
        """
        if self.is_final_phase:
            return False

        cfg = self.current_phase_config
        phase_step = step - self._phase_start_step

        # Hard maximum
        if phase_step >= cfg.max_steps:
            logger.info("Phase %r exceeded max_steps (%d), should transition",
                        cfg.name, cfg.max_steps)
            return True

        # Not yet at minimum
        if phase_step < cfg.min_steps:
            return False

        # Check metric threshold
        metric_val = metrics.get(cfg.transition_metric)
        if metric_val is None:
            return False

        if cfg.transition_direction == "below":
            crossed = metric_val < cfg.transition_threshold
        else:
            crossed = metric_val > cfg.transition_threshold

        if crossed:
            logger.info(
                "Phase %r transition condition met: %s=%.4f %s %.4f",
                cfg.name,
                cfg.transition_metric,
                metric_val,
                "<" if cfg.transition_direction == "below" else ">",
                cfg.transition_threshold,
            )
        return crossed

    def transition(self, step: int, metrics: Optional[Dict[str, float]] = None) -> str:
        """Advance to the next phase.

        Args:
            step: Current global step (becomes the start step of the new phase).
            metrics: Optional metrics snapshot at the time of transition.

        Returns:
            Name of the new phase.
        """
        old_phase = self.current_phase
        self._transition_history.append({
            "from_phase": old_phase,
            "step": step,
            "metrics": dict(metrics) if metrics else {},
        })

        if not self.is_final_phase:
            self._current_phase_idx += 1

        self._phase_start_step = step
        new_phase = self.current_phase
        logger.info("Curriculum transition: %s -> %s at step %d", old_phase, new_phase, step)
        return new_phase

    def set_phase(self, phase: str, step: int = 0) -> None:
        """Manually set the current phase.

        Args:
            phase: Phase name (must exist in ``self.phases``).
            step: Step to record as the phase start.
        """
        if phase not in self.phases:
            raise ValueError(f"Unknown phase {phase!r}. Known: {list(self.phases.keys())}")
        self._current_phase_idx = self.phase_order.index(phase)
        self._phase_start_step = step
        logger.info("Curriculum manually set to phase %r at step %d", phase, step)

    @property
    def transition_history(self) -> List[Dict[str, Any]]:
        """Return the full history of phase transitions."""
        return list(self._transition_history)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_phase_idx": self._current_phase_idx,
            "phase_start_step": self._phase_start_step,
            "transition_history": self._transition_history,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._current_phase_idx = state["current_phase_idx"]
        self._phase_start_step = state["phase_start_step"]
        self._transition_history = state.get("transition_history", [])
