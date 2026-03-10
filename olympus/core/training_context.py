"""
TrainingContext: Metadata about the current training step, passed to every
StatefulModule.forward() call.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class TrainingContext:
    """Immutable context for a single training step."""

    global_step: int = 0
    epoch: int = 0
    phase: str = "bootstrap"  # "bootstrap", "rl_pretrain", "flywheel", "vision"

    prev_loss: float = float('inf')
    prev_tier_activations: Optional[Dict[str, float]] = None

    is_training: bool = True
    gradient_accumulation_steps: int = 1
    current_accumulation_step: int = 0

    memory_bus: Optional[Any] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_accumulating(self) -> bool:
        return self.current_accumulation_step < self.gradient_accumulation_steps - 1

    @property
    def tokens_seen(self) -> int:
        return self.metadata.get("tokens_seen", 0)
