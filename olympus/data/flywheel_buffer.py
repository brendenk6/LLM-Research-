"""
FlywheelBuffer: Ring buffer for storing successful reasoning traces
generated during RL pretraining, to be replayed during flywheel training.
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """A single stored reasoning trace."""

    context: str
    trace: str
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class FlywheelBuffer:
    """Ring buffer for successful reasoning traces.

    Stores traces that exceed a reward threshold, evicting the oldest
    entries when the buffer reaches capacity.

    Args:
        max_size: Maximum number of traces to store.
        reward_threshold: Minimum reward for a trace to be stored.
        priority_sampling: If True, sample proportional to reward.
    """

    def __init__(
        self,
        max_size: int = 100_000,
        reward_threshold: float = 0.1,
        priority_sampling: bool = False,
    ) -> None:
        self.max_size = max_size
        self.reward_threshold = reward_threshold
        self.priority_sampling = priority_sampling
        self._buffer: deque[TraceEntry] = deque(maxlen=max_size)
        self._total_added: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        context: str,
        trace: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a reasoning trace if its reward exceeds the threshold.

        Args:
            context: The input context / prompt.
            trace: The generated reasoning trace.
            reward: The reward score for this trace.
            metadata: Optional metadata dict.

        Returns:
            True if the trace was stored, False if it was rejected.
        """
        if reward < self.reward_threshold:
            self._total_rejected += 1
            return False

        entry = TraceEntry(
            context=context,
            trace=trace,
            reward=reward,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self._buffer.append(entry)
        self._total_added += 1
        return True

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of traces from the buffer.

        Args:
            batch_size: Number of traces to sample.

        Returns:
            List of dicts with keys: ``context``, ``trace``, ``reward``,
            ``metadata``.
        """
        if len(self._buffer) == 0:
            return []

        n = min(batch_size, len(self._buffer))

        if self.priority_sampling and len(self._buffer) > 0:
            # Weighted sampling by reward
            entries = list(self._buffer)
            weights = [max(e.reward, 1e-8) for e in entries]
            sampled = random.choices(entries, weights=weights, k=n)
        else:
            sampled = random.sample(list(self._buffer), n)

        return [
            {
                "context": e.context,
                "trace": e.trace,
                "reward": e.reward,
                "metadata": e.metadata,
            }
            for e in sampled
        ]

    # ------------------------------------------------------------------
    # Properties and utilities
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of traces in the buffer."""
        return len(self._buffer)

    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics.

        Returns:
            Dict with ``size``, ``max_size``, ``total_added``,
            ``total_rejected``, ``avg_reward``, ``min_reward``,
            ``max_reward``.
        """
        rewards = [e.reward for e in self._buffer]
        return {
            "size": len(self._buffer),
            "max_size": self.max_size,
            "total_added": self._total_added,
            "total_rejected": self._total_rejected,
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }

    def clear(self) -> None:
        """Remove all traces from the buffer."""
        self._buffer.clear()
        logger.info("FlywheelBuffer cleared")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialize buffer state for checkpointing."""
        return {
            "entries": [
                {
                    "context": e.context,
                    "trace": e.trace,
                    "reward": e.reward,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for e in self._buffer
            ],
            "total_added": self._total_added,
            "total_rejected": self._total_rejected,
            "max_size": self.max_size,
            "reward_threshold": self.reward_threshold,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore buffer state from a checkpoint."""
        self._buffer.clear()
        self.max_size = state.get("max_size", self.max_size)
        self._buffer = deque(maxlen=self.max_size)
        self._total_added = state.get("total_added", 0)
        self._total_rejected = state.get("total_rejected", 0)
        for entry_dict in state.get("entries", []):
            self._buffer.append(TraceEntry(
                context=entry_dict["context"],
                trace=entry_dict["trace"],
                reward=entry_dict["reward"],
                metadata=entry_dict.get("metadata", {}),
                timestamp=entry_dict.get("timestamp", 0.0),
            ))
        logger.info("FlywheelBuffer loaded: %d entries", len(self._buffer))
