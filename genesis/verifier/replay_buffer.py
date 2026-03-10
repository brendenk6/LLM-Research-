"""Replay Buffer for ACT-V.

Stores historical Generator outputs so the Verifier can be trained on a
diverse mixture of old and new samples, improving stability and reducing
catastrophic forgetting.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch


class ReplayBuffer:
    """Ring-buffer that stores past Generator outputs for Verifier training.

    Each entry is a dict containing ``input_ids`` (token-ID tensor),
    ``scores`` (verification scores dict), and optional ``metadata``.

    The buffer has a fixed maximum capacity; once full, the oldest entries
    are overwritten in FIFO order.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialise ReplayBuffer.

        Args:
            max_size: Maximum number of entries the buffer can hold.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._buffer: List[Dict[str, Any]] = []
        self._write_idx: int = 0  # Next position to write in ring buffer

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries stored."""
        return len(self._buffer)

    def is_ready(self, min_size: int = 100) -> bool:
        """Return True if the buffer has at least *min_size* entries."""
        return self.size >= min_size

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        input_ids: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single entry to the buffer.

        Args:
            input_ids: 1-D or 2-D tensor of token IDs.  Detached and moved
                to CPU for storage efficiency.
            scores: Dict of verification score tensors (detached to CPU).
            metadata: Optional additional information (e.g. step number,
                corruption type, source).
        """
        entry: Dict[str, Any] = {
            "input_ids": input_ids.detach().cpu(),
            "scores": {k: v.detach().cpu() for k, v in scores.items()},
            "metadata": metadata or {},
        }

        if self.size < self.max_size:
            # Buffer not yet full: append
            self._buffer.append(entry)
        else:
            # Ring-buffer overwrite
            self._buffer[self._write_idx] = entry

        self._write_idx = (self._write_idx + 1) % self.max_size

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a random batch of entries from the buffer.

        Args:
            batch_size: Number of entries to sample.  If ``batch_size``
                exceeds the current buffer size, all entries are returned
                (shuffled).

        Returns:
            List of dicts, each containing ``input_ids``, ``scores``,
            and ``metadata``.
        """
        if self.size == 0:
            return []
        actual_size = min(batch_size, self.size)
        indices = random.sample(range(self.size), actual_size)
        return [self._buffer[i] for i in indices]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries from the buffer."""
        self._buffer.clear()
        self._write_idx = 0

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self.size}, max_size={self.max_size}, "
            f"ready={self.is_ready()})"
        )
