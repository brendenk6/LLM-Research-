"""
MemoryBus: A thread-safe, multi-scope memory system for sharing information
between models and across training steps.

Scopes:
  - step:      Cleared every training step.  For intra-step communication.
  - episode:   Cleared on demand (e.g., end of episode / rollout).
  - permanent: Never auto-cleared.  Persists across the full run and is
               included in checkpoints.
"""

import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


_VALID_SCOPES = {"step", "episode", "permanent"}


class MemoryBus:
    """Thread-safe memory bus with three scopes and cosine-similarity query."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stores: Dict[str, Dict[str, torch.Tensor]] = {
            "step": {},
            "episode": {},
            "permanent": {},
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def write(self, key: str, value: torch.Tensor, scope: str = "step") -> None:
        """Write a tensor into the bus under *key* in the given *scope*.

        Args:
            key: Identifier for the memory slot.
            value: Tensor to store (detached clone is kept).
            scope: One of ``"step"``, ``"episode"``, ``"permanent"``.
        """
        self._validate_scope(scope)
        with self._lock:
            self._stores[scope][key] = value.clone().detach()

    def read(self, key: str, scope: str = "step") -> Optional[torch.Tensor]:
        """Read a tensor from the bus.  Returns ``None`` if not found.

        Args:
            key: Identifier for the memory slot.
            scope: Scope to look in.  If ``None`` could be useful, callers
                   should iterate scopes themselves.
        """
        self._validate_scope(scope)
        with self._lock:
            val = self._stores[scope].get(key)
            return val.clone() if val is not None else None

    def query(
        self,
        query_vector: torch.Tensor,
        scope: str = "step",
        top_k: int = 5,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        """Cosine-similarity search over all tensors in *scope*.

        Every stored tensor is flattened to 1-D before comparison, so shapes
        need not match exactly (but the total number of elements should be
        the same as *query_vector* for meaningful results).

        Args:
            query_vector: 1-D query tensor.
            scope: Scope to search.
            top_k: Number of results to return.

        Returns:
            List of ``(key, tensor, similarity)`` tuples sorted descending by
            similarity.
        """
        self._validate_scope(scope)
        query_flat = query_vector.detach().flatten().float()

        results: List[Tuple[str, torch.Tensor, float]] = []
        with self._lock:
            for key, tensor in self._stores[scope].items():
                t_flat = tensor.flatten().float()
                # If sizes differ, pad the shorter one with zeros
                if t_flat.shape[0] != query_flat.shape[0]:
                    max_len = max(t_flat.shape[0], query_flat.shape[0])
                    t_flat = F.pad(t_flat, (0, max_len - t_flat.shape[0]))
                    q = F.pad(query_flat, (0, max_len - query_flat.shape[0]))
                else:
                    q = query_flat

                sim = F.cosine_similarity(q.unsqueeze(0), t_flat.unsqueeze(0)).item()
                results.append((key, tensor.clone(), sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def list_keys(self, scope: str = "step") -> List[str]:
        """Return all keys currently stored in *scope*."""
        self._validate_scope(scope)
        with self._lock:
            return list(self._stores[scope].keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Called at the end of every training step.  Clears the ``step`` scope."""
        with self._lock:
            self._stores["step"].clear()

    def clear_episode(self) -> None:
        """Manually clear the ``episode`` scope."""
        with self._lock:
            self._stores["episode"].clear()

    def clear_all(self) -> None:
        """Clear every scope (including permanent)."""
        with self._lock:
            for store in self._stores.values():
                store.clear()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a serialisable snapshot of the permanent scope."""
        with self._lock:
            return {
                "permanent": {k: v.cpu().clone() for k, v in self._stores["permanent"].items()},
                "episode": {k: v.cpu().clone() for k, v in self._stores["episode"].items()},
            }

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Restore permanent (and optionally episode) scope from a checkpoint."""
        with self._lock:
            if "permanent" in state:
                self._stores["permanent"] = {k: v.clone() for k, v in state["permanent"].items()}
            if "episode" in state:
                self._stores["episode"] = {k: v.clone() for k, v in state["episode"].items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_scope(scope: str) -> None:
        if scope not in _VALID_SCOPES:
            raise ValueError(f"Invalid scope '{scope}'. Must be one of {_VALID_SCOPES}.")

    def __repr__(self) -> str:
        counts = {s: len(store) for s, store in self._stores.items()}
        return f"MemoryBus(step={counts['step']}, episode={counts['episode']}, permanent={counts['permanent']})"
