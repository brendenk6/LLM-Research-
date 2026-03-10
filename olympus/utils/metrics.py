"""
MetricsAggregator: Tracks running averages, latest values, and counts for
training metrics.
"""

from typing import Dict, Optional
from collections import defaultdict


class MetricsAggregator:
    """
    Lightweight metric tracker with running average, windowed average, and
    last-value access.

    Usage::

        metrics = MetricsAggregator()
        metrics.update({"loss": 2.3, "acc": 0.8})
        metrics.update({"loss": 1.8, "acc": 0.85})
        metrics.get_average("loss")   # 2.05
        metrics.get_last("loss")      # 1.8
        metrics.to_dict()             # {"loss": 1.8, "acc": 0.85}

    Args:
        window_size: Maximum number of values to retain for windowed averaging.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._values: Dict[str, list] = defaultdict(list)
        self._last: Dict[str, float] = {}
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Record a batch of metrics.

        Args:
            metrics: Dictionary mapping metric names to values.
        """
        for key, value in metrics.items():
            self._last[key] = value
            self._values[key].append(value)
            if len(self._values[key]) > self.window_size:
                self._values[key] = self._values[key][-self.window_size:]
            self._sums[key] += value
            self._counts[key] += 1
            if key not in self._min or value < self._min[key]:
                self._min[key] = value
            if key not in self._max or value > self._max[key]:
                self._max[key] = value

    def get_last(self, key: str, default: float = 0.0) -> float:
        """Get the most recent value for a metric."""
        return self._last.get(key, default)

    def get_average(self, key: str, default: float = 0.0) -> float:
        """Get the windowed running average for a metric."""
        values = self._values.get(key, [])
        if not values:
            return default
        return sum(values) / len(values)

    def get_global_average(self, key: str, default: float = 0.0) -> float:
        """Get the global (all-time) average for a metric."""
        count = self._counts.get(key, 0)
        if count == 0:
            return default
        return self._sums[key] / count

    def get_min(self, key: str) -> Optional[float]:
        """Return the minimum recorded value for *key*."""
        return self._min.get(key)

    def get_max(self, key: str) -> Optional[float]:
        """Return the maximum recorded value for *key*."""
        return self._max.get(key)

    def get_count(self, key: str) -> int:
        """Return the total number of updates for *key*."""
        return self._counts.get(key, 0)

    def to_dict(self) -> Dict[str, float]:
        """Dump all last values as a dict."""
        return dict(self._last)

    def averages(self) -> Dict[str, float]:
        """Dump all windowed running averages as a dict."""
        return {k: self.get_average(k) for k in self._values}

    def dump(self, reset: bool = False) -> Dict[str, float]:
        """Dump comprehensive metrics as a flat dictionary.

        Keys are formatted as ``{name}/avg``, ``{name}/last``, etc.

        Args:
            reset: If True, reset all accumulators after dumping.
        """
        result: Dict[str, float] = {}
        for key in self._counts:
            result[f"{key}/avg"] = self.get_average(key)
            result[f"{key}/global_avg"] = self.get_global_average(key)
            if key in self._last:
                result[f"{key}/last"] = self._last[key]
            if key in self._min:
                result[f"{key}/min"] = self._min[key]
            if key in self._max:
                result[f"{key}/max"] = self._max[key]
            result[f"{key}/count"] = float(self._counts[key])

        if reset:
            self.reset()

        return result

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._values.clear()
        self._last.clear()
        self._sums.clear()
        self._counts.clear()
        self._min.clear()
        self._max.clear()

    def keys(self):
        """Return all tracked metric names."""
        return list(self._counts.keys())

    def __repr__(self) -> str:
        items = [f"{k}={self.get_average(k):.4g}" for k in self._counts]
        return f"MetricsAggregator({', '.join(items)})"
