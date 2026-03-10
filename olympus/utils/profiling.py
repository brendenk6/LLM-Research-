"""
Simple GPU memory profiling helpers.
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_stats(device: int = 0) -> Dict[str, float]:
    """Get GPU memory usage statistics in GB.

    Args:
        device: CUDA device index.

    Returns:
        Dict with memory stats, or {"available": False} if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def gpu_memory_stats_mb(device: Optional[int] = None) -> Dict[str, float]:
    """Return GPU memory statistics in megabytes.

    Args:
        device: CUDA device index (default: current device).

    Returns:
        Dict with keys: allocated_mb, reserved_mb, max_allocated_mb, free_mb, total_mb.
        Returns empty dict if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total = torch.cuda.get_device_properties(device).total_mem / (1024 ** 2)
    free = total - reserved

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "max_allocated_mb": round(max_allocated, 2),
        "free_mb": round(free, 2),
        "total_mb": round(total, 2),
    }


def reset_peak_memory(device: int = 0) -> None:
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters.

    Args:
        model: The PyTorch module.

    Returns:
        Dict with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def log_memory_snapshot(label: str = "", device: int = 0) -> None:
    """Print a memory snapshot with an optional label."""
    stats = get_gpu_memory_stats(device)
    if not stats.get("available", True):
        return
    logger.info(
        "[Memory %s] Allocated: %.2fGB, Reserved: %.2fGB, Peak: %.2fGB",
        label,
        stats["allocated_gb"],
        stats["reserved_gb"],
        stats["max_allocated_gb"],
    )


class MemoryTracker:
    """Context manager that tracks GPU memory delta.

    Usage::

        with MemoryTracker("forward_pass") as mt:
            output = model(input)
        print(mt.delta_mb)  # memory increase in MB
    """

    def __init__(self, label: str = "", device: Optional[int] = None) -> None:
        self.label = label
        self.device = device
        self.start_mb: float = 0.0
        self.end_mb: float = 0.0
        self.delta_mb: float = 0.0

    def __enter__(self) -> "MemoryTracker":
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.start_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        return self

    def __exit__(self, *args) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.end_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            self.delta_mb = self.end_mb - self.start_mb
            if self.label:
                logger.info(
                    "MemoryTracker[%s]: %.2fMB -> %.2fMB (delta=%.2fMB)",
                    self.label, self.start_mb, self.end_mb, self.delta_mb,
                )
