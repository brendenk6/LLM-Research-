"""
Reproducibility helpers: set all random seeds in one call.
"""

import random

import torch

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Set random seeds for reproducibility across torch, random, and numpy.

    Args:
        seed:                The seed value.
        deterministic_cudnn: If True, set cuDNN to deterministic mode
                             (may reduce performance).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if _HAS_NUMPY:
        np.random.seed(seed)

    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
