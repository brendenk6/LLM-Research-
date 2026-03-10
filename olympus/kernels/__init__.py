"""Custom CUDA/Triton kernels for Olympus.

Provides accelerated implementations with automatic PyTorch fallbacks
when Triton is unavailable (CPU, non-NVIDIA hardware).
"""

from olympus.kernels.fp4_quantize import (
    dequantize,
    quantize,
    quantize_dequantize,
)
from olympus.kernels.fused_gate_route import (
    HAS_TRITON,
    chunk_mean_pool,
    fused_gate_route,
    softmax_topk,
)

__all__ = [
    "HAS_TRITON",
    "chunk_mean_pool",
    "dequantize",
    "fused_gate_route",
    "quantize",
    "quantize_dequantize",
    "softmax_topk",
]
