"""Custom CUDA/Triton kernels for Olympus.

Provides accelerated implementations with automatic PyTorch fallbacks
when Triton is unavailable (CPU, non-NVIDIA hardware).
"""

from olympus.kernels.fp4_matmul import (
    FP4Linear,
    fp4_matmul,
)
from olympus.kernels.fp4_quantize import (
    dequantize,
    quantize,
    quantize_dequantize,
)
from olympus.kernels.memory_cross_attention import memory_cross_attention
from olympus.kernels.muon_step import muon_step
from olympus.kernels.sparse_expert_matmul import sparse_expert_matmul
from olympus.kernels.fused_gate_route import (
    HAS_TRITON,
    chunk_mean_pool,
    fused_gate_route,
    softmax_topk,
)

__all__ = [
    "FP4Linear",
    "HAS_TRITON",
    "chunk_mean_pool",
    "dequantize",
    "fp4_matmul",
    "fused_gate_route",
    "quantize",
    "quantize_dequantize",
    "memory_cross_attention",
    "muon_step",
    "softmax_topk",
    "sparse_expert_matmul",
]
