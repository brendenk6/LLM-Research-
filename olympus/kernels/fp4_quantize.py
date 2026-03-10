"""FP4 (INT4) quantization and dequantization kernels for GENESIS.

Symmetric INT4 quantization with per-group absmax scaling.  Values are
quantized to the [-8, 7] range using absmax / 7 as the scale factor.

Supports per-row (group_size=-1) and per-group quantization granularity.
Falls back to PyTorch when Triton is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INT4_MIN = -8
INT4_MAX = 7
SCALE_EPS = 1e-10


# ---------------------------------------------------------------------------
# PyTorch fallback implementations
# ---------------------------------------------------------------------------


def _quantize_pt(
    x_2d: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D float tensor to INT4 range with absmax scaling."""
    M, D = x_2d.shape
    if group_size <= 0 or group_size > D:
        group_size = D

    num_groups = (D + group_size - 1) // group_size
    D_padded = num_groups * group_size

    if D_padded > D:
        x_padded = F.pad(x_2d, (0, D_padded - D))
    else:
        x_padded = x_2d

    x_grouped = x_padded.view(M, num_groups, group_size)
    absmax = x_grouped.abs().amax(dim=-1, keepdim=True)  # (M, G, 1)
    scale = (absmax / INT4_MAX).clamp(min=SCALE_EPS)

    q = (x_grouped / scale).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
    q = q.view(M, D_padded)[:, :D]
    scale = scale.squeeze(-1)  # (M, num_groups)

    return q, scale


def _dequantize_pt(
    q: torch.Tensor, scale: torch.Tensor, group_size: int, orig_D: int
) -> torch.Tensor:
    """Dequantize INT4 values back to float."""
    M, D = q.shape
    if group_size <= 0 or group_size > D:
        group_size = D

    num_groups = scale.shape[1]
    D_padded = num_groups * group_size

    if D_padded > D:
        q_padded = F.pad(q, (0, D_padded - D))
    else:
        q_padded = q

    q_grouped = q_padded.view(M, num_groups, group_size).float()
    out = (q_grouped * scale.unsqueeze(-1)).view(M, D_padded)[:, :orig_D]
    return out


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _quantize_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        D,
        num_groups,
        GROUP_SIZE: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Per-group INT4 quantization.

        Each program handles one (row, group) pair.  Computes absmax over
        GROUP_SIZE elements, derives scale, quantizes to [-8, 7], stores
        int8 output and float32 scale.
        """
        pid = tl.program_id(0)
        row = pid // num_groups
        group = pid % num_groups

        offs = tl.arange(0, BLOCK_G)
        valid = offs < GROUP_SIZE
        col = group * GROUP_SIZE + offs
        col_valid = valid & (col < D)

        # Load values
        x_offs = row * D + col
        vals = tl.load(x_ptr + x_offs, mask=col_valid, other=0.0).to(tl.float32)

        # Per-group absmax
        absmax = tl.max(tl.abs(vals), axis=0)
        scale = absmax / 7.0
        scale = tl.where(scale < 1e-10, 1.0, scale)
        tl.store(scale_ptr + pid, scale)

        # Quantize: round-half-away-from-zero, clamp to [-8, 7]
        scaled = vals / scale
        offset = tl.where(scaled >= 0.0, 0.5, -0.5)
        q = (scaled + offset).to(tl.int32)
        q = tl.maximum(tl.minimum(q, 7), -8)

        tl.store(q_ptr + x_offs, q.to(tl.int8), mask=col_valid)

    @triton.jit
    def _dequantize_kernel(
        q_ptr,
        scale_ptr,
        out_ptr,
        D,
        num_groups,
        GROUP_SIZE: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Per-group INT4 dequantization.

        Each program handles one (row, group) pair.  Loads int8 quantized
        values and float32 scale, multiplies, stores float32 output.
        """
        pid = tl.program_id(0)
        row = pid // num_groups
        group = pid % num_groups

        offs = tl.arange(0, BLOCK_G)
        valid = offs < GROUP_SIZE
        col = group * GROUP_SIZE + offs
        col_valid = valid & (col < D)

        x_offs = row * D + col
        q = tl.load(q_ptr + x_offs, mask=col_valid, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + pid)

        out = q * scale
        tl.store(out_ptr + x_offs, out, mask=col_valid)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _resolve_group_size(group_size: int, D: int) -> int:
    """Normalise group_size: -1 or 0 means per-row (= D)."""
    if group_size <= 0 or group_size > D:
        return D
    return group_size


def _quantize_triton(
    x_2d: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    M, D = x_2d.shape
    group_size = _resolve_group_size(group_size, D)
    num_groups = (D + group_size - 1) // group_size

    q = torch.empty(M, D, device=x_2d.device, dtype=torch.int8)
    scales = torch.empty(M, num_groups, device=x_2d.device, dtype=torch.float32)

    BLOCK_G = triton.next_power_of_2(group_size)
    grid = (M * num_groups,)

    _quantize_kernel[grid](
        x_2d,
        q,
        scales,
        D,
        num_groups,
        GROUP_SIZE=group_size,
        BLOCK_G=BLOCK_G,
    )
    return q, scales


def _dequantize_triton(
    q: torch.Tensor, scales: torch.Tensor, group_size: int, orig_D: int
) -> torch.Tensor:
    M, D = q.shape
    group_size = _resolve_group_size(group_size, D)
    num_groups = scales.shape[1]

    out = torch.empty(M, D, device=q.device, dtype=torch.float32)
    BLOCK_G = triton.next_power_of_2(group_size)
    grid = (M * num_groups,)

    _dequantize_kernel[grid](
        q,
        scales,
        out,
        D,
        num_groups,
        GROUP_SIZE=group_size,
        BLOCK_G=BLOCK_G,
    )
    return out[:, :orig_D]


def _quantize_dispatch(
    x_2d: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if HAS_TRITON and x_2d.is_cuda:
        return _quantize_triton(x_2d, group_size)
    return _quantize_pt(x_2d, group_size)


def _dequantize_dispatch(
    q: torch.Tensor, scales: torch.Tensor, group_size: int, orig_D: int
) -> torch.Tensor:
    if HAS_TRITON and q.is_cuda:
        return _dequantize_triton(q, scales, group_size, orig_D)
    return _dequantize_pt(q, scales, group_size, orig_D)


# ---------------------------------------------------------------------------
# Autograd wrapper for fake quantization (STE)
# ---------------------------------------------------------------------------


class _QuantizeDequantizeSTE(torch.autograd.Function):
    """Fake quantisation with straight-through estimator.

    Forward: quantise to INT4 then immediately dequantise.
    Backward: pass gradient through unchanged (STE).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        orig_shape = x.shape
        D = x.shape[-1]
        x_2d = x.reshape(-1, D).contiguous()

        q, scales = _quantize_dispatch(x_2d, group_size)
        out = _dequantize_dispatch(q, scales, group_size, D)
        return out.reshape(orig_shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantize(
    x: torch.Tensor, group_size: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to symmetric INT4 with absmax scaling.

    Args:
        x: Input tensor of arbitrary shape ``(..., D)``.
        group_size: Number of contiguous elements sharing one scale factor
            along the last dimension.  ``-1`` means per-row (one scale per
            row of the last dimension).

    Returns:
        q: Quantized tensor (same shape as ``x``, dtype ``int8``,
            values in ``[-8, 7]``).
        scales: Scale factors, shape ``(*x.shape[:-1], num_groups)`` where
            ``num_groups = ceil(D / group_size)``.
    """
    orig_shape = x.shape
    D = x.shape[-1]
    batch_shape = x.shape[:-1]

    x_2d = x.reshape(-1, D).contiguous()
    gs = _resolve_group_size(group_size, D)

    q, scales = _quantize_dispatch(x_2d, gs)

    num_groups = scales.shape[1]
    return q.reshape(orig_shape), scales.reshape(*batch_shape, num_groups)


def dequantize(
    q: torch.Tensor, scales: torch.Tensor, group_size: int = -1
) -> torch.Tensor:
    """Dequantize an INT4 tensor back to float.

    Args:
        q: Quantized tensor ``(..., D)``, dtype ``int8``.
        scales: Scale factors ``(..., num_groups)``, dtype ``float32``.
        group_size: Must match the value used during quantization.

    Returns:
        Reconstructed float tensor with the same shape as ``q``.
    """
    orig_shape = q.shape
    D = q.shape[-1]
    gs = _resolve_group_size(group_size, D)

    q_2d = q.reshape(-1, D)
    scales_2d = scales.reshape(-1, scales.shape[-1])

    out = _dequantize_dispatch(q_2d, scales_2d, gs, D)
    return out.reshape(orig_shape)


def quantize_dequantize(
    x: torch.Tensor, group_size: int = -1
) -> torch.Tensor:
    """Fake quantisation for training: quantize then dequantize with STE.

    The forward pass simulates INT4 quantization noise.  The backward pass
    uses the straight-through estimator (gradient passes unchanged).

    Args:
        x: Input tensor of arbitrary shape ``(..., D)``.
        group_size: Group size for absmax scaling (``-1`` = per-row).

    Returns:
        Tensor with the same shape and dtype as ``x``, containing the
        quantize-then-dequantize approximation.
    """
    return _QuantizeDequantizeSTE.apply(x, group_size)
