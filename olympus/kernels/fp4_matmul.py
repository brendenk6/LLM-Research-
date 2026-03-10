"""FP4 quantized matrix multiplication kernel for GENESIS.

Replaces ``F.linear(x, weight, bias)`` with a quantize-on-the-fly matmul:
activations are quantized per-token, weights are pre-quantized per-channel,
and the output is produced in float32/bf16.

Falls back to standard ``F.linear`` when Triton is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.kernels.fp4_quantize import (
    _resolve_group_size,
    dequantize,
    quantize,
    quantize_dequantize,
)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------


def _fp4_matmul_pt(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_group_size: int = -1,
    weight_group_size: int = -1,
) -> torch.Tensor:
    """Simulated FP4 matmul via quantize-dequantize + standard matmul.

    Quantizes both activations and weights to INT4, dequantizes, then
    performs the matmul in float.  This matches the numerical behaviour of
    the fused Triton kernel but without the performance benefit.
    """
    # Fake-quantize activations (per-token)
    x_qdq = quantize_dequantize(x, group_size=act_group_size)

    # Fake-quantize weights (per-channel = per-row of weight matrix)
    w_qdq = quantize_dequantize(weight, group_size=weight_group_size)

    return F.linear(x_qdq, w_qdq, bias)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _fp4_matmul_kernel(
        # Pointers
        x_ptr,
        w_ptr,
        out_ptr,
        bias_ptr,
        # Matrix dimensions: x is (M, K), w is (N, K), out is (M, N)
        M,
        N,
        K,
        # Strides
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        # Quantization scales (pre-computed)
        x_scale_ptr,  # (M, x_num_groups)
        w_scale_ptr,  # (N, w_num_groups)
        x_num_groups,
        w_num_groups,
        X_GROUP_SIZE: tl.constexpr,
        W_GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Tiled FP4 matmul kernel.

        Each program computes a (BLOCK_M, BLOCK_N) tile of the output.
        Activations and weights are loaded as float, quantized to INT4 in
        registers (simulated via round+clamp), then accumulated in float32.

        The quantization scales are pre-computed per-group and loaded from
        global memory.  The effective computation is:

            out[m, n] = sum_k( qdq(x[m, k]) * qdq(w[n, k]) )

        where qdq denotes quantize-then-dequantize.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Offsets for this tile
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulator in float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Iterate over K dimension in tiles
        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)

            # --- Load activation tile ---
            x_offs = rm[:, None] * stride_xm + rk[None, :] * stride_xk
            x_mask = (rm[:, None] < M) & (rk[None, :] < K)
            x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(
                tl.float32
            )

            # --- Load weight tile ---
            w_offs = rn[:, None] * stride_wn + rk[None, :] * stride_wk
            w_mask = (rn[:, None] < N) & (rk[None, :] < K)
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0).to(
                tl.float32
            )

            # --- Quantize activations in-register ---
            # Per-token (row) quantization: load scale for each row in tile
            # x_scale shape: (M, x_num_groups)
            # For per-token: x_num_groups=1, group covers entire K
            x_group_idx = k_start // X_GROUP_SIZE
            x_scale_offs = rm * x_num_groups + x_group_idx
            x_scale_mask = rm < M
            x_scale = tl.load(
                x_scale_ptr + x_scale_offs, mask=x_scale_mask, other=1.0
            ).to(tl.float32)
            # x_scale: (BLOCK_M,) -> broadcast to (BLOCK_M, BLOCK_K)
            x_scaled = x_tile / x_scale[:, None]
            x_offset = tl.where(x_scaled >= 0.0, 0.5, -0.5)
            x_q = (x_scaled + x_offset).to(tl.int32)
            x_q = tl.maximum(tl.minimum(x_q, 7), -8)
            x_qdq = x_q.to(tl.float32) * x_scale[:, None]

            # --- Quantize weights in-register ---
            # Per-channel (row of W) quantization
            w_group_idx = k_start // W_GROUP_SIZE
            w_scale_offs = rn * w_num_groups + w_group_idx
            w_scale_mask = rn < N
            w_scale = tl.load(
                w_scale_ptr + w_scale_offs, mask=w_scale_mask, other=1.0
            ).to(tl.float32)
            # w_scale: (BLOCK_N,) -> broadcast to (BLOCK_N, BLOCK_K)
            w_scaled = w_tile / w_scale[:, None]
            w_offset = tl.where(w_scaled >= 0.0, 0.5, -0.5)
            w_q = (w_scaled + w_offset).to(tl.int32)
            w_q = tl.maximum(tl.minimum(w_q, 7), -8)
            w_qdq = w_q.to(tl.float32) * w_scale[:, None]

            # --- Accumulate: x_qdq (M,K) @ w_qdq^T (K,N) ---
            acc += tl.dot(x_qdq, tl.trans(w_qdq))

        # --- Add bias ---
        if HAS_BIAS:
            bias_offs = rn
            bias_mask = rn < N
            bias_val = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
            acc += bias_val[None, :]

        # --- Store output ---
        out_offs = rm[:, None] * stride_om + rn[None, :] * stride_on
        out_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(out_ptr + out_offs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class _FP4MatmulFn(torch.autograd.Function):
    """FP4 matmul with STE backward for quantization.

    Forward: quantize activations + weights in the Triton kernel, matmul.
    Backward: standard matmul gradients (STE for quantization steps).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        act_group_size: int,
        weight_group_size: int,
    ) -> torch.Tensor:
        orig_shape = x.shape
        M = x.reshape(-1, x.shape[-1]).shape[0]
        K = x.shape[-1]
        N = weight.shape[0]

        x_2d = x.reshape(M, K).contiguous()
        w_2d = weight.contiguous()  # (N, K)

        # Pre-compute quantization scales
        ags = _resolve_group_size(act_group_size, K)
        wgs = _resolve_group_size(weight_group_size, K)

        _, x_scales = quantize(x_2d, group_size=ags)  # (M, x_groups)
        _, w_scales = quantize(w_2d, group_size=wgs)  # (N, w_groups)
        x_scales = x_scales.reshape(M, -1).contiguous()
        w_scales = w_scales.reshape(N, -1).contiguous()

        x_num_groups = x_scales.shape[1]
        w_num_groups = w_scales.shape[1]

        out = torch.empty(M, N, device=x.device, dtype=torch.float32)

        # Grid: tile the (M, N) output
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = min(K, 64)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fp4_matmul_kernel[grid](
            x_2d,
            w_2d,
            out,
            bias if bias is not None else x_2d,  # dummy ptr when no bias
            M,
            N,
            K,
            x_2d.stride(0),
            x_2d.stride(1),
            w_2d.stride(0),
            w_2d.stride(1),
            out.stride(0),
            out.stride(1),
            x_scales,
            w_scales,
            x_num_groups,
            w_num_groups,
            X_GROUP_SIZE=ags,
            W_GROUP_SIZE=wgs,
            HAS_BIAS=bias is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        out_shape = (*orig_shape[:-1], N)
        out = out.reshape(out_shape).to(x.dtype)

        ctx.save_for_backward(x, weight, bias)
        ctx.act_group_size = act_group_size
        ctx.weight_group_size = weight_group_size

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias = ctx.saved_tensors

        # STE: treat quantization as identity in backward
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_x = grad_output @ weight
            grad_x = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output^T @ x
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_weight = go_2d.t().matmul(x_2d)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)

        return grad_x, grad_weight, grad_bias, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fp4_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_group_size: int = -1,
    weight_group_size: int = -1,
) -> torch.Tensor:
    """FP4-quantized linear projection (drop-in replacement for F.linear).

    Quantizes activations (per-token) and weights (per-channel) to INT4,
    performs the matmul, and returns the float result.

    Args:
        x: Input activations ``(..., K)``.
        weight: Weight matrix ``(N, K)`` (same layout as ``nn.Linear``).
        bias: Optional bias ``(N,)``.
        act_group_size: Quantization group size for activations (``-1`` =
            per-token, i.e. one scale per row).
        weight_group_size: Quantization group size for weights (``-1`` =
            per-channel, i.e. one scale per output neuron).

    Returns:
        Output tensor ``(..., N)``.
    """
    if HAS_TRITON and x.is_cuda:
        return _FP4MatmulFn.apply(x, weight, bias, act_group_size, weight_group_size)
    return _fp4_matmul_pt(x, weight, bias, act_group_size, weight_group_size)


class FP4Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with FP4 quantized matmul.

    Wraps an existing ``nn.Linear`` layer and routes its forward pass
    through :func:`fp4_matmul`.
    """

    def __init__(
        self,
        linear: nn.Linear,
        act_group_size: int = -1,
        weight_group_size: int = -1,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.act_group_size = act_group_size
        self.weight_group_size = weight_group_size

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp4_matmul(
            x,
            self.linear.weight,
            self.linear.bias,
            act_group_size=self.act_group_size,
            weight_group_size=self.weight_group_size,
        )
