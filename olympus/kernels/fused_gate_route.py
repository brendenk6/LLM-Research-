"""Fused gate routing kernels for GENESIS.

Provides Triton-accelerated implementations of:
1. Chunk mean pooling (used by TierGate for HLRT tier escalation)
2. Softmax + top-k selection + normalization (used by ExpertRouter for MoE)

Falls back to equivalent PyTorch ops when Triton is unavailable (CPU, testing).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# PyTorch fallback implementations
# ---------------------------------------------------------------------------


def _chunk_mean_pool_pt(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Chunk mean pooling via reshape + mean (PyTorch fallback)."""
    B, S, D = x.shape
    num_chunks = S // chunk_size
    return x.view(B, num_chunks, chunk_size, D).mean(dim=2)


def _softmax_topk_pt(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Softmax + topk + normalize (PyTorch fallback)."""
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, dim=-1)
    top_k_weights = top_k_weights / (
        top_k_weights.sum(dim=-1, keepdim=True) + 1e-9
    )
    return top_k_weights, top_k_indices, probs


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _chunk_mean_pool_kernel(
        x_ptr,
        out_ptr,
        S,
        D,
        chunk_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused chunk mean pooling.

        Each program handles one (batch, chunk) pair across a block of the D
        dimension.  Streams chunk_size tokens, accumulates the sum in float32,
        and writes the mean.  Avoids materializing the
        (B, num_chunks, chunk_size, D) intermediate tensor that the naive
        view + mean approach requires.
        """
        pid_bc = tl.program_id(0)  # batch_idx * num_chunks + chunk_idx
        pid_d = tl.program_id(1)  # block index over the D dimension

        num_chunks = S // chunk_size
        batch_idx = pid_bc // num_chunks
        chunk_idx = pid_bc % num_chunks

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        base = batch_idx * S * D + chunk_idx * chunk_size * D

        for t in range(chunk_size):
            ptr = base + t * D + d_off
            val = tl.load(x_ptr + ptr, mask=d_mask, other=0.0)
            acc += val.to(tl.float32)

        acc = acc / chunk_size

        out_off = batch_idx * num_chunks * D + chunk_idx * D + d_off
        tl.store(out_ptr + out_off, acc, mask=d_mask)

    @triton.jit
    def _fused_softmax_topk_kernel(
        logits_ptr,
        weights_ptr,
        indices_ptr,
        probs_ptr,
        N,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_E: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        """Fused softmax + top-k selection + weight normalization.

        Each program handles one token.  Loads NUM_EXPERTS logits into
        registers, computes stable softmax, selects top-k experts via
        iterative argmax, and normalizes the selected weights to sum to 1.

        Full softmax probabilities are written to probs_ptr for downstream
        load-balance loss computation and backward pass caching.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs = tl.arange(0, BLOCK_E)
        valid = offs < NUM_EXPERTS

        # Load logits; invalid expert slots get -inf (zero after softmax)
        logits = tl.load(
            logits_ptr + pid * NUM_EXPERTS + offs,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)

        # Numerically stable softmax
        max_val = tl.max(logits, axis=0)
        exp_vals = tl.exp(logits - max_val)
        sum_exp = tl.sum(exp_vals, axis=0)
        probs = exp_vals / sum_exp

        # Cache full softmax probs
        tl.store(probs_ptr + pid * NUM_EXPERTS + offs, probs, mask=valid)

        # Iterative top-k: find maximum, record it, mask it out, repeat
        remaining = tl.where(valid, probs, 0.0)
        expert_ids = offs

        for k in range(TOP_K):
            best_val = tl.max(remaining, axis=0)
            is_best = remaining == best_val
            # Tie-break: pick lowest expert index
            idx_candidates = tl.where(is_best, expert_ids, BLOCK_E)
            best_idx = tl.min(idx_candidates, axis=0)

            tl.store(weights_ptr + pid * TOP_K + k, best_val)
            tl.store(indices_ptr + pid * TOP_K + k, best_idx.to(tl.int64))

            remaining = tl.where(expert_ids == best_idx, 0.0, remaining)

        # Normalize selected weights to sum to 1
        topk_sum = tl.load(weights_ptr + pid * TOP_K).to(tl.float32)
        for k in range(1, TOP_K):
            topk_sum += tl.load(weights_ptr + pid * TOP_K + k).to(tl.float32)

        for k in range(TOP_K):
            w = tl.load(weights_ptr + pid * TOP_K + k).to(tl.float32)
            tl.store(weights_ptr + pid * TOP_K + k, w / (topk_sum + 1e-9))


# ---------------------------------------------------------------------------
# Autograd wrappers (forward: Triton, backward: PyTorch)
# ---------------------------------------------------------------------------


class _ChunkMeanPoolFn(torch.autograd.Function):
    """Differentiable Triton chunk mean pooling."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        B, S, D = x.shape
        num_chunks = S // chunk_size
        out = torch.empty(
            B, num_chunks, D, device=x.device, dtype=torch.float32
        )

        BLOCK_D = triton.next_power_of_2(min(D, 1024))
        grid = (B * num_chunks, triton.cdiv(D, BLOCK_D))

        _chunk_mean_pool_kernel[grid](
            x,
            out,
            S,
            D,
            chunk_size=chunk_size,
            BLOCK_D=BLOCK_D,
        )

        ctx.chunk_size = chunk_size
        ctx.input_shape = (B, S, D)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        B, S, D = ctx.input_shape
        chunk_size = ctx.chunk_size
        num_chunks = S // chunk_size
        # Mean backward: replicate chunk gradient across all positions
        grad_input = (
            grad_output.unsqueeze(2)
            .expand(B, num_chunks, chunk_size, D)
            .reshape(B, S, D)
            / chunk_size
        )
        return grad_input, None


class _FusedSoftmaxTopKFn(torch.autograd.Function):
    """Differentiable Triton fused softmax + topk + normalize."""

    @staticmethod
    def forward(
        ctx, logits: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, E = logits.shape

        weights = torch.empty(N, top_k, device=logits.device, dtype=torch.float32)
        indices = torch.empty(N, top_k, device=logits.device, dtype=torch.int64)
        probs = torch.empty(N, E, device=logits.device, dtype=torch.float32)

        BLOCK_E = triton.next_power_of_2(E)
        grid = (N,)

        _fused_softmax_topk_kernel[grid](
            logits,
            weights,
            indices,
            probs,
            N,
            NUM_EXPERTS=E,
            BLOCK_E=BLOCK_E,
            TOP_K=top_k,
        )

        ctx.save_for_backward(probs, indices)
        ctx.top_k = top_k
        ctx.num_experts = E
        return weights, indices, probs

    @staticmethod
    def backward(ctx, grad_weights, grad_indices, grad_probs):
        probs, indices = ctx.saved_tensors
        N, E = probs.shape

        # Gather the softmax values that were selected by topk
        selected_probs = probs.gather(1, indices)  # (N, top_k)
        prob_sum = selected_probs.sum(dim=-1, keepdim=True) + 1e-9

        # Backward through normalization: w_k = p_k / sum(p_j)
        weighted_grad = (grad_weights * selected_probs / prob_sum).sum(
            dim=-1, keepdim=True
        )
        grad_selected = (grad_weights - weighted_grad) / prob_sum

        # Scatter gradients back to the full expert dimension
        grad_full = torch.zeros(N, E, device=probs.device, dtype=torch.float32)
        grad_full.scatter_add_(1, indices, grad_selected)

        # Include any direct gradient on probs (e.g. from load-balance loss)
        if grad_probs is not None:
            grad_full = grad_full + grad_probs

        # Backward through softmax: J = diag(p) - p p^T
        # => grad_logits = p * (grad - sum(p * grad))
        sum_pg = (probs * grad_full).sum(dim=-1, keepdim=True)
        grad_logits = probs * (grad_full - sum_pg)

        return grad_logits, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_mean_pool(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Chunk mean pooling with Triton acceleration on CUDA.

    Computes the mean of each non-overlapping chunk of ``chunk_size`` tokens
    along the sequence dimension.

    Args:
        x: Input tensor of shape ``(B, S, D)``.  ``S`` must be divisible by
            ``chunk_size``.
        chunk_size: Number of tokens per chunk.

    Returns:
        Mean-pooled tensor of shape ``(B, S // chunk_size, D)``.
    """
    if HAS_TRITON and x.is_cuda:
        return _ChunkMeanPoolFn.apply(x, chunk_size)
    return _chunk_mean_pool_pt(x, chunk_size)


def softmax_topk(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused softmax + top-k selection + weight normalization.

    Computes softmax over the last dimension, selects the ``top_k`` highest
    probabilities, and normalizes the selected weights to sum to 1.

    Args:
        logits: Router logits of shape ``(N, num_experts)``.
        top_k: Number of experts to select per token.

    Returns:
        weights: Normalized routing weights ``(N, top_k)``.
        indices: Selected expert indices ``(N, top_k)`` as int64.
        probs: Full softmax probabilities ``(N, num_experts)``.
    """
    if HAS_TRITON and logits.is_cuda:
        return _FusedSoftmaxTopKFn.apply(logits, top_k)
    return _softmax_topk_pt(logits, top_k)


def fused_gate_route(
    x: torch.Tensor,
    gate_mlp: nn.Module,
    router_gate: nn.Linear,
    chunk_size: int = 32,
    threshold: float = 0.5,
    top_k: int = 2,
    jitter_noise: float = 0.0,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined fused chunk gating + expert routing.

    Replaces ``TierGate.forward()`` and ``ExpertRouter.forward()`` with
    Triton-accelerated chunk mean pooling and softmax+topk kernels.  The
    gate MLP and router linear projection use standard cuBLAS GEMMs.

    Args:
        x: Input tokens ``(B, S, D)``.
        gate_mlp: Gate MLP module (d_model -> hidden -> 1).
        router_gate: Router linear projection (d_model -> num_experts).
        chunk_size: Tokens per chunk for tier gating.
        threshold: Inference-time sigmoid threshold for chunk mask.
        top_k: Number of experts per token.
        jitter_noise: Multiplicative jitter during training (0 = disabled).
        training: Whether in training mode.

    Returns:
        chunk_mask: ``(B, num_chunks)`` soft float (training) or bool (eval).
        chunk_scores: ``(B, num_chunks)`` raw sigmoid scores.
        expert_weights: ``(B*S, top_k)`` normalized routing weights.
        expert_indices: ``(B*S, top_k)`` selected expert indices.
        router_probs: ``(B*S, num_experts)`` full softmax probabilities.
    """
    B, S, D = x.shape

    # Pad sequence to chunk boundary
    remainder = S % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        x = F.pad(x, (0, 0, 0, pad_len))
        S = x.size(1)

    # --- Chunk gating (Triton-accelerated mean pooling) ---
    chunk_repr = chunk_mean_pool(x, chunk_size)  # (B, num_chunks, D)
    chunk_scores = torch.sigmoid(gate_mlp(chunk_repr).squeeze(-1))

    if training:
        chunk_mask = chunk_scores
    else:
        chunk_mask = chunk_scores >= threshold

    # --- Expert routing (Triton-accelerated softmax + topk) ---
    x_flat = x.reshape(-1, D)

    if training and jitter_noise > 0.0:
        noise = torch.empty_like(x_flat).uniform_(
            1.0 - jitter_noise, 1.0 + jitter_noise
        )
        x_flat = x_flat * noise

    router_logits = router_gate(x_flat)  # cuBLAS GEMM
    expert_weights, expert_indices, router_probs = softmax_topk(
        router_logits, top_k
    )
    expert_weights = expert_weights.type_as(x)

    return chunk_mask, chunk_scores, expert_weights, expert_indices, router_probs
