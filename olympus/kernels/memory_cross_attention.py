"""Optimized cross-attention kernel for PHMA memory reads.

Replaces the manual matmul -> mask -> softmax -> matmul pipeline in
MemoryCrossAttention with PyTorch's ``scaled_dot_product_attention``
(which dispatches to FlashAttention / memory-efficient attention on CUDA)
and adds a Triton kernel for fused mask expansion.

Falls back to the manual implementation when SDPA is unavailable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Check for SDPA availability (PyTorch >= 2.0)
HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


# ---------------------------------------------------------------------------
# Triton kernel: fused memory mask expansion
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _expand_memory_mask_kernel(
        mask_ptr,  # (B, M) bool input
        out_ptr,   # (B, 1, 1, M) float output
        B,
        M,
        BLOCK_M: tl.constexpr,
    ):
        """Expand (B, M) bool mask to (B, 1, 1, M) float attention mask.

        True entries become 0.0 (attend), False entries become -inf (ignore).
        Fuses the unsqueeze + masked_fill into a single kernel.
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = offs < M

        # Load bool mask
        m_offs = pid_b * M + offs
        mask_val = tl.load(mask_ptr + m_offs, mask=valid, other=0)

        # Convert: True -> 0.0, False -> -inf
        attn_bias = tl.where(mask_val != 0, 0.0, float("-inf"))

        # Store to (B, 1, 1, M) layout: stride is M along last dim
        out_offs = pid_b * M + offs
        tl.store(out_ptr + out_offs, attn_bias, mask=valid)


def _expand_mask_triton(
    mask: torch.Tensor,
) -> torch.Tensor:
    """Convert (B, M) bool to (B, 1, 1, M) float attention bias via Triton."""
    B, M = mask.shape
    out = torch.empty(B, M, device=mask.device, dtype=torch.float32)

    BLOCK_M = triton.next_power_of_2(min(M, 1024))
    grid = (B, triton.cdiv(M, BLOCK_M))

    _expand_memory_mask_kernel[grid](
        mask, out, B, M, BLOCK_M=BLOCK_M,
    )
    return out.view(B, 1, 1, M)


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------


def _expand_mask_pt(mask: torch.Tensor) -> torch.Tensor:
    """Convert (B, M) bool to (B, 1, 1, M) float attention bias."""
    # True = attend (0.0), False = ignore (-inf)
    expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, M)
    return torch.where(expanded, 0.0, float("-inf"))


def _manual_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scale: float,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Manual cross-attention (reference implementation)."""
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    return torch.matmul(attn_weights, v)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def memory_cross_attention(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    num_heads: int,
    memory_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Optimized cross-attention from hidden states to memory bank.

    Uses ``F.scaled_dot_product_attention`` (FlashAttention / memory-efficient
    attention) when available on CUDA, falling back to manual attention
    otherwise.  Memory mask expansion is accelerated via Triton.

    Args:
        query: Hidden states ``(B, S_q, d_model)``.
        memory_keys: Memory key vectors ``(B, M, d_memory)`` or ``(M, d_memory)``.
        memory_values: Memory value vectors ``(B, M, d_memory)`` or ``(M, d_memory)``.
        q_proj: Query projection ``(d_model -> d_model)``.
        k_proj: Key projection ``(d_memory -> d_model)``.
        v_proj: Value projection ``(d_memory -> d_model)``.
        o_proj: Output projection ``(d_model -> d_model)``.
        num_heads: Number of attention heads.
        memory_mask: Optional ``(B, M)`` bool mask (True = valid entry).
        dropout_p: Attention dropout (training only).
        training: Whether in training mode.

    Returns:
        Attended output ``(B, S_q, d_model)``.
    """
    # Handle unbatched memory
    if memory_keys.dim() == 2:
        memory_keys = memory_keys.unsqueeze(0).expand(query.size(0), -1, -1)
    if memory_values.dim() == 2:
        memory_values = memory_values.unsqueeze(0).expand(query.size(0), -1, -1)

    B, S_q, d_model = query.shape
    M = memory_keys.size(1)
    head_dim = d_model // num_heads

    # Project Q, K, V
    q = q_proj(query).view(B, S_q, num_heads, head_dim).transpose(1, 2)
    k = k_proj(memory_keys).view(B, M, num_heads, head_dim).transpose(1, 2)
    v = v_proj(memory_values).view(B, M, num_heads, head_dim).transpose(1, 2)

    # Prepare attention mask
    attn_mask = None
    if memory_mask is not None:
        if HAS_TRITON and memory_mask.is_cuda:
            attn_mask = _expand_mask_triton(memory_mask)
        else:
            attn_mask = _expand_mask_pt(memory_mask)

    # Core attention
    if HAS_SDPA and q.is_cuda:
        # SDPA uses FlashAttention / memory-efficient attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,  # cross-attention, not causal
        )
    else:
        scale = 1.0 / math.sqrt(head_dim)
        attn_out = _manual_cross_attention(
            q, k, v, attn_mask, scale, dropout_p, training
        )

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, d_model)
    return o_proj(attn_out)
