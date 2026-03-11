"""Multi-Head Attention with RoPE and GQA support for GENESIS.

Supports:
- Rotary Position Embeddings on Q and K
- Group Query Attention (GQA) via configurable num_kv_heads
- Flash Attention 2 (optional, falls back to scaled dot-product)
- Causal masking for autoregressive generation
- KV caching for incremental decode
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from genesis.model.rotary import apply_rotary_emb

# Try to import flash attention (Dao-AILab implementation).
try:
    from flash_attn import flash_attn_func  # type: ignore[import-untyped]

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MultiHeadAttention(nn.Module):
    """Multi-head (optionally grouped-query) attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash_attention: bool = True,
    ) -> None:
        """Initialise MultiHeadAttention.

        Args:
            d_model: Model hidden dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads for GQA.  Defaults to
                ``num_heads`` (standard MHA).  Set to 1 for multi-query attention.
            dropout: Attention dropout probability (only used with vanilla attn).
            bias: Whether to include bias in linear projections.
            use_flash_attention: Attempt to use flash-attn if installed.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads  # heads per KV group
        self.dropout_p = dropout
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE

        assert num_heads % self.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[object] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, S, d_model).
            cos: RoPE cosine table — (total_seq_len, head_dim) or larger.
            sin: RoPE sine table — same shape as cos.
            attention_mask: Optional additive mask (B, 1, S_q, S_kv) or None.
            kv_cache: Optional LayerKVCache. When provided, new K/V are
                appended and the full cached K/V are used for attention.
            position_offset: Starting position index for RoPE when using
                KV cache (i.e. number of previously cached positions).

        Returns:
            Output tensor (B, S, d_model).
        """
        B, S, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, S, head_dim)
        # k, v: (B, num_kv_heads, S, head_dim)

        # Apply RoPE — slice cos/sin for the correct positions
        cos_slice = cos[position_offset : position_offset + S]
        sin_slice = sin[position_offset : position_offset + S]
        q = apply_rotary_emb(q, cos_slice, sin_slice)
        k = apply_rotary_emb(k, cos_slice, sin_slice)

        # KV cache: append new K/V and use full cached sequence
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Total KV sequence length (may differ from S when using cache)
        S_kv = k.size(2)

        # Expand KV heads for GQA: replicate each KV head for its group.
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, S_kv, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, S_kv, self.head_dim)

        # Compute attention
        if self.use_flash_attention and q.is_cuda and kv_cache is None:
            # Flash attention only for prefill (no cache) — it handles
            # causal masking internally but doesn't support cross-length Q/KV.
            attn_out = self._flash_attention(q, k, v)
        else:
            attn_out = self._vanilla_attention(q, k, v, attention_mask)

        # attn_out: (B, num_heads, S, head_dim) -> (B, S, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Flash Attention 2 path.

        flash_attn_func expects (B, S, H, D) layout.
        """
        # Transpose from (B, H, S, D) to (B, S, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=True,
        )
        # Back to (B, H, S, D)
        return out.transpose(1, 2)

    def _vanilla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with causal mask."""
        S_q = q.size(2)
        S_kv = k.size(2)
        scale = 1.0 / math.sqrt(self.head_dim)

        # (B, H, S_q, S_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask: each query position can only attend to positions <= itself.
        # When S_q < S_kv (decode with cache), the query at position i
        # (absolute position = S_kv - S_q + i) can attend to all positions
        # up to and including itself.
        causal_mask = torch.triu(
            torch.full((S_q, S_kv), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=S_kv - S_q + 1,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        return torch.matmul(attn_weights, v)
