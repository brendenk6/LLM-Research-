"""Tier 1 Token Processor for GENESIS.

A standard pre-norm Transformer encoder stack that operates on the full
token sequence.  Each TransformerBlock applies:

    RMSNorm -> MultiHeadAttention -> residual -> RMSNorm -> FFN -> residual
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from genesis.model.rmsnorm import RMSNorm
from genesis.model.attention import MultiHeadAttention
from genesis.model.ffn import SwiGLUFFN
from genesis.model.rotary import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Single pre-norm Transformer block with RMSNorm, attention, and SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)

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
            x: (B, S, d_model).
            cos, sin: RoPE tables.
            attention_mask: Optional additive attention mask.
            kv_cache: Optional LayerKVCache for this layer.
            position_offset: Starting position for RoPE with cache.

        Returns:
            (B, S, d_model).
        """
        # Attention sub-layer with pre-norm and residual.
        h = self.attn_norm(x)
        h = self.attn(
            h, cos, sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        x = x + h

        # FFN sub-layer with pre-norm and residual.
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x


class Tier1TokenProcessor(nn.Module):
    """Stack of TransformerBlock layers forming the Tier-1 token processor."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """Initialise Tier1TokenProcessor.

        Args:
            num_layers: Number of Transformer blocks.
            d_model: Hidden dimension.
            num_heads: Number of query attention heads.
            d_ff: FFN hidden dimension (defaults to SwiGLU heuristic).
            num_kv_heads: Number of KV heads for GQA.
            dropout: Dropout probability.
            use_flash_attention: Use flash-attn if available.
            max_seq_len: Maximum sequence length for RoPE cache.
            rope_base: Base frequency for RoPE.
            norm_eps: Epsilon for RMSNorm.
        """
        super().__init__()
        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    use_flash_attention=use_flash_attention,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[object] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Forward pass through all Transformer blocks.

        Args:
            x: (B, S, d_model).
            attention_mask: Optional additive mask.
            kv_cache: Optional KVCache with one LayerKVCache per layer.
            position_offset: Starting position for RoPE when using cache.

        Returns:
            (B, S, d_model).
        """
        S = x.size(1)
        # Get RoPE tables for the full possible range
        total_len = position_offset + S
        cos, sin = self.rotary_emb(total_len)

        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = layer(
                x, cos, sin,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                position_offset=position_offset,
            )
        return x
