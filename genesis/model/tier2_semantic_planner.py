"""Tier 2 Semantic Planner for GENESIS.

A deeper/wider Transformer stack that operates on the latent vectors produced
by LatentPooling.  Optionally uses Mixture-of-Experts FFN layers for
increased capacity without proportional compute cost.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from genesis.model.rmsnorm import RMSNorm
from genesis.model.attention import MultiHeadAttention
from genesis.model.ffn import SwiGLUFFN
from genesis.model.moe import MoEFFN
from genesis.model.rotary import RotaryEmbedding


class Tier2TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional MoE FFN for Tier 2."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        norm_eps: float = 1e-6,
        moe_num_experts: int | None = None,
        moe_top_k: int = 2,
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

        if moe_num_experts is not None and moe_num_experts > 1:
            self.ffn = MoEFFN(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                dropout=dropout,
            )
            self.is_moe = True
        else:
            self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
            self.is_moe = False

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, S, d_model).
            cos, sin: RoPE tables.
            attention_mask: Optional additive attention mask.

        Returns:
            (B, S, d_model).
        """
        h = self.attn_norm(x)
        h = self.attn(h, cos, sin, attention_mask=attention_mask)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x

    def load_balance_loss(self) -> torch.Tensor:
        """Return MoE load-balance loss if this block uses MoE, else 0."""
        if self.is_moe:
            return self.ffn.load_balance_loss()
        return torch.tensor(0.0)


class Tier2SemanticPlanner(nn.Module):
    """Stack of Tier2TransformerBlocks operating on latent representations.

    Takes the pooled latent vectors from LatentPooling, flattens them for
    attention across all escalated chunks, and produces refined semantic
    plan vectors.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        moe_num_experts: int | None = None,
        moe_top_k: int = 2,
    ) -> None:
        """Initialise Tier2SemanticPlanner.

        Args:
            num_layers: Number of Transformer blocks.
            d_model: Hidden dimension (typically wider than Tier 1).
            num_heads: Number of query attention heads.
            d_ff: FFN hidden dimension.
            num_kv_heads: Number of KV heads for GQA.
            dropout: Dropout probability.
            use_flash_attention: Use flash-attn if available.
            max_seq_len: Maximum flattened sequence length for RoPE.
            rope_base: Base frequency for RoPE.
            norm_eps: Epsilon for RMSNorm.
            moe_num_experts: Number of MoE experts (None for dense FFN).
            moe_top_k: Number of experts per token for MoE.
        """
        super().__init__()
        self.d_model = d_model

        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        self.layers = nn.ModuleList(
            [
                Tier2TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    use_flash_attention=use_flash_attention,
                    norm_eps=norm_eps,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(
        self,
        latent_vectors: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Tier 2 Transformer stack.

        Args:
            latent_vectors: Latent vectors of shape
                (B, num_chunks, latent_per_chunk, d_model) from LatentPooling.
            attention_mask: Optional additive mask.

        Returns:
            Refined latent vectors of the same shape as input.
        """
        B, num_chunks, latent_per_chunk, D = latent_vectors.shape

        # Flatten chunk and latent dims for attention: (B, num_chunks * L, D)
        S = num_chunks * latent_per_chunk
        x = latent_vectors.view(B, S, D)

        # Get RoPE tables
        cos, sin = self.rotary_emb(S)

        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask=attention_mask)

        x = self.final_norm(x)

        # Reshape back to (B, num_chunks, latent_per_chunk, D)
        x = x.view(B, num_chunks, latent_per_chunk, D)
        return x

    def load_balance_loss(self) -> torch.Tensor:
        """Aggregate load-balance loss from all MoE layers."""
        total = torch.tensor(0.0)
        for layer in self.layers:
            lb = layer.load_balance_loss()
            if lb.device != total.device:
                total = total.to(lb.device)
            total = total + lb
        return total
