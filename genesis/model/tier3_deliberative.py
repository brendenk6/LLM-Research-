"""Tier 3 Deliberative Reasoner for GENESIS.

A recurrent Transformer that runs the same set of layers multiple times,
progressively refining a residual representation.  This enables iterative
computation for harder reasoning tasks without adding more unique parameters.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from genesis.model.rmsnorm import RMSNorm
from genesis.model.attention import MultiHeadAttention
from genesis.model.ffn import SwiGLUFFN
from genesis.model.rotary import RotaryEmbedding


class Tier3TransformerBlock(nn.Module):
    """Single pre-norm Transformer block for Tier 3 recurrent processing."""

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
    ) -> torch.Tensor:
        h = self.attn_norm(x)
        h = self.attn(h, cos, sin, attention_mask=attention_mask)
        x = x + h
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        return x


class Tier3DeliberativeReasoner(nn.Module):
    """Recurrent Transformer for iterative deliberative reasoning.

    The same set of Transformer layers is applied ``recurrence_steps`` times
    to progressively refine a residual hidden state.  This provides adaptive
    compute depth without proportionally increasing parameters.
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        recurrence_steps: int = 4,
    ) -> None:
        """Initialise Tier3DeliberativeReasoner.

        Args:
            num_layers: Number of Transformer blocks per recurrence step.
            input_dim: Dimension of input representations (from Tier 2).
            d_model: Internal hidden dimension for Tier 3 processing.
            num_heads: Number of query attention heads.
            d_ff: FFN hidden dimension.
            num_kv_heads: Number of KV heads for GQA.
            dropout: Dropout probability.
            use_flash_attention: Use flash-attn if available.
            max_seq_len: Maximum sequence length for RoPE.
            rope_base: Base frequency for RoPE.
            norm_eps: Epsilon for RMSNorm.
            recurrence_steps: Number of times to repeat the layer stack.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.recurrence_steps = recurrence_steps

        # Project from Tier 2 dim to Tier 3 internal dim
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        # Project back from Tier 3 internal dim to Tier 2 dim
        self.output_proj = nn.Linear(d_model, input_dim, bias=False)

        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        # Shared layers applied recurrently
        self.layers = nn.ModuleList(
            [
                Tier3TransformerBlock(
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

        self.final_norm = RMSNorm(d_model, eps=norm_eps)

        # Learned step embedding to differentiate recurrence iterations
        self.step_embedding = nn.Parameter(
            torch.randn(recurrence_steps, d_model) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with recurrent Transformer processing.

        Args:
            x: Input latent vectors of shape (B, N, L, input_dim) where
                N = num escalated chunks, L = latent vectors per chunk.
            attention_mask: Optional additive mask.

        Returns:
            Refined latent vectors of shape (B, N, L, input_dim).
        """
        B, N, L, _ = x.shape

        # Flatten chunk and latent dims: (B, N*L, input_dim)
        x_flat = x.view(B, N * L, self.input_dim)

        # Project to internal dimension
        h = self.input_proj(x_flat)  # (B, N*L, d_model)

        S = h.size(1)
        cos, sin = self.rotary_emb(S)

        # Recurrent application of the same layers
        residual = h
        for step in range(self.recurrence_steps):
            # Add step-specific embedding to distinguish iterations
            step_emb = self.step_embedding[step].unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            h = h + step_emb

            # Apply all transformer layers
            for layer in self.layers:
                h = layer(h, cos, sin, attention_mask=attention_mask)

            # Residual connection across recurrence steps
            h = h + residual

        h = self.final_norm(h)

        # Project back to input_dim
        out = self.output_proj(h)  # (B, N*L, input_dim)

        # Reshape to (B, N, L, input_dim)
        out = out.view(B, N, L, self.input_dim)
        return out
