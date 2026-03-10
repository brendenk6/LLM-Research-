"""Rotary Position Embeddings (RoPE) for GENESIS.

Precomputes sin/cos tables and applies rotary embeddings to query and key
tensors inside attention layers.

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes and caches sin/cos positional embedding tables."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ) -> None:
        """Initialise RotaryEmbedding.

        Args:
            dim: Head dimension (must be even).
            max_seq_len: Maximum sequence length to precompute.
            base: Base frequency for the sinusoidal schedule.
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even."
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the inverse frequency vector: shape (dim // 2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build and cache the cos/sin tables.
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache up to *seq_len*."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # Outer product: (seq_len, dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate for pairs: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tables for positions 0..seq_len-1.

        Args:
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin) each of shape (seq_len, dim).
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension by negating and swapping."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a query or key tensor.

    Args:
        x: Tensor of shape (B, num_heads, S, head_dim).
        cos: Cosine table of shape (S, head_dim) or broadcastable.
        sin: Sine table of shape (S, head_dim) or broadcastable.

    Returns:
        Tensor of the same shape with RoPE applied.
    """
    # cos/sin: (S, head_dim) -> (1, 1, S, head_dim) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)
