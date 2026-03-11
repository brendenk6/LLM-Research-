"""KV Cache for autoregressive generation with HLRT.

Stores key/value tensors per layer so that during token-by-token generation
only the new token's K and V need to be computed and appended.  Supports
both standard MHA and GQA (stores num_kv_heads, expands on read).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LayerKVCache:
    """Cached K and V for a single attention layer."""

    key: Optional[torch.Tensor] = None    # (B, num_kv_heads, S_cached, head_dim)
    value: Optional[torch.Tensor] = None  # (B, num_kv_heads, S_cached, head_dim)

    @property
    def seq_len(self) -> int:
        """Number of cached positions."""
        if self.key is None:
            return 0
        return self.key.size(2)

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V and return the full cached K/V.

        Args:
            new_k: (B, num_kv_heads, S_new, head_dim)
            new_v: (B, num_kv_heads, S_new, head_dim)

        Returns:
            Full (key, value) tensors with all cached positions.
        """
        if self.key is None:
            self.key = new_k
            self.value = new_v
        else:
            self.key = torch.cat([self.key, new_k], dim=2)
            self.value = torch.cat([self.value, new_v], dim=2)
        return self.key, self.value

    def clear(self) -> None:
        """Reset the cache."""
        self.key = None
        self.value = None


class KVCache:
    """Full KV cache for all layers in a tier.

    Usage:
        cache = KVCache(num_layers=12)

        # Prefill (full sequence)
        for layer_idx, layer in enumerate(tier.layers):
            k, v = compute_kv(x)
            full_k, full_v = cache[layer_idx].update(k, v)

        # Decode (one token at a time)
        for step in range(max_new_tokens):
            for layer_idx, layer in enumerate(tier.layers):
                k_new, v_new = compute_kv(x_new)
                full_k, full_v = cache[layer_idx].update(k_new, v_new)
    """

    def __init__(self, num_layers: int) -> None:
        self.layers: list[LayerKVCache] = [LayerKVCache() for _ in range(num_layers)]

    def __getitem__(self, idx: int) -> LayerKVCache:
        return self.layers[idx]

    @property
    def seq_len(self) -> int:
        """Cached sequence length (from first layer)."""
        if not self.layers:
            return 0
        return self.layers[0].seq_len

    def clear(self) -> None:
        """Clear all layer caches."""
        for layer_cache in self.layers:
            layer_cache.clear()

    @staticmethod
    def estimate_memory(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Estimate KV cache memory in bytes.

        Args:
            num_layers: Number of attention layers.
            num_kv_heads: Number of KV heads per layer.
            head_dim: Dimension per head.
            max_seq_len: Maximum sequence length to cache.
            batch_size: Batch size.
            dtype: Data type for cached tensors.

        Returns:
            Estimated memory usage in bytes.
        """
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        # 2 for K and V
        return (
            2 * num_layers * batch_size * num_kv_heads
            * max_seq_len * head_dim * bytes_per_element
        )
