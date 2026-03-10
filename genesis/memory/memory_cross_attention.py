"""
MemoryCrossAttention: Efficient cross-attention mechanism for memory reads.

Provides standard multi-head cross-attention where queries come from hidden
states and keys/values come from memory entries. Supports variable-length
memory banks via an optional mask.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCrossAttention(nn.Module):
    """Multi-head cross-attention from hidden states to a memory bank.

    Q is projected from the query (hidden states), while K and V are
    projected from the memory bank. An optional mask allows variable
    numbers of valid memory entries per batch element.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_memory: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialise MemoryCrossAttention.

        Args:
            d_model: Dimension of the query (hidden state) vectors.
            num_heads: Number of attention heads.
            d_memory: Dimension of memory vectors.  Defaults to ``d_model``.
            dropout: Attention dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory if d_memory is not None else d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(self.d_memory, d_model)
        self.v_proj = nn.Linear(self.d_memory, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend from query to memory.

        Args:
            query: (B, S_q, d_model) hidden states used as queries.
            memory_keys: (B, M, d_memory) or (M, d_memory) memory key vectors.
            memory_values: (B, M, d_memory) or (M, d_memory) memory value vectors.
            memory_mask: Optional (B, M) boolean mask where ``True`` indicates
                a **valid** memory entry.  Entries marked ``False`` are ignored.

        Returns:
            (B, S_q, d_model) attended output.
        """
        # Handle unbatched memory (shared across all batch elements)
        if memory_keys.dim() == 2:
            memory_keys = memory_keys.unsqueeze(0).expand(query.size(0), -1, -1)
        if memory_values.dim() == 2:
            memory_values = memory_values.unsqueeze(0).expand(query.size(0), -1, -1)

        B, S_q, _ = query.shape
        M = memory_keys.size(1)

        # Project Q, K, V
        q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_keys).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_values).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, H, S_q, head_dim), k/v: (B, H, M, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S_q, M)

        # Apply memory mask: invalid entries get -inf
        if memory_mask is not None:
            # memory_mask: (B, M) -> (B, 1, 1, M)
            mask = memory_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # Weighted sum of values
        attn_out = torch.matmul(attn_weights, v)  # (B, H, S_q, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, -1)

        return self.o_proj(attn_out)
