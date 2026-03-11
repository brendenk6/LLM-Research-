"""Latent Pooling for GENESIS.

Cross-attention mechanism that compresses variable-length token chunks into
a fixed number of learned latent query vectors.  This bridges Tier 1 (token
level) and Tier 2 (semantic level) by creating compact chunk summaries.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPooling(nn.Module):
    """Cross-attention pooling from learned queries to token chunks.

    For each escalated chunk, a set of ``num_latent_vectors`` learned queries
    attend over the chunk's token representations to produce a compressed
    latent summary.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latent_vectors: int = 8,
        num_heads: int = 4,
        chunk_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """Initialise LatentPooling.

        Args:
            input_dim: Dimension of incoming token representations (Tier 1 output).
            output_dim: Dimension of latent vectors (Tier 2 input).
            num_latent_vectors: Number of learned query vectors per chunk.
            num_heads: Number of attention heads for cross-attention.
            chunk_size: Number of tokens per chunk (must match TierGate).
            dropout: Dropout probability for attention weights.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latent_vectors = num_latent_vectors
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Learned query vectors: (num_latent_vectors, output_dim)
        self.learned_queries = nn.Parameter(
            torch.randn(num_latent_vectors, output_dim) * 0.02
        )

        # Input projection from input_dim to output_dim for K and V
        self.input_proj = nn.Linear(input_dim, output_dim, bias=False)

        # Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(output_dim, output_dim, bias=False)
        self.k_proj = nn.Linear(output_dim, output_dim, bias=False)
        self.v_proj = nn.Linear(output_dim, output_dim, bias=False)
        self.o_proj = nn.Linear(output_dim, output_dim, bias=False)

        self.dropout_p = dropout
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token chunks into latent vectors via cross-attention.

        Args:
            x: Token representations of shape (B, S, input_dim).
            chunk_mask: Boolean or float mask of shape (B, num_chunks) indicating
                which chunks are escalated.

        Returns:
            Latent vectors of shape (B, num_escalated_chunks, num_latent_vectors, output_dim).
            If no chunks are escalated for a sample, that sample gets zero
            escalated chunks (ragged batching handled by padding).
        """
        B, S, _ = x.shape

        # Pad sequence to be divisible by chunk_size
        remainder = S % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            x = F.pad(x, (0, 0, 0, pad_len))
            S = x.size(1)

        num_chunks = S // self.chunk_size

        # Project input tokens to output_dim
        x_proj = self.input_proj(x)  # (B, S, output_dim)

        # Reshape into chunks: (B, num_chunks, chunk_size, output_dim)
        chunks = x_proj.view(B, num_chunks, self.chunk_size, self.output_dim)

        # Determine escalated chunks - use mask to select
        # For uniform batching, we find the max number of escalated chunks
        if chunk_mask.dtype == torch.bool:
            bool_mask = chunk_mask
        else:
            # Soft mask during training - use all chunks weighted
            bool_mask = chunk_mask > 0.5

        # Count escalated chunks per sample
        counts = bool_mask.sum(dim=-1)  # (B,)
        max_escalated = max(int(counts.max().item()), 1)

        # Gather escalated chunks with padding
        # Create indices for gathering
        escalated_chunks = torch.zeros(
            B, max_escalated, self.chunk_size, self.output_dim,
            device=x.device, dtype=x_proj.dtype,
        )
        escalated_valid = torch.zeros(B, max_escalated, device=x.device, dtype=torch.bool)

        for b in range(B):
            indices = bool_mask[b].nonzero(as_tuple=False).squeeze(-1)
            n = indices.size(0)
            if n > 0:
                escalated_chunks[b, :n] = chunks[b, indices]
                escalated_valid[b, :n] = True

        # Cross-attention: learned queries attend to each chunk's tokens
        # Queries: (num_latent_vectors, output_dim) -> broadcast to (B * max_escalated, L, output_dim)
        N = max_escalated
        L = self.num_latent_vectors

        # Flatten batch and chunk dims for efficient attention
        kv_input = escalated_chunks.view(B * N, self.chunk_size, self.output_dim)

        # Expand queries for all (batch, chunk) pairs
        queries = self.learned_queries.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, L, output_dim)

        # Project Q, K, V
        q = self.q_proj(queries)  # (B*N, L, output_dim)
        k = self.k_proj(kv_input)  # (B*N, chunk_size, output_dim)
        v = self.v_proj(kv_input)  # (B*N, chunk_size, output_dim)

        # Reshape for multi-head attention
        q = q.view(B * N, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * N, self.chunk_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * N, self.chunk_size, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B*N, H, L, head_dim), k/v: (B*N, H, chunk_size, head_dim)

        # Scaled dot-product attention (no causal mask needed for cross-attention)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*N, H, L, chunk_size)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        attn_out = torch.matmul(attn_weights, v)  # (B*N, H, L, head_dim)

        # Merge heads: (B*N, H, L, head_dim) -> (B*N, L, output_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B * N, L, self.output_dim)
        attn_out = self.o_proj(attn_out)
        attn_out = self.norm(attn_out)

        # Reshape back: (B, N, L, output_dim)
        latent_vectors = attn_out.view(B, N, L, self.output_dim)

        # Zero out padding positions
        latent_vectors = latent_vectors * escalated_valid.unsqueeze(-1).unsqueeze(-1).float()

        return latent_vectors
