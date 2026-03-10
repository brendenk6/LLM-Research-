"""SharedEmbeddings for GENESIS.

Token embeddings that can be shared (weight-tied) with the output projection.
RoPE is applied in the attention layers, not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedEmbeddings(nn.Module):
    """Token embedding layer with optional dropout.

    The embedding weight matrix can be reused as the output (un-embedding)
    projection via :pyattr:`weight` to implement weight tying.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.0,
        padding_idx: int | None = None,
    ) -> None:
        """Initialise SharedEmbeddings.

        Args:
            vocab_size: Size of the token vocabulary.
            d_model: Embedding / model dimension.
            dropout: Dropout probability applied after embedding lookup.
            padding_idx: Optional padding token index (embedding zeroed).
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:
        """Return the embedding weight matrix (for weight tying)."""
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids.

        Args:
            input_ids: Long tensor of shape (B, S).

        Returns:
            Embedded tokens of shape (B, S, d_model).
        """
        x = self.embedding(input_ids)
        x = self.dropout(x)
        return x
