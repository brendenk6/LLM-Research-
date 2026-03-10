"""Verifier Model for ACT-V (Adversarial Co-Training with Verifier).

A smaller Transformer encoder that scores text quality.  Sized at ~30-40%
of the Generator's parameter count, it shares the same vocabulary and uses
the same building blocks (TransformerBlock, SharedEmbeddings, RMSNorm).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from genesis.model.tier1_token_processor import TransformerBlock
from genesis.model.embeddings import SharedEmbeddings
from genesis.model.rmsnorm import RMSNorm
from genesis.model.rotary import RotaryEmbedding


class VerifierModel(nn.Module):
    """Transformer encoder for scoring text quality.

    The verifier processes an input sequence and produces:
      - ``hidden_states``: per-token representations (B, S, d_model).
      - ``pooled_output``: a single vector per sequence (B, d_model)
        obtained by mean-pooling over non-masked positions, used as
        input to the verification heads.

    Architecture mirrors the Generator's Tier-1 stack but with fewer
    layers and a narrower hidden dimension (~30-40% of Generator params).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
    ) -> None:
        """Initialise VerifierModel.

        Args:
            vocab_size: Vocabulary size (shared with Generator).
            d_model: Hidden dimension.
            num_layers: Number of Transformer blocks.
            num_heads: Number of attention heads.
            d_ff: FFN intermediate dimension.
            max_seq_len: Maximum sequence length for RoPE cache.
            dropout: Dropout probability.
            norm_eps: Epsilon for RMSNorm.
            rope_base: Base frequency for RoPE.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Token embeddings (shared vocabulary with Generator)
        self.embeddings = SharedEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
        )

        # Rotary position embeddings
        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        # Transformer encoder stack
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_flash_attention=True,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = RMSNorm(d_model, eps=norm_eps)

        # Pooling projection (maps mean-pooled hidden state to a clean vector)
        self.pool_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for the pooling projection."""
        nn.init.xavier_uniform_(self.pool_proj.weight)
        nn.init.zeros_(self.pool_proj.bias)

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-pool hidden states over non-masked positions.

        Args:
            hidden_states: (B, S, d_model).
            attention_mask: (B, S) with 1 for real tokens, 0 for padding.

        Returns:
            (B, d_model) pooled representation.
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # attention_mask: (B, S) -> (B, S, 1)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, S) token IDs.
            attention_mask: Optional (B, S) mask with 1 for real tokens and
                0 for padding.  Converted internally to additive attention
                mask for the Transformer layers.

        Returns:
            Dict with:
                - ``hidden_states``: (B, S, d_model) final layer representations.
                - ``pooled_output``: (B, d_model) mean-pooled + projected vector.
        """
        # Embed tokens
        x = self.embeddings(input_ids)  # (B, S, d_model)

        # Build additive attention mask from boolean/int mask
        S = x.size(1)
        cos, sin = self.rotary_emb(S)

        additive_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # Convert (B, S) of 0/1 to (B, 1, 1, S) additive mask
            # where 0-positions get -inf
            additive_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            additive_mask = additive_mask * torch.finfo(x.dtype).min

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask=additive_mask)

        # Final norm
        hidden_states = self.final_norm(x)

        # Pool and project
        pooled = self._mean_pool(hidden_states, attention_mask)
        pooled_output = torch.tanh(self.pool_proj(pooled))

        return {
            "hidden_states": hidden_states,
            "pooled_output": pooled_output,
        }
