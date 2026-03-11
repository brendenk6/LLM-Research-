"""HLRT: Hierarchical Latent Reasoning Transformer for GENESIS.

Top-level model that orchestrates the full multi-tier forward pass:

    Embed -> Tier 1 -> Gate -> Pool -> Tier 2 -> Gate -> Tier 3
         -> Condition -> Project logits

Inherits from StatefulModule to maintain persistent state (e.g., cached
plan vectors) across forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.core.stateful_module import StatefulModule

from genesis.model.embeddings import SharedEmbeddings
from genesis.model.rmsnorm import RMSNorm
from genesis.model.tier1_token_processor import Tier1TokenProcessor
from genesis.model.tier_gate import TierGate
from genesis.model.latent_pooling import LatentPooling
from genesis.model.tier2_semantic_planner import Tier2SemanticPlanner
from genesis.model.tier3_deliberative import Tier3DeliberativeReasoner
from genesis.model.conditioning import TopDownConditioning


@dataclass
class HLRTConfig:
    """Full configuration for the HLRT model."""

    # Vocabulary and embedding
    vocab_size: int = 100287
    d_model: int = 1024
    padding_idx: int | None = None
    embedding_dropout: float = 0.0
    tie_word_embeddings: bool = True

    # Tier 1: Token Processor
    tier1_num_layers: int = 12
    tier1_num_heads: int = 16
    tier1_num_kv_heads: int | None = None
    tier1_d_ff: int | None = None
    tier1_dropout: float = 0.0
    tier1_use_flash_attention: bool = True
    tier1_max_seq_len: int = 8192
    tier1_rope_base: float = 10000.0

    # Tier Gate 1->2
    gate1_chunk_size: int = 32
    gate1_threshold: float = 0.5

    # Latent Pooling
    num_latent_vectors: int = 8
    latent_pool_num_heads: int = 4
    latent_pool_dropout: float = 0.0

    # Tier 2: Semantic Planner
    tier2_d_model: int = 1536
    tier2_num_layers: int = 8
    tier2_num_heads: int = 16
    tier2_num_kv_heads: int | None = None
    tier2_d_ff: int | None = None
    tier2_dropout: float = 0.0
    tier2_use_flash_attention: bool = True
    tier2_max_seq_len: int = 2048
    tier2_rope_base: float = 10000.0
    tier2_moe_num_experts: int | None = None
    tier2_moe_top_k: int = 2

    # Tier Gate 2->3
    gate2_chunk_size: int = 1  # operates on latent vectors, not tokens
    gate2_threshold: float = 0.5

    # Tier 3: Deliberative Reasoner
    tier3_d_model: int = 1024
    tier3_num_layers: int = 5
    tier3_num_heads: int = 8
    tier3_num_kv_heads: int | None = None
    tier3_d_ff: int | None = None
    tier3_dropout: float = 0.0
    tier3_use_flash_attention: bool = True
    tier3_max_seq_len: int = 1024
    tier3_rope_base: float = 10000.0
    tier3_recurrence_steps: int = 4

    # General
    norm_eps: float = 1e-6

    # Loss weights
    gate_loss_weight: float = 0.01
    moe_loss_weight: float = 0.01


class HLRT(StatefulModule):
    """Hierarchical Latent Reasoning Transformer.

    Multi-tier architecture that progressively escalates difficult content
    through deeper processing stages:

    - **Tier 1** (Token Processor): Standard Transformer on full sequence.
    - **TierGate 1->2**: Selects which token chunks need deeper processing.
    - **Latent Pooling**: Compresses escalated chunks into latent vectors.
    - **Tier 2** (Semantic Planner): Deeper/wider Transformer on latents.
    - **TierGate 2->3**: Selects which latents need deliberative reasoning.
    - **Tier 3** (Deliberative Reasoner): Recurrent Transformer for hard cases.
    - **Top-Down Conditioning**: Projects higher-tier results back to tokens.
    """

    def __init__(self, config: HLRTConfig) -> None:
        """Initialise HLRT.

        Args:
            config: Full model configuration.
        """
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.embeddings = SharedEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            dropout=config.embedding_dropout,
            padding_idx=config.padding_idx,
        )

        # --- Tier 1: Token Processor ---
        self.tier1 = Tier1TokenProcessor(
            num_layers=config.tier1_num_layers,
            d_model=config.d_model,
            num_heads=config.tier1_num_heads,
            d_ff=config.tier1_d_ff,
            num_kv_heads=config.tier1_num_kv_heads,
            dropout=config.tier1_dropout,
            use_flash_attention=config.tier1_use_flash_attention,
            max_seq_len=config.tier1_max_seq_len,
            rope_base=config.tier1_rope_base,
            norm_eps=config.norm_eps,
        )

        # --- Gate 1->2 ---
        self.gate1 = TierGate(
            d_model=config.d_model,
            chunk_size=config.gate1_chunk_size,
            threshold=config.gate1_threshold,
        )

        # --- Latent Pooling ---
        self.latent_pool = LatentPooling(
            input_dim=config.d_model,
            output_dim=config.tier2_d_model,
            num_latent_vectors=config.num_latent_vectors,
            num_heads=config.latent_pool_num_heads,
            chunk_size=config.gate1_chunk_size,
            dropout=config.latent_pool_dropout,
        )

        # --- Tier 2: Semantic Planner ---
        self.tier2 = Tier2SemanticPlanner(
            num_layers=config.tier2_num_layers,
            d_model=config.tier2_d_model,
            num_heads=config.tier2_num_heads,
            d_ff=config.tier2_d_ff,
            num_kv_heads=config.tier2_num_kv_heads,
            dropout=config.tier2_dropout,
            use_flash_attention=config.tier2_use_flash_attention,
            max_seq_len=config.tier2_max_seq_len,
            rope_base=config.tier2_rope_base,
            norm_eps=config.norm_eps,
            moe_num_experts=config.tier2_moe_num_experts,
            moe_top_k=config.tier2_moe_top_k,
        )

        # --- Gate 2->3 ---
        self.gate2 = TierGate(
            d_model=config.tier2_d_model,
            chunk_size=config.gate2_chunk_size,
            threshold=config.gate2_threshold,
        )

        # --- Tier 3: Deliberative Reasoner ---
        self.tier3 = Tier3DeliberativeReasoner(
            num_layers=config.tier3_num_layers,
            input_dim=config.tier2_d_model,
            d_model=config.tier3_d_model,
            num_heads=config.tier3_num_heads,
            d_ff=config.tier3_d_ff,
            num_kv_heads=config.tier3_num_kv_heads,
            dropout=config.tier3_dropout,
            use_flash_attention=config.tier3_use_flash_attention,
            max_seq_len=config.tier3_max_seq_len,
            rope_base=config.tier3_rope_base,
            norm_eps=config.norm_eps,
            recurrence_steps=config.tier3_recurrence_steps,
        )

        # --- Top-Down Conditioning ---
        self.conditioning = TopDownConditioning(
            d_model=config.d_model,
            tier2_dim=config.tier2_d_model,
            chunk_size=config.gate1_chunk_size,
        )

        # --- Output projection ---
        self.output_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.tie_word_embeddings:
            # Weight-tied: reuse embedding weight for output projection
            self.output_proj = None
        else:
            self.output_proj = nn.Linear(
                config.d_model, config.vocab_size, bias=False
            )

        # --- Stateful: cache the last plan vector ---
        self.register_state(
            "last_plan_vector",
            torch.zeros(1, config.d_model),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply weight initialisation heuristics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_tier_activations: bool = False,
        kv_cache: Optional[Any] = None,
        position_offset: int = 0,
    ) -> Dict[str, Any]:
        """Full hierarchical forward pass.

        Args:
            input_ids: Token ids of shape (B, S).
            attention_mask: Optional additive attention mask.
            return_tier_activations: If True, include intermediate tier outputs
                in the return dict.
            kv_cache: Optional KVCache for Tier 1 incremental decode.
            position_offset: Starting position for RoPE when using cache.

        Returns:
            Dictionary with keys:
                - "logits": Output logits of shape (B, S, vocab_size).
                - "tier_activations": (optional) dict of intermediate outputs.
                - "aux_loss": Auxiliary loss from gates and MoE routers.
        """
        B, S = input_ids.shape
        tier_activations: Dict[str, Any] = {}

        # ---- Embedding ----
        x = self.embeddings(input_ids)  # (B, S, d_model)

        # ---- Tier 1: Token Processing ----
        tier1_out = self.tier1(
            x, attention_mask=attention_mask,
            kv_cache=kv_cache, position_offset=position_offset,
        )  # (B, S, d_model)
        if return_tier_activations:
            tier_activations["tier1"] = tier1_out

        # During single-token cached decode, skip tier gating entirely.
        # The gate needs chunk-level context to make meaningful decisions,
        # which isn't available when processing a single token. Use the
        # plan vector computed during prefill instead.
        if kv_cache is not None and S == 1:
            cached_plan = self.link_state("last_plan_vector")
            if cached_plan.shape[0] == 1 and B > 1:
                cached_plan = cached_plan.expand(B, -1)
            elif cached_plan.shape[0] != B:
                cached_plan = cached_plan[:1].expand(B, -1)
            tier1_out = self.conditioning.apply_cached_plan(tier1_out, cached_plan)

            h = self.output_norm(tier1_out)
            if self.output_proj is not None:
                logits = self.output_proj(h)
            else:
                logits = F.linear(h, self.embeddings.weight)

            result: Dict[str, Any] = {
                "logits": logits,
                "hidden_states": h,
                "aux_loss": torch.tensor(0.0, device=input_ids.device),
            }
            if return_tier_activations:
                result["tier_activations"] = tier_activations
            return result

        # ---- Gate 1->2: Select chunks for escalation ----
        gate1_mask, gate1_scores = self.gate1(tier1_out)  # (B, num_chunks), (B, num_chunks)
        if return_tier_activations:
            tier_activations["gate1_scores"] = gate1_scores

        # Check if any chunks are escalated
        if gate1_mask.dtype == torch.bool:
            any_escalated_t2 = gate1_mask.any().item()
        else:
            any_escalated_t2 = (gate1_mask > 0.5).any().item()

        aux_loss = self.config.gate_loss_weight * self.gate1.load_balance_loss()

        if any_escalated_t2:
            # ---- Latent Pooling ----
            latent_vectors = self.latent_pool(tier1_out, gate1_mask)
            # latent_vectors: (B, num_escalated_chunks, num_latent_vectors, tier2_d_model)
            if return_tier_activations:
                tier_activations["latent_vectors"] = latent_vectors

            # ---- Tier 2: Semantic Planning ----
            tier2_out = self.tier2(latent_vectors)
            # tier2_out: (B, num_escalated_chunks, num_latent_vectors, tier2_d_model)
            if return_tier_activations:
                tier_activations["tier2"] = tier2_out

            # Accumulate MoE load balance loss from Tier 2
            aux_loss = aux_loss + self.config.moe_loss_weight * self.tier2.load_balance_loss()

            # Compute and cache plan vector
            plan_vector = self.conditioning.compute_plan_vector(tier2_out)  # (B, d_model)
            self.set_state("last_plan_vector", plan_vector.detach().mean(dim=0, keepdim=True))

            # Apply plan to Tier 1 output
            tier1_out = self.conditioning.apply_cached_plan(tier1_out, plan_vector)

            # ---- Gate 2->3: Select latent chunks for deliberation ----
            # Flatten tier2 output for gating: (B, N*L, tier2_d_model)
            t2_flat = tier2_out.view(B, -1, self.config.tier2_d_model)
            gate2_mask, gate2_scores = self.gate2(t2_flat)
            if return_tier_activations:
                tier_activations["gate2_scores"] = gate2_scores

            aux_loss = aux_loss + self.config.gate_loss_weight * self.gate2.load_balance_loss()

            # Check if any chunks need Tier 3
            if gate2_mask.dtype == torch.bool:
                any_escalated_t3 = gate2_mask.any().item()
            else:
                any_escalated_t3 = (gate2_mask > 0.5).any().item()

            if any_escalated_t3:
                # ---- Tier 3: Deliberative Reasoning ----
                tier3_out = self.tier3(tier2_out)
                # tier3_out: (B, N, L, tier2_d_model)
                if return_tier_activations:
                    tier_activations["tier3"] = tier3_out

                # Integrate Tier 3 back into Tier 1
                tier1_out = self.conditioning.integrate_tier3(
                    tier1_out, tier3_out, gate1_mask, gate2_mask,
                )
        else:
            # No escalation: try to use cached plan vector
            cached_plan = self.link_state("last_plan_vector")
            if cached_plan.shape[0] == 1 and B > 1:
                cached_plan = cached_plan.expand(B, -1)
            elif cached_plan.shape[0] != B:
                cached_plan = cached_plan[:1].expand(B, -1)
            tier1_out = self.conditioning.apply_cached_plan(tier1_out, cached_plan)

        # ---- Output Projection ----
        h = self.output_norm(tier1_out)

        if self.output_proj is not None:
            logits = self.output_proj(h)
        else:
            # Weight-tied output projection
            logits = F.linear(h, self.embeddings.weight)

        result: Dict[str, Any] = {
            "logits": logits,
            "hidden_states": h,
            "aux_loss": aux_loss,
        }
        if return_tier_activations:
            result["tier_activations"] = tier_activations

        return result
