"""Text generator for HLRT with KV caching and cascade routing.

Handles the full autoregressive generation loop:
1. Prefill: process the prompt through all tiers, populate KV cache
2. Decode: generate tokens one at a time using cached K/V

The HLRT's multi-tier architecture means we only run Tier 2/3 when the
gate decides the current context is hard enough to warrant it.  During
decode, most tokens go through Tier 1 alone (cheap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from genesis.model.hlrt import HLRT, HLRTConfig
from genesis.inference.kv_cache import KVCache
from genesis.inference.sampling import SamplingConfig, sample
from genesis.inference.cascade_router import CascadeRouter


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    sampling: SamplingConfig = None  # type: ignore[assignment]
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    use_kv_cache: bool = True

    # Cascade routing overrides (None = use model defaults)
    gate1_threshold: float | None = None
    gate2_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.sampling is None:
            self.sampling = SamplingConfig()


class Generator:
    """Autoregressive text generator for HLRT models.

    Supports:
    - Greedy, top-k, top-p, and temperature sampling
    - KV caching for efficient decode
    - Cascade routing with tier activation tracking
    - EOS stopping
    - Batch generation

    Usage:
        model = HLRT(config)
        model.load_state_dict(...)
        model.eval()

        gen = Generator(model)
        output_ids = gen.generate(
            input_ids=prompt_ids,
            config=GenerationConfig(max_new_tokens=128),
        )
    """

    def __init__(self, model: HLRT) -> None:
        self.model = model
        self.config = model.config

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
    ) -> dict:
        """Generate text autoregressively.

        Args:
            input_ids: (B, S) prompt token ids.
            config: Generation configuration.

        Returns:
            Dictionary with:
                - "token_ids": (B, S + num_generated) full sequence
                - "new_token_ids": (B, num_generated) generated tokens only
                - "cascade_stats": CascadeStats with tier activation info
        """
        if config is None:
            config = GenerationConfig()

        self.model.eval()
        B, S = input_ids.shape

        router = CascadeRouter(
            gate1_threshold=config.gate1_threshold,
            gate2_threshold=config.gate2_threshold,
        )

        # Set up KV cache for Tier 1
        cache = None
        if config.use_kv_cache:
            cache = KVCache(num_layers=self.config.tier1_num_layers)

        # === Prefill: full forward pass on prompt ===
        prefill_out = self.model(
            input_ids,
            return_tier_activations=True,
            kv_cache=cache,
            position_offset=0,
        )
        logits = prefill_out["logits"]  # (B, S, vocab)

        # Track prefill tier activations
        activations = prefill_out.get("tier_activations", {})
        tier2_active = "tier2" in activations
        tier3_active = "tier3" in activations
        router.record(S, tier2_active, tier3_active)

        # Sample first new token from last position
        next_logits = logits[:, -1, :]  # (B, vocab)
        generated = []

        next_token = sample(next_logits, config.sampling)  # (B,)
        generated.append(next_token)

        # For uncached mode, track the full sequence
        all_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        # === Decode loop ===
        for step in range(1, config.max_new_tokens):
            # Check EOS
            if config.eos_token_id is not None:
                if (next_token == config.eos_token_id).all():
                    break

            if cache is not None:
                # Cached: only process the new token
                pos_offset = S + step - 1
                decode_ids = next_token.unsqueeze(1)  # (B, 1)
                out = self.model(
                    decode_ids,
                    return_tier_activations=True,
                    kv_cache=cache,
                    position_offset=pos_offset,
                )
            else:
                # Uncached: recompute full sequence
                out = self.model(
                    all_ids,
                    return_tier_activations=True,
                )
            logits = out["logits"]

            # Track tier activations
            activations = out.get("tier_activations", {})
            tier2_active = "tier2" in activations
            tier3_active = "tier3" in activations
            router.record(1, tier2_active, tier3_active)

            # Sample next token
            next_logits = logits[:, -1, :]
            generated_so_far = torch.stack(generated, dim=1) if generated else None
            next_token = sample(next_logits, config.sampling, generated_so_far)
            generated.append(next_token)
            all_ids = torch.cat([all_ids, next_token.unsqueeze(1)], dim=1)

        new_tokens = torch.stack(generated, dim=1)  # (B, num_generated)

        return {
            "token_ids": all_ids,
            "new_token_ids": new_tokens,
            "cascade_stats": router.stats,
        }

    @torch.no_grad()
    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
    ):
        """Generate tokens one at a time, yielding each as it's produced.

        Args:
            input_ids: (B, S) prompt token ids. B must be 1.
            config: Generation configuration.

        Yields:
            token_id (int) for each generated token.
        """
        if config is None:
            config = GenerationConfig()

        self.model.eval()
        B, S = input_ids.shape
        assert B == 1, "Streaming generation only supports batch_size=1"

        router = CascadeRouter(
            gate1_threshold=config.gate1_threshold,
            gate2_threshold=config.gate2_threshold,
        )

        cache = None
        if config.use_kv_cache:
            cache = KVCache(num_layers=self.config.tier1_num_layers)

        # Prefill
        out = self.model(
            input_ids,
            return_tier_activations=True,
            kv_cache=cache,
            position_offset=0,
        )
        logits = out["logits"]

        activations = out.get("tier_activations", {})
        router.record(S, "tier2" in activations, "tier3" in activations)

        next_logits = logits[:, -1, :]
        generated_ids = []

        next_token = sample(next_logits, config.sampling)
        token_id = next_token.item()
        generated_ids.append(token_id)
        yield token_id

        if config.eos_token_id is not None and token_id == config.eos_token_id:
            return

        for step in range(1, config.max_new_tokens):
            pos_offset = S + step - 1
            decode_ids = next_token.view(1, 1)

            out = self.model(
                decode_ids,
                return_tier_activations=True,
                kv_cache=cache,
                position_offset=pos_offset,
            )
            logits = out["logits"]

            activations = out.get("tier_activations", {})
            router.record(1, "tier2" in activations, "tier3" in activations)

            next_logits = logits[:, -1, :]
            gen_tensor = (
                torch.tensor([generated_ids], device=input_ids.device)
                if generated_ids else None
            )
            next_token = sample(next_logits, config.sampling, gen_tensor)
            token_id = next_token.item()

            generated_ids.append(token_id)
            yield token_id

            if config.eos_token_id is not None and token_id == config.eos_token_id:
                break

    @staticmethod
    def estimate_memory(
        config: HLRTConfig,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> dict:
        """Estimate memory requirements for generation.

        Args:
            config: Model configuration.
            max_seq_len: Maximum total sequence length (prompt + generation).
            batch_size: Batch size.
            dtype: Weight/activation dtype.

        Returns:
            Dictionary with memory estimates in bytes.
        """
        bytes_per = torch.tensor([], dtype=dtype).element_size()

        tier1_params = config.tier1_num_layers * (
            4 * config.d_model ** 2 +
            3 * config.d_model * int(8 / 3 * config.d_model)
        )
        tier2_params = config.tier2_num_layers * (
            4 * config.tier2_d_model ** 2 +
            3 * config.tier2_d_model * int(8 / 3 * config.tier2_d_model)
        )
        tier3_params = config.tier3_num_layers * (
            4 * config.tier3_d_model ** 2 +
            3 * config.tier3_d_model * int(8 / 3 * config.tier3_d_model)
        )
        embed_params = config.vocab_size * config.d_model
        total_params = tier1_params + tier2_params + tier3_params + embed_params

        head_dim = config.d_model // config.tier1_num_heads
        num_kv_heads = config.tier1_num_kv_heads or config.tier1_num_heads
        kv_cache = KVCache.estimate_memory(
            num_layers=config.tier1_num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            dtype=dtype,
        )

        activations = (
            config.tier1_num_layers * batch_size * max_seq_len
            * config.d_model * bytes_per
        )

        return {
            "model_weights": total_params * bytes_per,
            "kv_cache": kv_cache,
            "activations": activations,
            "total": total_params * bytes_per + kv_cache + activations,
        }
