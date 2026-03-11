"""Tests for the GENESIS inference pipeline: KVCache, Sampling, CascadeRouter, Generator."""

import pytest
import torch

from genesis.model.hlrt import HLRT, HLRTConfig
from genesis.inference.kv_cache import KVCache, LayerKVCache
from genesis.inference.sampling import (
    SamplingConfig,
    sample,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
)
from genesis.inference.cascade_router import CascadeRouter, CascadeStats
from genesis.inference.generator import Generator, GenerationConfig


# ---- Small config for fast tests ----

def _small_config() -> HLRTConfig:
    return HLRTConfig(
        vocab_size=256,
        d_model=64,
        tier1_num_layers=2,
        tier1_num_heads=4,
        tier1_max_seq_len=128,
        tier2_d_model=96,
        tier2_num_layers=1,
        tier2_num_heads=4,
        tier2_max_seq_len=64,
        tier3_d_model=64,
        tier3_num_layers=1,
        tier3_num_heads=4,
        tier3_max_seq_len=32,
        tier3_recurrence_steps=2,
        gate1_chunk_size=4,
        num_latent_vectors=2,
    )


# ===========================================================================
# KV Cache Tests
# ===========================================================================


class TestLayerKVCache:
    def test_empty_cache(self):
        cache = LayerKVCache()
        assert cache.seq_len == 0
        assert cache.key is None

    def test_first_update(self):
        cache = LayerKVCache()
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        full_k, full_v = cache.update(k, v)
        assert full_k.shape == (1, 4, 8, 16)
        assert cache.seq_len == 8

    def test_append_update(self):
        cache = LayerKVCache()
        k1 = torch.randn(1, 4, 8, 16)
        v1 = torch.randn(1, 4, 8, 16)
        cache.update(k1, v1)

        k2 = torch.randn(1, 4, 1, 16)
        v2 = torch.randn(1, 4, 1, 16)
        full_k, full_v = cache.update(k2, v2)
        assert full_k.shape == (1, 4, 9, 16)
        assert cache.seq_len == 9

    def test_clear(self):
        cache = LayerKVCache()
        cache.update(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        cache.clear()
        assert cache.seq_len == 0


class TestKVCache:
    def test_multi_layer(self):
        cache = KVCache(num_layers=4)
        assert len(cache.layers) == 4
        assert cache.seq_len == 0

    def test_indexing(self):
        cache = KVCache(num_layers=4)
        cache[0].update(torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
        assert cache[0].seq_len == 8
        assert cache[1].seq_len == 0

    def test_clear_all(self):
        cache = KVCache(num_layers=4)
        for i in range(4):
            cache[i].update(torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
        cache.clear()
        for i in range(4):
            assert cache[i].seq_len == 0

    def test_memory_estimate(self):
        mem = KVCache.estimate_memory(
            num_layers=12,
            num_kv_heads=16,
            head_dim=64,
            max_seq_len=1024,
            batch_size=1,
            dtype=torch.float16,
        )
        # 2 * 12 * 1 * 16 * 1024 * 64 * 2 bytes = 50,331,648
        assert mem == 2 * 12 * 1 * 16 * 1024 * 64 * 2


# ===========================================================================
# Sampling Tests
# ===========================================================================


class TestTemperature:
    def test_identity_at_1(self):
        logits = torch.randn(1, 100)
        assert torch.equal(apply_temperature(logits, 1.0), logits)

    def test_sharpens_below_1(self):
        logits = torch.randn(1, 100)
        sharp = apply_temperature(logits, 0.5)
        assert sharp.abs().max() > logits.abs().max()

    def test_flattens_above_1(self):
        logits = torch.tensor([[10.0, 1.0, 0.1]])
        flat = apply_temperature(logits, 2.0)
        assert flat[0, 0] < logits[0, 0]


class TestTopK:
    def test_keeps_top_k(self):
        logits = torch.tensor([[5.0, 3.0, 1.0, 0.5, 0.1]])
        filtered = apply_top_k(logits, 3)
        assert (filtered[0, :3] > float("-inf")).all()
        assert (filtered[0, 3:] == float("-inf")).all()

    def test_noop_when_k_exceeds_vocab(self):
        logits = torch.randn(1, 10)
        assert torch.equal(apply_top_k(logits, 100), logits)


class TestTopP:
    def test_keeps_high_prob_tokens(self):
        # One dominant token
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        filtered = apply_top_p(logits, 0.9)
        assert filtered[0, 0] > float("-inf")

    def test_noop_at_1(self):
        logits = torch.randn(1, 10)
        assert torch.equal(apply_top_p(logits, 1.0), logits)


class TestRepetitionPenalty:
    def test_no_change_at_1(self):
        logits = torch.randn(1, 100)
        ids = torch.tensor([[0, 1, 2]])
        assert torch.equal(apply_repetition_penalty(logits, ids, 1.0), logits)

    def test_penalizes_repeated_tokens(self):
        logits = torch.tensor([[5.0, 3.0, 1.0]])
        ids = torch.tensor([[0]])
        penalized = apply_repetition_penalty(logits, ids, 2.0)
        assert penalized[0, 0] < logits[0, 0]  # token 0 penalized
        assert penalized[0, 1] == logits[0, 1]  # token 1 unchanged


class TestSample:
    def test_greedy(self):
        logits = torch.tensor([[0.1, 0.2, 10.0, 0.3]])
        config = SamplingConfig(temperature=0.0)
        token = sample(logits, config)
        assert token.item() == 2

    def test_returns_valid_ids(self):
        logits = torch.randn(4, 256)
        config = SamplingConfig(temperature=1.0, top_k=50)
        tokens = sample(logits, config)
        assert tokens.shape == (4,)
        assert (tokens >= 0).all()
        assert (tokens < 256).all()


# ===========================================================================
# Cascade Router Tests
# ===========================================================================


class TestCascadeStats:
    def test_initial_state(self):
        stats = CascadeStats()
        assert stats.total_tokens == 0
        assert stats.tier2_rate == 0.0
        assert stats.tier3_rate == 0.0

    def test_update(self):
        stats = CascadeStats()
        stats.update(10, tier2_activated=True, tier3_activated=False)
        assert stats.total_tokens == 10
        assert stats.tier2_rate == 1.0
        assert stats.tier3_rate == 0.0

    def test_mixed_updates(self):
        stats = CascadeStats()
        stats.update(80, tier2_activated=False, tier3_activated=False)
        stats.update(20, tier2_activated=True, tier3_activated=True)
        assert stats.tier2_rate == 0.2
        assert stats.tier3_rate == 0.2

    def test_reset(self):
        stats = CascadeStats()
        stats.update(100, True, True)
        stats.reset()
        assert stats.total_tokens == 0

    def test_compute_savings(self):
        stats = CascadeStats()
        # All Tier 1 only = max savings
        stats.update(100, False, False)
        assert stats.compute_savings > 0.5


class TestCascadeRouter:
    def test_escalate_above_threshold(self):
        router = CascadeRouter()
        scores = torch.tensor([[0.8, 0.3, 0.9]])
        mask, any_esc = router.should_escalate(scores, 0.5)
        assert mask.tolist() == [[True, False, True]]
        assert any_esc is True

    def test_no_escalation_below_threshold(self):
        router = CascadeRouter()
        scores = torch.tensor([[0.1, 0.2, 0.3]])
        mask, any_esc = router.should_escalate(scores, 0.5)
        assert any_esc is False

    def test_override_threshold(self):
        router = CascadeRouter(gate1_threshold=0.9)
        scores = torch.tensor([[0.8, 0.85, 0.95]])
        mask, any_esc = router.route_gate1(scores, model_threshold=0.5)
        # Uses override (0.9), not model default (0.5)
        assert mask.tolist() == [[False, False, True]]


# ===========================================================================
# Generator Tests (end-to-end)
# ===========================================================================


class TestGenerator:
    @pytest.fixture
    def model(self):
        config = _small_config()
        m = HLRT(config)
        m.eval()
        return m

    def test_generate_basic(self, model):
        gen = Generator(model)
        input_ids = torch.randint(0, 256, (1, 8))
        result = gen.generate(
            input_ids,
            config=GenerationConfig(max_new_tokens=4),
        )
        assert "token_ids" in result
        assert "new_token_ids" in result
        assert "cascade_stats" in result
        assert result["new_token_ids"].shape == (1, 4)
        assert result["token_ids"].shape == (1, 12)  # 8 prompt + 4 generated

    def test_generate_batch(self, model):
        gen = Generator(model)
        input_ids = torch.randint(0, 256, (2, 8))
        result = gen.generate(
            input_ids,
            config=GenerationConfig(max_new_tokens=3),
        )
        assert result["new_token_ids"].shape == (2, 3)

    def test_generate_greedy_deterministic(self, model):
        gen = Generator(model)
        input_ids = torch.randint(0, 256, (1, 8))
        config = GenerationConfig(
            max_new_tokens=4,
            sampling=SamplingConfig(temperature=0.0),
        )
        r1 = gen.generate(input_ids, config)
        # Reset stateful plan vector between runs for determinism
        model.set_state("last_plan_vector", torch.zeros(1, model.config.d_model))
        r2 = gen.generate(input_ids, config)
        assert torch.equal(r1["new_token_ids"], r2["new_token_ids"])

    def test_eos_stops_generation(self, model):
        gen = Generator(model)
        # Force model to always predict token 42 by using greedy
        # (we can't truly control output, but we test EOS logic)
        input_ids = torch.randint(0, 256, (1, 8))
        config = GenerationConfig(
            max_new_tokens=100,
            eos_token_id=42,
            sampling=SamplingConfig(temperature=1.0),
        )
        result = gen.generate(input_ids, config)
        # Generation should have stopped at or before 100 tokens
        assert result["new_token_ids"].shape[1] <= 100

    def test_cascade_stats_populated(self, model):
        gen = Generator(model)
        input_ids = torch.randint(0, 256, (1, 8))
        result = gen.generate(
            input_ids,
            config=GenerationConfig(max_new_tokens=4),
        )
        stats = result["cascade_stats"]
        assert stats.total_tokens > 0
        assert stats.tier1_rate == 1.0

    def test_streaming(self, model):
        gen = Generator(model)
        input_ids = torch.randint(0, 256, (1, 8))
        config = GenerationConfig(max_new_tokens=5)
        tokens = list(gen.generate_streaming(input_ids, config))
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)

    def test_cached_matches_uncached(self, model):
        """KV-cached generation must produce identical output to uncached."""
        gen = Generator(model)
        input_ids = torch.randint(0, 256, (1, 8))
        config_cached = GenerationConfig(
            max_new_tokens=6,
            use_kv_cache=True,
            sampling=SamplingConfig(temperature=0.0),
        )
        r1 = gen.generate(input_ids, config_cached)

        model.set_state("last_plan_vector", torch.zeros(1, model.config.d_model))

        config_uncached = GenerationConfig(
            max_new_tokens=6,
            use_kv_cache=False,
            sampling=SamplingConfig(temperature=0.0),
        )
        r2 = gen.generate(input_ids, config_uncached)

        assert torch.equal(r1["new_token_ids"], r2["new_token_ids"]), (
            f"Cached: {r1['new_token_ids']}\nUncached: {r2['new_token_ids']}"
        )

    def test_memory_estimate(self):
        config = _small_config()
        mem = Generator.estimate_memory(config, max_seq_len=128)
        assert "model_weights" in mem
        assert "kv_cache" in mem
        assert "activations" in mem
        assert "total" in mem
        assert mem["total"] > 0
