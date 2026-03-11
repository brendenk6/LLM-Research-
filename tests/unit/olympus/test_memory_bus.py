"""Tests for MemoryBus."""

import pytest
import torch

from olympus.core.memory_bus import MemoryBus


@pytest.fixture
def bus():
    return MemoryBus()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStepScopeWriteRead:
    def test_write_then_read(self, bus: MemoryBus):
        t = torch.randn(16)
        bus.write("k1", t, scope="step")
        result = bus.read("k1", scope="step")
        assert result is not None
        assert torch.allclose(result, t)

    def test_read_nonexistent_returns_none(self, bus: MemoryBus):
        assert bus.read("missing", scope="step") is None

    def test_write_stores_detached_clone(self, bus: MemoryBus):
        t = torch.randn(8, requires_grad=True)
        bus.write("k1", t, scope="step")
        result = bus.read("k1", scope="step")
        assert not result.requires_grad

    def test_invalid_scope_raises(self, bus: MemoryBus):
        with pytest.raises(ValueError, match="Invalid scope"):
            bus.write("k1", torch.randn(4), scope="invalid")


class TestEpisodeScopePersistence:
    def test_episode_survives_step_clear(self, bus: MemoryBus):
        bus.write("ep_key", torch.ones(4), scope="episode")
        bus.step()  # clears step scope
        result = bus.read("ep_key", scope="episode")
        assert result is not None
        assert torch.allclose(result, torch.ones(4))


class TestEpisodeScopeExpiry:
    def test_clear_episode_removes_episode_data(self, bus: MemoryBus):
        bus.write("ep_key", torch.ones(4), scope="episode")
        bus.clear_episode()
        assert bus.read("ep_key", scope="episode") is None

    def test_clear_episode_does_not_touch_permanent(self, bus: MemoryBus):
        bus.write("perm_key", torch.ones(4), scope="permanent")
        bus.clear_episode()
        assert bus.read("perm_key", scope="permanent") is not None


class TestPermanentScope:
    def test_permanent_survives_step_and_episode_clear(self, bus: MemoryBus):
        bus.write("perm", torch.tensor([1.0, 2.0]), scope="permanent")
        bus.step()
        bus.clear_episode()
        result = bus.read("perm", scope="permanent")
        assert result is not None
        assert torch.allclose(result, torch.tensor([1.0, 2.0]))

    def test_clear_all_removes_permanent(self, bus: MemoryBus):
        bus.write("perm", torch.tensor([1.0]), scope="permanent")
        bus.clear_all()
        assert bus.read("perm", scope="permanent") is None


class TestQueryCosineSimilarity:
    def test_query_returns_most_similar(self, bus: MemoryBus):
        # Write two vectors in step scope.
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        bus.write("v1", v1, scope="step")
        bus.write("v2", v2, scope="step")

        # Query with something close to v1.
        query = torch.tensor([0.9, 0.1, 0.0, 0.0])
        results = bus.query(query, scope="step", top_k=2)

        assert len(results) == 2
        # First result should be v1 (higher cosine similarity).
        assert results[0][0] == "v1"
        assert results[0][2] > results[1][2]  # similarity ordering

    def test_query_empty_scope(self, bus: MemoryBus):
        query = torch.tensor([1.0, 0.0])
        results = bus.query(query, scope="step", top_k=5)
        assert results == []

    def test_query_top_k_limits(self, bus: MemoryBus):
        for i in range(10):
            bus.write(f"k{i}", torch.randn(8), scope="step")
        results = bus.query(torch.randn(8), scope="step", top_k=3)
        assert len(results) == 3


class TestStepCleanup:
    def test_step_clears_step_scope(self, bus: MemoryBus):
        bus.write("a", torch.randn(4), scope="step")
        bus.write("b", torch.randn(4), scope="step")
        bus.step()
        assert bus.read("a", scope="step") is None
        assert bus.read("b", scope="step") is None
        assert bus.list_keys(scope="step") == []


class TestListKeys:
    def test_list_keys_returns_stored_keys(self, bus: MemoryBus):
        bus.write("x", torch.randn(4), scope="step")
        bus.write("y", torch.randn(4), scope="step")
        keys = bus.list_keys(scope="step")
        assert set(keys) == {"x", "y"}

    def test_list_keys_empty(self, bus: MemoryBus):
        assert bus.list_keys(scope="episode") == []


class TestStateDictRoundtrip:
    def test_roundtrip_preserves_permanent_and_episode(self, bus: MemoryBus):
        bus.write("perm_a", torch.tensor([1.0, 2.0, 3.0]), scope="permanent")
        bus.write("ep_b", torch.tensor([4.0, 5.0]), scope="episode")
        bus.write("step_c", torch.tensor([6.0]), scope="step")

        sd = bus.state_dict()

        new_bus = MemoryBus()
        new_bus.load_state_dict(sd)

        # Permanent should be restored.
        perm = new_bus.read("perm_a", scope="permanent")
        assert perm is not None
        assert torch.allclose(perm, torch.tensor([1.0, 2.0, 3.0]))

        # Episode should be restored.
        ep = new_bus.read("ep_b", scope="episode")
        assert ep is not None
        assert torch.allclose(ep, torch.tensor([4.0, 5.0]))

        # Step data is NOT in state_dict.
        assert new_bus.read("step_c", scope="step") is None

    def test_state_dict_keys(self, bus: MemoryBus):
        bus.write("k", torch.zeros(2), scope="permanent")
        sd = bus.state_dict()
        assert "permanent" in sd
        assert "episode" in sd
