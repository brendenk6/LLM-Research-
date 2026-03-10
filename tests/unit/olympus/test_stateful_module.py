"""Tests for StatefulModule and StateLink."""


import pytest
import torch
import torch.nn as nn

from olympus.core.stateful_module import StatefulModule, StateLink


# ---------------------------------------------------------------------------
# Fixture: a minimal StatefulModule subclass
# ---------------------------------------------------------------------------

class SimpleStateful(StatefulModule):
    """Tiny module that carries one state tensor and one memory tensor."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.register_state("running_hidden", torch.zeros(dim))
        self.register_memory("context_vector", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.link_state("running_hidden")
        out = self.linear(x + h)
        self.set_state("running_hidden", out.detach())
        return out


@pytest.fixture
def module():
    torch.manual_seed(42)
    return SimpleStateful(dim=64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStateRegistration:
    def test_state_registered(self, module: SimpleStateful):
        assert "running_hidden" in module.state_names()
        state = module.get_state("running_hidden")
        assert state.shape == (64,)
        assert torch.all(state == 0)

    def test_duplicate_state_raises(self, module: SimpleStateful):
        with pytest.raises(ValueError, match="already registered"):
            module.register_state("running_hidden", torch.zeros(64))

    def test_memory_registered(self, module: SimpleStateful):
        assert "context_vector" in module.memory_names()
        mem = module.get_memory("context_vector")
        assert mem.shape == (64,)

    def test_duplicate_memory_raises(self, module: SimpleStateful):
        with pytest.raises(ValueError, match="already registered"):
            module.register_memory("context_vector", torch.zeros(64))


class TestStatePersistence:
    def test_state_persists_across_forward_passes(self, module: SimpleStateful):
        x = torch.randn(64)
        _ = module(x)
        state_after_first = module.get_state("running_hidden").clone()

        _ = module(x)
        state_after_second = module.get_state("running_hidden").clone()

        # State should have changed between passes (since set_state is called
        # with the output of the linear layer).
        assert not torch.allclose(state_after_first, state_after_second)

    def test_state_value_matches_output(self, module: SimpleStateful):
        x = torch.randn(64)
        out = module(x)
        # After forward, running_hidden == out.detach()
        assert torch.allclose(module.get_state("running_hidden"), out.detach())


class TestGradientThroughState:
    def test_gradient_flows_through_state_link(self, module: SimpleStateful):
        # Directly use StateLink with a requires_grad tensor so backward fires
        # and populates _state_grads on the module.
        state_tensor = module.get_state("running_hidden").clone().detach().requires_grad_(True)
        linked = StateLink.apply(state_tensor, module, "running_hidden")
        loss = (linked ** 2).sum()
        loss.backward()

        # The state gradient buffer should have been populated.
        grad = module._state_grads["running_hidden"]
        assert grad is not None
        assert grad.shape == (64,)

    def test_zero_state_grads(self, module: SimpleStateful):
        state_tensor = module.get_state("running_hidden").clone().detach().requires_grad_(True)
        linked = StateLink.apply(state_tensor, module, "running_hidden")
        (linked ** 2).sum().backward()
        assert module._state_grads["running_hidden"] is not None

        module.zero_state_grads()
        assert module._state_grads["running_hidden"] is None

    def test_apply_state_grads_updates_state(self, module: SimpleStateful):
        # Manually populate a gradient in _state_grads so apply_state_grads
        # has something to apply.
        fake_grad = torch.ones(64)
        module._state_grads["running_hidden"] = fake_grad

        state_before = module.get_state("running_hidden").clone()
        module.apply_state_grads(lr=0.1)
        state_after = module.get_state("running_hidden")

        # State should have been updated: new = old - lr * grad
        expected = state_before - 0.1 * fake_grad
        assert torch.allclose(state_after, expected)
        # Grads should be cleared after apply
        assert module._state_grads["running_hidden"] is None


class TestMemoryRegistration:
    def test_memory_works(self, module: SimpleStateful):
        new_val = torch.randn(64)
        module.update_memory("context_vector", new_val)
        retrieved = module.get_memory("context_vector")
        assert torch.allclose(retrieved, new_val)

    def test_memory_is_non_differentiable(self, module: SimpleStateful):
        mem = module.get_memory("context_vector")
        assert not mem.requires_grad


class TestShapes:
    @pytest.mark.parametrize("dim", [32, 64, 128])
    def test_various_dims(self, dim: int):
        m = SimpleStateful(dim=dim)
        x = torch.randn(dim)
        out = m(x)
        assert out.shape == (dim,)

    def test_batched_input(self):
        m = SimpleStateful(dim=64)
        x = torch.randn(8, 64)
        out = m(x)
        assert out.shape == (8, 64)


class TestStateDictWithState:
    def test_save_load_preserves_state_and_memory(self, module: SimpleStateful):
        # Populate state and memory with non-zero values.
        x = torch.randn(64)
        _ = module(x)
        module.update_memory("context_vector", torch.randn(64))

        state_before = module.get_state("running_hidden").clone()
        memory_before = module.get_memory("context_vector").clone()

        sd = module.state_dict_with_state()

        # Create a fresh module and load.
        fresh = SimpleStateful(dim=64)
        fresh.load_state_dict_with_state(sd)

        assert torch.allclose(fresh.get_state("running_hidden"), state_before)
        assert torch.allclose(fresh.get_memory("context_vector"), memory_before)

    def test_state_dict_contains_olympus_keys(self, module: SimpleStateful):
        sd = module.state_dict_with_state()
        assert "_olympus_state_tensors" in sd
        assert "_olympus_memory_tensors" in sd
        assert "_olympus_state_meta" in sd
        assert "_olympus_memory_meta" in sd


class TestToDevice:
    def test_to_moves_state_and_memory(self, module: SimpleStateful):
        # Populate with non-zero data.
        module.set_state("running_hidden", torch.randn(64))
        module.update_memory("context_vector", torch.randn(64))

        # Move to CPU explicitly (the only device always available).
        module = module.to("cpu")

        state = module.get_state("running_hidden")
        memory = module.get_memory("context_vector")

        assert state.device.type == "cpu"
        assert memory.device.type == "cpu"

    @pytest.mark.gpu
    def test_to_cuda_if_available(self, module: SimpleStateful):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        module = module.to("cuda")
        assert module.get_state("running_hidden").device.type == "cuda"
        assert module.get_memory("context_vector").device.type == "cuda"
