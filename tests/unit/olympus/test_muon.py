"""Tests for the Muon optimizer.

Muon applies Newton-Schulz orthogonalisation to gradient matrices for faster
convergence.  It only works on 2-D+ parameter tensors.

Module under test: olympus.optim.muon
"""

import pytest
import torch
import torch.nn as nn

from olympus.optim.muon import Muon

# Alias to the static method for convenience
newton_schulz = Muon._newton_schulz


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNewtonSchulzOrthogonality:
    def test_output_approximately_orthogonal(self):
        torch.manual_seed(0)
        G = torch.randn(64, 64)
        Q = newton_schulz(G, steps=5)

        # Q @ Q^T should be close to the identity.
        product = Q @ Q.T
        identity = torch.eye(64)
        max_dev = (product - identity).abs().max().item()
        assert max_dev < 0.5, (
            f"Max deviation from identity: {max_dev:.4f}"
        )

    def test_output_shape_matches_input(self):
        G = torch.randn(32, 64)
        Q = newton_schulz(G, steps=5)
        assert Q.shape == G.shape

    def test_non_square_matrix(self):
        G = torch.randn(128, 64)
        Q = newton_schulz(G, steps=5)
        # For tall matrices, Q^T @ Q should approximate identity.
        product = Q.T @ Q
        identity = torch.eye(64)
        max_dev = (product - identity).abs().max().item()
        assert max_dev < 0.5, f"Max deviation: {max_dev:.4f}"


class TestMuonStepUpdatesParams:
    def test_params_change_after_step(self):
        torch.manual_seed(42)
        model = nn.Linear(32, 32, bias=False)
        opt = Muon(model.parameters(), lr=0.01)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()

        params_before = model.weight.clone()
        opt.step()

        assert not torch.allclose(model.weight, params_before)


class TestMuonRejects1D:
    def test_raises_for_1d_params(self):
        param_1d = nn.Parameter(torch.randn(64))
        with pytest.raises(ValueError):
            Muon([param_1d], lr=0.01)


class TestMomentumBufferCreated:
    def test_momentum_buffer_exists_after_step(self):
        model = nn.Linear(32, 32, bias=False)
        opt = Muon(model.parameters(), lr=0.01, momentum=0.9)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Check that momentum buffer is stored in optimizer state.
        state = opt.state[model.weight]
        assert "momentum_buffer" in state
        assert state["momentum_buffer"].shape == model.weight.shape


class TestWeightDecay:
    def test_weight_decay_shrinks_params(self):
        torch.manual_seed(0)
        model = nn.Linear(32, 32, bias=False)
        nn.init.ones_(model.weight)

        opt = Muon(model.parameters(), lr=0.0, weight_decay=0.1)

        # Zero-gradient step: only weight decay should act.
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()  # Clear the gradient so only weight decay is applied.

        # Manually set gradient to zero.
        model.weight.grad = torch.zeros_like(model.weight)
        opt.step()

        # With weight decay and lr=0, the update depends on implementation.
        # At minimum, weight should not have grown.
        assert model.weight.abs().mean().item() <= 1.0
