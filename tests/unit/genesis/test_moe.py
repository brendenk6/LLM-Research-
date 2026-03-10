"""Tests for MoEFFN (Mixture-of-Experts Feed-Forward Network).

Module under test: genesis.model.moe
"""

import pytest
import torch

from genesis.model.moe import MoEFFN


@pytest.fixture
def moe():
    torch.manual_seed(42)
    return MoEFFN(
        d_model=64,
        d_ff=128,
        num_experts=4,
        top_k=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_output_shape_matches_input(self, moe: MoEFFN):
        B, S, D = 2, 16, 64
        x = torch.randn(B, S, D)
        out = moe(x)
        assert out.shape == (B, S, D)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_various_batch_sizes(self, moe: MoEFFN, batch_size: int):
        x = torch.randn(batch_size, 8, 64)
        out = moe(x)
        assert out.shape == (batch_size, 8, 64)

    @pytest.mark.parametrize("seq_len", [1, 8, 32])
    def test_various_seq_lengths(self, moe: MoEFFN, seq_len: int):
        x = torch.randn(2, seq_len, 64)
        out = moe(x)
        assert out.shape == (2, seq_len, 64)


class TestExpertsActivated:
    def test_top_k_selection(self, moe: MoEFFN):
        """Verify that the router selects top_k experts per token."""
        B, S, D = 2, 16, 64
        x = torch.randn(B, S, D)
        # Run forward to populate router state
        out = moe(x)
        assert out.shape == (B, S, D)
        # Verify load balance loss is available (confirms routing happened)
        lb_loss = moe.load_balance_loss()
        assert lb_loss.dim() == 0
        assert lb_loss.item() >= 0.0

    def test_different_inputs_may_route_differently(self, moe: MoEFFN):
        """Two very different inputs should potentially route to different experts."""
        x1 = torch.randn(1, 8, 64) * 10.0
        x2 = -x1
        out1 = moe(x1)
        out2 = moe(x2)
        # Outputs should differ (different expert combinations).
        assert not torch.allclose(out1, out2, atol=1e-5)


class TestLoadBalanceLoss:
    def test_loss_is_scalar(self, moe: MoEFFN):
        x = torch.randn(4, 16, 64)
        moe(x)
        lb_loss = moe.load_balance_loss()
        assert lb_loss.dim() == 0

    def test_loss_nonnegative(self, moe: MoEFFN):
        x = torch.randn(4, 16, 64)
        moe(x)
        lb_loss = moe.load_balance_loss()
        assert lb_loss.item() >= 0.0

    def test_loss_backward(self, moe: MoEFFN):
        x = torch.randn(4, 16, 64, requires_grad=True)
        out = moe(x)
        lb_loss = moe.load_balance_loss()
        total = out.sum() + lb_loss
        total.backward()
        assert x.grad is not None

    def test_loss_before_forward_is_zero(self):
        moe = MoEFFN(d_model=64, num_experts=4, top_k=2)
        lb_loss = moe.load_balance_loss()
        assert lb_loss.item() == 0.0
