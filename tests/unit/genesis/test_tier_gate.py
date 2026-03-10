"""Tests for TierGate.

The TierGate learns to classify which chunks of tokens should be escalated
to higher-tier processing. It produces per-chunk scores in [0, 1] and an
auxiliary load-balancing loss.

Module under test: genesis.model.tier_gate
"""

import pytest
import torch

from genesis.model.tier_gate import TierGate


@pytest.fixture
def gate():
    torch.manual_seed(42)
    return TierGate(d_model=64, chunk_size=8, threshold=0.5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_scores_shape(self, gate: TierGate):
        # seq_len=16 with chunk_size=8 -> 2 chunks
        x = torch.randn(4, 16, 64)  # (B, S, D)
        mask, scores = gate(x)
        assert scores.shape == (4, 2)
        assert mask.shape == (4, 2)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_sizes(self, gate: TierGate, batch_size: int):
        x = torch.randn(batch_size, 16, 64)
        mask, scores = gate(x)
        assert scores.shape[0] == batch_size
        assert mask.shape[0] == batch_size

    def test_seq_len_not_divisible_by_chunk_size(self, gate: TierGate):
        """Sequence length not divisible by chunk_size should be padded."""
        x = torch.randn(2, 10, 64)  # 10 not divisible by 8 -> pads to 16 -> 2 chunks
        mask, scores = gate(x)
        assert scores.shape == (2, 2)


class TestScoresInRange:
    def test_scores_between_0_and_1(self, gate: TierGate):
        x = torch.randn(4, 16, 64)
        mask, scores = gate(x)
        assert scores.min().item() >= 0.0 - 1e-6
        assert scores.max().item() <= 1.0 + 1e-6

    def test_scores_with_extreme_inputs(self, gate: TierGate):
        x = torch.randn(2, 16, 64) * 100.0
        mask, scores = gate(x)
        assert scores.min().item() >= 0.0 - 1e-6
        assert scores.max().item() <= 1.0 + 1e-6


class TestTrainVsEval:
    def test_train_returns_soft_mask(self, gate: TierGate):
        gate.train()
        x = torch.randn(2, 16, 64)
        mask, scores = gate(x)
        # In training mode, mask is the soft sigmoid scores (float)
        assert mask.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_eval_returns_bool_mask(self, gate: TierGate):
        gate.eval()
        x = torch.randn(2, 16, 64)
        mask, scores = gate(x)
        assert mask.dtype == torch.bool


class TestLoadBalanceLoss:
    def test_loss_is_scalar(self, gate: TierGate):
        x = torch.randn(4, 16, 64)
        mask, scores = gate(x)
        lb_loss = gate.load_balance_loss()
        assert lb_loss.dim() == 0

    def test_loss_nonnegative(self, gate: TierGate):
        x = torch.randn(4, 16, 64)
        mask, scores = gate(x)
        lb_loss = gate.load_balance_loss()
        assert lb_loss.item() >= 0.0

    def test_loss_backward(self, gate: TierGate):
        x = torch.randn(4, 16, 64, requires_grad=True)
        mask, scores = gate(x)
        lb_loss = gate.load_balance_loss()
        lb_loss.backward()
        assert x.grad is not None

    def test_loss_before_forward_is_zero(self):
        gate = TierGate(d_model=64, chunk_size=8)
        lb_loss = gate.load_balance_loss()
        assert lb_loss.item() == 0.0
