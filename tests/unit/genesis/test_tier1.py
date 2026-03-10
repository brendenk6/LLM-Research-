"""Tests for Tier1TokenProcessor."""

import pytest
import torch

from genesis.model.tier1_token_processor import Tier1TokenProcessor


@pytest.fixture
def tier1():
    torch.manual_seed(42)
    return Tier1TokenProcessor(
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=128,
        dropout=0.0,
        use_flash_attention=False,
        max_seq_len=128,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_output_shape_matches_input(self, tier1: Tier1TokenProcessor):
        B, S, D = 2, 16, 64
        x = torch.randn(B, S, D)
        out = tier1(x)
        assert out.shape == (B, S, D)

    @pytest.mark.parametrize("seq_len", [1, 8, 32, 64])
    def test_various_sequence_lengths(self, tier1: Tier1TokenProcessor, seq_len: int):
        x = torch.randn(1, seq_len, 64)
        out = tier1(x)
        assert out.shape == (1, seq_len, 64)

    def test_batch_size_1(self, tier1: Tier1TokenProcessor):
        x = torch.randn(1, 8, 64)
        out = tier1(x)
        assert out.shape == (1, 8, 64)


class TestCausalMasking:
    def test_future_tokens_do_not_affect_past(self, tier1: Tier1TokenProcessor):
        """Changing a future token should not alter the output at earlier positions."""
        torch.manual_seed(0)
        B, S, D = 1, 16, 64
        x = torch.randn(B, S, D)
        out_full = tier1(x).detach()

        # Modify the last token.
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(D)
        out_modified = tier1(x_modified).detach()

        # All positions except the last should be identical.
        assert torch.allclose(out_full[:, :-1, :], out_modified[:, :-1, :], atol=1e-5), (
            "Causal masking violated: future tokens affected earlier positions."
        )

    def test_past_tokens_affect_future(self, tier1: Tier1TokenProcessor):
        """Changing an early token should alter later outputs."""
        torch.manual_seed(0)
        B, S, D = 1, 16, 64
        x = torch.randn(B, S, D)
        out_full = tier1(x).detach()

        x_modified = x.clone()
        x_modified[:, 0, :] = torch.randn(D)
        out_modified = tier1(x_modified).detach()

        # Later positions should differ.
        diff = (out_full[:, -1, :] - out_modified[:, -1, :]).abs().max().item()
        assert diff > 1e-4, "Changing past token should affect future positions."


class TestGradientFlow:
    def test_all_params_receive_gradients(self, tier1: Tier1TokenProcessor):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = tier1(x)
        loss = out.sum()
        loss.backward()

        # Input should have gradients.
        assert x.grad is not None

        # All learnable parameters should have gradients.
        for name, param in tier1.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient."
