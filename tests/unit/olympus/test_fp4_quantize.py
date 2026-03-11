"""Tests for FP4 (INT4) quantization kernels.

Verifies PyTorch fallback correctness and, when CUDA + Triton are available,
numerical equivalence between the Triton kernels and the PyTorch reference.

Module under test: olympus.kernels.fp4_quantize
"""

import pytest
import torch

from olympus.kernels.fp4_quantize import (
    HAS_TRITON,
    _dequantize_pt,
    _quantize_pt,
    dequantize,
    quantize,
    quantize_dequantize,
)

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = HAS_TRITON and CUDA_AVAILABLE
skip_no_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE, reason="CUDA + Triton required"
)


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------


class TestQuantizePyTorch:
    def test_output_shapes_per_row(self):
        x = torch.randn(4, 128)
        q, scales = _quantize_pt(x, group_size=-1)
        assert q.shape == (4, 128)
        assert q.dtype == torch.int8
        assert scales.shape == (4, 1)

    def test_output_shapes_per_group(self):
        x = torch.randn(4, 128)
        q, scales = _quantize_pt(x, group_size=32)
        assert q.shape == (4, 128)
        assert scales.shape == (4, 4)

    def test_values_in_range(self):
        x = torch.randn(8, 256)
        q, _ = _quantize_pt(x, group_size=64)
        assert (q >= -8).all()
        assert (q <= 7).all()

    def test_zero_input(self):
        x = torch.zeros(2, 64)
        q, scales = _quantize_pt(x, group_size=-1)
        assert (q == 0).all()

    def test_constant_input(self):
        x = torch.full((2, 64), 3.5)
        q, scales = _quantize_pt(x, group_size=-1)
        # scale = 3.5 / 7 = 0.5, q = round(3.5 / 0.5) = 7
        assert (q == 7).all()
        assert torch.allclose(scales, torch.full_like(scales, 0.5))

    def test_non_divisible_group_size(self):
        # D=100, group_size=32 -> 4 groups (last group has 4 elements)
        x = torch.randn(2, 100)
        q, scales = _quantize_pt(x, group_size=32)
        assert q.shape == (2, 100)
        assert scales.shape == (2, 4)  # ceil(100/32) = 4

    def test_3d_input_via_public_api(self):
        x = torch.randn(2, 16, 64)
        q, scales = quantize(x, group_size=-1)
        assert q.shape == (2, 16, 64)
        assert scales.shape == (2, 16, 1)

    def test_group_size_larger_than_d(self):
        x = torch.randn(4, 32)
        q, scales = _quantize_pt(x, group_size=256)
        # Falls back to per-row
        assert scales.shape == (4, 1)


# ---------------------------------------------------------------------------
# dequantize
# ---------------------------------------------------------------------------


class TestDequantizePyTorch:
    def test_output_shape(self):
        q = torch.randint(-8, 8, (4, 128), dtype=torch.int8)
        scales = torch.rand(4, 1)
        out = _dequantize_pt(q, scales, group_size=-1, orig_D=128)
        assert out.shape == (4, 128)

    def test_identity_scale(self):
        # scale=1 means dequantized = quantized values as float
        q = torch.tensor([[1, -3, 7, -8]], dtype=torch.int8)
        scales = torch.ones(1, 1)
        out = _dequantize_pt(q, scales, group_size=-1, orig_D=4)
        expected = torch.tensor([[1.0, -3.0, 7.0, -8.0]])
        assert torch.allclose(out, expected)

    def test_scaling(self):
        q = torch.tensor([[4, -2]], dtype=torch.int8)
        scales = torch.tensor([[0.5]])
        out = _dequantize_pt(q, scales, group_size=-1, orig_D=2)
        expected = torch.tensor([[2.0, -1.0]])
        assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# roundtrip (quantize -> dequantize)
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_reconstruction_error_bounded(self):
        torch.manual_seed(42)
        x = torch.randn(8, 256)
        q, scales = _quantize_pt(x, group_size=-1)
        x_hat = _dequantize_pt(q, scales, group_size=-1, orig_D=256)

        # Max error per row bounded by scale / 2
        row_max = x.abs().amax(dim=-1)
        max_step = row_max / 7.0 / 2.0  # half a quantization step
        row_err = (x - x_hat).abs().amax(dim=-1)
        assert (row_err <= max_step + 1e-6).all()

    def test_public_api_roundtrip(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16, 128)
        q, scales = quantize(x, group_size=32)
        x_hat = dequantize(q, scales, group_size=32)
        assert x_hat.shape == x.shape
        # Should be reasonably close
        assert (x - x_hat).abs().max() < 1.0

    def test_per_group_more_accurate_than_per_row(self):
        torch.manual_seed(42)
        x = torch.randn(4, 256)

        q_row, s_row = _quantize_pt(x, group_size=-1)
        x_row = _dequantize_pt(q_row, s_row, group_size=-1, orig_D=256)

        q_grp, s_grp = _quantize_pt(x, group_size=32)
        x_grp = _dequantize_pt(q_grp, s_grp, group_size=32, orig_D=256)

        err_row = (x - x_row).abs().mean()
        err_grp = (x - x_grp).abs().mean()
        # Per-group should be at least as accurate as per-row
        assert err_grp <= err_row + 1e-6


# ---------------------------------------------------------------------------
# quantize_dequantize (STE)
# ---------------------------------------------------------------------------


class TestQuantizeDequantize:
    def test_output_shape_and_dtype(self):
        x = torch.randn(4, 64)
        out = quantize_dequantize(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_ste_gradient(self):
        x = torch.randn(4, 64, requires_grad=True)
        out = quantize_dequantize(x, group_size=32)
        loss = out.sum()
        loss.backward()
        # STE: gradient passes through unchanged
        assert x.grad is not None
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_not_identity(self):
        # quantize_dequantize should introduce quantization noise
        torch.manual_seed(42)
        x = torch.randn(4, 128)
        out = quantize_dequantize(x)
        # Very unlikely to be exactly equal (would need all values
        # to land exactly on quantization grid)
        assert not torch.equal(x, out)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


class TestQuantizeTriton:
    @skip_no_triton
    def test_matches_pytorch_per_row(self):
        torch.manual_seed(42)
        x = torch.randn(8, 256, device="cuda")
        q_pt, s_pt = _quantize_pt(x, group_size=-1)
        q_tr, s_tr = quantize(x, group_size=-1)
        # Scales should match closely
        assert torch.allclose(s_tr, s_pt, atol=1e-5)
        # Dequantized roundtrip should match
        x_pt = _dequantize_pt(q_pt, s_pt, group_size=-1, orig_D=256)
        x_tr = dequantize(q_tr, s_tr, group_size=-1)
        assert torch.allclose(x_tr, x_pt, atol=1e-5)

    @skip_no_triton
    def test_matches_pytorch_per_group(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128, device="cuda")
        q_pt, s_pt = _quantize_pt(x, group_size=32)
        q_tr, s_tr = quantize(x, group_size=32)
        assert torch.allclose(s_tr, s_pt, atol=1e-5)
        x_pt = _dequantize_pt(q_pt, s_pt, group_size=32, orig_D=128)
        x_tr = dequantize(q_tr, s_tr, group_size=32)
        assert torch.allclose(x_tr, x_pt, atol=1e-5)

    @skip_no_triton
    def test_large_tensor(self):
        torch.manual_seed(0)
        x = torch.randn(64, 4096, device="cuda")
        q, scales = quantize(x, group_size=128)
        assert q.shape == (64, 4096)
        assert scales.shape == (64, 32)
        assert (q >= -8).all()
        assert (q <= 7).all()

    @skip_no_triton
    def test_ste_gradient_triton(self):
        x = torch.randn(4, 128, device="cuda", requires_grad=True)
        out = quantize_dequantize(x, group_size=32)
        out.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))
