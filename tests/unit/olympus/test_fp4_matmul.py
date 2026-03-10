"""Tests for FP4 quantized matrix multiplication kernel.

Verifies that the FP4 matmul produces results close to the standard
F.linear (within quantization noise) and that gradients flow correctly.

Module under test: olympus.kernels.fp4_matmul
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.kernels.fp4_matmul import (
    FP4Linear,
    _fp4_matmul_pt,
    fp4_matmul,
)
from olympus.kernels.fp4_quantize import HAS_TRITON

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = HAS_TRITON and CUDA_AVAILABLE
skip_no_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE, reason="CUDA + Triton required"
)


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------


class TestFP4MatmulPyTorch:
    def test_output_shape_2d(self):
        x = torch.randn(8, 64)
        w = torch.randn(32, 64)
        out = _fp4_matmul_pt(x, w)
        assert out.shape == (8, 32)

    def test_output_shape_3d(self):
        x = torch.randn(2, 16, 64)
        w = torch.randn(32, 64)
        out = _fp4_matmul_pt(x, w)
        assert out.shape == (2, 16, 32)

    def test_with_bias(self):
        x = torch.randn(4, 64)
        w = torch.randn(32, 64)
        b = torch.randn(32)
        out = _fp4_matmul_pt(x, w, bias=b)
        assert out.shape == (4, 32)

    def test_close_to_f_linear(self):
        torch.manual_seed(42)
        x = torch.randn(8, 64)
        w = torch.randn(32, 64)
        b = torch.randn(32)

        fp4_out = _fp4_matmul_pt(x, w, bias=b)
        ref_out = F.linear(x, w, b)

        # FP4 quantization introduces noise, but should be reasonably close
        rel_err = (fp4_out - ref_out).abs().mean() / ref_out.abs().mean()
        assert rel_err < 0.15, f"Relative error too high: {rel_err:.4f}"

    def test_zero_input(self):
        x = torch.zeros(4, 64)
        w = torch.randn(32, 64)
        out = _fp4_matmul_pt(x, w)
        assert torch.allclose(out, torch.zeros(4, 32), atol=1e-6)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        x = torch.randn(4, 64, requires_grad=True)
        w = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(32, requires_grad=True)

        out = _fp4_matmul_pt(x, w, bias=b)
        out.sum().backward()

        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape

    def test_per_group_quantization(self):
        torch.manual_seed(42)
        x = torch.randn(8, 128)
        w = torch.randn(64, 128)

        out_row = _fp4_matmul_pt(x, w, act_group_size=-1)
        out_grp = _fp4_matmul_pt(x, w, act_group_size=32)

        ref = F.linear(x, w)
        err_row = (out_row - ref).abs().mean()
        err_grp = (out_grp - ref).abs().mean()

        # Per-group should be at least as accurate
        assert err_grp <= err_row + 0.1


# ---------------------------------------------------------------------------
# FP4Linear wrapper
# ---------------------------------------------------------------------------


class TestFP4Linear:
    def test_wraps_nn_linear(self):
        linear = nn.Linear(64, 32)
        fp4 = FP4Linear(linear)

        x = torch.randn(4, 64)
        out = fp4(x)
        assert out.shape == (4, 32)

    def test_weight_and_bias_accessible(self):
        linear = nn.Linear(64, 32)
        fp4 = FP4Linear(linear)
        assert fp4.weight is linear.weight
        assert fp4.bias is linear.bias

    def test_no_bias(self):
        linear = nn.Linear(64, 32, bias=False)
        fp4 = FP4Linear(linear)
        x = torch.randn(4, 64)
        out = fp4(x)
        assert out.shape == (4, 32)
        assert fp4.bias is None

    def test_gradient_through_wrapper(self):
        linear = nn.Linear(64, 32)
        fp4 = FP4Linear(linear)
        x = torch.randn(4, 64, requires_grad=True)
        out = fp4(x)
        out.sum().backward()
        assert x.grad is not None
        assert linear.weight.grad is not None


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


class TestFP4MatmulTriton:
    @skip_no_triton
    def test_output_shape(self):
        x = torch.randn(8, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = fp4_matmul(x, w)
        assert out.shape == (8, 32)

    @skip_no_triton
    def test_with_bias(self):
        x = torch.randn(8, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        b = torch.randn(32, device="cuda")
        out = fp4_matmul(x, w, bias=b)
        assert out.shape == (8, 32)

    @skip_no_triton
    def test_matches_pytorch_fallback(self):
        torch.manual_seed(42)
        x = torch.randn(16, 128, device="cuda")
        w = torch.randn(64, 128, device="cuda")
        b = torch.randn(64, device="cuda")

        pt_out = _fp4_matmul_pt(x, w, bias=b)
        tr_out = fp4_matmul(x, w, bias=b)

        # Both paths do quantize-dequantize; results should be very close
        assert torch.allclose(tr_out, pt_out, atol=0.5)

    @skip_no_triton
    def test_close_to_f_linear(self):
        torch.manual_seed(0)
        x = torch.randn(16, 128, device="cuda")
        w = torch.randn(64, 128, device="cuda")

        fp4_out = fp4_matmul(x, w)
        ref_out = F.linear(x, w)

        rel_err = (fp4_out - ref_out).abs().mean() / ref_out.abs().mean()
        assert rel_err < 0.15

    @skip_no_triton
    def test_3d_input(self):
        x = torch.randn(2, 16, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        out = fp4_matmul(x, w)
        assert out.shape == (2, 16, 32)

    @skip_no_triton
    def test_gradient(self):
        x = torch.randn(8, 64, device="cuda", requires_grad=True)
        w = torch.randn(32, 64, device="cuda", requires_grad=True)
        b = torch.randn(32, device="cuda", requires_grad=True)

        out = fp4_matmul(x, w, bias=b)
        out.sum().backward()

        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None

    @skip_no_triton
    def test_large_matmul(self):
        torch.manual_seed(0)
        x = torch.randn(64, 1024, device="cuda")
        w = torch.randn(512, 1024, device="cuda")
        out = fp4_matmul(x, w)
        assert out.shape == (64, 512)
