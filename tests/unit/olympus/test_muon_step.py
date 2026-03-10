"""Tests for fused Muon optimizer step kernel.

Verifies that the fused step produces the same parameter updates as
the original Muon optimizer class.

Module under test: olympus.kernels.muon_step
"""

import pytest
import torch
import torch.nn as nn

from olympus.kernels.muon_step import (
    HAS_TRITON,
    _muon_step_pt,
    _newton_schulz,
    muon_step,
)
from olympus.optim.muon import Muon

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = HAS_TRITON and CUDA_AVAILABLE
skip_no_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE, reason="CUDA + Triton required"
)


class TestNewtonSchulz:
    def test_approximately_orthogonal(self):
        torch.manual_seed(0)
        G = torch.randn(64, 64)
        Q = _newton_schulz(G, steps=5)
        product = Q @ Q.T
        identity = torch.eye(64)
        assert (product - identity).abs().max().item() < 0.5

    def test_preserves_shape(self):
        G = torch.randn(32, 64)
        Q = _newton_schulz(G, steps=5)
        assert Q.shape == (32, 64)


class TestMuonStepPyTorch:
    def test_param_changes(self):
        torch.manual_seed(42)
        param = torch.randn(32, 32)
        grad = torch.randn(32, 32)
        buf = torch.zeros(32, 32)
        param_before = param.clone()

        _muon_step_pt(param, grad, buf, lr=0.02, momentum=0.95,
                       weight_decay=0.0, ns_steps=5)

        assert not torch.equal(param, param_before)

    def test_momentum_buffer_updated(self):
        torch.manual_seed(0)
        param = torch.randn(32, 32)
        grad = torch.randn(32, 32)
        buf = torch.zeros(32, 32)

        _muon_step_pt(param, grad, buf, lr=0.02, momentum=0.95,
                       weight_decay=0.0, ns_steps=5)

        # Buffer should no longer be zero
        assert buf.abs().sum() > 0

    def test_weight_decay_shrinks(self):
        param = torch.ones(16, 16)
        grad = torch.zeros(16, 16)
        buf = torch.zeros(16, 16)

        _muon_step_pt(param, grad, buf, lr=0.1, momentum=0.95,
                       weight_decay=0.5, ns_steps=5)

        # Weight decay should reduce param norm
        assert param.abs().mean().item() < 1.0

    def test_matches_muon_class(self):
        """Fused PT step should match the Muon optimizer class output."""
        torch.manual_seed(42)

        # Reference: Muon class
        model_ref = nn.Linear(32, 32, bias=False)
        opt = Muon(model_ref.parameters(), lr=0.02, momentum=0.95)
        x = torch.randn(4, 32)
        loss = model_ref(x).sum()
        loss.backward()
        grad_ref = model_ref.weight.grad.clone()
        opt.step()
        param_ref = model_ref.weight.clone()

        # Test: muon_step function
        torch.manual_seed(42)
        model_test = nn.Linear(32, 32, bias=False)
        # Copy initial weights to match
        with torch.no_grad():
            model_test.weight.copy_(model_ref.weight + (param_ref - model_ref.weight))
            # Actually, let's just use the same starting weights
        torch.manual_seed(42)
        model_test2 = nn.Linear(32, 32, bias=False)
        x2 = torch.randn(4, 32)
        loss2 = model_test2(x2).sum()
        loss2.backward()

        param = model_test2.weight.data.clone()
        grad = model_test2.weight.grad.clone()
        buf = torch.zeros_like(param)

        _muon_step_pt(param, grad, buf, lr=0.02, momentum=0.95,
                       weight_decay=0.0, ns_steps=5)

        assert torch.allclose(param, param_ref, atol=1e-5)


class TestMuonStepTriton:
    @skip_no_triton
    def test_matches_pytorch(self):
        torch.manual_seed(42)

        # PT path
        param_pt = torch.randn(64, 64, device="cuda")
        grad = torch.randn(64, 64, device="cuda")
        buf_pt = torch.zeros(64, 64, device="cuda")

        param_tr = param_pt.clone()
        buf_tr = buf_pt.clone()

        _muon_step_pt(param_pt, grad, buf_pt, lr=0.02, momentum=0.95,
                       weight_decay=0.01, ns_steps=5)
        muon_step(param_tr, grad, buf_tr, lr=0.02, momentum=0.95,
                  weight_decay=0.01, ns_steps=5)

        assert torch.allclose(param_tr, param_pt, atol=1e-4)
        assert torch.allclose(buf_tr, buf_pt, atol=1e-5)

    @skip_no_triton
    def test_large_matrix(self):
        torch.manual_seed(0)
        param = torch.randn(512, 256, device="cuda")
        grad = torch.randn(512, 256, device="cuda")
        buf = torch.zeros(512, 256, device="cuda")

        param_before = param.clone()
        muon_step(param, grad, buf, lr=0.02, momentum=0.95, ns_steps=5)

        assert not torch.equal(param, param_before)
