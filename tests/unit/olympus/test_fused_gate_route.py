"""Tests for fused gate routing kernels.

Verifies PyTorch fallback correctness and, when CUDA + Triton are available,
numerical equivalence between the Triton kernels and the PyTorch reference.

Module under test: olympus.kernels.fused_gate_route
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.kernels.fused_gate_route import (
    HAS_TRITON,
    _chunk_mean_pool_pt,
    _softmax_topk_pt,
    chunk_mean_pool,
    fused_gate_route,
    softmax_topk,
)

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = HAS_TRITON and CUDA_AVAILABLE
skip_no_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE, reason="CUDA + Triton required"
)


# ---------------------------------------------------------------------------
# chunk_mean_pool
# ---------------------------------------------------------------------------


class TestChunkMeanPoolPyTorch:
    def test_basic_shape(self):
        x = torch.randn(2, 64, 128)
        out = _chunk_mean_pool_pt(x, chunk_size=32)
        assert out.shape == (2, 2, 128)

    def test_matches_view_mean(self):
        torch.manual_seed(42)
        x = torch.randn(4, 128, 256)
        result = _chunk_mean_pool_pt(x, 32)
        expected = x.view(4, 4, 32, 256).mean(dim=2)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_single_chunk(self):
        x = torch.randn(1, 32, 64)
        out = _chunk_mean_pool_pt(x, chunk_size=32)
        expected = x.mean(dim=1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_gradient_flow(self):
        x = torch.randn(2, 64, 32, requires_grad=True)
        out = _chunk_mean_pool_pt(x, chunk_size=32)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        expected_grad = torch.full_like(x, 1.0 / 32)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)


class TestChunkMeanPoolTriton:
    @skip_no_triton
    def test_matches_pytorch(self):
        torch.manual_seed(42)
        x = torch.randn(4, 128, 256, device="cuda")
        pt_result = _chunk_mean_pool_pt(x, 32)
        triton_result = chunk_mean_pool(x, 32)
        assert torch.allclose(triton_result, pt_result, atol=1e-5)

    @skip_no_triton
    def test_large_d_model(self):
        torch.manual_seed(0)
        x = torch.randn(2, 64, 2048, device="cuda")
        pt_result = _chunk_mean_pool_pt(x, 32)
        triton_result = chunk_mean_pool(x, 32)
        assert torch.allclose(triton_result, pt_result, atol=1e-5)

    @skip_no_triton
    def test_gradient(self):
        x = torch.randn(2, 64, 128, device="cuda", requires_grad=True)
        out = chunk_mean_pool(x, chunk_size=32)
        out.sum().backward()
        assert x.grad is not None
        expected_grad = torch.full_like(x, 1.0 / 32)
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)


# ---------------------------------------------------------------------------
# softmax_topk
# ---------------------------------------------------------------------------


class TestSoftmaxTopKPyTorch:
    def test_basic_shape(self):
        logits = torch.randn(16, 8)
        weights, indices, probs = _softmax_topk_pt(logits, top_k=2)
        assert weights.shape == (16, 2)
        assert indices.shape == (16, 2)
        assert probs.shape == (16, 8)

    def test_weights_sum_to_one(self):
        logits = torch.randn(32, 4)
        weights, _, _ = _softmax_topk_pt(logits, top_k=2)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_in_range(self):
        num_experts = 8
        logits = torch.randn(16, num_experts)
        _, indices, _ = _softmax_topk_pt(logits, top_k=2)
        assert (indices >= 0).all()
        assert (indices < num_experts).all()

    def test_selects_highest_probs(self):
        logits = torch.tensor([[0.0, 5.0, 0.0, 10.0]])
        _, indices, _ = _softmax_topk_pt(logits, top_k=2)
        assert set(indices[0].tolist()) == {1, 3}

    def test_probs_sum_to_one(self):
        logits = torch.randn(16, 8)
        _, _, probs = _softmax_topk_pt(logits, top_k=2)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_topk_1(self):
        logits = torch.randn(8, 4)
        weights, indices, _ = _softmax_topk_pt(logits, top_k=1)
        assert weights.shape == (8, 1)
        assert torch.allclose(
            weights, torch.ones_like(weights), atol=1e-5
        )


class TestSoftmaxTopKTriton:
    @skip_no_triton
    def test_matches_pytorch(self):
        torch.manual_seed(42)
        logits = torch.randn(64, 8, device="cuda")
        pt_w, pt_i, pt_p = _softmax_topk_pt(logits, top_k=2)
        tr_w, tr_i, tr_p = softmax_topk(logits, top_k=2)
        assert torch.allclose(tr_w, pt_w, atol=1e-5)
        assert (tr_i == pt_i).all()
        assert torch.allclose(tr_p, pt_p, atol=1e-5)

    @skip_no_triton
    def test_num_experts_16(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 16, device="cuda")
        pt_w, pt_i, pt_p = _softmax_topk_pt(logits, top_k=2)
        tr_w, tr_i, tr_p = softmax_topk(logits, top_k=2)
        assert torch.allclose(tr_w, pt_w, atol=1e-5)
        assert (tr_i == pt_i).all()

    @skip_no_triton
    def test_gradient(self):
        logits = torch.randn(16, 4, device="cuda", requires_grad=True)
        logits_ref = logits.detach().clone().requires_grad_(True)

        w, _, _ = softmax_topk(logits, top_k=2)
        w.sum().backward()

        w_ref, _, _ = _softmax_topk_pt(logits_ref, top_k=2)
        w_ref.sum().backward()

        assert torch.allclose(logits.grad, logits_ref.grad, atol=1e-4)


# ---------------------------------------------------------------------------
# fused_gate_route end-to-end
# ---------------------------------------------------------------------------


class TestFusedGateRoute:
    @staticmethod
    def _make_modules(d_model=64, num_experts=4):
        gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        router_gate = nn.Linear(d_model, num_experts, bias=False)
        return gate_mlp, router_gate

    def test_output_shapes(self):
        torch.manual_seed(0)
        d_model, num_experts, chunk_size, top_k = 64, 4, 16, 2
        gate_mlp, router_gate = self._make_modules(d_model, num_experts)

        x = torch.randn(2, 64, d_model)
        mask, scores, weights, indices, probs = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=chunk_size, top_k=top_k, training=True,
        )

        num_chunks = 64 // chunk_size
        assert mask.shape == (2, num_chunks)
        assert scores.shape == (2, num_chunks)
        assert weights.shape == (2 * 64, top_k)
        assert indices.shape == (2 * 64, top_k)
        assert probs.shape == (2 * 64, num_experts)

    def test_inference_mask_is_bool(self):
        torch.manual_seed(0)
        gate_mlp, router_gate = self._make_modules()
        x = torch.randn(1, 32, 64)
        mask, _, _, _, _ = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=16, training=False,
        )
        assert mask.dtype == torch.bool

    def test_training_mask_is_float(self):
        torch.manual_seed(0)
        gate_mlp, router_gate = self._make_modules()
        x = torch.randn(1, 32, 64)
        mask, _, _, _, _ = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=16, training=True,
        )
        assert mask.is_floating_point()

    def test_padding_non_divisible_seq(self):
        torch.manual_seed(0)
        gate_mlp, router_gate = self._make_modules()
        x = torch.randn(1, 30, 64)  # 30 not divisible by 16
        mask, scores, weights, indices, probs = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=16, top_k=2, training=True,
        )
        # Pads to 32 -> 2 chunks
        assert mask.shape == (1, 2)
        # Expert routing over padded length
        assert weights.shape == (32, 2)

    def test_expert_weights_sum_to_one(self):
        torch.manual_seed(0)
        gate_mlp, router_gate = self._make_modules()
        x = torch.randn(2, 32, 64)
        _, _, weights, _, _ = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=16, top_k=2, training=True,
        )
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_jitter_noise_changes_routing(self):
        torch.manual_seed(0)
        gate_mlp, router_gate = self._make_modules()
        x = torch.randn(1, 32, 64)

        gate_mlp.eval()
        router_gate.eval()

        _, _, w1, _, _ = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=16, top_k=2, jitter_noise=0.0, training=True,
        )

        torch.manual_seed(99)
        _, _, w2, _, _ = fused_gate_route(
            x, gate_mlp, router_gate,
            chunk_size=16, top_k=2, jitter_noise=0.5, training=True,
        )

        # With substantial jitter, weights should differ
        assert not torch.allclose(w1, w2)
