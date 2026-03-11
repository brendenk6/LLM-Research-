"""Tests for sparse expert dispatch kernel.

Verifies that the sorted gather/scatter dispatch produces the same results
as the naive loop-based implementation.

Module under test: olympus.kernels.sparse_expert_matmul
"""

import pytest
import torch
import torch.nn as nn

from genesis.model.ffn import SwiGLUFFN
from olympus.kernels.sparse_expert_matmul import (
    _naive_expert_dispatch,
    _sorted_expert_dispatch,
    sparse_expert_matmul,
)


def _make_experts(d_model=64, d_ff=128, num_experts=4):
    return nn.ModuleList(
        [SwiGLUFFN(d_model=d_model, d_ff=d_ff) for _ in range(num_experts)]
    )


def _make_routing(N, num_experts, top_k, device="cpu"):
    """Generate random but valid routing indices and weights."""
    indices = torch.randint(0, num_experts, (N, top_k), device=device)
    # Random weights, normalized per token
    raw = torch.rand(N, top_k, device=device)
    weights = raw / (raw.sum(dim=-1, keepdim=True) + 1e-9)
    return indices, weights


class TestSortedVsNaive:
    def test_output_matches_naive(self):
        torch.manual_seed(42)
        d_model, num_experts, top_k = 64, 4, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 32
        x = torch.randn(N, d_model)
        indices, weights = _make_routing(N, num_experts, top_k)

        naive_out = _naive_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )
        sorted_out = _sorted_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )

        assert torch.allclose(naive_out, sorted_out, atol=1e-5)

    def test_single_expert(self):
        torch.manual_seed(0)
        d_model, num_experts, top_k = 32, 4, 1
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 16
        x = torch.randn(N, d_model)
        indices, weights = _make_routing(N, num_experts, top_k)

        naive_out = _naive_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )
        sorted_out = _sorted_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )

        assert torch.allclose(naive_out, sorted_out, atol=1e-5)

    def test_all_tokens_same_expert(self):
        torch.manual_seed(0)
        d_model, num_experts, top_k = 32, 4, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 8
        x = torch.randn(N, d_model)
        # All tokens routed to expert 2
        indices = torch.full((N, top_k), 2, dtype=torch.long)
        weights = torch.ones(N, top_k) / top_k

        naive_out = _naive_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )
        sorted_out = _sorted_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )

        assert torch.allclose(naive_out, sorted_out, atol=1e-5)

    def test_empty_experts(self):
        """Some experts receive zero tokens."""
        torch.manual_seed(42)
        d_model, num_experts, top_k = 32, 8, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 4  # very few tokens, many experts will be empty
        x = torch.randn(N, d_model)
        indices, weights = _make_routing(N, num_experts, top_k)

        naive_out = _naive_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )
        sorted_out = _sorted_expert_dispatch(
            x, indices, weights, experts, num_experts, top_k
        )

        assert torch.allclose(naive_out, sorted_out, atol=1e-5)


class TestSparseExpertMatmul:
    def test_output_shape(self):
        torch.manual_seed(0)
        d_model, num_experts, top_k = 64, 4, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 32
        x = torch.randn(N, d_model)
        indices, weights = _make_routing(N, num_experts, top_k)

        out = sparse_expert_matmul(
            x, indices, weights, experts, num_experts, top_k
        )
        assert out.shape == (N, d_model)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        d_model, num_experts, top_k = 32, 4, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 16
        x = torch.randn(N, d_model, requires_grad=True)
        indices, weights = _make_routing(N, num_experts, top_k)

        out = sparse_expert_matmul(
            x, indices, weights, experts, num_experts, top_k
        )
        out.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        # At least some expert parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in experts.parameters()
        )
        assert has_grad

    def test_large_batch(self):
        torch.manual_seed(0)
        d_model, num_experts, top_k = 64, 8, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 512
        x = torch.randn(N, d_model)
        indices, weights = _make_routing(N, num_experts, top_k)

        out = sparse_expert_matmul(
            x, indices, weights, experts, num_experts, top_k
        )
        assert out.shape == (N, d_model)

    def test_deterministic(self):
        torch.manual_seed(42)
        d_model, num_experts, top_k = 32, 4, 2
        experts = _make_experts(d_model, num_experts=num_experts)
        N = 16
        x = torch.randn(N, d_model)
        indices, weights = _make_routing(N, num_experts, top_k)

        out1 = sparse_expert_matmul(
            x, indices, weights, experts, num_experts, top_k
        )
        out2 = sparse_expert_matmul(
            x, indices, weights, experts, num_experts, top_k
        )

        assert torch.allclose(out1, out2, atol=1e-6)
