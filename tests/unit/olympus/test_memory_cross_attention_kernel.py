"""Tests for optimized memory cross-attention kernel.

Module under test: olympus.kernels.memory_cross_attention
"""

import pytest
import torch
import torch.nn as nn

from genesis.memory.memory_cross_attention import (
    MemoryCrossAttention as OriginalMemoryCrossAttention,
)
from olympus.kernels.memory_cross_attention import (
    HAS_SDPA,
    _expand_mask_pt,
    _manual_cross_attention,
    memory_cross_attention,
)

CUDA_AVAILABLE = torch.cuda.is_available()


class TestExpandMask:
    def test_shape(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        out = _expand_mask_pt(mask)
        assert out.shape == (2, 1, 1, 3)

    def test_values(self):
        mask = torch.tensor([[True, False]])
        out = _expand_mask_pt(mask)
        assert out[0, 0, 0, 0].item() == 0.0
        assert out[0, 0, 0, 1].item() == float("-inf")

    def test_all_valid(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = _expand_mask_pt(mask)
        assert (out == 0.0).all()


class TestManualCrossAttention:
    def test_output_shape(self):
        B, H, S_q, M, D = 2, 4, 8, 16, 32
        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, M, D)
        v = torch.randn(B, H, M, D)
        out = _manual_cross_attention(q, k, v, None, 1.0 / D**0.5, 0.0, False)
        assert out.shape == (B, H, S_q, D)

    def test_with_mask(self):
        B, H, S_q, M, D = 1, 2, 4, 8, 16
        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, M, D)
        v = torch.randn(B, H, M, D)
        mask = torch.zeros(B, 1, 1, M)
        mask[..., :4] = 0.0
        mask[..., 4:] = float("-inf")
        out = _manual_cross_attention(q, k, v, mask, 1.0 / D**0.5, 0.0, False)
        assert out.shape == (B, H, S_q, D)


class TestMemoryCrossAttention:
    @staticmethod
    def _make_modules(d_model=64, d_memory=64, num_heads=4):
        q_proj = nn.Linear(d_model, d_model)
        k_proj = nn.Linear(d_memory, d_model)
        v_proj = nn.Linear(d_memory, d_model)
        o_proj = nn.Linear(d_model, d_model)
        return q_proj, k_proj, v_proj, o_proj

    def test_basic_output_shape(self):
        torch.manual_seed(0)
        d_model, num_heads = 64, 4
        q_proj, k_proj, v_proj, o_proj = self._make_modules(d_model)

        query = torch.randn(2, 8, d_model)
        mem_k = torch.randn(2, 16, d_model)
        mem_v = torch.randn(2, 16, d_model)

        out = memory_cross_attention(
            query, mem_k, mem_v,
            q_proj, k_proj, v_proj, o_proj,
            num_heads=num_heads,
        )
        assert out.shape == (2, 8, d_model)

    def test_unbatched_memory(self):
        torch.manual_seed(0)
        d_model, num_heads = 64, 4
        q_proj, k_proj, v_proj, o_proj = self._make_modules(d_model)

        query = torch.randn(2, 8, d_model)
        mem_k = torch.randn(16, d_model)  # unbatched
        mem_v = torch.randn(16, d_model)  # unbatched

        out = memory_cross_attention(
            query, mem_k, mem_v,
            q_proj, k_proj, v_proj, o_proj,
            num_heads=num_heads,
        )
        assert out.shape == (2, 8, d_model)

    def test_with_memory_mask(self):
        torch.manual_seed(0)
        d_model, num_heads = 64, 4
        q_proj, k_proj, v_proj, o_proj = self._make_modules(d_model)

        query = torch.randn(2, 8, d_model)
        mem_k = torch.randn(2, 16, d_model)
        mem_v = torch.randn(2, 16, d_model)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 12:] = False  # last 4 slots invalid

        out = memory_cross_attention(
            query, mem_k, mem_v,
            q_proj, k_proj, v_proj, o_proj,
            num_heads=num_heads,
            memory_mask=mask,
        )
        assert out.shape == (2, 8, d_model)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        d_model, num_heads = 64, 4
        q_proj, k_proj, v_proj, o_proj = self._make_modules(d_model)

        query = torch.randn(2, 4, d_model, requires_grad=True)
        mem_k = torch.randn(2, 8, d_model)
        mem_v = torch.randn(2, 8, d_model)

        out = memory_cross_attention(
            query, mem_k, mem_v,
            q_proj, k_proj, v_proj, o_proj,
            num_heads=num_heads,
        )
        out.sum().backward()

        assert query.grad is not None
        assert q_proj.weight.grad is not None

    def test_matches_original_module(self):
        """Output should match the original MemoryCrossAttention module."""
        torch.manual_seed(42)
        d_model, num_heads = 64, 4

        orig = OriginalMemoryCrossAttention(d_model, num_heads, dropout=0.0)

        query = torch.randn(2, 8, d_model)
        mem_k = torch.randn(2, 16, d_model)
        mem_v = torch.randn(2, 16, d_model)

        orig.eval()
        orig_out = orig(query, mem_k, mem_v)

        # Use the same weights
        kernel_out = memory_cross_attention(
            query, mem_k, mem_v,
            orig.q_proj, orig.k_proj, orig.v_proj, orig.o_proj,
            num_heads=num_heads,
            dropout_p=0.0,
            training=False,
        )

        assert torch.allclose(orig_out, kernel_out, atol=1e-5)
