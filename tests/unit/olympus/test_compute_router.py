"""Tests for ComputeRouter.

ComputeRouter provides dynamic routing with load-balancing loss for
variable-compute architectures (e.g., mixture-of-experts).

The module under test is olympus.core.compute_router.
"""

import pytest
import torch
import torch.nn as nn

from olympus.core.compute_router import ComputeRouter, RoutePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(
    input_dim: int = 64,
    num_experts: int = 3,
    top_k: int = 2,
    gate_type: str = "linear",
) -> ComputeRouter:
    return ComputeRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        top_k=top_k,
        gate_type=gate_type,
    )


def _dummy_experts(num: int, dim: int) -> nn.ModuleList:
    """Create simple linear experts for testing routing execution."""
    return nn.ModuleList([nn.Linear(dim, dim) for _ in range(num)])


# ---------------------------------------------------------------------------
# Tests — plan() returns correct shapes and data
# ---------------------------------------------------------------------------

class TestPlanShapes:
    def test_plan_returns_route_plan(self):
        router = _make_router(input_dim=64, num_experts=3)
        x = torch.randn(4, 16, 64)  # (B, S, D)
        plan = router.plan(x)
        assert isinstance(plan, RoutePlan)

    def test_plan_expert_indices_shape(self):
        router = _make_router(input_dim=64, num_experts=4, top_k=2)
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        # 3-D input is flattened: N = B * S = 64
        assert plan.expert_indices.shape == (64, 2)

    def test_plan_expert_weights_shape(self):
        router = _make_router(input_dim=64, num_experts=4, top_k=2)
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        assert plan.expert_weights.shape == (64, 2)

    def test_plan_probabilities_shape(self):
        router = _make_router(input_dim=64, num_experts=4, top_k=2)
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        assert plan.probabilities.shape == (64, 4)

    def test_plan_raw_logits_shape(self):
        router = _make_router(input_dim=64, num_experts=4, top_k=2)
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        assert plan.raw_logits.shape == (64, 4)

    def test_plan_weights_sum_to_one(self):
        router = _make_router(input_dim=64, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        plan = router.plan(x)
        sums = plan.expert_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_plan_probabilities_sum_to_one(self):
        router = _make_router(input_dim=64, num_experts=3)
        x = torch.randn(8, 64)
        plan = router.plan(x)
        sums = plan.probabilities.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_plan_2d_input(self):
        router = _make_router(input_dim=64, num_experts=3, top_k=2)
        x = torch.randn(8, 64)
        plan = router.plan(x)
        assert plan.expert_indices.shape == (8, 2)

    def test_plan_num_experts_property(self):
        router = _make_router(input_dim=64, num_experts=5, top_k=2)
        x = torch.randn(4, 64)
        plan = router.plan(x)
        assert plan.num_experts == 5

    def test_plan_top_k_property(self):
        router = _make_router(input_dim=64, num_experts=5, top_k=3)
        x = torch.randn(4, 64)
        plan = router.plan(x)
        assert plan.top_k == 3

    def test_plan_indices_in_valid_range(self):
        router = _make_router(input_dim=64, num_experts=4, top_k=2)
        x = torch.randn(16, 64)
        plan = router.plan(x)
        assert (plan.expert_indices >= 0).all()
        assert (plan.expert_indices < 4).all()


# ---------------------------------------------------------------------------
# Tests — execute() routes tokens through correct paths
# ---------------------------------------------------------------------------

class TestExecuteRoutesCorrectly:
    def test_execute_output_shape_3d(self):
        dim = 64
        num_experts = 3
        router = _make_router(input_dim=dim, num_experts=num_experts, top_k=2)
        experts = _dummy_experts(num_experts, dim)
        x = torch.randn(2, 8, dim)
        out = router.execute(x, experts)
        assert out.shape == x.shape

    def test_execute_output_shape_2d(self):
        dim = 64
        num_experts = 3
        router = _make_router(input_dim=dim, num_experts=num_experts, top_k=2)
        experts = _dummy_experts(num_experts, dim)
        x = torch.randn(8, dim)
        out = router.execute(x, experts)
        assert out.shape == x.shape

    def test_execute_with_precomputed_plan(self):
        dim = 64
        num_experts = 2
        router = _make_router(input_dim=dim, num_experts=num_experts, top_k=2)
        experts = _dummy_experts(num_experts, dim)
        x = torch.randn(4, dim)
        plan = router.plan(x)
        out = router.execute(x, experts, route_plan=plan)
        assert out.shape == x.shape

    def test_execute_without_plan(self):
        dim = 64
        num_experts = 2
        router = _make_router(input_dim=dim, num_experts=num_experts, top_k=2)
        experts = _dummy_experts(num_experts, dim)
        x = torch.randn(4, dim)
        # execute without route_plan should compute plan internally
        out = router.execute(x, experts)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Tests — all gate types work
# ---------------------------------------------------------------------------

class TestAllGateTypes:
    @pytest.mark.parametrize("gate_type", ["linear", "mlp", "entropy"])
    def test_gate_type_constructs_and_plans(self, gate_type: str):
        router = _make_router(input_dim=64, num_experts=3, gate_type=gate_type)
        x = torch.randn(2, 8, 64)
        plan = router.plan(x)
        assert plan.probabilities.shape[-1] == 3

    @pytest.mark.parametrize("gate_type", ["linear", "mlp", "entropy"])
    def test_gate_type_execute(self, gate_type: str):
        dim = 64
        num_experts = 2
        router = _make_router(input_dim=dim, num_experts=num_experts, top_k=2, gate_type=gate_type)
        experts = _dummy_experts(num_experts, dim)
        x = torch.randn(4, dim)
        out = router.execute(x, experts)
        assert out.shape == x.shape

    @pytest.mark.parametrize("gate_type", ["linear", "mlp", "entropy"])
    def test_gate_type_backward(self, gate_type: str):
        dim = 64
        num_experts = 2
        router = _make_router(input_dim=dim, num_experts=num_experts, top_k=2, gate_type=gate_type)
        router.train()
        experts = _dummy_experts(num_experts, dim)
        x = torch.randn(4, dim, requires_grad=True)
        out = router.execute(x, experts)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_invalid_gate_type_raises(self):
        with pytest.raises(ValueError, match="Unknown gate_type"):
            ComputeRouter(input_dim=64, num_experts=3, gate_type="invalid")


# ---------------------------------------------------------------------------
# Tests — load_balance_loss is a non-negative scalar
# ---------------------------------------------------------------------------

class TestLoadBalanceLoss:
    def test_load_balance_loss_is_scalar(self):
        router = _make_router(input_dim=64, num_experts=3)
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        lb_loss = router.load_balance_loss(plan)
        assert lb_loss.dim() == 0

    def test_load_balance_loss_nonnegative_linear(self):
        router = _make_router(input_dim=64, num_experts=3, gate_type="linear")
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        lb_loss = router.load_balance_loss(plan)
        assert lb_loss.item() >= 0.0

    def test_load_balance_loss_nonnegative_mlp(self):
        router = _make_router(input_dim=64, num_experts=3, gate_type="mlp")
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        lb_loss = router.load_balance_loss(plan)
        assert lb_loss.item() >= 0.0

    def test_load_balance_loss_is_scalar_entropy(self):
        router = _make_router(input_dim=64, num_experts=3, gate_type="entropy")
        x = torch.randn(4, 16, 64)
        plan = router.plan(x)
        lb_loss = router.load_balance_loss(plan)
        # entropy gate subtracts entropy bonus, so loss may be negative
        assert lb_loss.dim() == 0
        assert torch.isfinite(lb_loss)

    def test_load_balance_loss_differentiable(self):
        router = _make_router(input_dim=64, num_experts=3)
        router.train()
        x = torch.randn(8, 64, requires_grad=True)
        plan = router.plan(x)
        lb_loss = router.load_balance_loss(plan)
        lb_loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Tests — get_routing_stats returns expected data
# ---------------------------------------------------------------------------

class TestRoutingStats:
    def test_stats_empty_before_any_plan(self):
        router = _make_router(input_dim=64, num_experts=3)
        stats = router.get_routing_stats()
        assert isinstance(stats, dict)
        assert stats["total_tokens"] == 0
        assert stats["total_steps"] == 0

    def test_stats_populated_after_plan(self):
        router = _make_router(input_dim=64, num_experts=3)
        x = torch.randn(4, 16, 64)
        router.plan(x)
        stats = router.get_routing_stats()
        assert stats["total_tokens"] == 4 * 16  # B * S = 64
        assert stats["total_steps"] == 1
        assert "expert_utilization" in stats
        assert len(stats["expert_utilization"]) == 3
        assert "max_load_ratio" in stats
        assert "min_load_ratio" in stats

    def test_stats_accumulate_over_multiple_plans(self):
        router = _make_router(input_dim=64, num_experts=3)
        x = torch.randn(8, 64)
        router.plan(x)
        router.plan(x)
        stats = router.get_routing_stats()
        assert stats["total_tokens"] == 16  # 8 + 8
        assert stats["total_steps"] == 2

    def test_stats_expert_utilization_sums_to_one(self):
        router = _make_router(input_dim=64, num_experts=4)
        x = torch.randn(32, 64)
        router.plan(x)
        stats = router.get_routing_stats()
        util_sum = sum(stats["expert_utilization"].values())
        assert abs(util_sum - 1.0) < 1e-5

    def test_reset_stats(self):
        router = _make_router(input_dim=64, num_experts=3)
        x = torch.randn(8, 64)
        router.plan(x)
        router.reset_stats()
        stats = router.get_routing_stats()
        assert stats["total_tokens"] == 0
        assert stats["total_steps"] == 0
