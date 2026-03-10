"""Sparse expert dispatch + batched matmul kernel for GENESIS MoE.

Replaces the naive Python loop over experts with an efficient
gather-compute-scatter pattern:

1. **Gather**: Sort/group tokens by expert assignment.
2. **Compute**: Batch-process each expert's tokens as a contiguous GEMM.
3. **Scatter**: Accumulate weighted results back to the original positions.

Falls back to the loop-based implementation when Triton is unavailable.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core dispatch logic (works on both CPU and CUDA)
# ---------------------------------------------------------------------------


def _sorted_expert_dispatch(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: nn.ModuleList,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Dispatch tokens to experts using sorted gather/scatter.

    Instead of looping over every (k, expert) pair and masking, this:
    1. Expands tokens by top_k (each token appears top_k times).
    2. Sorts by expert index so each expert's tokens are contiguous.
    3. Runs each expert on its contiguous batch (efficient GEMM).
    4. Scatters weighted outputs back via index_add.

    This eliminates per-token boolean masking and enables cuBLAS to
    operate on larger contiguous batches.
    """
    N, D = x.shape  # N = B * S

    # --- Expand: replicate each token top_k times ---
    # flat_indices: (N * top_k,) -- which expert each expanded token goes to
    # flat_weights: (N * top_k,) -- the routing weight
    # flat_token_idx: (N * top_k,) -- original token index
    flat_indices = expert_indices.reshape(-1)  # (N * top_k)
    flat_weights = expert_weights.reshape(-1)  # (N * top_k)
    flat_token_idx = (
        torch.arange(N, device=x.device)
        .unsqueeze(1)
        .expand(N, top_k)
        .reshape(-1)
    )  # (N * top_k)

    # --- Sort by expert index for contiguous batches ---
    sorted_order = flat_indices.argsort(stable=True)
    sorted_expert_ids = flat_indices[sorted_order]
    sorted_weights = flat_weights[sorted_order]
    sorted_token_idx = flat_token_idx[sorted_order]

    # Gather the input tokens in expert-sorted order
    sorted_x = x[sorted_token_idx]  # (N * top_k, D)

    # --- Compute expert boundaries ---
    # Count how many tokens go to each expert
    expert_counts = torch.zeros(
        num_experts, dtype=torch.long, device=x.device
    )
    expert_counts.scatter_add_(
        0, sorted_expert_ids.long(), torch.ones_like(sorted_expert_ids, dtype=torch.long)
    )
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.long, device=x.device
    )
    expert_offsets[1:] = expert_counts.cumsum(0)

    # --- Run each expert on its contiguous slice ---
    sorted_out = torch.empty_like(sorted_x)

    for e in range(num_experts):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()
        if start == end:
            continue
        expert_input = sorted_x[start:end]  # contiguous (count_e, D)
        sorted_out[start:end] = experts[e](expert_input)

    # --- Scatter weighted outputs back ---
    sorted_out = sorted_out * sorted_weights.unsqueeze(-1)

    output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
    output.index_add_(0, sorted_token_idx, sorted_out)

    return output


def _naive_expert_dispatch(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: nn.ModuleList,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Original loop-based dispatch (reference implementation)."""
    N, D = x.shape
    output = torch.zeros_like(x)

    for k in range(top_k):
        indices_k = expert_indices[:, k]
        weights_k = expert_weights[:, k]

        for expert_idx in range(num_experts):
            token_mask = indices_k == expert_idx
            if not token_mask.any():
                continue
            expert_input = x[token_mask]
            expert_output = experts[expert_idx](expert_input)
            output[token_mask] += (
                weights_k[token_mask].unsqueeze(-1) * expert_output
            )

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sparse_expert_matmul(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: nn.ModuleList,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Efficient sparse expert dispatch and computation.

    Sorts tokens by expert assignment for contiguous GEMM batches, then
    scatters weighted results back.  This is significantly faster than the
    naive loop when experts have variable load, because each expert
    processes a single contiguous tensor rather than many small masked
    slices.

    Args:
        x: Input tokens ``(N, D)`` where ``N = B * S``.
        expert_indices: Selected expert indices ``(N, top_k)``, int64.
        expert_weights: Normalized routing weights ``(N, top_k)``, float.
        experts: ``nn.ModuleList`` of expert modules (each maps ``D -> D``).
        num_experts: Total number of experts.
        top_k: Number of experts per token.

    Returns:
        Output tensor ``(N, D)``.
    """
    return _sorted_expert_dispatch(
        x, expert_indices, expert_weights, experts, num_experts, top_k
    )
