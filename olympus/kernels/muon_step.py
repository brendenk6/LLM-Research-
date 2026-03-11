"""Fused Muon optimizer step kernel for GENESIS.

Fuses the element-wise operations of the Muon optimizer (weight decay,
momentum update, Nesterov look-ahead, norm scaling, parameter update) into
minimal kernel launches.  Newton-Schulz matrix multiplications use cuBLAS.

Falls back to the separate-operation implementation on CPU.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Newton-Schulz (cuBLAS — not fused, matmul is already optimal)
# ---------------------------------------------------------------------------

NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz orthogonalization (same as Muon._newton_schulz)."""
    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        transposed = True

    norm = G.norm() + eps
    X = G / norm

    for _ in range(steps):
        A = X @ X.T
        X = NS_A * X + NS_B * (A @ X) + NS_C * (A @ (A @ X))

    if transposed:
        X = X.T
    return X


# ---------------------------------------------------------------------------
# Triton kernels for element-wise fusion
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _fused_momentum_nesterov_kernel(
        grad_ptr,
        buf_ptr,
        out_ptr,
        n_elements,
        mu,
        BLOCK: tl.constexpr,
    ):
        """Fused momentum update + Nesterov computation.

        buf = mu * buf + grad
        out = grad + mu * buf   (Nesterov look-ahead)

        Two reads (grad, buf), two writes (buf, out) instead of four
        separate kernel launches.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(buf_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Momentum update
        b_new = mu * b + g
        tl.store(buf_ptr + offs, b_new, mask=mask)

        # Nesterov
        nesterov = g + mu * b_new
        tl.store(out_ptr + offs, nesterov, mask=mask)

    @triton.jit
    def _fused_scale_update_kernel(
        param_ptr,
        orth_grad_ptr,
        n_elements,
        scale,
        lr,
        BLOCK: tl.constexpr,
    ):
        """Fused norm-scaling + parameter update.

        param -= lr * (scale * orth_grad)

        One read of orth_grad, one read-modify-write of param, instead of
        separate mul + add kernels.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        p = tl.load(param_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        og = tl.load(orth_grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        p_new = p - lr * scale * og
        tl.store(param_ptr + offs, p_new, mask=mask)

    @triton.jit
    def _weight_decay_kernel(
        param_ptr,
        n_elements,
        decay_factor,
        BLOCK: tl.constexpr,
    ):
        """In-place weight decay: param *= decay_factor."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        p = tl.load(param_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(param_ptr + offs, p * decay_factor, mask=mask)


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 1024


def _triton_grid(n: int) -> tuple:
    return (triton.cdiv(n, BLOCK_SIZE),)


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------


@torch.no_grad()
def _muon_step_pt(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    ns_steps: int,
) -> None:
    """Single Muon parameter update (pure PyTorch, in-place)."""
    # Weight decay
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)

    # Momentum + Nesterov
    momentum_buffer.mul_(momentum).add_(grad)
    nesterov_grad = grad + momentum * momentum_buffer

    # Orthogonalize
    orth_grad = _newton_schulz(nesterov_grad, steps=ns_steps)

    # Scale to match gradient norm
    scale = grad.norm() / (orth_grad.norm() + 1e-8)
    orth_grad.mul_(scale)

    # Update
    param.add_(orth_grad, alpha=-lr)


# ---------------------------------------------------------------------------
# Fused step (Triton element-wise + cuBLAS Newton-Schulz)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _muon_step_fused(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    ns_steps: int,
) -> None:
    """Fused Muon step: Triton for element-wise, cuBLAS for matmuls."""
    n = param.numel()

    # 1. Weight decay (single fused kernel)
    if weight_decay != 0.0:
        _weight_decay_kernel[_triton_grid(n)](
            param, n, 1.0 - lr * weight_decay, BLOCK=BLOCK_SIZE,
        )

    # 2. Fused momentum + Nesterov (single kernel, 2 reads + 2 writes)
    nesterov_grad = torch.empty_like(grad)
    _fused_momentum_nesterov_kernel[_triton_grid(n)](
        grad, momentum_buffer, nesterov_grad, n, momentum,
        BLOCK=BLOCK_SIZE,
    )

    # 3. Newton-Schulz orthogonalization (cuBLAS matmuls — optimal as-is)
    orth_grad = _newton_schulz(nesterov_grad, steps=ns_steps)

    # 4. Compute norm scale (reductions — PyTorch is efficient)
    scale = grad.norm().item() / (orth_grad.norm().item() + 1e-8)

    # 5. Fused scale + update (single kernel)
    _fused_scale_update_kernel[_triton_grid(n)](
        param, orth_grad, n, scale, lr, BLOCK=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def muon_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
) -> None:
    """Perform a single fused Muon optimizer step on one parameter.

    Modifies ``param`` and ``momentum_buffer`` in-place.  Uses Triton
    kernels for element-wise operations on CUDA, cuBLAS for the
    Newton-Schulz matrix multiplications.

    Args:
        param: 2-D parameter tensor ``(M, N)``.
        grad: Gradient tensor matching ``param`` shape.
        momentum_buffer: Momentum state tensor matching ``param`` shape.
        lr: Learning rate.
        momentum: Nesterov momentum coefficient.
        weight_decay: Decoupled weight decay.
        ns_steps: Newton-Schulz iteration count.
    """
    if HAS_TRITON and param.is_cuda:
        _muon_step_fused(
            param, grad, momentum_buffer, lr, momentum, weight_decay, ns_steps
        )
    else:
        _muon_step_pt(
            param, grad, momentum_buffer, lr, momentum, weight_decay, ns_steps
        )
