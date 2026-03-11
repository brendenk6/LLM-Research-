"""
Muon optimizer: Orthogonalized gradient descent via Newton-Schulz iteration,
combined with Nesterov momentum and decoupled weight decay.

Only operates on 2-D parameter tensors.  1-D / other shapes are skipped
(use MuonAdamWHybrid if you need mixed handling).
"""


import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.

    Applies Newton-Schulz orthogonalization to the gradient (or momentum
    buffer) before updating the parameters, which has the effect of
    preconditioning with the inverse square root of the gradient covariance.

    Args:
        params:       Iterable of parameters (must be 2-D tensors).
        lr:           Learning rate (default: 0.02).
        momentum:     Nesterov momentum coefficient (default: 0.95).
        weight_decay: Decoupled weight decay (default: 0.0).
        ns_steps:     Number of Newton-Schulz iterations (default: 5).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

        # Validate: all params must be 2-D
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon only supports 2-D parameters, got shape {p.shape}. "
                        f"Use MuonAdamWHybrid for mixed-dimension parameter sets."
                    )

    @staticmethod
    @torch.no_grad()
    def _newton_schulz(
        G: torch.Tensor,
        steps: int = 5,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """Approximate orthogonalization via Newton-Schulz iteration.

        Computes an approximation of ``G @ (G^T G)^{-1/2}`` which
        orthogonalizes the rows of G.

        For a matrix X, the iteration is:
            X_{k+1} = 1.5 * X_k  -  0.5 * X_k @ X_k^T @ X_k

        We start with X_0 = G / ||G||_F and iterate *steps* times.

        Args:
            G:     2-D gradient tensor.
            steps: Number of iterations.
            eps:   Small constant for numerical stability.

        Returns:
            Orthogonalized gradient tensor with the same shape.
        """
        assert G.dim() == 2, f"Newton-Schulz requires 2-D input, got {G.dim()}-D"

        # Ensure we operate on the "tall" orientation for stability
        transposed = False
        if G.shape[0] < G.shape[1]:
            G = G.T
            transposed = True

        # Normalize
        norm = G.norm() + eps
        X = G / norm

        # Newton-Schulz iteration: X <- 1.5 X - 0.5 X X^T X
        # Using pre-computed coefficients for the cubic polynomial that
        # converges to the polar factor:
        #   a, b, c tuned for faster convergence
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))

        if transposed:
            X = X.T

        return X

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
                     (optional).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]

                # Initialize momentum buffer
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]

                # Nesterov momentum: update buffer, then use look-ahead
                buf.mul_(mu).add_(grad)
                nesterov_grad = grad + mu * buf

                # Orthogonalize
                orth_grad = self._newton_schulz(nesterov_grad, steps=ns_steps)

                # Scale to match original gradient norm (preserves effective LR semantics)
                scale = grad.norm() / (orth_grad.norm() + 1e-8)
                orth_grad.mul_(scale)

                # Update parameters
                p.add_(orth_grad, alpha=-lr)

        return loss
