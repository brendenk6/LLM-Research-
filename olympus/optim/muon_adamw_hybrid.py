"""
MuonAdamWHybrid: Automatically routes 2-D parameters to Muon (Newton-Schulz
orthogonalized SGD) and 1-D parameters (biases, LayerNorm weights, embeddings)
to AdamW.

This gives the best of both worlds: Muon's superior conditioning for weight
matrices and AdamW's robustness for everything else.
"""


import torch
from torch.optim import Optimizer


class MuonAdamWHybrid(Optimizer):
    """
    Hybrid optimizer: Muon for 2-D params, AdamW for everything else.

    Args:
        params:           Iterable of parameters or param groups.
        lr_muon:          Learning rate for Muon (2-D params).  Default: 0.02.
        lr_adamw:         Learning rate for AdamW (non-2-D params).  Default: 3e-4.
        momentum:         Nesterov momentum for Muon.  Default: 0.95.
        betas:            AdamW beta coefficients.  Default: (0.9, 0.999).
        eps:              AdamW epsilon.  Default: 1e-8.
        weight_decay_muon:  Decoupled weight decay for Muon params.  Default: 0.0.
        weight_decay_adamw: Weight decay for AdamW params.  Default: 0.01.
        ns_steps:         Newton-Schulz iterations for Muon.  Default: 5.
    """

    def __init__(
        self,
        params,
        lr_muon: float = 0.02,
        lr_adamw: float = 3e-4,
        momentum: float = 0.95,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay_muon: float = 0.0,
        weight_decay_adamw: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr_muon=lr_muon,
            lr_adamw=lr_adamw,
            momentum=momentum,
            betas=betas,
            eps=eps,
            weight_decay_muon=weight_decay_muon,
            weight_decay_adamw=weight_decay_adamw,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @staticmethod
    @torch.no_grad()
    def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """Newton-Schulz orthogonalization (same as Muon._newton_schulz)."""
        assert G.dim() == 2

        transposed = False
        if G.shape[0] < G.shape[1]:
            G = G.T
            transposed = True

        norm = G.norm() + eps
        X = G / norm

        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))

        if transposed:
            X = X.T

        return X

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_muon = group["lr_muon"]
            lr_adamw = group["lr_adamw"]
            mu = group["momentum"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd_muon = group["weight_decay_muon"]
            wd_adamw = group["weight_decay_adamw"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if p.dim() == 2:
                    # ---- Muon path ----
                    if wd_muon != 0.0:
                        p.mul_(1.0 - lr_muon * wd_muon)

                    state = self.state[p]
                    if len(state) == 0:
                        state["type"] = "muon"
                        state["momentum_buffer"] = torch.zeros_like(grad)

                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(grad)
                    nesterov_grad = grad + mu * buf

                    orth_grad = self._newton_schulz(nesterov_grad, steps=ns_steps)
                    scale = grad.norm() / (orth_grad.norm() + 1e-8)
                    orth_grad.mul_(scale)

                    p.add_(orth_grad, alpha=-lr_muon)

                else:
                    # ---- AdamW path ----
                    state = self.state[p]
                    if len(state) == 0:
                        state["type"] = "adamw"
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    t = state["step"]

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Decoupled weight decay
                    if wd_adamw != 0.0:
                        p.mul_(1.0 - lr_adamw * wd_adamw)

                    # Update biased first and second moment estimates
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # Bias correction
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t

                    corrected_avg = exp_avg / bc1
                    corrected_avg_sq = exp_avg_sq / bc2

                    # Update
                    denom = corrected_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(corrected_avg, denom, value=-lr_adamw)

        return loss
