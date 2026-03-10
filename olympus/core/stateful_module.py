"""
StatefulModule: Base class for modules that maintain persistent state across
training steps with gradient flow through that state via StateLink.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple


class StateLink(torch.autograd.Function):
    """
    Straight-through autograd Function that allows gradients to flow through
    persistent state tensors. On the forward pass it returns a clone of the
    state; on the backward pass it writes the gradient into a buffer on the
    module so the state can be updated.
    """

    @staticmethod
    def forward(ctx, state_tensor: torch.Tensor, module_ref: Any, state_name: str) -> torch.Tensor:
        """
        Forward pass: return a clone of the state tensor that participates in
        the compute graph.

        Args:
            state_tensor: The persistent state tensor.
            module_ref: Reference to the owning StatefulModule (used for grad routing).
            state_name: Name of the state buffer being linked.

        Returns:
            A clone of the state tensor with grad enabled.
        """
        ctx.module_ref = module_ref
        ctx.state_name = state_name
        # Clone so downstream ops don't mutate the original buffer
        output = state_tensor.clone()
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        """
        Backward pass: store the gradient in the module's _state_grads dict
        so it can be consumed by an optimizer or manual update rule.
        """
        module_ref = ctx.module_ref
        state_name = ctx.state_name

        if hasattr(module_ref, "_state_grads"):
            if state_name in module_ref._state_grads and module_ref._state_grads[state_name] is not None:
                module_ref._state_grads[state_name] = module_ref._state_grads[state_name] + grad_output
            else:
                module_ref._state_grads[state_name] = grad_output.clone()

        # Return gradient for state_tensor, None for module_ref and state_name
        return grad_output, None, None


class StatefulModule(nn.Module):
    """
    Base class for neural-network modules that carry persistent state and
    shared memory across training steps.

    State tensors are registered as non-parameter buffers but can still receive
    gradients through StateLink.  Memory tensors are key-value pairs that can
    be shared across modules via the MemoryBus.

    Usage::

        class MyModule(StatefulModule):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)
                self.register_state("running_hidden", torch.zeros(dim))
                self.register_memory("context_vector", torch.zeros(dim))

            def forward(self, x, ctx):
                h = self.link_state("running_hidden")
                out = self.linear(x + h)
                self.set_state("running_hidden", out.detach())
                return out
    """

    def __init__(self) -> None:
        super().__init__()
        # Persistent state tensors (participate in grad flow via StateLink)
        self._state_tensors: Dict[str, torch.Tensor] = {}
        # Accumulated gradients for state tensors from backward passes
        self._state_grads: Dict[str, Optional[torch.Tensor]] = {}
        # Shared memory tensors (read/written via MemoryBus or direct access)
        self._memory_tensors: Dict[str, torch.Tensor] = {}
        # Metadata for state/memory (e.g., dtype, shape info for reconstruction)
        self._state_meta: Dict[str, Dict[str, Any]] = {}
        self._memory_meta: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # State registration and access
    # ------------------------------------------------------------------

    def register_state(
        self,
        name: str,
        tensor: torch.Tensor,
        persistent: bool = True,
        **meta: Any,
    ) -> None:
        """
        Register a persistent state tensor.

        Args:
            name: Unique name for this state.
            tensor: Initial value (will be cloned).
            persistent: Whether to include in state_dict.
            **meta: Arbitrary metadata stored alongside the state.
        """
        if name in self._state_tensors:
            raise ValueError(f"State '{name}' is already registered.")
        self._state_tensors[name] = tensor.clone().detach()
        self._state_grads[name] = None
        self._state_meta[name] = {"persistent": persistent, **meta}
        # Also register as a buffer so it moves with .to() / .cuda() etc.
        self.register_buffer(f"_state_buf_{name}", self._state_tensors[name], persistent=persistent)

    def link_state(self, name: str) -> torch.Tensor:
        """
        Retrieve a state tensor wrapped through StateLink so that gradients
        flow back into ``_state_grads[name]``.

        Args:
            name: The registered state name.

        Returns:
            A tensor that is part of the autograd graph.
        """
        if name not in self._state_tensors:
            raise KeyError(f"State '{name}' has not been registered.")
        # Sync from buffer in case .to() moved it
        buf = getattr(self, f"_state_buf_{name}")
        self._state_tensors[name] = buf
        return StateLink.apply(buf, self, name)

    def get_state(self, name: str) -> torch.Tensor:
        """Return the raw state tensor (no autograd link)."""
        if name not in self._state_tensors:
            raise KeyError(f"State '{name}' has not been registered.")
        buf = getattr(self, f"_state_buf_{name}")
        self._state_tensors[name] = buf
        return buf

    def set_state(self, name: str, value: torch.Tensor) -> None:
        """
        Update a state tensor in-place.

        Args:
            name: The registered state name.
            value: New value (will be detached and cloned).
        """
        if name not in self._state_tensors:
            raise KeyError(f"State '{name}' has not been registered.")
        new_val = value.detach().clone()
        self._state_tensors[name] = new_val
        # Keep the registered buffer in sync
        buf_name = f"_state_buf_{name}"
        delattr(self, buf_name)
        self.register_buffer(buf_name, new_val, persistent=self._state_meta[name].get("persistent", True))

    def get_state_grad(self, name: str) -> Optional[torch.Tensor]:
        """Return the accumulated gradient for a state tensor, or None."""
        return self._state_grads.get(name)

    def zero_state_grads(self) -> None:
        """Clear all accumulated state gradients."""
        for name in self._state_grads:
            self._state_grads[name] = None

    def apply_state_grads(self, lr: float = 1e-3) -> None:
        """
        Simple SGD-style update of state tensors using accumulated gradients.
        For more sophisticated updates, use an external optimizer.

        Args:
            lr: Learning rate for the state update.
        """
        for name, grad in self._state_grads.items():
            if grad is not None:
                current = self.get_state(name)
                self.set_state(name, current - lr * grad)
        self.zero_state_grads()

    # ------------------------------------------------------------------
    # Memory registration and access
    # ------------------------------------------------------------------

    def register_memory(
        self,
        name: str,
        tensor: torch.Tensor,
        **meta: Any,
    ) -> None:
        """
        Register a shared memory tensor.

        Args:
            name: Unique name for this memory slot.
            tensor: Initial value (will be cloned).
            **meta: Arbitrary metadata.
        """
        if name in self._memory_tensors:
            raise ValueError(f"Memory '{name}' is already registered.")
        self._memory_tensors[name] = tensor.clone().detach()
        self._memory_meta[name] = dict(meta)
        self.register_buffer(f"_mem_buf_{name}", self._memory_tensors[name], persistent=True)

    def get_memory(self, name: str) -> torch.Tensor:
        """Return the memory tensor."""
        if name not in self._memory_tensors:
            raise KeyError(f"Memory '{name}' has not been registered.")
        buf = getattr(self, f"_mem_buf_{name}")
        self._memory_tensors[name] = buf
        return buf

    def update_memory(self, name: str, value: torch.Tensor) -> None:
        """
        Update a memory tensor in-place.

        Args:
            name: The registered memory name.
            value: New value (will be detached and cloned).
        """
        if name not in self._memory_tensors:
            raise KeyError(f"Memory '{name}' has not been registered.")
        new_val = value.detach().clone()
        self._memory_tensors[name] = new_val
        buf_name = f"_mem_buf_{name}"
        delattr(self, buf_name)
        self.register_buffer(buf_name, new_val, persistent=True)

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def state_dict_with_state(self, **kwargs) -> Dict[str, Any]:
        """
        Return a state dict that includes both parameters/buffers AND the
        explicit state/memory tensors with their metadata.
        """
        base = super().state_dict(**kwargs)
        extra = {
            "_olympus_state_tensors": {k: v.cpu() for k, v in self._state_tensors.items()},
            "_olympus_state_meta": self._state_meta,
            "_olympus_memory_tensors": {k: v.cpu() for k, v in self._memory_tensors.items()},
            "_olympus_memory_meta": self._memory_meta,
        }
        return {**base, **extra}

    def load_state_dict_with_state(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Load a state dict produced by ``state_dict_with_state``.

        Restores state tensors, memory tensors, and their metadata before
        loading the standard nn.Module state dict.
        """
        # Extract Olympus-specific keys
        state_tensors = state_dict.pop("_olympus_state_tensors", {})
        state_meta = state_dict.pop("_olympus_state_meta", {})
        memory_tensors = state_dict.pop("_olympus_memory_tensors", {})
        memory_meta = state_dict.pop("_olympus_memory_meta", {})

        # Restore state tensors
        for name, tensor in state_tensors.items():
            if name in self._state_tensors:
                self.set_state(name, tensor)
            else:
                self.register_state(name, tensor, **state_meta.get(name, {}))
        self._state_meta.update(state_meta)

        # Restore memory tensors
        for name, tensor in memory_tensors.items():
            if name in self._memory_tensors:
                self.update_memory(name, tensor)
            else:
                self.register_memory(name, tensor, **memory_meta.get(name, {}))
        self._memory_meta.update(memory_meta)

        # Load the standard state dict (parameters + buffers)
        super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> "StatefulModule":
        """
        Override to() so that state and memory tensors are moved along with
        the module's parameters and buffers.
        """
        result = super().to(*args, **kwargs)
        # Re-sync internal dicts from buffers (which super().to() already moved)
        for name in list(self._state_tensors.keys()):
            buf_name = f"_state_buf_{name}"
            if hasattr(self, buf_name):
                self._state_tensors[name] = getattr(self, buf_name)
        for name in list(self._memory_tensors.keys()):
            buf_name = f"_mem_buf_{name}"
            if hasattr(self, buf_name):
                self._memory_tensors[name] = getattr(self, buf_name)
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def state_names(self):
        """Return list of registered state names."""
        return list(self._state_tensors.keys())

    def memory_names(self):
        """Return list of registered memory names."""
        return list(self._memory_tensors.keys())

    def __repr__(self) -> str:
        base = super().__repr__()
        states = ", ".join(f"{k}: {tuple(v.shape)}" for k, v in self._state_tensors.items())
        memories = ", ".join(f"{k}: {tuple(v.shape)}" for k, v in self._memory_tensors.items())
        extra = []
        if states:
            extra.append(f"  (states): {{{states}}}")
        if memories:
            extra.append(f"  (memories): {{{memories}}}")
        if extra:
            return base[:-1] + "\n" + "\n".join(extra) + "\n)"
        return base
