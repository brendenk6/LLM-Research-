"""
MemoryController: Orchestrator for the 3-level PHMA memory hierarchy.

Coordinates WorkingMemory (L1), EpisodicMemory (L2), and SemanticMemory (L3)
through a unified read/write/erase cycle.  Outputs from the three levels are
combined via learned gating weights that allow the model to dynamically
adjust how much it relies on each memory tier.

Auxiliary losses (utilization, consistency, retrieval) are collected from the
sub-modules and returned as a dict by ``compute_aux_losses()`` for the
training loop to weight and add to the main language-modelling loss.

Inherits from StatefulModule so that controller-level state (e.g. gating
statistics) can persist and receive gradients through StateLink.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.core.stateful_module import StatefulModule
from olympus.core.memory_bus import MemoryBus
from genesis.memory.working_memory import WorkingMemory
from genesis.memory.episodic_memory import EpisodicMemory
from genesis.memory.semantic_memory import SemanticMemory
from genesis.memory.memory_consistency import MemoryConsistency


class MemoryController(StatefulModule):
    """Orchestrates the 3-level PHMA memory system with learned gating."""

    def __init__(
        self,
        d_model: int = 768,
        num_working_slots: int = 256,
        num_episodic_slots: int = 1024,
        num_semantic_entries: int = 10000,
        d_key: int = 256,
        num_heads: int = 8,
        importance_threshold: float = 1.5,
        consistency_threshold: float = 0.8,
        memory_bus: Optional[MemoryBus] = None,
    ) -> None:
        """Initialise MemoryController.

        Args:
            d_model: Hidden dimension of the model.
            num_working_slots: Number of slots for Level-1 working memory.
            num_episodic_slots: Number of slots for Level-2 episodic memory.
            num_semantic_entries: Maximum entries for Level-3 semantic memory.
            d_key: Key dimension for semantic memory retrieval.
            num_heads: Number of attention heads for memory reads.
            importance_threshold: Surprise z-score threshold for episodic writes.
            consistency_threshold: Cosine similarity threshold for contradiction
                detection.
            memory_bus: Optional shared MemoryBus instance.  If ``None``, a
                local bus is created.
        """
        super().__init__()
        self.d_model = d_model
        self.memory_bus = memory_bus if memory_bus is not None else MemoryBus()

        # ----- Level 1: Working Memory ------------------------------------
        self.working_memory = WorkingMemory(
            num_slots=num_working_slots,
            d_model=d_model,
            num_heads=num_heads,
        )

        # ----- Level 2: Episodic Memory -----------------------------------
        self.episodic_memory = EpisodicMemory(
            num_slots=num_episodic_slots,
            d_model=d_model,
            num_heads=num_heads,
            importance_threshold=importance_threshold,
            memory_bus=self.memory_bus,
        )

        # ----- Level 3: Semantic Memory -----------------------------------
        self.semantic_memory = SemanticMemory(
            num_entries=num_semantic_entries,
            d_key=d_key,
            d_value=d_model,
            d_model=d_model,
        )

        # ----- Consistency checker ----------------------------------------
        self.consistency = MemoryConsistency(
            d_model=d_model,
            threshold=consistency_threshold,
        )

        # ----- Learned gating over the 3 levels ---------------------------
        # Produces per-position 3-way softmax mixing weights from the
        # concatenation of the three level outputs.
        self.gate_proj = nn.Linear(d_model * 3, 3)

        # ----- Semantic write projections ---------------------------------
        self.semantic_key_proj = nn.Linear(d_model, d_key)
        self.semantic_value_proj = nn.Linear(d_model, d_model)

        # ----- Output projection ------------------------------------------
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # ----- Controller-level persistent state --------------------------
        # Exponential moving average of gating weights for monitoring
        self.register_state("gate_ema", torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))

        # Cache for consistency loss computation (set during forward)
        self._last_write_hidden: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run all three memory levels and combine with learned gating.

        Args:
            hidden_states: (B, S, d_model) input hidden states.
            context: Optional dict that may contain ``"loss_per_token"``
                (B, S) for surprise-based episodic writes.

        Returns:
            (B, S, d_model) memory-augmented hidden states.
        """
        # --- Level 1: Working memory (per-sequence typed slots) -----------
        working_out = self.working_memory(hidden_states, context=context)

        # --- Level 2: Episodic memory (cross-sequence, surprise-gated) ----
        episodic_out = self.episodic_memory(hidden_states, context=context)

        # --- Level 3: Semantic memory (persistent KV store) ---------------
        semantic_out = self.semantic_memory(hidden_states)

        # --- Learned gating -----------------------------------------------
        gate_input = torch.cat(
            [working_out, episodic_out, semantic_out], dim=-1,
        )  # (B, S, d_model * 3)
        gate_logits = self.gate_proj(gate_input)         # (B, S, 3)
        gate_weights = F.softmax(gate_logits, dim=-1)    # (B, S, 3)

        # Weighted combination of the three level outputs
        stacked = torch.stack(
            [working_out, episodic_out, semantic_out], dim=-1,
        )  # (B, S, d_model, 3)
        mixed = (stacked * gate_weights.unsqueeze(-2)).sum(dim=-1)  # (B, S, d_model)

        # Update gate EMA for monitoring (detached, no grad)
        with torch.no_grad():
            batch_gate_mean = gate_weights.mean(dim=(0, 1))  # (3,)
            old_ema = self.get_state("gate_ema")
            new_ema = 0.99 * old_ema + 0.01 * batch_gate_mean
            self.set_state("gate_ema", new_ema)

        # --- Semantic writes (training only) ------------------------------
        if self.training:
            h_summary = hidden_states.mean(dim=1)  # (B, d_model)
            sem_keys = self.semantic_key_proj(h_summary)      # (B, d_key)
            sem_values = self.semantic_value_proj(h_summary)   # (B, d_model)

            # Check for contradictions before writing
            n_written = self.semantic_memory.num_written.item()
            if n_written > 0:
                existing = self.semantic_memory.values[:n_written]
                for b in range(h_summary.size(0)):
                    has_contradiction, _ = self.consistency.check_consistency(
                        sem_values[b], existing,
                    )
                    if not has_contradiction:
                        self.semantic_memory.write(sem_keys[b], sem_values[b])
            else:
                self.semantic_memory.write(sem_keys, sem_values)

            # Cache write vectors for consistency loss
            self._last_write_hidden = sem_values.detach()

        # --- Residual + projection + layer norm ---------------------------
        output = self.layer_norm(hidden_states + self.out_proj(mixed))
        return output

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def compute_aux_losses(self) -> Dict[str, torch.Tensor]:
        """Collect auxiliary losses from all memory sub-modules.

        Returns:
            Dict with keys:
                ``"utilization"``: Working memory slot utilization loss.
                ``"consistency"``: Contradiction detection loss for recent
                    writes against episodic memory.
                ``"retrieval"``: Semantic memory retrieval accuracy loss
                    (zero when no supervised signal is available).
        """
        device = next(self.parameters()).device

        # 1. Working memory utilization loss
        utilization = self.working_memory.utilization_loss()

        # 2. Consistency loss: recent writes vs episodic memory
        if self._last_write_hidden is not None:
            episodic_slots = self.episodic_memory.get_state("episodic_slots")
            consistency = self.consistency(
                self._last_write_hidden, episodic_slots,
            )
        else:
            consistency = torch.tensor(0.0, device=device, requires_grad=True)

        # 3. Semantic retrieval loss (requires external targets; placeholder)
        retrieval = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            "utilization": utilization,
            "consistency": consistency,
            "retrieval": retrieval,
        }

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        """Return state dict including controller and sub-module state."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict restoring controller and sub-module state."""
        super().load_state_dict(state_dict, strict=strict)
