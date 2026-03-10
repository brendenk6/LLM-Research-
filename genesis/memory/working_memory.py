"""
WorkingMemory (Level 1): Per-sequence typed memory slots.

Provides a fixed bank of memory slots with learned type embeddings (entity,
relation, quantity, temporal, spatial).  Hidden states read from memory via
cross-attention and write via a gated MLP.  Stale slots are decayed based on
learned relevance scores.

Inherits from StatefulModule so that slot contents persist across training
steps with gradient flow through StateLink.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olympus.core.stateful_module import StatefulModule
from genesis.memory.memory_cross_attention import MemoryCrossAttention


class WorkingMemory(StatefulModule):
    """Level-1 working memory with typed slots and gated write."""

    # Canonical type names in slot order
    TYPE_NAMES = ("entity", "relation", "quantity", "temporal", "spatial")

    def __init__(
        self,
        num_slots: int = 256,
        d_model: int = 768,
        d_type: int = 32,
        num_types: int = 5,
        num_heads: int = 8,
    ) -> None:
        """Initialise WorkingMemory.

        Args:
            num_slots: Number of memory slots.
            d_model: Hidden dimension of each slot (and the model).
            d_type: Dimension of per-slot type embedding.
            num_types: Number of distinct type categories.
            num_heads: Number of heads for the cross-attention reader.
        """
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.d_type = d_type
        self.num_types = num_types

        # ----- Persistent state (survives across forward calls) ----------
        self.register_state("memory_slots", torch.zeros(num_slots, d_model))
        self.register_state("type_vectors", torch.zeros(num_slots, d_type))

        # ----- Type embeddings -------------------------------------------
        self.type_embeddings = nn.Embedding(num_types, d_type)

        # Learned assignment of each slot to a type (logits)
        self.slot_type_logits = nn.Parameter(torch.randn(num_slots, num_types) * 0.02)

        # ----- Read pathway (cross-attention) ----------------------------
        self.read_attn = MemoryCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_memory=d_model,
        )
        self.read_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # ----- Write pathway (gated MLP) --------------------------------
        self.write_gate_proj = nn.Linear(d_model, d_model)
        self.write_content_proj = nn.Linear(d_model, d_model)

        # Project hidden to a slot-selection score
        self.slot_selector = nn.Linear(d_model, num_slots)

        # ----- Erase / relevance pathway --------------------------------
        self.relevance_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # ----- Output projection ----------------------------------------
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Read from memory via cross-attention.

        Args:
            hidden_states: (B, S, d_model) current hidden states.

        Returns:
            (B, S, d_model) memory-augmented representation.
        """
        slots = self.link_state("memory_slots")  # (num_slots, d_model)

        # Cross-attention: Q=hidden, K=V=slots
        read_out = self.read_attn(
            query=hidden_states,
            memory_keys=slots,
            memory_values=slots,
        )
        return read_out

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, hidden_states: torch.Tensor) -> None:
        """Write to memory slots using a gated MLP.

        For each token, compute a write gate and content vector, then
        update the selected slot with a soft gated blend.

        Args:
            hidden_states: (B, S, d_model) current hidden states.
        """
        slots = self.link_state("memory_slots")  # (num_slots, d_model)

        # Average over batch and sequence to get a summary write signal
        # (in practice, one could do per-token writes; we aggregate for
        # efficiency in the per-step setting)
        h_mean = hidden_states.mean(dim=(0, 1))  # (d_model,)

        write_gate = torch.sigmoid(self.write_gate_proj(h_mean))      # (d_model,)
        write_content = self.write_content_proj(h_mean)                # (d_model,)

        # Soft slot selection (which slots to update)
        slot_weights = torch.softmax(self.slot_selector(h_mean), dim=0)  # (num_slots,)

        # Gated update per slot
        new_content = write_gate * write_content  # (d_model,)
        # Broadcast: slot_weights (num_slots,1) * new_content (d_model,) -> (num_slots, d_model)
        write_delta = slot_weights.unsqueeze(-1) * new_content.unsqueeze(0)

        updated_slots = (1 - slot_weights.unsqueeze(-1)) * slots + write_delta
        self.set_state("memory_slots", updated_slots)

        # Update type vectors
        type_probs = torch.softmax(self.slot_type_logits, dim=-1)  # (num_slots, num_types)
        new_type_vectors = type_probs @ self.type_embeddings.weight  # (num_slots, d_type)
        self.set_state("type_vectors", new_type_vectors)

    # ------------------------------------------------------------------
    # Erase
    # ------------------------------------------------------------------

    def erase(self, relevance_threshold: float = 0.1) -> None:
        """Decay old slots whose learned relevance falls below threshold.

        Args:
            relevance_threshold: Slots with relevance below this are decayed.
        """
        slots = self.link_state("memory_slots")  # (num_slots, d_model)

        # Compute per-slot relevance score
        relevance = self.relevance_proj(slots).squeeze(-1)  # (num_slots,)

        # Decay factor: 1.0 for relevant slots, small for irrelevant
        decay = torch.where(
            relevance > relevance_threshold,
            torch.ones_like(relevance),
            relevance / relevance_threshold,  # smooth decay toward 0
        )
        decayed_slots = slots * decay.unsqueeze(-1)
        self.set_state("memory_slots", decayed_slots)

    # ------------------------------------------------------------------
    # Utilization loss
    # ------------------------------------------------------------------

    def utilization_loss(self) -> torch.Tensor:
        """Penalise unused slots to encourage full utilization.

        Returns:
            Scalar loss: negative entropy of slot norms (higher entropy =
            more uniform usage = lower loss).
        """
        slots = self.get_state("memory_slots")  # (num_slots, d_model)
        slot_norms = slots.norm(dim=-1)  # (num_slots,)

        # Normalize to a distribution
        probs = F.softmax(slot_norms, dim=0)

        # Negative entropy (minimize to maximize entropy / uniformity)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum()

        # We want to *maximize* entropy, so the loss is its negation
        max_entropy = torch.log(torch.tensor(float(self.num_slots), device=slots.device))
        return 1.0 - (entropy / max_entropy)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[dict] = None,
    ) -> torch.Tensor:
        """Full working-memory cycle: read, write, erase, return augmented hidden.

        Args:
            hidden_states: (B, S, d_model) input hidden states.
            context: Optional context dict (unused but kept for API compat).

        Returns:
            (B, S, d_model) memory-augmented hidden states.
        """
        # Read from memory
        read_out = self.read(hidden_states)

        # Gated fusion of read output with original hidden states
        combined = torch.cat([hidden_states, read_out], dim=-1)
        gate = self.read_gate(combined)
        augmented = hidden_states + gate * read_out

        # Write new information into memory
        self.write(hidden_states)

        # Erase stale entries
        self.erase()

        # Project and normalize
        output = self.layer_norm(augmented + self.out_proj(augmented))
        return output
