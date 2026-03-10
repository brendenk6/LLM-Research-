"""
EpisodicMemory (Level 2): Cross-sequence persistent memory.

Stores important information that persists ACROSS sequences within a
batch/session.  Write decisions are driven by surprise-based importance
scoring -- tokens with unusually high loss are considered novel/important
and trigger memory writes.

Reads are performed via cross-attention from hidden states to episodic
memory slots.  The module communicates across sequences using the MemoryBus
with ``"episode"`` scope.

Inherits from StatefulModule so slot contents persist across training steps
with gradient flow through StateLink.
"""

from typing import Optional

import torch
import torch.nn as nn

from olympus.core.stateful_module import StatefulModule
from olympus.core.memory_bus import MemoryBus
from genesis.memory.memory_cross_attention import MemoryCrossAttention


class EpisodicMemory(StatefulModule):
    """Level-2 episodic memory with surprise-based writes and cross-attention reads."""

    def __init__(
        self,
        num_slots: int = 1024,
        d_model: int = 768,
        num_heads: int = 8,
        importance_threshold: float = 1.5,
        memory_bus: Optional[MemoryBus] = None,
    ) -> None:
        """Initialise EpisodicMemory.

        Args:
            num_slots: Number of episodic memory slots.
            d_model: Hidden dimension of each slot (and the model).
            num_heads: Number of heads for the cross-attention reader.
            importance_threshold: Loss multiplier threshold above which a
                token is considered surprising enough to write.
            memory_bus: Optional MemoryBus instance for cross-sequence
                communication.  If ``None``, a local bus is created.
        """
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.importance_threshold = importance_threshold
        self.memory_bus = memory_bus if memory_bus is not None else MemoryBus()

        # ----- Persistent state (survives across sequences) ---------------
        self.register_state("episodic_slots", torch.zeros(num_slots, d_model))
        self.register_state("slot_ages", torch.zeros(num_slots))
        self.register_state("write_pointer", torch.zeros(1, dtype=torch.long))

        # Running statistics for surprise detection
        self.register_state("running_loss_mean", torch.ones(1))
        self.register_state("running_loss_var", torch.ones(1))

        # ----- Read pathway (cross-attention) -----------------------------
        self.read_attn = MemoryCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_memory=d_model,
        )
        self.read_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # ----- Write pathway ---------------------------------------------
        # Importance scorer: takes hidden + loss signal -> importance weight
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Content compressor for writing to memory
        self.write_proj = nn.Linear(d_model, d_model)

        # ----- Output projection ------------------------------------------
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Surprise detection
    # ------------------------------------------------------------------

    def _compute_surprise(self, loss_per_token: torch.Tensor) -> torch.Tensor:
        """Compute surprise scores from per-token loss.

        Surprise is measured as how many standard deviations the loss is
        above the running mean.

        Args:
            loss_per_token: (...) per-token loss values.

        Returns:
            (...) surprise scores (z-scores, clamped to >= 0).
        """
        mean = self.get_state("running_loss_mean").item()
        var = self.get_state("running_loss_var").item()
        std = max(var ** 0.5, 1e-6)

        surprise = (loss_per_token - mean) / std
        return surprise.clamp(min=0.0)

    def _update_loss_statistics(self, loss_per_token: torch.Tensor) -> None:
        """Update running mean and variance of per-token loss.

        Uses exponential moving average with momentum 0.01.

        Args:
            loss_per_token: (...) per-token loss values.
        """
        momentum = 0.01
        batch_mean = loss_per_token.mean().detach()
        batch_var = loss_per_token.var().detach().clamp(min=1e-6)

        old_mean = self.get_state("running_loss_mean")
        old_var = self.get_state("running_loss_var")

        new_mean = (1 - momentum) * old_mean + momentum * batch_mean
        new_var = (1 - momentum) * old_var + momentum * batch_var

        self.set_state("running_loss_mean", new_mean)
        self.set_state("running_loss_var", new_var)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from episodic memory via cross-attention.

        Args:
            query: (B, S, d_model) current hidden states.

        Returns:
            (B, S, d_model) memory-attended representation.
        """
        slots = self.link_state("episodic_slots")  # (num_slots, d_model)

        # Also check memory bus for cross-sequence episodic data
        bus_data = self.memory_bus.read("episodic_context", scope="episode")
        if bus_data is not None and bus_data.dim() == 2 and bus_data.size(-1) == self.d_model:
            # Concatenate bus data with local slots
            combined_keys = torch.cat([slots, bus_data.to(slots.device)], dim=0)
            combined_values = combined_keys
        else:
            combined_keys = slots
            combined_values = slots

        read_out = self.read_attn(
            query=query,
            memory_keys=combined_keys,
            memory_values=combined_values,
        )
        return read_out

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_if_important(
        self,
        hidden_states: torch.Tensor,
        loss_per_token: torch.Tensor,
    ) -> None:
        """Write to episodic memory if tokens are sufficiently surprising.

        Tokens whose loss exceeds the importance threshold (in z-score
        terms) are compressed and written to the next available slot.

        Args:
            hidden_states: (B, S, d_model) current hidden states.
            loss_per_token: (B, S) per-token loss values.
        """
        # Update running statistics
        self._update_loss_statistics(loss_per_token)

        # Compute surprise scores
        surprise = self._compute_surprise(loss_per_token)  # (B, S)

        # Identify surprising tokens
        mask = surprise > self.importance_threshold  # (B, S)

        if not mask.any():
            return

        # Gather surprising token hidden states
        # Use mask to select and aggregate
        # Expand loss as a feature for importance scoring
        loss_feature = loss_per_token.unsqueeze(-1)  # (B, S, 1)
        scorer_input = torch.cat([hidden_states, loss_feature], dim=-1)  # (B, S, d_model+1)
        importance_weights = self.importance_scorer(scorer_input).squeeze(-1)  # (B, S)

        # Zero out non-surprising tokens
        importance_weights = importance_weights * mask.float()

        # Weighted average of surprising tokens per batch element
        weight_sum = importance_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        weighted_hidden = (importance_weights.unsqueeze(-1) * hidden_states).sum(dim=1)  # (B, d_model)
        weighted_hidden = weighted_hidden / weight_sum  # (B, d_model)

        # Project to write content
        write_content = self.write_proj(weighted_hidden)  # (B, d_model)

        # Write each batch element's summary to the next slot
        slots = self.get_state("episodic_slots")  # (num_slots, d_model)
        pointer = self.get_state("write_pointer").item()
        ages = self.get_state("slot_ages")

        B = write_content.size(0)
        for b in range(B):
            slot_idx = int(pointer + b) % self.num_slots
            slots[slot_idx] = write_content[b].detach()
            ages[slot_idx] = 0.0

        new_pointer = torch.tensor(
            [(int(pointer) + B) % self.num_slots],
            dtype=torch.long,
            device=slots.device,
        )
        self.set_state("episodic_slots", slots)
        self.set_state("write_pointer", new_pointer)

        # Age all other slots
        ages = ages + 1.0
        self.set_state("slot_ages", ages)

        # Publish to memory bus for cross-sequence access
        self.memory_bus.write(
            "episodic_context",
            write_content.detach().mean(dim=0, keepdim=True),
            scope="episode",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[dict] = None,
    ) -> torch.Tensor:
        """Full episodic-memory cycle: read, optionally write, return augmented hidden.

        Args:
            hidden_states: (B, S, d_model) input hidden states.
            context: Optional dict that may contain ``"loss_per_token"``
                (B, S) for surprise-based writes.

        Returns:
            (B, S, d_model) memory-augmented hidden states.
        """
        # Read from episodic memory
        read_out = self.read(hidden_states)

        # Gated fusion
        combined = torch.cat([hidden_states, read_out], dim=-1)
        gate = self.read_gate(combined)
        augmented = hidden_states + gate * read_out

        # Conditionally write if loss info is available
        if context is not None and "loss_per_token" in context:
            self.write_if_important(hidden_states, context["loss_per_token"])

        # Project and normalize
        output = self.layer_norm(augmented + self.out_proj(augmented))
        return output
