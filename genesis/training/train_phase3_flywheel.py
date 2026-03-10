"""
Phase 3: Data Flywheel with Self-Generated Reasoning Traces.

Creates a positive feedback loop: the model generates reasoning traces,
scores them with a reward function, stores successful ones in a
FlywheelBuffer, and trains on the best traces mixed with standard data.
As the model improves, it generates better traces, further improving
training quality.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from genesis.model.hlrt import HLRT
from olympus.core.training_orchestrator import (
    TrainingOrchestrator,
    ModelConfig,
    ObjectiveConfig,
)
from olympus.core.training_context import TrainingContext
from olympus.data.flywheel_buffer import FlywheelBuffer
from olympus.data.tokenizer import TokenizerWrapper
from olympus.optim.muon_adamw_hybrid import MuonAdamWHybrid

logger = logging.getLogger(__name__)


# ======================================================================
# NTP loss for flywheel mixed data
# ======================================================================

def flywheel_ntp_objective(
    models: Dict[str, nn.Module],
    ctx: TrainingContext,
    batch: Any,
) -> torch.Tensor:
    """NTP loss for flywheel training (same as bootstrap but phase-tagged)."""
    model = models["hlrt"]
    input_ids = batch["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    out = model(input_ids)
    logits = out["logits"]
    aux_loss = out.get("aux_loss", 0.0)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
    )

    if isinstance(aux_loss, torch.Tensor):
        loss = loss + aux_loss
    elif isinstance(aux_loss, (int, float)) and aux_loss > 0:
        loss = loss + aux_loss

    return loss


# ======================================================================
# FlywheelTrainer
# ======================================================================

class FlywheelTrainer:
    """Phase 3: Flywheel training with self-generated reasoning traces.

    The flywheel loop:
    1. Train on mixed batches (standard data + successful traces).
    2. Periodically generate new reasoning traces from prompts.
    3. Score traces with the reward function and store good ones.
    4. Repeat -- the model improves, generating better traces, which
       further improve training.

    Args:
        model: The HLRT model to train.
        flywheel_buffer: Buffer for storing successful reasoning traces.
        reward_fn: Callable ``(model, context_ids, trace_ids) -> (B,)``
            reward tensor.
        optimizer: MuonAdamWHybrid optimizer.
        config: Training configuration dict with keys:
            - reward_threshold (float): Min reward to store a trace.
            - trace_mix_ratio (float): Fraction of batch from buffer.
            - generation_interval (int): Steps between trace generation.
            - max_trace_tokens (int): Max tokens per generated trace.
            - temperature (float): Sampling temperature for generation.
            - gradient_accumulation_steps (int)
            - max_grad_norm (float)
            - log_interval (int)
            - save_interval (int)
            - checkpoint_dir (str)
            - device (str)
    """

    def __init__(
        self,
        model: HLRT,
        flywheel_buffer: FlywheelBuffer,
        reward_fn: Callable[..., torch.Tensor],
        optimizer: MuonAdamWHybrid,
        config: dict,
    ) -> None:
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.model = model.to(self.device)
        self.flywheel_buffer = flywheel_buffer
        self.reward_fn = reward_fn
        self.optimizer = optimizer

        self.reward_threshold: float = config.get("reward_threshold", 0.1)
        self.trace_mix_ratio: float = config.get("trace_mix_ratio", 0.3)
        self.generation_interval: int = config.get("generation_interval", 100)
        self.max_trace_tokens: int = config.get("max_trace_tokens", 64)
        self.temperature: float = config.get("temperature", 0.8)

        # Tokenizer for encoding / decoding traces
        self.tokenizer = TokenizerWrapper(
            backend="tiktoken",
            vocab_size=config.get("vocab_size", 32000),
        )

        # Build orchestrator
        grad_accum = config.get("gradient_accumulation_steps", 4)
        self.orchestrator = TrainingOrchestrator(
            gradient_accumulation_steps=grad_accum,
            device=self.device,
        )
        self.orchestrator.add_model(ModelConfig(
            name="hlrt",
            model=self.model,
            optimizer=self.optimizer,
            max_grad_norm=config.get("max_grad_norm", 1.0),
            enabled_phases=["flywheel"],
        ))
        self.orchestrator.add_objective(ObjectiveConfig(
            name="ntp",
            compute_fn=flywheel_ntp_objective,
            weight=1.0,
            enabled_phases=["flywheel"],
        ))
        self.orchestrator.set_phase("flywheel")

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "FlywheelTrainer initialised: %.2fM params, buffer=%d traces",
            param_count / 1e6, self.flywheel_buffer.size,
        )

    # ------------------------------------------------------------------
    # Trace generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_traces(
        self,
        prompts: List[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """Generate reasoning traces and score them.

        Args:
            prompts: List of context token tensors, each of shape (S,).

        Returns:
            List of dicts with keys ``"context_ids"``, ``"trace_ids"``,
            ``"reward"``, ``"context_text"``, ``"trace_text"``.
        """
        self.model.eval()
        results: List[Dict[str, Any]] = []

        for context_ids in prompts:
            context_ids = context_ids.unsqueeze(0).to(self.device)  # (1, S)
            generated = context_ids

            # Autoregressive generation
            for _ in range(self.max_trace_tokens):
                out = self.model(generated)
                next_logits = out["logits"][:, -1, :]  # (1, V)
                scaled = next_logits / max(self.temperature, 1e-8)
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                generated = torch.cat([generated, next_token], dim=1)

            trace_ids = generated[:, context_ids.shape[1]:]  # (1, T)

            # Score the trace
            reward = self.reward_fn(self.model, context_ids, trace_ids)
            reward_val = reward.item() if isinstance(reward, torch.Tensor) else float(reward)

            context_text = self.tokenizer.decode(context_ids[0].tolist())
            trace_text = self.tokenizer.decode(trace_ids[0].tolist())

            results.append({
                "context_ids": context_ids[0].cpu(),
                "trace_ids": trace_ids[0].cpu(),
                "reward": reward_val,
                "context_text": context_text,
                "trace_text": trace_text,
            })

        self.model.train()
        return results

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single flywheel training step with mixed data.

        Replaces a portion of the batch with traces sampled from the
        FlywheelBuffer, then runs the NTP objective through the
        orchestrator.

        Args:
            batch: Dict with ``"input_ids"`` (B, S).

        Returns:
            Dict of loss values from the orchestrator.
        """
        mixed_batch = self._mix_with_buffer(batch)
        return self.orchestrator.step(mixed_batch)

    def flywheel_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Full flywheel iteration: generate, store, and train.

        1. Generate traces from the batch contexts.
        2. Store traces above the reward threshold in the buffer.
        3. Train on a mixed batch of standard data + buffer traces.

        Args:
            batch: Dict with ``"input_ids"`` (B, S).

        Returns:
            Dict with ``"loss"``, ``"traces_generated"``,
            ``"traces_stored"``, ``"avg_reward"``, ``"buffer_size"``.
        """
        input_ids = batch["input_ids"]

        # Generate traces from first half of each sequence as context
        context_len = min(input_ids.shape[1] // 2, 512)
        prompts = [ids[:context_len] for ids in input_ids]

        trace_results = self.generate_traces(prompts)

        # Store successful traces
        num_stored = 0
        total_reward = 0.0
        for result in trace_results:
            total_reward += result["reward"]
            stored = self.flywheel_buffer.add(
                context=result["context_text"],
                trace=result["trace_text"],
                reward=result["reward"],
            )
            if stored:
                num_stored += 1

        # Train on mixed batch
        loss_dict = self.train_step(batch)
        total_loss = loss_dict.get("total", loss_dict.get("ntp", 0.0))

        avg_reward = total_reward / max(len(trace_results), 1)

        return {
            "loss": total_loss,
            "traces_generated": len(trace_results),
            "traces_stored": num_stored,
            "avg_reward": avg_reward,
            "buffer_size": self.flywheel_buffer.size,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mix_with_buffer(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Replace a portion of the batch with traces from the buffer.

        Args:
            batch: Original batch dict with ``"input_ids"`` (B, S).

        Returns:
            Mixed batch dict.
        """
        input_ids = batch["input_ids"]  # (B, S)
        B, S = input_ids.shape

        if self.flywheel_buffer.size == 0:
            return batch

        num_replace = max(1, int(B * self.trace_mix_ratio))
        num_replace = min(num_replace, self.flywheel_buffer.size, B)

        traces = self.flywheel_buffer.sample(num_replace)
        mixed_ids = input_ids.clone()

        for i, trace_data in enumerate(traces):
            if i >= B:
                break
            combined = trace_data["context"] + " " + trace_data["trace"]
            token_ids = self.tokenizer.encode(combined)

            if len(token_ids) > S:
                token_ids = token_ids[:S]
            elif len(token_ids) < S:
                token_ids = token_ids + [self.tokenizer.pad_id] * (S - len(token_ids))

            mixed_ids[i] = torch.tensor(token_ids, dtype=torch.long)

        return {"input_ids": mixed_ids.to(self.device)}
