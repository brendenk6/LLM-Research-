"""
GRPO: Group Relative Policy Optimization for GENESIS.

Implements the GRPO algorithm from the DeepSeek-R1 paper. For each prompt,
G completions are generated, rewards are computed for each, normalized within
the group via z-score, and used as advantages for a clipped policy gradient
update with KL divergence penalty against a reference model.

Key insight: GRPO eliminates the value network entirely by using group-relative
normalization of rewards as the advantage estimate. This reduces memory overhead
and training instability associated with critic networks.

Reference: DeepSeek-R1 (2024) - https://arxiv.org/abs/2401.02954
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization.

    Attributes:
        group_size: Number of completions to generate per prompt (G in the paper).
        kl_coeff: Coefficient for the KL divergence penalty against the reference
            policy. Higher values keep the policy closer to the reference.
        clip_range: Epsilon for PPO-style clipped surrogate objective. Limits
            the magnitude of policy ratio updates.
        temperature: Sampling temperature for completion generation. Higher
            values produce more diverse completions within each group.
        max_completion_tokens: Maximum number of tokens per generated completion.
        min_group_std: Minimum standard deviation for group normalization to
            prevent division by near-zero values.
    """

    group_size: int = 8
    kl_coeff: float = 0.1
    clip_range: float = 0.2
    temperature: float = 1.0
    max_completion_tokens: int = 256
    min_group_std: float = 1e-8


class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    Trains a language model using RL without a value network. Instead of
    learning a value function for advantage estimation, GRPO generates a
    group of completions per prompt and normalizes rewards within each group.

    This integrates with the Olympus TrainingOrchestrator as a drop-in
    RL training component.

    Args:
        model: The policy model to train.
        ref_model: A frozen copy of the policy model used for KL penalty
            computation. Should not be updated during training.
        reward_fn: A callable that takes (prompt, completion) and returns
            a scalar reward float.
        config: GRPO hyperparameters.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[str, str], float],
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config or GRPOConfig()

        # Freeze the reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def generate_group(
        self,
        prompts: List[str],
        tokenizer: Any,
    ) -> List[List[str]]:
        """Generate a group of completions for each prompt.

        For each prompt, samples G independent completions from the current
        policy using temperature sampling.

        Args:
            prompts: List of input prompt strings.
            tokenizer: Tokenizer with encode/decode methods.

        Returns:
            List of lists, where each inner list contains G completions
            for the corresponding prompt.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        G = self.config.group_size
        all_completions: List[List[str]] = []

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            # Repeat input for the full group
            batch_input = input_ids.expand(G, -1)  # (G, seq_len)

            completions: List[str] = []
            # Generate each completion independently for diversity
            for i in range(G):
                generated_ids = batch_input[i : i + 1]  # (1, seq_len)
                for _ in range(self.config.max_completion_tokens):
                    outputs = self.model(generated_ids)
                    next_logits = outputs["logits"][:, -1, :]  # (1, V)
                    scaled = next_logits / max(self.config.temperature, 1e-8)
                    probs = F.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    # Check for EOS
                    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                        if next_token.item() == tokenizer.eos_token_id:
                            break

                # Decode only the generated portion
                completion_ids = generated_ids[0, input_ids.shape[1]:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completions.append(completion_text)

            all_completions.append(completions)

        return all_completions

    def compute_advantages(
        self,
        rewards_per_group: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize rewards within each group using z-score.

        For each prompt's group of rewards, subtracts the group mean and
        divides by the group standard deviation. This is the core GRPO trick:
        intra-group normalization provides a baseline without a learned value
        function.

        Args:
            rewards_per_group: (batch_size, G) tensor of rewards, where each
                row contains rewards for G completions of a single prompt.

        Returns:
            (batch_size, G) tensor of normalized advantages.
        """
        # Compute per-group statistics along the group dimension
        group_mean = rewards_per_group.mean(dim=1, keepdim=True)  # (B, 1)
        group_std = rewards_per_group.std(dim=1, keepdim=True)    # (B, 1)
        group_std = group_std.clamp(min=self.config.min_group_std)

        advantages = (rewards_per_group - group_mean) / group_std
        return advantages

    def compute_loss(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the GRPO policy loss with clipped surrogate and KL penalty.

        Uses the PPO-style clipped objective to prevent overly large updates,
        combined with a KL divergence penalty against the reference model to
        maintain generation quality.

        Args:
            logprobs: (B, G) log-probabilities of completions under the
                current policy.
            ref_logprobs: (B, G) log-probabilities of the same completions
                under the reference (frozen) policy.
            advantages: (B, G) normalized advantages from compute_advantages.

        Returns:
            Tuple of (scalar loss, dict of component metrics).
        """
        # Policy ratio: exp(log_pi - log_pi_old)
        # Here ref_logprobs serves as the old policy logprobs
        log_ratio = logprobs - ref_logprobs
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective (PPO-style)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence penalty: approximate KL(pi || pi_ref)
        # Using the Schulman (2020) approximation: (ratio - 1) - log(ratio)
        approx_kl = (ratio - 1.0) - log_ratio
        kl_penalty = approx_kl.mean()

        total_loss = policy_loss + self.config.kl_coeff * kl_penalty

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "kl_div": approx_kl.mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > self.config.clip_range).float().mean().item(),
            "mean_ratio": ratio.mean().item(),
        }

        return total_loss, metrics

    def step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one GRPO training step.

        Full pipeline: generate completions, score them, compute advantages,
        and update the policy. This is the main entry point called by the
        Olympus TrainingOrchestrator.

        Args:
            batch: Dictionary containing:
                - "prompts": List of prompt strings.
                - "tokenizer": Tokenizer instance for encoding/decoding.
                - Optionally "optimizer" for parameter updates.

        Returns:
            Dictionary of training metrics for logging.
        """
        prompts = batch["prompts"]
        tokenizer = batch["tokenizer"]
        B = len(prompts)
        G = self.config.group_size
        device = next(self.model.parameters()).device

        # Step 1: Generate G completions per prompt
        group_completions = self.generate_group(prompts, tokenizer)

        # Step 2: Compute rewards for each completion
        rewards = torch.zeros(B, G, device=device)
        for i, (prompt, completions) in enumerate(zip(prompts, group_completions)):
            for j, completion in enumerate(completions):
                rewards[i, j] = self.reward_fn(prompt, completion)

        # Step 3: Compute group-normalized advantages
        advantages = self.compute_advantages(rewards)

        # Step 4: Compute log-probabilities under current and reference policies
        self.model.train()
        logprobs = torch.zeros(B, G, device=device)
        ref_logprobs = torch.zeros(B, G, device=device)

        for i, (prompt, completions) in enumerate(zip(prompts, group_completions)):
            for j, completion in enumerate(completions):
                full_text = prompt + completion
                input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
                prompt_len = len(tokenizer.encode(prompt))

                # Current policy log-probs
                outputs = self.model(input_ids)
                logits = outputs["logits"][0, prompt_len - 1 : -1, :]
                target = input_ids[0, prompt_len:]
                token_logprobs = F.log_softmax(logits, dim=-1)
                selected = token_logprobs.gather(1, target.unsqueeze(-1)).squeeze(-1)
                logprobs[i, j] = selected.sum()

                # Reference policy log-probs
                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids)
                    ref_logits = ref_outputs["logits"][0, prompt_len - 1 : -1, :]
                    ref_token_logprobs = F.log_softmax(ref_logits, dim=-1)
                    ref_selected = ref_token_logprobs.gather(1, target.unsqueeze(-1)).squeeze(-1)
                    ref_logprobs[i, j] = ref_selected.sum()

        # Step 5: Compute loss and update
        loss, loss_metrics = self.compute_loss(logprobs, ref_logprobs, advantages)

        if "optimizer" in batch:
            optimizer = batch["optimizer"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Aggregate metrics
        metrics = {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_max": rewards.max().item(),
            "reward_min": rewards.min().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            **loss_metrics,
        }

        return metrics
