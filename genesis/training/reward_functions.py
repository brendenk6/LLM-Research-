"""
Reward functions for GENESIS RL training.

Provides composable reward functions for Group Relative Policy Optimization
(GRPO) and other RL training pipelines. Each reward function scores a
(prompt, completion) pair and returns a scalar float.

Designed to integrate with the Olympus TrainingOrchestrator and the
GRPOTrainer in genesis.training.grpo.

Available rewards:
- InformationGainReward: Perplexity-based information gain measurement.
- CorrectnessReward: Exact-match checking for math/code answers.
- FormatReward: Structural format compliance checking.
- CompositeReward: Weighted combination of multiple reward functions.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardFunction(ABC):
    """Base class for all reward functions.

    Subclasses must implement __call__ to score a (prompt, completion) pair.
    All reward functions return a float, typically in a bounded range
    (e.g., [0, 1] or [-1, 1]) for stable RL training.
    """

    @abstractmethod
    def __call__(self, prompt: str, completion: str) -> float:
        """Score a completion given its prompt.

        Args:
            prompt: The input prompt that was given to the model.
            completion: The model's generated completion.

        Returns:
            A scalar reward value.
        """
        ...


class InformationGainReward(RewardFunction):
    """Measures information gain by comparing perplexity with and without context.

    Computes how much the completion's perplexity improves when conditioned
    on the full prompt versus a baseline (empty or minimal context). A higher
    information gain means the model is leveraging the prompt effectively
    to produce a more informed completion.

    The reward is the normalized log-perplexity reduction:
        reward = clamp((baseline_ppl - contextual_ppl) / baseline_ppl, -1, 1)

    Args:
        ref_model: A language model used to compute perplexities.
        tokenizer: Tokenizer compatible with the reference model.
        baseline_context: Minimal context string for baseline perplexity.
            Defaults to empty string.
        max_reward: Upper clamp for the reward value.
        min_reward: Lower clamp for the reward value.
    """

    def __init__(
        self,
        ref_model: nn.Module,
        tokenizer: Any,
        baseline_context: str = "",
        max_reward: float = 1.0,
        min_reward: float = -1.0,
    ) -> None:
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.baseline_context = baseline_context
        self.max_reward = max_reward
        self.min_reward = min_reward

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _compute_perplexity(self, context: str, completion: str) -> float:
        """Compute perplexity of completion conditioned on context.

        Args:
            context: The conditioning text.
            completion: The text whose perplexity we measure.

        Returns:
            Perplexity as a float. Returns a large value if computation fails.
        """
        device = next(self.ref_model.parameters()).device
        full_text = context + completion
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(device)
        context_len = len(self.tokenizer.encode(context))

        if input_ids.shape[1] <= context_len:
            return 1e6  # No completion tokens to evaluate

        with torch.no_grad():
            outputs = self.ref_model(input_ids)
            logits = outputs["logits"][0, context_len - 1 : -1, :]  # (T, V)
            targets = input_ids[0, context_len:]  # (T,)

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
            avg_neg_log_prob = -token_log_probs.mean().item()

        return math.exp(avg_neg_log_prob)

    def __call__(self, prompt: str, completion: str) -> float:
        """Compute information gain reward.

        Args:
            prompt: The full input prompt.
            completion: The model's generated text.

        Returns:
            Reward based on perplexity reduction from baseline to contextual.
        """
        if not completion.strip():
            return self.min_reward

        baseline_ppl = self._compute_perplexity(self.baseline_context, completion)
        contextual_ppl = self._compute_perplexity(prompt, completion)

        if baseline_ppl < 1e-8:
            return 0.0

        # Normalized perplexity reduction
        gain = (baseline_ppl - contextual_ppl) / baseline_ppl
        return max(self.min_reward, min(self.max_reward, gain))


class CorrectnessReward(RewardFunction):
    """Checks if the completion contains the correct answer for math/code tasks.

    Extracts the final answer from the completion using configurable patterns
    and compares it against the expected output. Supports both exact string
    matching and numeric comparison with tolerance.

    Args:
        expected_answers: Dictionary mapping prompt strings (or prompt hashes)
            to their expected answer strings.
        answer_pattern: Regex pattern to extract the answer from completions.
            Should contain a named group 'answer'. Defaults to matching text
            after common answer markers.
        numeric_tolerance: Tolerance for numeric comparisons. If both the
            extracted and expected answers are numeric, they are compared
            within this tolerance.
        correct_reward: Reward value for a correct answer.
        incorrect_reward: Reward value for an incorrect answer.
        partial_reward: Reward for partially correct answers (contains the
            answer but with extra content).
    """

    def __init__(
        self,
        expected_answers: Optional[Dict[str, str]] = None,
        answer_pattern: str = r"(?:answer is|answer:|=)\s*(?P<answer>[^\n,.]+)",
        numeric_tolerance: float = 1e-6,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        partial_reward: float = 0.5,
    ) -> None:
        self.expected_answers = expected_answers or {}
        self.answer_pattern = re.compile(answer_pattern, re.IGNORECASE)
        self.numeric_tolerance = numeric_tolerance
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.partial_reward = partial_reward

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the answer from completion text using the pattern."""
        match = self.answer_pattern.search(text)
        if match:
            return match.group("answer").strip()
        return None

    def _is_numeric_match(self, a: str, b: str) -> bool:
        """Check if two strings represent the same number within tolerance."""
        try:
            val_a = float(a.replace(",", "").strip())
            val_b = float(b.replace(",", "").strip())
            return abs(val_a - val_b) <= self.numeric_tolerance
        except (ValueError, TypeError):
            return False

    def __call__(self, prompt: str, completion: str) -> float:
        """Score completion correctness against expected answer.

        Args:
            prompt: The input prompt (used to look up expected answer).
            completion: The model's generated completion.

        Returns:
            Reward based on answer correctness.
        """
        expected = self.expected_answers.get(prompt)
        if expected is None:
            return self.incorrect_reward

        extracted = self._extract_answer(completion)
        if extracted is None:
            # Check if the expected answer appears anywhere in the completion
            if expected.strip().lower() in completion.lower():
                return self.partial_reward
            return self.incorrect_reward

        # Exact string match (case-insensitive)
        if extracted.lower() == expected.strip().lower():
            return self.correct_reward

        # Numeric match
        if self._is_numeric_match(extracted, expected):
            return self.correct_reward

        # Partial match: extracted contains expected or vice versa
        if expected.strip().lower() in extracted.lower():
            return self.partial_reward

        return self.incorrect_reward


class FormatReward(RewardFunction):
    """Checks if the completion follows expected structural format.

    Verifies the presence of reasoning tags and structural elements in
    the completion. Designed to encourage models to produce well-formatted
    chain-of-thought reasoning within designated tags.

    Args:
        reason_start_tag: Opening tag for reasoning sections.
        reason_end_tag: Closing tag for reasoning sections.
        required_sections: Additional regex patterns that should appear
            in the completion.
        tag_reward: Reward for having properly matched reasoning tags.
        structure_reward: Reward per matched required section.
        max_reward: Maximum total reward.
    """

    def __init__(
        self,
        reason_start_tag: str = "<|reason_start|>",
        reason_end_tag: str = "<|reason_end|>",
        required_sections: Optional[List[str]] = None,
        tag_reward: float = 0.5,
        structure_reward: float = 0.25,
        max_reward: float = 1.0,
    ) -> None:
        self.reason_start_tag = reason_start_tag
        self.reason_end_tag = reason_end_tag
        self.required_sections = required_sections or []
        self.tag_reward = tag_reward
        self.structure_reward = structure_reward
        self.max_reward = max_reward

    def __call__(self, prompt: str, completion: str) -> float:
        """Score completion format compliance.

        Args:
            prompt: The input prompt (unused, kept for interface consistency).
            completion: The model's generated completion to check.

        Returns:
            Reward in [0, max_reward] based on format compliance.
        """
        if not completion.strip():
            return 0.0

        reward = 0.0

        # Check for properly paired reasoning tags
        has_start = self.reason_start_tag in completion
        has_end = self.reason_end_tag in completion

        if has_start and has_end:
            start_idx = completion.index(self.reason_start_tag)
            end_idx = completion.index(self.reason_end_tag)
            if end_idx > start_idx:
                # Tags are properly ordered and content exists between them
                inner = completion[start_idx + len(self.reason_start_tag):end_idx]
                if inner.strip():
                    reward += self.tag_reward
                else:
                    # Tags present but empty reasoning
                    reward += self.tag_reward * 0.25
        elif has_start or has_end:
            # Only one tag present: partial credit
            reward += self.tag_reward * 0.1

        # Check required sections
        if self.required_sections:
            per_section = self.structure_reward / len(self.required_sections)
            for pattern in self.required_sections:
                if re.search(pattern, completion, re.IGNORECASE):
                    reward += per_section

        return min(self.max_reward, reward)


class CompositeReward(RewardFunction):
    """Combines multiple reward functions with configurable weights.

    Computes a weighted sum of rewards from multiple functions. Useful for
    multi-objective RL training where you want to balance correctness,
    format compliance, and information quality.

    Args:
        rewards_and_weights: List of (RewardFunction, weight) tuples.
            Weights do not need to sum to 1; they are used as-is in
            the weighted sum.
        normalize: If True, divide the total by the sum of weights to
            produce a weighted average instead of a weighted sum.
    """

    def __init__(
        self,
        rewards_and_weights: List[Tuple[RewardFunction, float]],
        normalize: bool = True,
    ) -> None:
        if not rewards_and_weights:
            raise ValueError("Must provide at least one (reward_fn, weight) pair.")
        self.rewards_and_weights = rewards_and_weights
        self.normalize = normalize

    def __call__(self, prompt: str, completion: str) -> float:
        """Compute weighted combination of all reward functions.

        Args:
            prompt: The input prompt.
            completion: The model's generated completion.

        Returns:
            Weighted sum (or average) of individual rewards.
        """
        total_reward = 0.0
        total_weight = 0.0

        for reward_fn, weight in self.rewards_and_weights:
            score = reward_fn(prompt, completion)
            total_reward += weight * score
            total_weight += weight

        if self.normalize and total_weight > 0:
            return total_reward / total_weight

        return total_reward
