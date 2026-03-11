"""Sampling strategies for text generation.

Supports top-k, top-p (nucleus), temperature scaling, repetition penalty,
and combinations thereof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    """Configuration for sampling behavior."""

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    min_tokens: int = 0


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature.

    Args:
        logits: (B, vocab_size) or (vocab_size,).
        temperature: Scaling factor. Lower = more deterministic.

    Returns:
        Scaled logits.
    """
    if temperature == 1.0:
        return logits
    if temperature <= 0.0:
        return logits  # greedy handled at sample time
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside the top-k highest values.

    Args:
        logits: (B, vocab_size).
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with -inf for non-top-k positions.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens whose cumulative
    probability exceeds p.

    Args:
        logits: (B, vocab_size).
        p: Cumulative probability threshold.

    Returns:
        Filtered logits.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Shift right so the first token above threshold is kept
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original ordering
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalize tokens that have already appeared.

    Args:
        logits: (B, vocab_size).
        generated_ids: (B, S) token ids generated so far.
        penalty: Multiplicative penalty (>1.0 to discourage repetition).

    Returns:
        Penalized logits.
    """
    if penalty == 1.0:
        return logits

    # Gather logits for generated tokens
    score = torch.gather(logits, -1, generated_ids)
    # Penalize: divide positive logits, multiply negative logits
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits = logits.scatter(-1, generated_ids, score)
    return logits


def sample(
    logits: torch.Tensor,
    config: SamplingConfig,
    generated_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample next token ids from logits.

    Args:
        logits: (B, vocab_size) logits for the next token.
        config: Sampling configuration.
        generated_ids: (B, S) previously generated tokens for repetition penalty.

    Returns:
        (B,) sampled token ids.
    """
    # Repetition penalty
    if generated_ids is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

    # Temperature
    logits = apply_temperature(logits, config.temperature)

    # Greedy
    if config.temperature <= 0.0:
        return logits.argmax(dim=-1)

    # Top-k
    logits = apply_top_k(logits, config.top_k)

    # Top-p
    logits = apply_top_p(logits, config.top_p)

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
