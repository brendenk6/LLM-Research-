"""
FSDP wrapping utilities for GENESIS models.

Provides auto-wrap policies that shard at tier boundaries and
Transformer block boundaries for optimal memory distribution.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Optional, Set, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

logger = logging.getLogger(__name__)


@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapping."""

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"
    min_num_params: int = 1_000_000  # Min params for auto-wrap


def get_sharding_strategy(name: str) -> ShardingStrategy:
    """Convert string to ShardingStrategy enum."""
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    if name not in strategies:
        raise ValueError(f"Unknown sharding strategy: {name}. Options: {list(strategies)}")
    return strategies[name]


def get_mixed_precision(name: str) -> Optional[MixedPrecision]:
    """Build MixedPrecision policy from string."""
    if name == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif name == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif name in ("fp32", "none", None):
        return None
    else:
        raise ValueError(f"Unknown precision: {name}. Options: bf16, fp16, fp32")


def get_backward_prefetch(name: str) -> Optional[BackwardPrefetch]:
    """Convert string to BackwardPrefetch enum."""
    options = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }
    return options.get(name)


def get_hlrt_wrap_policy(min_num_params: int = 1_000_000):
    """Auto-wrap policy for HLRT models.

    Wraps each major tier and Transformer block as a separate FSDP unit.
    This gives fine-grained sharding without wrapping tiny modules.
    """
    # Import HLRT components to identify wrap boundaries
    from genesis.model.tier1_token_processor import Tier1TokenProcessor
    from genesis.model.tier2_semantic_planner import Tier2SemanticPlanner
    from genesis.model.tier3_deliberative import Tier3DeliberativeReasoner
    from genesis.model.embeddings import SharedEmbeddings

    wrap_classes: Set[Type[nn.Module]] = {
        Tier1TokenProcessor,
        Tier2SemanticPlanner,
        Tier3DeliberativeReasoner,
        SharedEmbeddings,
    }

    # Also try to get TransformerBlock for inner wrapping
    try:
        from genesis.model.transformer_block import TransformerBlock
        wrap_classes.add(TransformerBlock)
    except ImportError:
        pass

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=wrap_classes,
    )


def wrap_model_fsdp(
    model: nn.Module,
    fsdp_config: FSDPConfig,
    device_id: Optional[int] = None,
) -> FSDP:
    """Wrap a model in FSDP with the given configuration.

    Args:
        model: The model to wrap (typically HLRT).
        fsdp_config: FSDP configuration.
        device_id: Local CUDA device ID.

    Returns:
        FSDP-wrapped model.
    """
    if device_id is None:
        device_id = torch.cuda.current_device()

    sharding = get_sharding_strategy(fsdp_config.sharding_strategy)
    mp = get_mixed_precision(fsdp_config.mixed_precision)
    prefetch = get_backward_prefetch(fsdp_config.backward_prefetch)
    wrap_policy = get_hlrt_wrap_policy(fsdp_config.min_num_params)

    wrapped = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp,
        auto_wrap_policy=wrap_policy,
        backward_prefetch=prefetch,
        device_id=device_id,
        limit_all_gathers=True,
        use_orig_params=True,  # Required for optimizer param groups
    )

    # Apply activation checkpointing to transformer blocks
    if fsdp_config.activation_checkpointing:
        _apply_activation_checkpointing(wrapped)

    param_count = sum(p.numel() for p in wrapped.parameters())
    logger.info(
        "FSDP wrapped: %s strategy, %s precision, %.1fM params, "
        "activation_ckpt=%s, device=%d",
        fsdp_config.sharding_strategy,
        fsdp_config.mixed_precision,
        param_count / 1e6,
        fsdp_config.activation_checkpointing,
        device_id,
    )

    return wrapped


def _apply_activation_checkpointing(model: FSDP) -> None:
    """Apply activation checkpointing to Transformer blocks inside FSDP."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    # Try to find TransformerBlock class
    check_classes = set()
    try:
        from genesis.model.transformer_block import TransformerBlock
        check_classes.add(TransformerBlock)
    except ImportError:
        pass

    if not check_classes:
        logger.warning("No TransformerBlock found for activation checkpointing")
        return

    def check_fn(module: nn.Module) -> bool:
        return isinstance(module, tuple(check_classes))

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )
    logger.info("Activation checkpointing applied to %s", check_classes)
