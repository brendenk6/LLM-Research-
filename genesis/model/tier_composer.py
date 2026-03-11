"""TierComposer: LEGO-style composition utilities for HLRT models.

Enables building larger models from independently trained tiers:

    # Train a 50M model
    small = HLRT(small_config)
    train(small)

    # Train a 100M model
    medium = HLRT(medium_config)
    train(medium)

    # Compose: take Tier 1 from small, Tier 2+3 from medium
    composer = TierComposer(target_config)
    composer.load_tier(1, "small_checkpoint.pt")
    composer.load_tier(2, "medium_checkpoint.pt")
    composer.load_tier(3, "medium_checkpoint.pt")
    composed_model = composer.build(freeze_tiers=[1])  # freeze the transplanted tier
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn

from genesis.model.hlrt import HLRT, HLRTConfig

logger = logging.getLogger(__name__)

# Maps tier IDs to the attribute prefixes on the HLRT model.
_TIER_PREFIXES: Dict[int, List[str]] = {
    1: ["tier1.", "embeddings.", "gate1."],
    2: ["tier2.", "latent_pool.", "gate2."],
    3: ["tier3."],
}

# Connector modules that bridge tiers — loaded separately or rebuilt.
_CONNECTOR_PREFIXES = ["conditioning.", "output_norm.", "output_proj."]


class TierComposer:
    """Build an HLRT model by composing independently trained tiers.

    Args:
        target_config: The HLRTConfig for the composed model.
    """

    def __init__(self, target_config: HLRTConfig) -> None:
        self.config = target_config
        self._tier_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self._connector_state: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_tier(
        self,
        tier_id: int,
        checkpoint_path: Union[str, Path],
        map_location: Optional[str] = None,
    ) -> None:
        """Load a single tier's weights from a checkpoint.

        The checkpoint can be a full HLRT checkpoint — only the relevant
        tier's parameters will be extracted.

        Args:
            tier_id: Which tier to load (1, 2, or 3).
            checkpoint_path: Path to a .pt checkpoint file.
            map_location: Device mapping for torch.load.
        """
        if tier_id not in _TIER_PREFIXES:
            raise ValueError(f"tier_id must be 1, 2, or 3, got {tier_id}")

        checkpoint = torch.load(
            str(checkpoint_path), map_location=map_location, weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        prefixes = _TIER_PREFIXES[tier_id]
        tier_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in prefixes)
        }

        if not tier_state:
            raise ValueError(
                f"No parameters found for tier {tier_id} in {checkpoint_path}. "
                f"Expected keys starting with: {prefixes}"
            )

        self._tier_states[tier_id] = tier_state
        logger.info(
            "Loaded tier %d from %s (%d parameters, %.2fM values)",
            tier_id, checkpoint_path, len(tier_state),
            sum(v.numel() for v in tier_state.values()) / 1e6,
        )

    def load_tier_from_state_dict(
        self,
        tier_id: int,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        """Load a tier directly from an in-memory state dict.

        Args:
            tier_id: Which tier to load (1, 2, or 3).
            state_dict: Full model state dict (tier params will be extracted).
        """
        if tier_id not in _TIER_PREFIXES:
            raise ValueError(f"tier_id must be 1, 2, or 3, got {tier_id}")

        prefixes = _TIER_PREFIXES[tier_id]
        tier_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in prefixes)
        }
        self._tier_states[tier_id] = tier_state

    def load_connectors(
        self,
        checkpoint_path: Union[str, Path],
        map_location: Optional[str] = None,
    ) -> None:
        """Load connector modules (conditioning, output) from a checkpoint.

        Args:
            checkpoint_path: Path to a .pt checkpoint file.
            map_location: Device mapping for torch.load.
        """
        checkpoint = torch.load(
            str(checkpoint_path), map_location=map_location, weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        self._connector_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in _CONNECTOR_PREFIXES)
        }

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        freeze_tiers: Optional[List[int]] = None,
        reinit_connectors: bool = False,
    ) -> HLRT:
        """Construct the composed HLRT model.

        Args:
            freeze_tiers: List of tier IDs to freeze (no gradient updates).
                Use this when transplanting a pretrained tier into a new model.
            reinit_connectors: If True, randomly initialize the connector modules
                (conditioning, output) even if they were loaded. Useful when
                tier dimensions changed and old connectors are incompatible.

        Returns:
            A fully constructed HLRT model with composed weights.
        """
        model = HLRT(self.config)
        freeze_tiers = set(freeze_tiers or [])

        # Load tier weights (with dimension compatibility checks)
        for tier_id, tier_state in self._tier_states.items():
            _load_partial_state_dict(model, tier_state, tier_id)

        # Load or reinit connectors
        if self._connector_state and not reinit_connectors:
            _load_partial_state_dict(model, self._connector_state, label="connectors")

        # Freeze requested tiers
        for tier_id in freeze_tiers:
            freeze_tier(model, tier_id)

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Composed model: %.2fM total params, %.2fM trainable",
            total_params / 1e6, trainable / 1e6,
        )

        return model

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of what will be composed."""
        lines = [f"TierComposer target: {self.config.d_model}d model"]
        for tier_id in [1, 2, 3]:
            if tier_id in self._tier_states:
                n = sum(v.numel() for v in self._tier_states[tier_id].values())
                lines.append(f"  Tier {tier_id}: loaded ({n/1e6:.2f}M params)")
            else:
                lines.append(f"  Tier {tier_id}: fresh init")
        if self._connector_state:
            lines.append("  Connectors: loaded")
        else:
            lines.append("  Connectors: fresh init")
        return "\n".join(lines)


# ======================================================================
# Standalone utilities (usable without TierComposer)
# ======================================================================


def freeze_tier(model: HLRT, tier_id: int) -> None:
    """Freeze all parameters belonging to a specific tier.

    Args:
        model: An HLRT model instance.
        tier_id: Which tier to freeze (1, 2, or 3).
    """
    if tier_id not in _TIER_PREFIXES:
        raise ValueError(f"tier_id must be 1, 2, or 3, got {tier_id}")

    prefixes = _TIER_PREFIXES[tier_id]
    count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
            count += 1

    logger.info("Froze %d parameter tensors in tier %d", count, tier_id)


def unfreeze_tier(model: HLRT, tier_id: int) -> None:
    """Unfreeze all parameters belonging to a specific tier.

    Args:
        model: An HLRT model instance.
        tier_id: Which tier to unfreeze (1, 2, or 3).
    """
    if tier_id not in _TIER_PREFIXES:
        raise ValueError(f"tier_id must be 1, 2, or 3, got {tier_id}")

    prefixes = _TIER_PREFIXES[tier_id]
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = True


def save_tier(
    model: HLRT,
    tier_id: int,
    path: Union[str, Path],
    extra: Optional[Dict] = None,
) -> None:
    """Save a single tier's weights to a checkpoint file.

    Args:
        model: An HLRT model instance.
        tier_id: Which tier to save (1, 2, or 3).
        path: Output path for the .pt file.
        extra: Optional extra metadata to include.
    """
    if tier_id not in _TIER_PREFIXES:
        raise ValueError(f"tier_id must be 1, 2, or 3, got {tier_id}")

    prefixes = _TIER_PREFIXES[tier_id]
    tier_state = {
        k: v for k, v in model.state_dict().items()
        if any(k.startswith(p) for p in prefixes)
    }

    checkpoint = {"tier_id": tier_id, "model_state_dict": tier_state}
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, str(path))
    n_params = sum(v.numel() for v in tier_state.values())
    logger.info("Saved tier %d to %s (%.2fM params)", tier_id, path, n_params / 1e6)


def count_parameters(model: HLRT) -> Dict[str, int]:
    """Count parameters per tier and overall.

    Returns:
        Dictionary with keys 'tier1', 'tier2', 'tier3', 'connectors',
        'total', and 'trainable'.
    """
    counts: Dict[str, int] = {"tier1": 0, "tier2": 0, "tier3": 0, "connectors": 0}

    for name, param in model.named_parameters():
        assigned = False
        for tier_id, prefixes in _TIER_PREFIXES.items():
            if any(name.startswith(p) for p in prefixes):
                counts[f"tier{tier_id}"] += param.numel()
                assigned = True
                break
        if not assigned:
            counts["connectors"] += param.numel()

    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


# ======================================================================
# Internal helpers
# ======================================================================


def _load_partial_state_dict(
    model: nn.Module,
    partial_state: Dict[str, torch.Tensor],
    label: Union[int, str] = "",
) -> None:
    """Load a partial state dict with shape-mismatch warnings."""
    model_state = model.state_dict()
    loaded, skipped = 0, 0

    for key, value in partial_state.items():
        if key not in model_state:
            logger.warning("Key '%s' not found in target model (skipped)", key)
            skipped += 1
            continue
        if model_state[key].shape != value.shape:
            logger.warning(
                "Shape mismatch for '%s': source %s vs target %s (skipped)",
                key, value.shape, model_state[key].shape,
            )
            skipped += 1
            continue
        model_state[key] = value
        loaded += 1

    model.load_state_dict(model_state)
    logger.info(
        "Loaded %d/%d tensors for %s (%d skipped due to shape mismatch)",
        loaded, loaded + skipped, label, skipped,
    )
