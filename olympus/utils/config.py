"""
Simple configuration loading with OmegaConf.

Falls back to plain YAML if OmegaConf is not installed.
"""

import os
from typing import Any, Dict, Optional, Union

try:
    from omegaconf import OmegaConf, DictConfig

    _HAS_OMEGACONF = True
except ImportError:
    _HAS_OMEGACONF = False


def load_config(
    path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Union["DictConfig", Dict[str, Any]]:
    """Load a YAML configuration file.

    Args:
        path:      Path to a ``.yaml`` / ``.yml`` file.
        overrides: Optional dict of key-value overrides applied after loading.
                   Supports dotted keys (e.g. ``{"model.hidden": 768}``).

    Returns:
        An OmegaConf DictConfig if OmegaConf is installed, otherwise a plain
        dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    if _HAS_OMEGACONF:
        cfg = OmegaConf.load(path)
        if overrides:
            override_cfg = OmegaConf.create(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
        return cfg
    else:
        import yaml

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        if overrides:
            _apply_overrides(cfg, overrides)

        return cfg


def _apply_overrides(cfg: dict, overrides: Dict[str, Any]) -> None:
    """Apply dotted-key overrides to a plain dict in-place."""
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
