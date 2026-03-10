"""
Structured logging setup with optional Weights & Biases integration.
"""

import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger("olympus")

_WANDB_RUN = None


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    wandb_run_name: Optional[str] = None,
) -> None:
    """Configure Olympus logging.

    Args:
        level:         Python logging level.
        log_file:      If provided, also write logs to this file.
        use_wandb:     Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity:  W&B entity (team / user).
        wandb_config:  Config dict to log with the W&B run.
        wandb_run_name: Display name for the W&B run.
    """
    global _WANDB_RUN

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("olympus")
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates on repeated calls
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # W&B init (optional)
    if use_wandb:
        try:
            import wandb

            _WANDB_RUN = wandb.init(
                project=wandb_project or "olympus",
                entity=wandb_entity,
                config=wandb_config or {},
                name=wandb_run_name,
                reinit=True,
            )
            root.info("Weights & Biases initialized (project=%s)", wandb_project)
        except ImportError:
            root.warning("wandb not installed; W&B logging disabled.")
        except Exception as e:
            root.warning("wandb init failed: %s", e)

    root.info("Olympus logging configured (level=%s)", logging.getLevelName(level))


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log a dict of metrics to the Python logger and optionally to W&B.

    Args:
        metrics: Dictionary of metric name -> value.
        step:    Global training step (for W&B x-axis).
        prefix:  String prepended to every metric key.
    """
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    parts = [f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    logger.info("step=%s  %s", step, "  ".join(parts))

    if _WANDB_RUN is not None:
        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception:
            pass
