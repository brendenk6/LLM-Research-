"""Phase 0: 50M parameter proof-of-concept training.

End-to-end training script for the small HLRT model. Reads config from
phase0_50m.yaml, loads packed data, and trains with full logging.

Usage:
    # Step 1: Prepare data (streams from HuggingFace, packs into binary)
    python -m genesis.training.prepare_data --config genesis/training/configs/phase0_50m.yaml

    # Step 2: Train
    python -m genesis.training.train_phase0 --config genesis/training/configs/phase0_50m.yaml

    # Resume from checkpoint
    python -m genesis.training.train_phase0 --config genesis/training/configs/phase0_50m.yaml --resume checkpoints/phase0/checkpoint_step1000.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genesis.model.hlrt import HLRT, HLRTConfig
from genesis.model.tier_composer import count_parameters
from olympus.data.packed_dataset import PackedDataset
from olympus.data.tokenizer import TokenizerWrapper
from olympus.optim.muon_adamw_hybrid import MuonAdamWHybrid
from olympus.optim.schedulers import WSDScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(cfg: dict) -> HLRT:
    """Construct the HLRT model from config."""
    model_cfg = cfg["model"]
    t1, t2, t3 = model_cfg["tier1"], model_cfg["tier2"], model_cfg["tier3"]
    gate = model_cfg.get("gate", {})

    config = HLRTConfig(
        vocab_size=model_cfg["vocab_size"],
        d_model=t1["d_model"],
        embedding_dropout=model_cfg.get("dropout", 0.0),
        # Tier 1
        tier1_num_layers=t1["layers"],
        tier1_num_heads=t1["heads"],
        tier1_d_ff=t1["d_ff"],
        tier1_dropout=model_cfg.get("dropout", 0.0),
        tier1_use_flash_attention=model_cfg.get("use_flash_attention", True),
        tier1_max_seq_len=model_cfg["max_seq_len"],
        tier1_rope_base=model_cfg.get("rope_base", 10000.0),
        # Gate 1->2
        gate1_chunk_size=t2.get("chunk_size", 32),
        gate1_threshold=gate.get("tier2_threshold", 0.3),
        # Latent pooling
        num_latent_vectors=t2.get("latent_vectors_per_chunk", 4),
        latent_pool_num_heads=4,
        # Tier 2
        tier2_d_model=t2["d_model"],
        tier2_num_layers=t2["layers"],
        tier2_num_heads=t2["heads"],
        tier2_d_ff=t2["d_ff"],
        tier2_dropout=model_cfg.get("dropout", 0.0),
        tier2_use_flash_attention=model_cfg.get("use_flash_attention", True),
        tier2_max_seq_len=model_cfg["max_seq_len"] // 2,
        tier2_rope_base=model_cfg.get("rope_base", 10000.0),
        # Gate 2->3
        gate2_threshold=gate.get("tier3_threshold", 0.7),
        # Tier 3
        tier3_d_model=t3["d_model"],
        tier3_num_layers=t3["layers"],
        tier3_num_heads=t3["heads"],
        tier3_d_ff=t3["d_ff"],
        tier3_dropout=model_cfg.get("dropout", 0.0),
        tier3_use_flash_attention=model_cfg.get("use_flash_attention", True),
        tier3_max_seq_len=model_cfg["max_seq_len"] // 4,
        tier3_rope_base=model_cfg.get("rope_base", 10000.0),
        tier3_recurrence_steps=t3.get("recurrence_steps", 2),
    )

    model = HLRT(config)

    params = count_parameters(model)
    logger.info("Model created:")
    for k, v in params.items():
        logger.info("  %-12s: %.2fM", k, v / 1e6)

    return model


def build_optimizer(model: HLRT, cfg: dict) -> MuonAdamWHybrid:
    """Build the MuonAdamWHybrid optimizer."""
    opt_cfg = cfg["optimizer"]
    return MuonAdamWHybrid(
        model.parameters(),
        lr_muon=float(opt_cfg.get("muon_lr", 0.02)),
        lr_adamw=float(opt_cfg.get("adamw_lr", 3e-4)),
        momentum=float(opt_cfg.get("muon_momentum", 0.95)),
        betas=tuple(float(b) for b in opt_cfg.get("adamw_betas", [0.9, 0.999])),
        weight_decay_adamw=float(opt_cfg.get("weight_decay", 0.01)),
        ns_steps=int(opt_cfg.get("ns_iterations", 5)),
    )


def build_scheduler(optimizer: MuonAdamWHybrid, cfg: dict) -> WSDScheduler:
    """Build the WSD learning rate scheduler."""
    sched_cfg = cfg["scheduler"]
    train_cfg = cfg["training"]
    return WSDScheduler(
        optimizer=optimizer,
        base_lr=sched_cfg.get("peak_lr", 0.02),
        min_lr=sched_cfg.get("min_lr", 2e-5),
        warmup_steps=sched_cfg.get("warmup_steps", 500),
        total_steps=train_cfg.get("max_steps", 10000),
        decay_steps=sched_cfg.get("decay_steps", 500),
    )


def train(
    cfg: dict,
    resume_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints/phase0",
) -> None:
    """Main training loop for Phase 0."""
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    hw_cfg = cfg.get("hardware", {})

    requested_device = hw_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    logger.info("Device: %s", device)

    # --- Data ---
    packed_dir = data_cfg.get("packed_dir", "data/packed_phase0")
    if not Path(packed_dir).exists():
        logger.error(
            "Packed data not found at '%s'. Run data preparation first:\n"
            "  python -m genesis.training.prepare_data --config genesis/training/configs/phase0_50m.yaml",
            packed_dir,
        )
        return

    train_ds = PackedDataset(packed_dir, split="train")
    val_ds = PackedDataset(packed_dir, split="val")

    logger.info("Train: %s", train_ds)
    logger.info("Val:   %s", val_ds)

    batch_size = train_cfg.get("batch_size", 64)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=data_cfg.get("num_workers", 4), pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # --- Model ---
    model = build_model(cfg)
    model = model.to(device)

    # Mixed precision
    use_amp = hw_cfg.get("precision", "bf16") == "bf16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and hw_cfg.get("precision") != "bf16"))
    amp_dtype = torch.bfloat16 if hw_cfg.get("precision") == "bf16" else torch.float16

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- Resume ---
    start_step = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        logger.info("Resumed from step %d", start_step)

    # --- Training ---
    max_steps = train_cfg.get("max_steps", 10000)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("checkpoint_interval", 500)
    eval_interval = train_cfg.get("eval_interval", 250)
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    early_stop_patience = train_cfg.get("early_stop_patience", 1500)

    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = start_step
    best_val_loss = float("inf")
    steps_without_improvement = 0
    tokens_seen = 0
    epoch = 0

    logger.info("=" * 60)
    logger.info("Phase 0 Training — 50M HLRT")
    logger.info("  Max steps:     %d", max_steps)
    logger.info("  Batch size:    %d × %d accum = %d effective", batch_size, grad_accum, batch_size * grad_accum)
    logger.info("  Seq len:       %d", data_cfg.get("max_seq_len", 2048))
    logger.info("  AMP:           %s (%s)", use_amp, amp_dtype)
    logger.info("=" * 60)

    model.train()
    train_start = time.time()

    while global_step < max_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)

            # Forward + backward with gradient accumulation
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids)
                logits = out["logits"][:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                aux_loss = out.get("aux_loss", 0.0)
                if isinstance(aux_loss, torch.Tensor):
                    loss = loss + aux_loss
                loss = loss / grad_accum

            loss.backward()
            tokens_seen += input_ids.numel()

            if (global_step + 1) % grad_accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # --- Logging ---
            if global_step % log_interval == 0:
                real_loss = loss.item() * grad_accum
                ppl = min(math.exp(real_loss), 1e6) if real_loss < 20 else 1e6
                elapsed = time.time() - train_start
                tok_per_sec = tokens_seen / max(elapsed, 1)
                logger.info(
                    "Step %5d/%d | loss=%.4f | ppl=%.1f | lr=%.2e | %.0f tok/s | %.1fM tokens",
                    global_step, max_steps, real_loss, ppl,
                    scheduler.last_lr, tok_per_sec, tokens_seen / 1e6,
                )

            # --- Eval ---
            if global_step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
                val_ppl = min(math.exp(val_loss), 1e6) if val_loss < 20 else 1e6
                logger.info(
                    "  [EVAL] Step %d | val_loss=%.4f | val_ppl=%.1f",
                    global_step, val_loss, val_ppl,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    # Save best model
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, "best.pt")
                else:
                    steps_without_improvement += eval_interval
                    if steps_without_improvement >= early_stop_patience:
                        logger.info(
                            "Early stopping at step %d (no improvement for %d steps)",
                            global_step, early_stop_patience,
                        )
                        break

                model.train()

            # --- Checkpoint ---
            if global_step % save_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir,
                                f"checkpoint_step{global_step}.pt")

        # Check early stop from inner loop break
        if steps_without_improvement >= early_stop_patience:
            break

    elapsed = time.time() - train_start
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Steps:       %d", global_step)
    logger.info("  Tokens seen: %.2fM", tokens_seen / 1e6)
    logger.info("  Wall time:   %.1f min", elapsed / 60)
    logger.info("  Best val:    %.4f", best_val_loss)
    logger.info("=" * 60)

    # Final save
    save_checkpoint(model, optimizer, global_step, checkpoint_dir, "final.pt")


def evaluate(
    model: HLRT,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids)
                logits = out["logits"][:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(
    model: HLRT,
    optimizer: MuonAdamWHybrid,
    step: int,
    checkpoint_dir: str,
    filename: str,
) -> None:
    """Save a checkpoint."""
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    logger.info("Checkpoint saved: %s", path)


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Train 50M HLRT model")
    parser.add_argument("--config", type=str, default="genesis/training/configs/phase0_50m.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase0")
    args = parser.parse_args()

    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        logger.error("PyYAML required. Install with: pip install pyyaml")
        return

    train(cfg, resume_path=args.resume, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
