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

# DDP imports — optional, only needed for multi-GPU
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int]:
    """Initialise DDP if launched via torchrun. Returns (rank, world_size)."""
    if not DDP_AVAILABLE:
        return 0, 1
    if "RANK" not in os.environ:
        return 0, 1  # Single-GPU, no torchrun
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp() -> None:
    """Destroy DDP process group if active."""
    if DDP_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """True if this is rank 0 (or single-GPU)."""
    return rank == 0


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
    """Build the WSD learning rate scheduler.

    All scheduler step counts are in optimizer steps (not micro-steps).
    total_steps = max_steps / gradient_accumulation_steps.
    """
    sched_cfg = cfg["scheduler"]
    train_cfg = cfg["training"]
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    total_opt_steps = train_cfg.get("max_steps", 10000) // grad_accum
    return WSDScheduler(
        optimizer=optimizer,
        base_lr=sched_cfg.get("peak_lr", 0.02),
        min_lr=sched_cfg.get("min_lr", 2e-5),
        warmup_steps=sched_cfg.get("warmup_steps", 500),
        total_steps=total_opt_steps,
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

    # --- DDP setup ---
    rank, world_size = setup_ddp()
    ddp = world_size > 1

    requested_device = hw_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        requested_device = "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available — falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device if not ddp else f"cuda:{rank}")
    if is_main_process(rank):
        logger.info("Device: %s (world_size=%d)", device, world_size)

    # --- CUDA speed optimizations ---
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Data ---
    packed_dir = data_cfg.get("packed_dir", "data/packed_phase0")
    if not Path(packed_dir).exists():
        logger.error(
            "Packed data not found at '%s'. Run data preparation first:\n"
            "  python -m genesis.training.prepare_data --config genesis/training/configs/phase0_50m.yaml",
            packed_dir,
        )
        cleanup_ddp()
        return

    train_ds = PackedDataset(packed_dir, split="train")
    val_ds = PackedDataset(packed_dir, split="val")

    if is_main_process(rank):
        logger.info("Train: %s", train_ds)
        logger.info("Val:   %s", val_ds)

    batch_size = train_cfg.get("batch_size", 64)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    pin = device.type == "cuda"
    n_workers = data_cfg.get("num_workers", 4)
    prefetch = 2 if pin and n_workers > 0 else None

    # DDP: each rank gets a different data shard
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None),  # Shuffle only if no DDP sampler
        sampler=train_sampler,
        num_workers=n_workers, pin_memory=pin,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=min(n_workers, 2), pin_memory=pin,
        prefetch_factor=prefetch,
    )

    # --- Model ---
    model = build_model(cfg)
    model = model.to(device)

    # torch.compile — massive speedup from kernel fusion (CUDA only)
    if device.type == "cuda" and hw_cfg.get("use_torch_compile", True):
        if is_main_process(rank):
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Mixed precision (only on CUDA — MPS bf16 autocast is unreliable)
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
        state_dict = ckpt["model_state_dict"]
        _unwrap_model(model).load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        if is_main_process(rank):
            logger.info("Resumed from step %d", start_step)

    # --- DDP wrapping (after resume, so state dict keys match) ---
    if ddp:
        model = DDP(model, device_ids=[rank])

    # --- Training ---
    max_steps = train_cfg.get("max_steps", 10000)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("checkpoint_interval", 500)
    eval_interval = train_cfg.get("eval_interval", 250)
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    early_stop_patience = train_cfg.get("early_stop_patience", 1500)

    if is_main_process(rank):
        os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = start_step
    best_val_loss = float("inf")
    steps_without_improvement = 0
    tokens_seen = 0
    epoch = 0

    if is_main_process(rank):
        eff_batch = batch_size * grad_accum * world_size
        logger.info("=" * 60)
        logger.info("Phase 0 Training — HLRT")
        logger.info("  Max steps:     %d", max_steps)
        logger.info("  Batch size:    %d × %d accum × %d GPU = %d effective",
                     batch_size, grad_accum, world_size, eff_batch)
        logger.info("  Seq len:       %d", data_cfg.get("max_seq_len", 2048))
        logger.info("  AMP:           %s (%s)", use_amp, amp_dtype)
        logger.info("  torch.compile: %s", device.type == "cuda" and hw_cfg.get("use_torch_compile", True))
        logger.info("  DDP:           %s (world_size=%d)", ddp, world_size)
        logger.info("=" * 60)

    model.train()
    train_start = time.time()

    while global_step < max_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # Proper shuffling per epoch in DDP

        for batch in train_loader:
            if global_step >= max_steps:
                break

            input_ids = batch["input_ids"]
            # Slice to max_seq_len if data blocks are longer (e.g. 4096 -> 1024)
            max_seq = cfg["model"].get("max_seq_len", input_ids.shape[1])
            if input_ids.shape[1] > max_seq:
                input_ids = input_ids[:, :max_seq]
            input_ids = input_ids.to(device)

            # Forward + backward with gradient accumulation
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids)
                logits = out["logits"][:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                aux_loss = out.get("aux_loss", 0.0)
                if isinstance(aux_loss, torch.Tensor):
                    loss = loss + aux_loss
                loss = loss / grad_accum

            loss.backward()
            tokens_seen += input_ids.numel() * world_size  # Count across all GPUs

            if (global_step + 1) % grad_accum == 0:
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    grad_norm = None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            global_step += 1

            # --- Logging (rank 0 only) ---
            if global_step % log_interval == 0 and is_main_process(rank):
                real_loss = loss.item() * grad_accum
                ppl = min(math.exp(real_loss), 1e6) if real_loss < 20 else 1e6
                elapsed = time.time() - train_start
                tok_per_sec = tokens_seen / max(elapsed, 1)
                gn_str = f" | grad_norm={grad_norm:.2f}" if grad_norm is not None else ""
                logger.info(
                    "Step %5d/%d | loss=%.4f | ppl=%.1f | lr=%.2e | %.0f tok/s | %.1fM tok%s",
                    global_step, max_steps, real_loss, ppl,
                    scheduler.last_lr, tok_per_sec, tokens_seen / 1e6, gn_str,
                )

            # --- Eval ---
            if global_step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)

                # Average val loss across ranks for DDP
                if ddp:
                    val_loss_t = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_t.item()

                if is_main_process(rank):
                    val_ppl = min(math.exp(val_loss), 1e6) if val_loss < 20 else 1e6
                    logger.info(
                        "  [EVAL] Step %d | val_loss=%.4f | val_ppl=%.1f",
                        global_step, val_loss, val_ppl,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    if is_main_process(rank):
                        save_checkpoint(model, optimizer, global_step, checkpoint_dir, "best.pt", scheduler)
                else:
                    steps_without_improvement += eval_interval
                    if steps_without_improvement >= early_stop_patience:
                        if is_main_process(rank):
                            logger.info(
                                "Early stopping at step %d (no improvement for %d steps)",
                                global_step, early_stop_patience,
                            )
                        break

                model.train()

            # --- Checkpoint ---
            if global_step % save_interval == 0 and is_main_process(rank):
                save_checkpoint(model, optimizer, global_step, checkpoint_dir,
                                f"checkpoint_step{global_step}.pt", scheduler)

        # Check early stop from inner loop break
        if steps_without_improvement >= early_stop_patience:
            break

    elapsed = time.time() - train_start
    if is_main_process(rank):
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info("  Steps:       %d", global_step)
        logger.info("  Tokens seen: %.2fM", tokens_seen / 1e6)
        logger.info("  Wall time:   %.1f min", elapsed / 60)
        logger.info("  Best val:    %.4f", best_val_loss)
        logger.info("=" * 60)

        save_checkpoint(model, optimizer, global_step, checkpoint_dir, "final.pt", scheduler)

    cleanup_ddp()


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

    max_seq = 1024  # Match training seq len
    max_eval_batches = 100  # Cap eval to avoid hours-long val passes
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"][:, :max_seq].to(device)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids)
                logits = out["logits"][:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= max_eval_batches:
                break

    return total_loss / max(n_batches, 1)


def _unwrap_model(model):
    """Unwrap DDP and torch.compile wrappers to get the raw model."""
    if hasattr(model, "module"):  # DDP
        model = model.module
    if hasattr(model, "_orig_mod"):  # torch.compile
        model = model._orig_mod
    return model


def save_checkpoint(
    model,
    optimizer: MuonAdamWHybrid,
    step: int,
    checkpoint_dir: str,
    filename: str,
    scheduler=None,
) -> None:
    """Save a checkpoint (auto-unwraps DDP/compile wrappers)."""
    path = os.path.join(checkpoint_dir, filename)
    raw = _unwrap_model(model)
    data = {
        "step": step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(data, path)
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
