"""
Phase 1: Distributed Bootstrap Training (FSDP Multi-GPU).

Drop-in replacement for train_phase1_bootstrap that adds multi-GPU
support via FSDP. Falls back to single-GPU if world_size == 1.

Launch with torchrun:
    torchrun --nproc_per_node=8 -m genesis.training.train_phase1_distributed \
        --config genesis/training/configs/phase1_1b.yaml \
        --data-dir packed/ \
        --checkpoint-dir checkpoints/phase1
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from genesis.model.hlrt import HLRT, HLRTConfig
from genesis.training.train_phase1_bootstrap import ntp_objective
from olympus.core.training_orchestrator import ModelConfig, ObjectiveConfig
from olympus.core.training_context import TrainingContext
from olympus.data.tokenizer import TokenizerWrapper
from olympus.optim.muon_adamw_hybrid import MuonAdamWHybrid
from olympus.optim.schedulers import WSDScheduler
from olympus.distributed.fsdp_wrapper import FSDPConfig
from olympus.distributed.distributed_orchestrator import DistributedTrainingOrchestrator

logger = logging.getLogger(__name__)


class DistributedBootstrapTrainer:
    """Phase 1 trainer with FSDP multi-GPU support.

    Automatically detects distributed environment from torchrun
    and wraps the model in FSDP when multiple GPUs are available.

    Args:
        model: HLRT model (will be wrapped in FSDP if distributed).
        optimizer_cls: Optimizer class (default MuonAdamWHybrid).
        optimizer_kwargs: Kwargs for optimizer construction.
        scheduler_cls: Scheduler class (default WSDScheduler).
        scheduler_kwargs: Kwargs for scheduler construction.
        tokenizer: TokenizerWrapper instance.
        config: Training config dict.
        fsdp_config: Optional FSDP config (auto-detected if None).
    """

    def __init__(
        self,
        model: HLRT,
        optimizer_cls=MuonAdamWHybrid,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_cls=WSDScheduler,
        scheduler_kwargs: Optional[dict] = None,
        tokenizer: Optional[TokenizerWrapper] = None,
        config: Optional[dict] = None,
        fsdp_config: Optional[FSDPConfig] = None,
    ) -> None:
        config = config or {}
        optimizer_kwargs = optimizer_kwargs or {}
        scheduler_kwargs = scheduler_kwargs or {}

        # Detect distributed environment
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.is_distributed = self.world_size > 1

        if self.is_distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(f"cuda:{self.local_rank}")

        # Auto-detect FSDP config
        if fsdp_config is None and self.is_distributed:
            fsdp_config = FSDPConfig(
                sharding_strategy="FULL_SHARD",
                mixed_precision="bf16",
                activation_checkpointing=True,
            )

        # Build optimizer BEFORE FSDP wrapping (orchestrator rebuilds it after)
        self.model = model.to(self.device)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

        self.tokenizer = tokenizer or TokenizerWrapper(backend="tiktoken")

        # Build distributed orchestrator
        grad_accum = config.get("gradient_accumulation_steps", 4)
        self.orchestrator = DistributedTrainingOrchestrator(
            fsdp_config=fsdp_config if self.is_distributed else None,
            local_rank=self.local_rank,
            world_size=self.world_size,
            gradient_accumulation_steps=grad_accum,
            device=self.device,
        )

        self.orchestrator.add_model(ModelConfig(
            name="hlrt",
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=config.get("max_grad_norm", 1.0),
            enabled_phases=["bootstrap"],
        ))
        self.orchestrator.add_objective(ObjectiveConfig(
            name="ntp",
            compute_fn=ntp_objective,
            weight=1.0,
            enabled_phases=["bootstrap"],
        ))
        self.orchestrator.set_phase("bootstrap")

        # After FSDP wrapping, update model reference
        self.model = self.orchestrator._models["hlrt"].model
        self.scheduler = self.orchestrator._models["hlrt"].scheduler

        self.config = config
        self._tokens_seen: int = 0

        if self.is_main_rank:
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "DistributedBootstrapTrainer: %d GPUs, %.1fM params, "
                "grad_accum=%d, effective_batch=%d",
                self.world_size, param_count / 1e6, grad_accum,
                config.get("batch_size", 32) * grad_accum * self.world_size,
            )

    @property
    def is_main_rank(self) -> bool:
        return self.global_rank == 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training micro-step."""
        step_start = time.time()

        loss_dict = self.orchestrator.step(batch)

        total_loss = loss_dict.get("total", loss_dict.get("ntp", 0.0))
        perplexity = min(math.exp(total_loss), 1e6) if total_loss < 20 else 1e6
        lr = self.scheduler.last_lr if hasattr(self.scheduler, "last_lr") else 0.0

        batch_tokens = batch["input_ids"].numel()
        self._tokens_seen += batch_tokens
        elapsed = time.time() - step_start
        tokens_per_sec = batch_tokens / max(elapsed, 1e-6)

        # Scale tokens/sec by world_size for total throughput
        if self.is_distributed:
            tokens_per_sec *= self.world_size

        return {
            "loss": total_loss,
            "perplexity": perplexity,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with distributed logging."""
        epoch_loss = 0.0
        epoch_tps = 0.0
        num_steps = 0
        log_interval = self.config.get("log_interval", 100)
        save_interval = self.config.get("save_interval", 5000)
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints/phase1")

        epoch_start = time.time()

        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_loss += metrics["loss"]
            epoch_tps += metrics["tokens_per_sec"]
            num_steps += 1

            global_step = self.orchestrator.global_step

            if self.is_main_rank and num_steps % log_interval == 0:
                logger.info(
                    "Step %d | loss=%.4f | ppl=%.2f | lr=%.2e | %.0f tok/s",
                    global_step, metrics["loss"], metrics["perplexity"],
                    metrics["lr"], metrics["tokens_per_sec"],
                )

            if (save_interval > 0 and global_step > 0
                    and global_step % save_interval == 0):
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_step{global_step}.pt")
                )

        avg_loss = epoch_loss / max(num_steps, 1)
        elapsed = time.time() - epoch_start

        if self.is_main_rank:
            logger.info(
                "Epoch complete | avg_loss=%.4f | %d steps in %.1fs",
                avg_loss, num_steps, elapsed,
            )

        return {
            "epoch_loss": avg_loss,
            "epoch_perplexity": min(math.exp(avg_loss), 1e6) if avg_loss < 20 else 1e6,
            "avg_tokens_per_sec": epoch_tps / max(num_steps, 1),
            "total_steps": num_steps,
        }

    def save_checkpoint(self, path: str) -> None:
        self.orchestrator.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.orchestrator.load_checkpoint(path)

    def make_dataloader(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create a DataLoader with DistributedSampler if multi-GPU."""
        sampler = None
        shuffle = True

        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


# ======================================================================
# CLI entry point (for torchrun)
# ======================================================================

def main():
    """Entry point for distributed training via torchrun."""
    parser = argparse.ArgumentParser(description="GENESIS Phase 1 Distributed Training")
    parser.add_argument("--data-dir", type=str, default="packed/",
                        help="Directory with packed binary data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase1",
                        help="Checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {os.environ.get('RANK', 0)}] %(name)s | %(message)s",
    )

    # Build model
    tokenizer = TokenizerWrapper(backend="tiktoken")
    config = HLRTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
    )
    model = HLRT(config)

    # Build trainer
    trainer = DistributedBootstrapTrainer(
        model=model,
        optimizer_cls=MuonAdamWHybrid,
        optimizer_kwargs={
            "lr_muon": args.muon_lr,
            "lr_adamw": args.lr,
            "weight_decay_adamw": 0.01,
            "ns_steps": 5,
        },
        scheduler_cls=WSDScheduler,
        scheduler_kwargs={
            "base_lr": args.lr,
            "min_lr": 2e-5,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.max_steps,
            "decay_start": int(args.max_steps * 0.9),
        },
        tokenizer=tokenizer,
        config={
            "gradient_accumulation_steps": args.grad_accum,
            "max_grad_norm": 1.0,
            "log_interval": args.log_interval,
            "save_interval": args.save_interval,
            "checkpoint_dir": args.checkpoint_dir,
            "batch_size": args.batch_size,
            "device": f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}",
        },
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        if trainer.is_main_rank:
            logger.info("Resumed from step %d", trainer.orchestrator.global_step)

    # Load data
    from olympus.data.packed_dataset import PackedDataset

    train_ds = PackedDataset(args.data_dir, split="train")
    val_ds = PackedDataset(args.data_dir, split="val")

    train_loader = trainer.make_dataloader(
        train_ds, batch_size=args.batch_size, num_workers=4,
    )
    val_loader = trainer.make_dataloader(
        val_ds, batch_size=args.batch_size, num_workers=2,
    )

    if trainer.is_main_rank:
        logger.info("Train: %s", train_ds)
        logger.info("Val: %s", val_ds)
        logger.info(
            "Effective batch: %d (batch=%d x accum=%d x gpus=%d)",
            args.batch_size * args.grad_accum * trainer.world_size,
            args.batch_size, args.grad_accum, trainer.world_size,
        )

    # Training loop
    epoch = 0
    while trainer.orchestrator.global_step < args.max_steps:
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        trainer.orchestrator.set_epoch(epoch)
        metrics = trainer.train_epoch(train_loader)

        # Stop if we've hit max steps
        if trainer.orchestrator.global_step >= args.max_steps:
            break

        epoch += 1

    # Final checkpoint
    trainer.save_checkpoint(
        os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    )

    if trainer.is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
