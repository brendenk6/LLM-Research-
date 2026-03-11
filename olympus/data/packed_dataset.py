"""Memory-mapped dataset for reading pre-packed binary token blocks.

Reads uint32 binary files written by genesis-pack (Rust).
Format: flat array of uint32, every `block_size` tokens = 1 training sample.

Usage:
    from olympus.data.packed_dataset import PackedDataset
    ds = PackedDataset("packed/", split="train")
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for batch in loader:
        input_ids = batch["input_ids"]  # (B, block_size) long tensor
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """Memory-mapped dataset over pre-packed token blocks.

    Args:
        data_dir: Path to directory containing ``*_input_ids.bin`` and ``meta.json``.
        split: ``"train"`` or ``"val"``.
    """

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        data_dir = Path(data_dir)

        meta_path = data_dir / "meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.block_size: int = self.meta["block_size"]

        if split == "train":
            self.n_blocks = self.meta["train_blocks"]
        elif split == "val":
            self.n_blocks = self.meta["val_blocks"]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        bin_path = data_dir / f"{split}_input_ids.bin"
        # Memory-map: zero RAM, random access
        self.data = np.memmap(
            bin_path, dtype=np.uint32, mode="r",
            shape=(self.n_blocks, self.block_size),
        )

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx].astype(np.int64)
        return {"input_ids": torch.from_numpy(tokens)}

    def __repr__(self) -> str:
        return (
            f"PackedDataset(blocks={self.n_blocks}, "
            f"block_size={self.block_size}, "
            f"tokens={self.n_blocks * self.block_size:,})"
        )
