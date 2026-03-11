"""Prepare training data: HuggingFace → tokenize → pack into binary format.

Streams data from HuggingFace (no full download), tokenizes with the project
tokenizer, and writes packed binary files compatible with PackedDataset.

Usage:
    python -m genesis.training.prepare_data \
        --config genesis/training/configs/phase0_50m.yaml

    # Or with explicit arguments:
    python -m genesis.training.prepare_data \
        --dataset HuggingFaceFW/fineweb-edu \
        --subset sample-10BT \
        --out-dir data/packed_phase0 \
        --seq-len 2048 \
        --num-tokens 1100000000 \
        --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_tokenizer(vocab_size: int = 32000):
    """Get the best available tokenizer."""
    from olympus.data.tokenizer import TokenizerWrapper

    # For 32k vocab, prefer sentencepiece or char fallback
    # For 100k+ vocab, prefer tiktoken
    if vocab_size <= 50000:
        # Try tiktoken first (it's fast), but the vocab will be larger
        # than requested — that's fine for data packing, we just need
        # consistent token IDs
        tok = TokenizerWrapper(backend="tiktoken", vocab_size=vocab_size)
    else:
        tok = TokenizerWrapper(backend="tiktoken", vocab_size=vocab_size)

    logger.info("Using tokenizer: %s (vocab=%d)", tok.backend_name, tok.vocab_size)
    return tok


def _stream_texts(
    dataset_name: str,
    subset: Optional[str] = None,
    streaming: bool = True,
) -> Iterator[str]:
    """Yield text documents from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is required. Install with:\n"
            "  pip install datasets"
        )
        sys.exit(1)

    logger.info("Loading dataset: %s (subset=%s, streaming=%s)", dataset_name, subset, streaming)

    kwargs = {"split": "train", "streaming": streaming}
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(dataset_name, **kwargs)

    for example in ds:
        text = example.get("text", "")
        if text and len(text) >= 100:  # Skip very short documents
            yield text


def _pack_tokens(
    texts: Iterator[str],
    tokenizer,
    block_size: int,
    max_tokens: int,
    eos_id: int,
) -> np.ndarray:
    """Tokenize texts and pack into fixed-size blocks.

    Documents are concatenated with EOS separators and sliced into
    blocks of exactly `block_size` tokens.
    """
    buffer = []
    blocks = []
    total_tokens = 0

    for text in texts:
        ids = tokenizer.encode(text)
        buffer.extend(ids)
        buffer.append(eos_id)

        # Slice off complete blocks
        while len(buffer) >= block_size:
            block = buffer[:block_size]
            blocks.append(np.array(block, dtype=np.uint32))
            buffer = buffer[block_size:]
            total_tokens += block_size

            if len(blocks) % 5000 == 0:
                logger.info(
                    "Packed %d blocks (%.2fM tokens / %.2fM target)",
                    len(blocks), total_tokens / 1e6, max_tokens / 1e6,
                )

            if total_tokens >= max_tokens:
                break

        if total_tokens >= max_tokens:
            break

    logger.info(
        "Packing complete: %d blocks, %d tokens (%.2fM)",
        len(blocks), total_tokens, total_tokens / 1e6,
    )
    return np.stack(blocks) if blocks else np.empty((0, block_size), dtype=np.uint32)


def prepare(
    dataset_name: str,
    out_dir: str,
    seq_len: int = 2048,
    num_tokens: int = 1_100_000_000,
    val_ratio: float = 0.01,
    subset: Optional[str] = None,
    streaming: bool = True,
    vocab_size: int = 32000,
) -> Path:
    """Full data preparation pipeline.

    Args:
        dataset_name: HuggingFace dataset identifier.
        out_dir: Output directory for packed binary files.
        seq_len: Sequence length (block size) for training.
        num_tokens: Total tokens to pack (train + val).
        val_ratio: Fraction of blocks to hold out for validation.
        subset: Dataset subset/config name.
        streaming: Whether to stream (True) or download fully (False).
        vocab_size: Target vocabulary size for tokenizer selection.

    Returns:
        Path to the output directory.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = _get_tokenizer(vocab_size)
    eos_id = tokenizer.eos_id

    # Stream and pack
    texts = _stream_texts(dataset_name, subset=subset, streaming=streaming)
    all_blocks = _pack_tokens(texts, tokenizer, seq_len, num_tokens, eos_id)

    if len(all_blocks) == 0:
        logger.error("No data was packed. Check dataset name and network connection.")
        sys.exit(1)

    # Split train/val
    n_total = len(all_blocks)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    # Shuffle before splitting
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(n_total)
    train_blocks = all_blocks[indices[:n_train]]
    val_blocks = all_blocks[indices[n_train:]]

    # Write binary files
    train_path = out_path / "train_input_ids.bin"
    val_path = out_path / "val_input_ids.bin"

    train_blocks.tofile(str(train_path))
    val_blocks.tofile(str(val_path))

    # Write metadata
    meta = {
        "block_size": seq_len,
        "train_blocks": n_train,
        "val_blocks": n_val,
        "total_tokens": n_total * seq_len,
        "dataset": dataset_name,
        "subset": subset,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_backend": tokenizer.backend_name,
    }
    meta_path = out_path / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("  Output dir:    %s", out_path)
    logger.info("  Train blocks:  %d (%.2fM tokens)", n_train, n_train * seq_len / 1e6)
    logger.info("  Val blocks:    %d (%.2fM tokens)", n_val, n_val * seq_len / 1e6)
    logger.info("  Block size:    %d", seq_len)
    logger.info("  Tokenizer:     %s (vocab=%d)", tokenizer.backend_name, tokenizer.vocab_size)
    logger.info("=" * 60)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from HuggingFace")
    parser.add_argument("--config", type=str, help="Path to YAML config (reads data section)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="data/packed_phase0")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-tokens", type=int, default=1_100_000_000,
                        help="Total tokens to prepare (default: 1.1B for Chinchilla-optimal 50M)")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--no-streaming", action="store_true",
                        help="Download full dataset instead of streaming")
    args = parser.parse_args()

    # If config provided, read data section from it
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg.get("data", {})
            model_cfg = cfg.get("model", {})

            args.dataset = data_cfg.get("dataset", args.dataset)
            args.subset = data_cfg.get("subset", args.subset)
            args.out_dir = data_cfg.get("packed_dir", args.out_dir)
            args.seq_len = data_cfg.get("max_seq_len", args.seq_len)
            args.vocab_size = model_cfg.get("vocab_size", args.vocab_size)
            if data_cfg.get("val_ratio"):
                args.val_ratio = data_cfg["val_ratio"]
            if data_cfg.get("streaming") is False:
                args.no_streaming = True
        except ImportError:
            logger.warning("PyYAML not installed, ignoring --config. Install with: pip install pyyaml")

    prepare(
        dataset_name=args.dataset,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        num_tokens=args.num_tokens,
        val_ratio=args.val_ratio,
        subset=args.subset,
        streaming=not args.no_streaming,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
