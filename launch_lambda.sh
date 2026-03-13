#!/bin/bash
# Launch GENESIS training on Lambda Labs
#
# Single GPU:
#   bash launch_lambda.sh
#
# Multi-GPU (auto-detects available GPUs):
#   bash launch_lambda.sh --multi
#
# Resume from checkpoint:
#   bash launch_lambda.sh --resume checkpoints/phase0/best.pt

set -e

CONFIG="genesis/training/configs/phase0_124m_lambda.yaml"
CHECKPOINT_DIR="checkpoints/phase0_124m"
RESUME_ARG=""

# Parse args
MULTI=false
for arg in "$@"; do
    case $arg in
        --multi) MULTI=true ;;
        --resume) shift; RESUME_ARG="--resume $1" ;;
    esac
done

echo "============================================"
echo "  GENESIS Phase 0 — 124M Lambda Training"
echo "============================================"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPUs detected"
echo ""

if [ "$MULTI" = true ]; then
    NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Launching DDP with $NGPU GPUs..."
    torchrun --nproc_per_node=$NGPU \
        -m genesis.training.train_phase0 \
        --config $CONFIG \
        --checkpoint-dir $CHECKPOINT_DIR \
        $RESUME_ARG
else
    echo "Launching single-GPU training..."
    python -m genesis.training.train_phase0 \
        --config $CONFIG \
        --checkpoint-dir $CHECKPOINT_DIR \
        $RESUME_ARG
fi
