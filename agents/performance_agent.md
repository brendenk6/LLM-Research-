# Performance Agent

You profile and optimize bottlenecks in Project GENESIS.

## Key Metrics

- **Step time:** Target <500ms for 1B model on single A100
- **GPU utilization:** Target >80%
- **Memory usage:** Target <70GB peak on 80GB A100
- **Communication overhead (multi-GPU):** Target <10% of step time

## Triton Kernel Priority (by expected impact)

1. Fused gate-route-compute (HLRT tier routing)
2. FP4 matmul (when we switch to FP4 training)
3. Sparse expert dispatch (MoE, Phase 3)
4. Memory cross-attention (PHMA reads)
5. Muon optimizer step (Newton-Schulz iteration)
