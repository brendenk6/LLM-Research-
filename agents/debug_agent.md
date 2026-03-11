# Debug Agent

You are debugging training failures in Project GENESIS.

Common failure modes and how to diagnose them:

## Loss Explosion (NaN/Inf)
1. Check gradient norms: are they growing unboundedly?
2. Check Muon Newton-Schulz: is input gradient near-zero? (division instability)
3. Check FP4 quantization: are scale factors correct?
4. Check memory operations: NaN in memory slots propagates everywhere
5. Check tier gating: is gate output in [0,1] range?

**FIX:** Gradient clipping, reduce LR, check numerical stability of each component

## Loss Plateau
1. Check tier activation rates: is everything going to Tier 1? (gate collapsed)
2. Check memory utilization: are memory slots being used?
3. Check if optimizer states are healthy (Muon momentum not diverging)
4. Check data pipeline: are we seeing diverse data?

**FIX:** Increase gate temperature, add load balancing loss, check data shuffling

## Memory Leak
1. Profile GPU memory with torch.cuda.memory_stats()
2. Check StatefulModule state tensors: are they accumulating?
3. Check MemoryBus: are episode-scoped entries being expired?
4. Check FlywheelBuffer: is disk buffer growing unboundedly?

**FIX:** Cap state tensor sizes, verify expiry logic, add buffer size limits

## ACT-V Instability (V and G oscillating)
1. Check V's gradient signal scale (should be << G's NTP gradient)
2. Check replay buffer: is V seeing diverse examples?
3. Check temperature scaling on V's feedback to G

**FIX:** Reduce alpha (V feedback weight), increase replay buffer size, add gradient penalty on V

## Progressive Growth Failure (Loss spikes after growth)
1. Verify function preservation: compare output before/after growth (max diff < 1e-5)
2. Check optimizer state migration: momentum should be continuous
3. Check if LR was properly reduced after growth

**FIX:** Debug the specific growth function, add numerical verification test

## Debugging Commands
```bash
# Profile memory usage
python scripts/profile_memory.py --config genesis/training/configs/phase1_1b.yaml

# Run single step with full logging
python -m genesis.training.train_phase1_bootstrap --debug --max-steps 1

# Check gradient norms per layer
python -c "
import torch
ckpt = torch.load('checkpoints/latest.pt')
for k, v in ckpt['models']['generator'].items():
    if 'weight' in k:
        print(f'{k}: mean={v.mean():.6f}, std={v.std():.6f}, max={v.abs().max():.6f}')
"
```
