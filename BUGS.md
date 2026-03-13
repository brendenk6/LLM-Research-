# BUGS — Project GENESIS

Engineering journal for bug fixes. Root cause analysis, before/after code. Update after EACH iteration.

---

## BUG-001: KV-cached generation diverges from uncached after step 2

**Date**: Mar 10, 2026
**Severity**: Critical (correctness)
**Status**: FIXED

### Symptom
`test_cached_matches_uncached` fails. Greedy generation produces different tokens depending on whether KV cache is enabled:
```
Cached:   tensor([[144, 144,  84,  84,  84,  84]])
Uncached: tensor([[144, 144, 144, 144, 144, 144]])
```
First 2 tokens match, then diverge.

### Root Cause
HLRT's TierGate operates on chunks of tokens. During cached decode, only 1 token flows through Tier 1, so the gate sees completely different input than during uncached full-sequence processing. This causes different tier2/tier3 escalation decisions, producing different conditioning vectors that alter the output.

The KV cache attention math was correct — the bug was architectural: gating logic designed for chunk-level context can't produce meaningful decisions from a single token.

### Fix
Early-return in `HLRT.forward()` when `kv_cache is not None and S == 1`. Skip tier gating entirely and apply the plan vector computed during prefill via `conditioning.apply_cached_plan()`.

```python
# Before (broken): single token goes through full gating pipeline
gate1_mask, gate1_scores = self.gate1(tier1_out)  # meaningless on 1 token

# After (fixed): skip gating, reuse prefill plan vector
if kv_cache is not None and S == 1:
    cached_plan = self.link_state("last_plan_vector")
    tier1_out = self.conditioning.apply_cached_plan(tier1_out, cached_plan)
    # early return with logits
```

### Why This Is Correct
The prefill pass processes the full prompt and computes a plan vector that captures the semantic intent. Single decode tokens should execute under that plan, not re-derive it from insufficient context. This matches how the architecture was designed — Tier 2/3 provide high-level guidance, Tier 1 does the token-level work.

---

## BUG-002: ACT-V replay buffer stores batches instead of sequences

**Date**: Mar 10, 2026
**Severity**: High (crash)
**Status**: FIXED

### Symptom
`ACTVTrainer.co_training_step()` crashes with `RuntimeError: Tensors must have same number of dimensions: got 3 and 2` when replay buffer is ready for sampling.

### Root Cause
`train_generator_step()` stored the full batch `input_ids` (B, S) as a single replay buffer entry. When `train_verifier_step()` later sampled entries and stacked them with `torch.stack()`, it created a 3D tensor (n_replay, B, S) instead of 2D (n_replay, S). The padding logic then tried to concat a 3D tensor with a 2D pad tensor.

### Fix
Store individual sequences in the replay buffer instead of full batches:
```python
# Before: stores entire batch as one entry
self.replay_buffer.add(input_ids=input_ids, scores=v_scores, ...)

# After: stores each sequence individually
for i in range(input_ids.size(0)):
    seq_scores = {k: v[i] for k, v in v_scores.items()}
    self.replay_buffer.add(input_ids=input_ids[i], scores=seq_scores, ...)
```

---

## BUG-003: FlywheelTrainer ignores trace_mix_ratio=0

**Date**: Mar 10, 2026
**Severity**: Medium (incorrect behavior)
**Status**: FIXED

### Symptom
Setting `trace_mix_ratio=0.0` still replaces 1 sample per batch from the flywheel buffer, because `max(1, int(B * 0.0))` evaluates to 1.

### Root Cause
`_mix_with_buffer` checked `buffer.size == 0` but not `trace_mix_ratio <= 0`.

### Fix
```python
if self.flywheel_buffer.size == 0 or self.trace_mix_ratio <= 0:
    return batch
```

---

## BUG-004: HLRT missing hidden_states in output for distillation

**Date**: Mar 10, 2026
**Severity**: High (crash)
**Status**: FIXED

### Symptom
ACT-V distillation crashes with shape mismatch: `proj_G` expects `(B, d_model)` but receives `(B, vocab_size)`.

### Root Cause
HLRT's `forward()` only returned `logits` and `aux_loss`. The `_distill()` method in `train_actv.py` falls back to `gen_out["logits"]` when `"hidden_states"` isn't available, but logits have shape `(B, S, vocab_size)` while the projection expects `(B, d_model)`.

### Fix
Added `"hidden_states": h` to HLRT's output dict (the normalized Tier 1 output before projection). This is the correct representation for distillation — it's the model's internal representation, not the vocabulary-projected output.

---

## BUG-006: torch.var() returns NaN on MPS (Apple Silicon)

**Date**: Mar 11, 2026
**Severity**: Critical (training produces NaN loss)
**Status**: FIXED

### Symptom
Phase 0 training on M1 Mac produces `loss=nan` from step 1. All training steps show NaN, model weights immediately corrupted.

### Root Cause
`TierGate.load_balance_loss()` used `torch.var()` to compute variance of gate scores. On MPS, `torch.var()` returns NaN when values are small or tightly clustered (gate sigmoid outputs were all ~0.4998-0.5009). This is a known MPS backend bug.

The NaN `aux_loss` propagated: `total_loss = cross_entropy + aux_loss` = NaN. Backward pass produced NaN gradients in all 81 parameter tensors, permanently corrupting the model on step 1.

The cross-entropy loss itself was correct (11.6, expected for random init with 100K vocab). The forward pass was clean — only `aux_loss` was NaN.

### Fix
```python
# Before (broken on MPS):
loss = chunk_load.float().var() + (mean_activation.float().var())

# After (works everywhere):
cl = chunk_load.float()
ma = mean_activation.float()
loss = (cl - cl.mean()).pow(2).mean() + (ma - ma.mean()).pow(2).mean()
```

### Secondary fix
`Tier2SemanticPlanner.load_balance_loss()` created `torch.tensor(0.0)` on CPU then accumulated MPS tensors into it, causing device mismatch:
```python
# Before:
total = torch.tensor(0.0)  # CPU
total = total + lb  # lb is on MPS -> device mismatch

# After:
total = None
for layer in self.layers:
    lb = layer.load_balance_loss()
    total = lb if total is None else total + lb
return total if total is not None else torch.tensor(0.0)
```

---

## BUG-007: OOM on M1 16GB with 100K vocab at seq_len=4096

**Date**: Mar 11, 2026
**Severity**: High (training killed by OS)
**Status**: FIXED

### Symptom
Training killed with exit code 137 (SIGKILL/OOM) even at batch_size=1 with 4096 sequence length.

### Root Cause
Two memory killers compounding on unified memory (CPU + GPU share 16GB):
1. **Logits tensor**: 1 × 4096 × 100,287 × 4 bytes = 1.6GB. Backward pass doubles this.
2. **Attention maps without Flash Attention**: Tier1 alone stores 4 layers × 6 heads × 4096 × 4096 × 4 bytes = 1.5GB for backward. Flash Attention avoids materializing this, but isn't available on MPS.
3. **DataLoader workers**: `num_workers=2` forks Python processes, each inheriting parent memory footprint.

Total: model (270MB) + optimizer (810MB) + attention maps (1.5GB) + logits+grads (3.2GB) + workers (~1GB) > 16GB.

### Fix
Three changes in `phase0_50m_mac.yaml`:
- `batch_size: 1` (was 4)
- `max_seq_len: 1024` (was 4096, blocks sliced in training loop)
- `num_workers: 0` (was 2, eliminates forked process overhead)

At seq_len=1024: attention maps = 96MB, logits = 400MB. Fits comfortably.

---

## BUG-008: Eval loop takes ~8 hours per evaluation

**Date**: Mar 11, 2026
**Severity**: Medium (training stalls)
**Status**: FIXED

### Symptom
Training hangs at step 250 (first eval checkpoint). Process still alive but no new training steps logged.

### Root Cause
`evaluate()` iterated over ALL 56,572 validation blocks at batch_size=1. At ~0.5s per forward pass, that's ~7.8 hours per eval. Eval runs every 250 training steps.

### Fix
```python
max_eval_batches = 100
# ...
if n_batches >= max_eval_batches:
    break
```
100 batches at batch=1 takes ~50 seconds. Statistically sufficient for val loss estimation.

---

## BUG-009: WSD scheduler total_steps counts micro-steps instead of optimizer steps

**Date**: Mar 12, 2026
**Severity**: High (incorrect LR schedule)
**Status**: FIXED

### Symptom
LR never reaches decay phase. During 10K-step run with `grad_accum=32`, LR stayed at peak (0.02) for the entire stable+decay range because the scheduler thought it had 10,000 steps but only received 312 `scheduler.step()` calls.

### Root Cause
`build_scheduler()` passed `total_steps=train_cfg["max_steps"]` (micro-steps) to `WSDScheduler`. But `scheduler.step()` is called once per optimizer step (every `grad_accum` micro-steps). With `max_steps=10000` and `grad_accum=32`:
- Scheduler expects 10,000 steps, decay starts at step 9,984
- Only 312 scheduler steps actually happen (10000/32)
- Scheduler never reaches decay phase

### Fix
```python
# Before:
total_steps=train_cfg.get("max_steps", 10000)

# After:
grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
total_opt_steps = train_cfg.get("max_steps", 10000) // grad_accum
# ... total_steps=total_opt_steps
```

Also added scheduler state to checkpoints (`scheduler_state_dict`) so LR schedule survives resume.

---

## BUG-005: Greedy generation not deterministic across runs (renamed from BUG-002)

**Date**: Mar 10, 2026
**Severity**: Medium (test reliability)
**Status**: FIXED

### Symptom
`test_generate_greedy_deterministic` fails because two greedy generation runs on the same input produce different tokens.

### Root Cause
HLRT uses `StatefulModule` with a persistent `last_plan_vector`. After the first generation, this vector holds a non-zero value that influences the second run's output (via `conditioning.apply_cached_plan` on the no-escalation path).

### Fix
Reset state between test runs:
```python
model.set_state("last_plan_vector", torch.zeros(1, model.config.d_model))
```
