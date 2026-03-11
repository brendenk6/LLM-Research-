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
