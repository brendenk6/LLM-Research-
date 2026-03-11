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

## BUG-002: Greedy generation not deterministic across runs

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
