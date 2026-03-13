# PROGRESS — Project GENESIS

Session-level work tracking. Most recent session first.

---

## Session 6 — Mar 12, 2026
**Focus**: Lambda H100 speed optimizations + 124M training config

### Done
- Added 7 speed optimizations to `train_phase0.py` (all backward-compatible with MPS):
  - TF32 tensor cores (`set_float32_matmul_precision("high")`)
  - `torch.compile` model compilation (CUDA only, ~2x expected speedup)
  - Full DDP multi-GPU support via `torchrun` (DistributedDataParallel, DistributedSampler, rank-guarded logging/saving, distributed val loss averaging)
  - DataLoader prefetch for async data loading
  - Gradient norm logging (track `grad_norm` every log step)
  - `zero_grad(set_to_none=True)` micro-optimization
  - `_unwrap_model()` for clean save/resume through DDP + torch.compile wrappers
- Upgraded `_vanilla_attention` to use `F.scaled_dot_product_attention` on CUDA/MPS — gets PyTorch's built-in flash attention kernel without the external `flash_attn` package. Manual fallback only for MPS decode or CPU.
- Created 124M Lambda config (`phase0_124m_lambda.yaml`):
  - Tier 1: 8L/8H/512D, Tier 2: 4L/8H/768D, Tier 3: 2L/8H/512D
  - Vocab padded 100,287 → 100,352 (128-aligned for GPU kernel efficiency)
  - GPT-3 hyperparams: adamw_lr=6e-4, beta2=0.95, weight_decay=0.1
  - 40K steps, ~20B tokens target, ~$10 on single H100
- Created `launch_lambda.sh` (single-GPU and multi-GPU DDP launch)
- Analyzed Karpathy's "Let's reproduce GPT-2 (124M)" transcript for training best practices

### Training ladder plan
1. **124M** → pipeline validation, match Karpathy's val loss (~$10)
2. **500M** → real capacity test, see if HLRT gating helps (~$150-200)
3. **1B** → the real thing, if 500M proves the architecture (~$400-700)

### Status
- [x] Speed optimizations (TF32, compile, DDP, SDPA, prefetch)
- [x] 124M Lambda config
- [x] Launch script
- [ ] Run 124M on Lambda H100
- [ ] Run 500M on Lambda H100
- [ ] Run 1B on Lambda H100

## Session 5 — Mar 12, 2026
**Focus**: Extended pretraining + SFT fine-tuning pipeline

### Done
- Extended pretraining from 10K to 50K steps (val_loss 9.57 → 5.84, PPL 14,362 → 345)
  - 6.6 hours wall time, 41M tokens seen, ~1,729 tok/s at steady state
  - Loss curve still descending at end — no plateau
- Fixed WSD scheduler bug (BUG-009): `total_steps` was micro-steps but scheduler.step() called per optimizer step
  - `build_scheduler` now divides `max_steps` by `gradient_accumulation_steps`
- Added scheduler state to checkpoints (save + restore on resume)
- Built SFT training pipeline (`genesis/training/train_sft.py`):
  - Instruction masking: loss only on response tokens (IGNORE_INDEX=-100)
  - Format: `<|bos|> instruction <|reason_start|> output <|reason_end|> <|eos|>`
  - AdamW at 2e-5 LR, grad_accum=8, batch=1
- Converted GirlyPopQuartz social data for GENESIS SFT (`convert_gpq_to_sft.py`):
  - 45,225 examples from 9 files (multiturn, reflect, clarify, chat, voice_style, etc.)
  - Stripped GPQ special tokens, extracted instruction/output, re-formatted
- Combined SFT dataset: 70,151 examples (24.9K Platypus reasoning + 45.2K social)
- SFT training complete: 3K steps, 27 min, val_loss 5.06 (PPL 157)
  - Steady improvement throughout, no overfitting
- Generation testing: model produces grammatical English with SFT formatting cues
  - Still repetition-prone and semantically weak at 67M params — expected
  - Full pipeline validated: pretrain → extend → SFT → generate

### Status
- [x] Extended pretraining (50K steps, val_loss 5.84)
- [x] Fix scheduler total_steps bug
- [x] Build SFT pipeline
- [x] Convert GirlyPopQuartz social data
- [x] SFT training (70K examples, val_loss 5.06)
- [x] Generation testing (pretrained + SFT)
- [ ] Analyze gate routing behavior in detail
- [ ] Consider further pretraining (100K+ steps) or move to Lambda 1B

## Session 4 — Mar 11, 2026
**Focus**: Phase 0 M1 Mac training run (architecture validation)

### Done
- Pulled `claude/small-model-modular-test-ySo07` branch from GitHub (Phase 0: 50M config, TierComposer, data pipeline, Colab notebook)
- Created M1 Mac training config (`genesis/training/configs/phase0_50m_mac.yaml`):
  - 67.4M params (Tier1: 48M, Tier2: 14M, Tier3: 5M), 100287 vocab, 1024 seq len
  - batch=1, grad_accum=32, MPS device, float32, no flash attention, no DataLoader workers
- Fixed 3 MPS-specific bugs:
  - BUG-006: `torch.var()` NaN on MPS — manual variance in `tier_gate.py`
  - BUG-007: OOM at 4096 seq len — reduced to 1024, batch=1, workers=0
  - BUG-008: 8-hour eval loop — capped at 100 batches
- Fixed MPS device detection in `train_phase0.py`
- Fixed GradScaler device type (hardcoded "cuda" in autocast)
- Fixed CPU/MPS tensor mismatch in `tier2_semantic_planner.py`
- Pretraining complete: 10K steps, 1.8 hrs, val_loss 9.57 (PPL 14,362), ~1,760 tok/s

### Status
- [x] Pull Phase 0 branch
- [x] Create M1 Mac config
- [x] Fix MPS compatibility bugs
- [x] Initial pretraining (10K steps)
- [x] Generation testing (word salad at val_loss 9.57 — expected)

## Session 1 — Mar 10, 2026
**Focus**: Repository setup + documentation scaffold

### Done
- Cloned repo from GitHub (`brendenk6/LLM-Research-`)
- Switched to `claude/llm-training-brainstorm-hT5lb` branch (main work branch)
- Audited full codebase: 27K+ lines across 89 Python files
- Created project documentation suite:
  - CLAUDE.md (project instructions)
  - BUILD_PLAN.md (9-phase master plan with checkboxes)
  - PROGRESS.md (this file)
  - BUGS.md (engineering journal)
  - CHANGELOG.md (user-facing changes)
  - FILE_MAP.md (codebase guide)
  - Updated README.md

### Status Assessment
- **Phases 1-2 COMPLETE**: Olympus framework + GENESIS model/memory/verifier/training all implemented
- **Phase 3+ NOT STARTED**: Distributed, vision, inference, conversion, MLX all empty stubs
- **Next up**: Phase 3 (DiLoCo distributed training) or Phase 7 (CUDA inference pipeline) — depends on priority

## Session 3 — Mar 10, 2026
**Focus**: Phase 8 integration tests + bug fixes

### Done
- Wrote 50 integration tests across 13 test classes:
  - Full training step (BootstrapTrainer → Orchestrator → HLRT)
  - ACT-V adversarial co-training loop (Generator + Verifier + distillation)
  - Progressive growth scheduling + adaptive triggers
  - Flywheel cycle (generate traces → score → store → retrain)
  - Memory persistence across sequences (working, episodic, semantic, controller)
  - Tier routing activation rate benchmarks
  - Optimizer comparison (Muon vs AdamW convergence)
  - Memory overhead benchmark
  - Reward function composition
  - Data pipeline (quality filter + batcher + tokenizer + buffer)
  - Checkpoint save/load roundtrip
  - Train-then-generate pipeline
  - WSD scheduler integration
- Fixed 3 pre-existing bugs found by integration tests:
  - BUG-002: ACT-V replay buffer stored batches, not sequences
  - BUG-003: FlywheelTrainer ignored trace_mix_ratio=0
  - BUG-004: HLRT missing hidden_states output for distillation
- All 85 tests pass (50 integration + 35 unit) in 3.65s on CPU

## Session 2 — Mar 10, 2026
**Focus**: KV cache integration + inference pipeline completion

### Done
- Built CUDA inference pipeline from scratch (4 files, ~630 lines):
  - `genesis/inference/kv_cache.py` — LayerKVCache + KVCache with memory estimation
  - `genesis/inference/sampling.py` — temperature, top-k, top-p, repetition penalty
  - `genesis/inference/cascade_router.py` — tier activation tracking + threshold overrides
  - `genesis/inference/generator.py` — batch + streaming generation with KV cache
- Wired KV cache into `MultiHeadAttention.forward()`, `Tier1TokenProcessor.forward()`, `HLRT.forward()`
- Fixed KV-cached decode: skip tier gating on single-token decode, reuse prefill plan vector (BUG-001)
- Fixed greedy determinism: reset `last_plan_vector` state between runs (BUG-005)
- Scaled Tier 3 from 4 to 5 layers (483M → 496M params)
- Wrote 35 unit tests (all passing)

### Notes
- All Triton kernels have PyTorch fallback paths — can test locally without GPU
- GrowthController architecture ops are intentionally stubbed for subclass override
- The `agents/` directory has role cards (markdown), not executable agents
