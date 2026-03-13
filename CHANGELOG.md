# Changelog — Project GENESIS

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

---

## [Unreleased]

### Added
- Lambda H100 speed optimizations in training pipeline:
  - TF32 tensor cores (auto-enabled on CUDA)
  - `torch.compile` model compilation (configurable via `use_torch_compile`)
  - DDP multi-GPU support via `torchrun` with `DistributedDataParallel`
  - DataLoader prefetch (`prefetch_factor=2`) for async data loading
  - Gradient norm logging for spike detection
  - `zero_grad(set_to_none=True)` for faster gradient clearing
- SDPA attention backend: `_vanilla_attention` now uses `F.scaled_dot_product_attention` on CUDA/MPS — gets flash attention without the external `flash_attn` package
- 124M Lambda training config (`genesis/training/configs/phase0_124m_lambda.yaml`):
  - GPT-2 124M scale validation run (~$10 on H100)
  - Vocab padded to 100,352 (128-aligned for GPU kernel efficiency)
  - GPT-3 hyperparams: LR 6e-4, beta2=0.95, weight_decay=0.1, 715 warmup steps
- Launch script (`launch_lambda.sh`) for single-GPU and multi-GPU DDP training
- `_unwrap_model()` helper for clean checkpoint save/resume through DDP + torch.compile wrappers
- Phase 0 M1 Mac training pipeline (`genesis/training/train_phase0.py`, `genesis/training/prepare_data.py`)
- 50M param config for M1 Mac (`genesis/training/configs/phase0_50m_mac.yaml`)
- TierComposer: LEGO-style tier composition for building models from independently trained tiers (`genesis/model/tier_composer.py`)
- MPS device support in training pipeline (fallback detection, AMP disabled)
- Eval loop batch cap (100 batches) for reasonable eval times on small hardware
- SFT fine-tuning pipeline (`genesis/training/train_sft.py`) — instruction masking, GENESIS special token formatting
- GirlyPopQuartz data converter (`convert_gpq_to_sft.py`) — 45K social/conversational examples
- Combined SFT dataset: 70K examples (Platypus reasoning + GirlyPopQuartz social)
- Generation sampling scripts (`sample_phase0.py`, `sample_sft.py`)
- Scheduler state persistence in checkpoints (save + restore on resume)

### Changed
- Extended Phase 0 pretraining from 10K to 50K steps (val_loss 9.57 → 5.84)
- Updated config for 50K steps with corrected scheduler parameters

### Fixed
- `torch.var()` NaN on MPS in `TierGate.load_balance_loss()` — replaced with manual variance
- CPU/MPS device mismatch in `Tier2SemanticPlanner.load_balance_loss()` — eliminated CPU tensor creation
- OOM on M1 16GB with 100K vocab — reduced batch to 1, seq_len to 1024, disabled DataLoader workers
- WSD scheduler `total_steps` counted micro-steps instead of optimizer steps — LR never reached decay phase

### Previously added
- Olympus training framework: StatefulModule, MemoryBus, ComputeRouter, TrainingOrchestrator, GrowthController
- Muon optimizer + MuonAdamWHybrid + WSD scheduler
- 6 Triton kernels with PyTorch fallbacks (FP4 quantize/matmul, fused gate routing, sparse expert matmul, memory cross-attention, Muon step)
- Data pipeline: tokenizer, curriculum, memory-aware batcher, quality filter, flywheel buffer
- Full utils suite: checkpointing, config, logging, metrics, profiling, seeds
- HLRT 3-tier model architecture (token processor, semantic planner, deliberative reasoner)
- Tier gating, latent pooling, conditioning, MoE routing
- PHMA memory system (working, episodic, semantic + controller + consistency)
- ACT-V verifier (model, verification head, negative generator, distillation, replay buffer)
- Training pipelines: Phase 1 bootstrap, Phase 2 RL/GRPO, Phase 3 flywheel, ACT-V co-training
- GRPO (Group Relative Policy Optimization) + composable reward functions
- Unit tests for Olympus core, kernels, and GENESIS model components
- Development agent role cards (architect, debug, test, integration, completeness, documentation, performance)
- Project documentation: GENESIS_BLUEPRINT.md, PROPOSAL.md
- Documentation scaffold: CLAUDE.md, BUILD_PLAN.md, PROGRESS.md, BUGS.md, CHANGELOG.md, FILE_MAP.md
- CUDA inference pipeline: Generator (batch + streaming), KVCache, Sampling (top-k/top-p/temperature/repetition penalty), CascadeRouter with tier activation stats
- Tier 3 Deliberative Reasoner scaled from 4 to 5 layers (483M -> 496M total params)
- 35 unit tests for inference pipeline (all passing)
- KV cache wired into MultiHeadAttention, Tier1TokenProcessor, and HLRT forward pass

- 50 integration tests covering all major subsystems (all passing)
- HLRT now returns `hidden_states` in forward output for distillation compatibility

### Fixed
- KV-cached generation now matches uncached output (skip tier gating during single-token decode, reuse prefill plan vector)
- Greedy generation determinism (reset stateful plan vector between runs)
- ACT-V replay buffer stored full batches instead of individual sequences (3D tensor on stack)
- FlywheelTrainer `_mix_with_buffer` ignored `trace_mix_ratio=0` (always replaced at least 1 sample)
