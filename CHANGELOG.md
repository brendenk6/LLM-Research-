# Changelog — Project GENESIS

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

---

## [Unreleased]

### Added
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

### Fixed
- KV-cached generation now matches uncached output (skip tier gating during single-token decode, reuse prefill plan vector)
- Greedy generation determinism (reset stateful plan vector between runs)
