# Project GENESIS: Complete Implementation Blueprint

## For Claude Code Agents

**Version:** 1.0
**Date:** March 10, 2026
**Author:** Brenden (Human Architect) + Claude (AI Co-Architect)
**Purpose:** This document is the SOLE source of truth for implementing Project GENESIS. Every component, file, function, test, and dependency is specified here. If something is not in this document, it does not exist in the project. If something IS in this document, it MUST be implemented exactly as described.

-----

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
1. [Repository Structure](#2-repository-structure)
1. [Environment Setup](#3-environment-setup)
1. [Olympus Framework (Custom Training Runtime)](#4-olympus-framework)
1. [GENESIS Model Architecture (HLRT)](#5-genesis-model-architecture)
1. [PHMA Memory System](#6-phma-memory-system)
1. [ACT-V Verifier Co-Training](#7-act-v-verifier-co-training)
1. [Training Objectives and RL Flywheel](#8-training-objectives-and-rl-flywheel)
1. [Optimization Stack (Muon + FP4 + Progressive Growing)](#9-optimization-stack)
1. [Distributed Training (DiLoCo)](#10-distributed-training)
1. [Vision Integration](#11-vision-integration)
1. [MLX Inference Runtime](#12-mlx-inference-runtime)
1. [Data Pipeline](#13-data-pipeline)
1. [Testing and Validation](#14-testing-and-validation)
1. [Agent System](#15-agent-system)
1. [Concrete Build Order](#16-concrete-build-order)
1. [Cost and Compute Budget](#17-cost-and-compute-budget)
1. [Troubleshooting Guide](#18-troubleshooting-guide)

-----

## 1. PROJECT OVERVIEW

### 1.1 What We Are Building

Project GENESIS (Generative Engine with Networked Expert Systems, Iterative Self-improvement) is a frontier-class LLM trained from scratch using seven composable innovations:

1. **HLRT** (Hierarchical Latent Reasoning Transformer): A 3-tier variable-compute architecture
1. **Reinforcement Pretraining Flywheel**: RL-based training from day 1 with self-generating data
1. **ACT-V** (Adversarial Co-Training with Verifier): Built-in hallucination resistance
1. **PHMA** (Persistent Hierarchical Memory Architecture): 3-level differentiable memory
1. **Optimization Stack**: Muon + FP4 + Progressive Growing for 3-4x compute savings
1. **DiLoCo Distribution**: Geo-distributed training on consumer/cloud hardware
1. **Compound Inference Cascade**: Tiered inference at 5-8x less average compute

All of this runs on **Olympus**, a custom training framework built as a superset of PyTorch that adds the missing primitives (stateful modules, dynamic routing, multi-model orchestration, progressive growth, memory bus).

### 1.2 Training vs Inference Split

- **Training:** Cloud CUDA hardware (Lambda Labs or similar). PyTorch + Olympus. Multi-GPU (8x A100 or 8x H100).
- **Inference:** Apple Silicon (M1 16GB target). MLX framework. Quantized to Q4/Q6.

### 1.3 Scale Targets

|Phase           |Model Size             |Active Params|Hardware|Duration|
|----------------|-----------------------|-------------|--------|--------|
|Phase 1 (PoC)   |1B total               |1B (dense)   |1x A100 |4 weeks |
|Phase 2 (Scale) |7B total               |7B (dense)   |8x A100 |6 weeks |
|Phase 3 (MoE)   |40B total (MoE)        |8B active    |8x H100 |8 weeks |
|Phase 4 (Vision)|+400M vision encoder   |8B + 400M    |8x A100 |4 weeks |
|Phase 5 (MLX)   |Same weights, quantized|8B active Q4 |M1 16GB |4 weeks |

### 1.4 Key Design Principles

- **Olympus wraps PyTorch, never forks it.** All existing CUDA kernels, FlashAttention, NCCL remain untouched. We add new abstractions on top.
- **Every component is testable in isolation.** Each of the 7 innovations has its own test suite and benchmark.
- **Progressive complexity.** Start with the simplest version of each component, validate, then add complexity.
- **Cloud train, local deploy.** Training happens on CUDA. Inference happens on MLX. The conversion pipeline is a first-class citizen.

-----

## 2. REPOSITORY STRUCTURE

```
genesis/
|
|-- README.md                          # Project overview, quickstart
|-- pyproject.toml                     # Python package config (PEP 621)
|-- setup.cfg                          # Package metadata
|-- Makefile                           # Common commands (train, test, lint, convert)
|-- .github/
|   |-- workflows/
|       |-- ci.yml                     # CI pipeline (lint, test, type-check)
|       |-- gpu-test.yml               # GPU-specific tests (runs on self-hosted runner)
|
|-- olympus/                           # === OLYMPUS FRAMEWORK ===
|   |-- __init__.py                    # Public API exports
|   |-- version.py                     # Semantic versioning
|   |
|   |-- core/                          # Core abstractions
|   |   |-- __init__.py
|   |   |-- stateful_module.py         # StatefulModule base class
|   |   |-- memory_bus.py              # Cross-module communication channel
|   |   |-- compute_router.py          # Dynamic routing with compiled fast paths
|   |   |-- training_orchestrator.py   # Multi-model training loop
|   |   |-- growth_controller.py       # Progressive model expansion
|   |   |-- training_context.py        # Per-step metadata (batch idx, epoch, loss, etc.)
|   |
|   |-- optim/                         # Custom optimizers
|   |   |-- __init__.py
|   |   |-- muon.py                    # Muon optimizer (Newton-Schulz orthogonalization)
|   |   |-- muon_adamw_hybrid.py       # Hybrid: Muon for 2D, AdamW for 1D params
|   |   |-- schedulers.py             # WSD (Warmup-Stable-Decay) + growth-aware scheduling
|   |
|   |-- kernels/                       # Custom CUDA/Triton kernels
|   |   |-- __init__.py
|   |   |-- fp4_matmul.py              # FP4 quantized matrix multiplication (Triton)
|   |   |-- fp4_quantize.py            # FP4 quantize/dequantize ops
|   |   |-- fused_gate_route.py        # Fused gate-evaluate + route + compute (Triton)
|   |   |-- sparse_expert_matmul.py    # MoE expert selection + batched matmul (Triton)
|   |   |-- memory_cross_attention.py  # Optimized cross-attention for memory reads (Triton)
|   |   |-- muon_step.py              # Fused Muon optimizer step (Triton)
|   |
|   |-- distributed/                   # DiLoCo and multi-node training
|   |   |-- __init__.py
|   |   |-- diloco.py                  # DiLoCo outer optimizer (Nesterov momentum)
|   |   |-- island.py                  # Single compute island abstraction
|   |   |-- coordinator.py             # Global coordination via Hivemind DHT
|   |   |-- sparse_sync.py            # Sparse gradient synchronization (MoE-aware)
|   |   |-- quantized_comm.py          # 4-bit gradient quantization for communication
|   |   |-- fault_tolerance.py         # Island failure detection and recovery
|   |
|   |-- data/                          # Data pipeline components
|   |   |-- __init__.py
|   |   |-- quality_filter.py          # FineWeb-style quality classifier
|   |   |-- memory_aware_batcher.py    # Clusters related documents for episodic memory
|   |   |-- flywheel_buffer.py         # Stores and serves successful reasoning traces
|   |   |-- curriculum.py              # Training curriculum (phase transitions, data mixing)
|   |   |-- tokenizer.py              # BPE tokenizer wrapper (tiktoken or sentencepiece)
|   |
|   |-- utils/                         # Shared utilities
|   |   |-- __init__.py
|   |   |-- checkpointing.py           # Save/load with StatefulModule state + memory
|   |   |-- logging.py                 # Structured logging (W&B integration)
|   |   |-- profiling.py               # Memory + compute profiling helpers
|   |   |-- config.py                  # Hydra/OmegaConf configuration system
|   |   |-- metrics.py                 # Training metrics aggregation
|   |   |-- seeds.py                   # Reproducibility utilities
|
|-- genesis/                           # === GENESIS MODEL ===
|   |-- __init__.py
|   |
|   |-- model/                         # Model architecture
|   |   |-- __init__.py
|   |   |-- hlrt.py                    # Top-level HLRT model (combines all tiers)
|   |   |-- tier1_token_processor.py   # Tier 1: lightweight fast Transformer
|   |   |-- tier2_semantic_planner.py  # Tier 2: deep Transformer on latent chunks
|   |   |-- tier3_deliberative.py      # Tier 3: recurrent deep reasoner
|   |   |-- tier_gate.py               # Learned gating classifier for tier escalation
|   |   |-- latent_pooling.py          # Cross-attention pooling (tokens -> latent vectors)
|   |   |-- conditioning.py            # Top-down conditioning from higher tiers
|   |   |-- embeddings.py              # Token + position embeddings (shared across tiers)
|   |   |-- attention.py               # Multi-head attention with MLA option
|   |   |-- ffn.py                     # Feed-forward network (dense or MoE)
|   |   |-- moe.py                     # Mixture-of-Experts layer
|   |   |-- moe_router.py             # Expert routing (top-k, auxiliary load balancing)
|   |   |-- rmsnorm.py                 # RMSNorm layer
|   |   |-- rotary.py                  # Rotary position embeddings (RoPE)
|   |
|   |-- memory/                        # PHMA Memory System
|   |   |-- __init__.py
|   |   |-- working_memory.py          # Level 1: per-sequence typed memory slots
|   |   |-- episodic_memory.py         # Level 2: cross-sequence persistent memory
|   |   |-- semantic_memory.py         # Level 3: persistent key-value knowledge store
|   |   |-- memory_controller.py       # Write/read/erase gating logic
|   |   |-- memory_cross_attention.py  # Cross-attention interface for memory reads
|   |   |-- memory_consistency.py      # Auxiliary loss for contradiction detection
|   |
|   |-- verifier/                      # ACT-V Verifier
|   |   |-- __init__.py
|   |   |-- verifier_model.py          # Verifier Transformer (30-40% of G params)
|   |   |-- verification_head.py       # Classification heads (factual, logical, stylistic)
|   |   |-- negative_generator.py      # Generates corrupted passages for V training
|   |   |-- distillation.py            # Verification capability distillation into G
|   |   |-- replay_buffer.py           # Historical G outputs for V stability
|   |
|   |-- vision/                        # Vision Integration
|   |   |-- __init__.py
|   |   |-- vision_encoder.py          # SigLIP vision encoder wrapper
|   |   |-- projection.py              # Vision-to-HLRT projection layer
|   |   |-- visual_token_injector.py   # Injects visual tokens into HLRT input stream
|   |   |-- image_preprocessing.py     # Image resize, normalize, patch extraction
|   |
|   |-- training/                      # Training scripts and configs
|   |   |-- __init__.py
|   |   |-- train_phase1_bootstrap.py  # Phase 1: NTP bootstrap (tokens 0-500B equivalent)
|   |   |-- train_phase2_rl.py         # Phase 2: Reinforcement pretraining
|   |   |-- train_phase3_flywheel.py   # Phase 3: Data flywheel with self-generated traces
|   |   |-- train_actv.py              # ACT-V adversarial co-training integration
|   |   |-- train_vision.py            # Vision alignment + joint fine-tuning
|   |   |-- grpo.py                    # Group Relative Policy Optimization
|   |   |-- reward_functions.py        # Information gain + domain verifiers
|   |   |-- configs/                   # Hydra config files
|   |       |-- phase1_1b.yaml
|   |       |-- phase2_7b.yaml
|   |       |-- phase3_moe_40b.yaml
|   |       |-- phase4_vision.yaml
|   |       |-- optimizer/
|   |       |   |-- muon_fp4.yaml
|   |       |   |-- adamw_bf16.yaml    # Fallback config
|   |       |-- distributed/
|   |       |   |-- single_gpu.yaml
|   |       |   |-- multi_gpu_8.yaml
|   |       |   |-- diloco.yaml
|   |       |-- data/
|   |           |-- bootstrap.yaml
|   |           |-- rl_pretrain.yaml
|   |           |-- flywheel.yaml
|   |           |-- vision.yaml
|   |
|   |-- inference/                     # Inference pipeline (CUDA, for validation)
|   |   |-- __init__.py
|   |   |-- generator.py               # Text generation with HLRT cascade
|   |   |-- cascade_router.py          # Inference-time tier routing
|   |   |-- kv_cache.py                # KV cache management (MLA-compatible)
|   |   |-- sampling.py                # Top-k, top-p, temperature sampling
|   |
|   |-- conversion/                    # Model format conversion
|       |-- __init__.py
|       |-- export_safetensors.py      # PyTorch -> safetensors
|       |-- convert_to_mlx.py          # safetensors -> MLX format
|       |-- quantize.py                # Post-training quantization (Q4_K, Q6_K)
|       |-- validate_conversion.py     # Numerical equivalence testing
|
|-- genesis_mlx/                       # === MLX INFERENCE RUNTIME ===
|   |-- __init__.py
|   |-- model/
|   |   |-- hlrt.py                    # HLRT in MLX (inference only)
|   |   |-- tier1.py                   # Tier 1 token processor (MLX)
|   |   |-- tier2.py                   # Tier 2 semantic planner (MLX)
|   |   |-- tier3.py                   # Tier 3 deliberative reasoner (MLX)
|   |   |-- gate.py                    # Tier gating (MLX)
|   |   |-- attention.py               # MLX attention with MLA
|   |   |-- moe.py                     # MLX MoE with lazy expert loading
|   |   |-- embeddings.py              # MLX embeddings
|   |
|   |-- memory/
|   |   |-- working_memory.py          # MLX working memory (persistent across turns)
|   |   |-- semantic_memory.py         # MLX semantic memory (mmap from SSD)
|   |   |-- memory_manager.py          # Manages memory lifecycle during chat
|   |
|   |-- vision/
|   |   |-- encoder.py                 # SigLIP in MLX
|   |   |-- projection.py              # Vision projection in MLX
|   |
|   |-- generate.py                    # Main generation loop (handles tiers + memory)
|   |-- serve.py                       # Local HTTP server for chat interface
|   |-- load.py                        # Model loading with lazy weight init
|   |-- quantize.py                    # MLX quantization utilities
|
|-- tests/                             # === TEST SUITE ===
|   |-- unit/
|   |   |-- olympus/
|   |   |   |-- test_stateful_module.py
|   |   |   |-- test_memory_bus.py
|   |   |   |-- test_compute_router.py
|   |   |   |-- test_growth_controller.py
|   |   |   |-- test_training_orchestrator.py
|   |   |   |-- test_muon.py
|   |   |   |-- test_fp4.py
|   |   |   |-- test_diloco.py
|   |   |
|   |   |-- genesis/
|   |   |   |-- test_hlrt.py
|   |   |   |-- test_tier1.py
|   |   |   |-- test_tier2.py
|   |   |   |-- test_tier3.py
|   |   |   |-- test_tier_gate.py
|   |   |   |-- test_working_memory.py
|   |   |   |-- test_episodic_memory.py
|   |   |   |-- test_semantic_memory.py
|   |   |   |-- test_verifier.py
|   |   |   |-- test_moe.py
|   |   |   |-- test_vision_encoder.py
|   |   |   |-- test_grpo.py
|   |   |
|   |   |-- genesis_mlx/
|   |       |-- test_mlx_hlrt.py
|   |       |-- test_mlx_memory.py
|   |       |-- test_mlx_generation.py
|   |       |-- test_conversion_equivalence.py
|   |
|   |-- integration/
|   |   |-- test_full_training_step.py   # One complete train step through all components
|   |   |-- test_actv_loop.py            # Generator + Verifier adversarial loop
|   |   |-- test_progressive_growth.py   # Grow model and verify loss continuity
|   |   |-- test_flywheel_cycle.py       # Generate traces -> filter -> retrain
|   |   |-- test_memory_persistence.py   # Memory state across sequences
|   |   |-- test_mlx_end_to_end.py       # Full inference on MLX
|   |
|   |-- benchmarks/
|       |-- bench_tier_routing.py         # Measure tier activation rates
|       |-- bench_muon_vs_adamw.py        # Compare optimizer convergence
|       |-- bench_fp4_vs_bf16.py          # Numerical accuracy comparison
|       |-- bench_memory_scaling.py       # Memory overhead at various scales
|       |-- bench_mlx_throughput.py       # Tokens/sec on M1
|
|-- scripts/                           # === UTILITY SCRIPTS ===
|   |-- download_data.py               # Download and prepare training data
|   |-- prepare_tokenizer.py           # Train BPE tokenizer on corpus
|   |-- launch_training.py             # Launch training on Lambda Labs
|   |-- monitor_training.py            # W&B dashboard + live monitoring
|   |-- convert_model.py               # Full conversion pipeline (PyTorch -> MLX)
|   |-- run_benchmarks.py              # Run all benchmark suites
|   |-- profile_memory.py              # GPU memory profiling during training
|
|-- agents/                            # === DEVELOPMENT AGENTS ===
|   |-- architect_agent.md             # Reviews architecture decisions
|   |-- debug_agent.md                 # Debugs training failures
|   |-- completeness_agent.md          # Checks for unfinished/stub code
|   |-- test_agent.md                  # Writes and runs tests
|   |-- performance_agent.md           # Profiles and optimizes bottlenecks
|   |-- integration_agent.md           # Validates cross-component compatibility
|   |-- documentation_agent.md         # Keeps docs and code in sync
|
|-- docs/                              # === DOCUMENTATION ===
|   |-- architecture.md                # Deep dive on HLRT architecture
|   |-- olympus_api.md                 # Olympus framework API reference
|   |-- training_guide.md              # Step-by-step training instructions
|   |-- mlx_deployment.md              # MLX conversion and deployment guide
|   |-- troubleshooting.md             # Common issues and fixes
|   |-- research_notes.md              # Links to papers, decisions rationale
```

-----

## 3. ENVIRONMENT SETUP

### 3.1 Cloud Training Environment (Lambda Labs)

**Required Instance:** 8x A100 80GB (Phase 1-2, 4) or 8x H100 80GB (Phase 3)

```bash
# System packages
sudo apt update && sudo apt install -y git tmux htop nvtop

# Python environment
conda create -n genesis python=3.11 -y
conda activate genesis

# PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install \
    triton==3.1.0 \
    flash-attn==2.7.0 \
    transformers==4.47.0 \
    datasets==3.2.0 \
    tokenizers==0.21.0 \
    safetensors==0.4.5 \
    wandb==0.19.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    einops==0.8.0 \
    hivemind==1.2.0 \
    bitsandbytes==0.45.0 \
    sentencepiece==0.2.0 \
    tiktoken==0.8.0 \
    scipy==1.14.0 \
    tqdm \
    rich \
    pytest \
    pytest-gpu

# Install Olympus + GENESIS in dev mode
git clone <repo_url> genesis
cd genesis
pip install -e ".[dev]"

# Verify GPU setup
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"
```

### 3.2 Local Development Environment (M1 Mac)

```bash
# MLX inference runtime
pip install \
    mlx==0.22.0 \
    mlx-lm==0.21.0 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    tiktoken==0.8.0 \
    numpy==2.1.0 \
    tqdm \
    rich \
    pytest

# Install genesis_mlx in dev mode
cd genesis
pip install -e ".[mlx]"

# Verify MLX
python -c "import mlx.core as mx; print(f'MLX device: {mx.default_device()}')"
```

### 3.3 pyproject.toml

```toml
[project]
name = "genesis-llm"
version = "0.1.0"
description = "GENESIS: Frontier LLM with Hierarchical Latent Reasoning"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4.0",
    "einops>=0.8.0",
    "safetensors>=0.4.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",
    "wandb>=0.18.0",
    "tqdm",
    "rich",
]

[project.optional-dependencies]
cuda = [
    "triton>=3.0.0",
    "flash-attn>=2.6.0",
    "bitsandbytes>=0.44.0",
    "hivemind>=1.2.0",
]
mlx = [
    "mlx>=0.20.0",
    "mlx-lm>=0.20.0",
]
data = [
    "datasets>=3.0.0",
    "tokenizers>=0.20.0",
    "sentencepiece>=0.2.0",
    "tiktoken>=0.7.0",
]
vision = [
    "torchvision>=0.19.0",
    "Pillow>=10.0.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-xdist",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
```

-----

## 4. OLYMPUS FRAMEWORK

Olympus is a training runtime that sits ABOVE PyTorch. It adds five primitives that PyTorch lacks. Every primitive is implemented as a Python class that internally dispatches to PyTorch operations and custom Triton kernels.

**CRITICAL: Olympus never modifies PyTorch internals.** It wraps, extends, and composes PyTorch. All existing PyTorch functionality remains available.

### 4.1 StatefulModule

**File:** `olympus/core/stateful_module.py`

Key class that replaces `torch.nn.Module` for any module that needs persistent state across forward passes. State is differentiable (gradients flow through cross-iteration connections). Memory is non-differentiable persistent storage.

### 4.2 TrainingContext

**File:** `olympus/core/training_context.py`

Metadata about the current training step, passed to every StatefulModule.forward() call.

### 4.3 MemoryBus

**File:** `olympus/core/memory_bus.py`

Cross-module, cross-iteration communication channel with three scopes: step, episode, permanent.

### 4.4 ComputeRouter

**File:** `olympus/core/compute_router.py`

Dynamic routing with compiled fast paths for variable-compute architectures.

### 4.5 TrainingOrchestrator

**File:** `olympus/core/training_orchestrator.py`

Manages multi-model, multi-objective training with full state lifecycle management.

### 4.6 GrowthController

**File:** `olympus/core/growth_controller.py`

Progressive model expansion with function-preserving transformations.

### 4.7 Muon Optimizer

**File:** `olympus/optim/muon.py`

Newton-Schulz orthogonalization of gradient matrices for faster convergence.

-----

## 5. GENESIS MODEL ARCHITECTURE

### 5.1 HLRT Overview

The Hierarchical Latent Reasoning Transformer has three tiers:

```
Input tokens
    |
    v
[Tier 1: Token Processor] -----> Output (easy tokens, ~60-70%)
    |                                 ^
    v (escalation gate)               |
[Latent Pooling] -------> [Tier 2: Semantic Planner] ----> Conditioning
    |                                 ^                      vectors back
    v (escalation gate)               |                      to Tier 1
[Tier 3: Deliberative Reasoner] ------+
    (activated 5-10% of time)
```

### 5.2 Model Sizes

**Phase 1 PoC (~610M params):**
- Tier 1: 8 layers, d_model=768, 8 heads (~120M)
- Tier 2: 16 layers, d_model=1024, 16 heads (~350M)
- Tier 3: 8 layers, d_model=512, 8 heads, with recurrence (~80M)

**Phase 2 (~3.4B growing to 7B):**
- Tier 1: 12 layers, d_model=1024, 16 heads (~400M)
- Tier 2: 24 layers, d_model=2048, 32 heads (~2.5B)
- Tier 3: 12 layers, d_model=1024, 16 heads (~400M)

**Phase 3 MoE (~40B total, 8B active):**
- Tier 2: 24 layers, MoE FFN (16 experts, top-2), d_model=2048

-----

## 6-18. See full implementation details in source code.

Each section is implemented according to the specifications in the source files with comprehensive docstrings.

-----

## APPENDIX A: Key Research Papers

1. **Muon/Moonlight:** "Moonlight: Muon Optimizer at 100B Scale" (2025)
1. **NVFP4:** "NVFP4: A Training-Friendly FP4 Format" (2024)
1. **DiLoCo:** "DiLoCo: Distributed Low-Communication Training" (2024)
1. **RPT:** "Reinforcement Pretraining" - Microsoft (2025)
1. **DeepSeek-V3:** Architecture details for MoE, MLA (2024)
1. **GRPO:** Group Relative Policy Optimization (DeepSeek-R1, 2025)
1. **SigLIP:** "SigLIP: Sigmoid Loss for Image-Language Pretraining" (2023)
1. **FlashAttention-2:** "FlashAttention-2: Faster Attention" (2023)
1. **Net2Net:** "Net2Net: Accelerating Learning via Knowledge Transfer" (2016)
1. **RxT:** "Reactive Transformer: Stateful Real-Time Processing" (2025)
1. **Letta/MemGPT:** Stateful agent architecture (2024-2025)

## APPENDIX B: Naming Conventions

- **Files:** snake_case.py
- **Classes:** PascalCase
- **Functions:** snake_case
- **Constants:** UPPER_SNAKE_CASE
- **Config keys:** snake_case
- **Metrics:** slash-separated: "generator/loss", "tier1/activation_rate"
- **Memory bus keys:** "module_name/data_name"
- **Checkpoint files:** `genesis_step{step}_phase{phase}.pt`

-----

*End of GENESIS Blueprint v1.0*
*This document is the source of truth. When in doubt, read the blueprint.*
