# FILE MAP — Project GENESIS

Quick reference for navigating the codebase.

---

## Root
| File | Purpose |
|------|---------|
| `GENESIS_BLUEPRINT.md` | Complete implementation spec — source of truth |
| `PROPOSAL.md` | Research rationale + design decisions |
| `CLAUDE.md` | Project instructions for Claude Code |
| `BUILD_PLAN.md` | Master build plan with phase checkboxes |
| `PROGRESS.md` | Session-level work tracking |
| `BUGS.md` | Bug fix engineering journal |
| `CHANGELOG.md` | User-facing change log |
| `FILE_MAP.md` | This file |
| `pyproject.toml` | Python package config (PEP 621) |
| `Makefile` | Common commands (train, test, lint) |

---

## `olympus/` — Custom Training Framework

### `olympus/core/` — Core Abstractions
| File | Lines | What It Does |
|------|-------|--------------|
| `stateful_module.py` | 331 | `StatefulModule` base class with `StateLink` autograd for gradient flow through persistent state |
| `training_context.py` | 35 | Dataclass: step metadata (phase, batch idx, grad accum state, memory bus ref) |
| `memory_bus.py` | 161 | Thread-safe cross-module communication (step/episode/permanent scopes, cosine similarity retrieval) |
| `compute_router.py` | 257 | Dynamic routing with 3 gate types (linear/MLP/entropy), load-balance loss, straight-through estimator |
| `training_orchestrator.py` | 278 | Multi-model training loop with objective chaining and gradient accumulation |
| `growth_controller.py` | 210 | Progressive model expansion scheduling (architecture ops stubbed for subclass override) |

### `olympus/optim/` — Optimizers
| File | Lines | What It Does |
|------|-------|--------------|
| `muon.py` | 157 | Newton-Schulz orthogonalized gradient descent with Nesterov momentum |
| `muon_adamw_hybrid.py` | 157 | Hybrid: Muon for 2D params, AdamW for 1D/embeddings/norms |
| `schedulers.py` | 147 | WSD (Warmup-Stable-Decay) scheduler with cosine annealing + polynomial decay |

### `olympus/kernels/` — Triton Kernels (all have PyTorch fallback)
| File | Lines | What It Does |
|------|-------|--------------|
| `fp4_quantize.py` | 233 | INT4 quantization/dequantization with absmax per-group scaling |
| `fp4_matmul.py` | 273 | FP4 linear layer forward/backward pass |
| `fused_gate_route.py` | 298 | Chunk pooling + softmax-topk + fused gating for tier routing |
| `sparse_expert_matmul.py` | 188 | Sorted expert dispatch for MoE (routing, batching, weighted combination) |
| `memory_cross_attention.py` | 251 | Optimized cross-attention for memory reads with SDPA |
| `muon_step.py` | 168 | Fused Newton-Schulz orthogonalization step |

### `olympus/data/` — Data Pipeline
| File | Lines | What It Does |
|------|-------|--------------|
| `tokenizer.py` | 230 | BPE tokenizer wrapper with special tokens, encode/decode, vocab management |
| `curriculum.py` | 318 | Multi-phase data mixing with metric-based phase transitions |
| `memory_aware_batcher.py` | 208 | TF-IDF similarity clustering for document batching |
| `quality_filter.py` | 212 | Heuristic text quality scoring (word length, TTR, punctuation, variance) |
| `flywheel_buffer.py` | 194 | Priority-weighted replay buffer for successful reasoning traces |

### `olympus/utils/` — Shared Utilities
| File | Lines | What It Does |
|------|-------|--------------|
| `checkpointing.py` | 166 | Save/load model + optimizer + EMA state |
| `config.py` | 66 | Dataclass-based config management with validation |
| `logging.py` | 105 | Structured logging with metrics aggregation |
| `metrics.py` | 133 | Running stats (mean/var/std), loss tracking |
| `profiling.py` | 130 | GPU memory, wall-clock, throughput profiling |
| `seeds.py` | 36 | Determinism setup (torch, numpy, Python, CUDA) |

### `olympus/distributed/` — DiLoCo (STUB)
Empty. Future Phase 3.

---

## `genesis/` — GENESIS Model

### `genesis/model/` — HLRT Architecture
| File | Lines | What It Does |
|------|-------|--------------|
| `hlrt.py` | 356 | Top-level HLRT: Tier 1 -> Gate -> Pool -> Tier 2 -> Gate -> Tier 3 |
| `tier1_token_processor.py` | 147 | Lightweight fast Transformer (pre-norm, RMSNorm, SwiGLU) |
| `tier2_semantic_planner.py` | 204 | Deep Transformer on latent chunks, optional MoE |
| `tier3_deliberative.py` | 188 | Recurrent deep reasoner with multi-step reasoning |
| `tier_gate.py` | 122 | Learned gating for tier escalation (chunk pool + threshold) |
| `latent_pooling.py` | 173 | Cross-attention: tokens -> latent vectors |
| `conditioning.py` | 174 | Top-down conditioning from higher tiers to lower |
| `embeddings.py` | 56 | Token + position embeddings with dropout |
| `attention.py` | 173 | Multi-head attention with flash attention + RoPE + causal masking |
| `ffn.py` | 66 | SwiGLU feedforward network |
| `moe.py` | 109 | Mixture-of-Experts layer with router dispatch |
| `moe_router.py` | 120 | Top-k expert routing with auxiliary load-balance loss |
| `rmsnorm.py` | 49 | Root-mean-square layer normalization |
| `rotary.py` | 99 | Rotary Position Embeddings (RoPE) precomputation |

### `genesis/memory/` — PHMA Memory System
| File | Lines | What It Does |
|------|-------|--------------|
| `working_memory.py` | 237 | Level 1: per-sequence typed slots with attention-based I/O gating |
| `episodic_memory.py` | 278 | Level 2: cross-sequence persistent with surprise-based writes |
| `semantic_memory.py` | 259 | Level 3: persistent key-value store with k-NN cosine lookup |
| `memory_controller.py` | 237 | Orchestrates all three levels, routes queries, manages bus |
| `memory_cross_attention.py` | 107 | Cross-attention interface for reading from memory slots |
| `memory_consistency.py` | 127 | Consistency checking (episodic <-> semantic alignment) |

### `genesis/verifier/` — ACT-V
| File | Lines | What It Does |
|------|-------|--------------|
| `verifier_model.py` | 176 | Classification head for correctness scoring |
| `verification_head.py` | 154 | Token-level classification head |
| `negative_generator.py` | 323 | Corruption strategies (entity swap, negation, numbers, facts, temporal) |
| `distillation.py` | 133 | Knowledge distillation with temperature scaling |
| `replay_buffer.py` | 123 | Priority-weighted replay of historical outputs |

### `genesis/training/` — Training Pipelines
| File | Lines | What It Does |
|------|-------|--------------|
| `train_phase1_bootstrap.py` | 260 | Phase 1: NTP pretraining with TrainingOrchestrator + MuonAdamW |
| `train_phase2_rl.py` | 276 | Phase 2: RL pretraining with GRPO, generates completions + rewards |
| `train_phase3_flywheel.py` | 325 | Phase 3: self-improvement data loop |
| `train_actv.py` | 447 | ACT-V adversarial co-training integration |
| `grpo.py` | 303 | Group Relative Policy Optimization (DeepSeek-R1 style) |
| `reward_functions.py` | 356 | Composable rewards: information gain, correctness, format, composite |

### `genesis/vision/` — Vision (STUB)
Empty. Future Phase 4.

### `genesis/inference/` — Inference Pipeline
| File | Lines | What It Does |
|------|-------|--------------|
| `generator.py` | 230 | Autoregressive text generation with HLRT cascade (batch + streaming modes) |
| `kv_cache.py` | 120 | Per-layer KV cache for incremental decoding, with memory estimation |
| `sampling.py` | 140 | Top-k, top-p, temperature, repetition penalty, greedy decoding |
| `cascade_router.py` | 140 | Inference-time tier routing with activation stats and threshold overrides |

### `genesis/conversion/` — Conversion (STUB)
Empty. Future Phase 6.

---

## `genesis_mlx/` — MLX Inference Runtime (STUB)
Empty. Future Phase 5.

---

## `tests/` — Test Suite

### `tests/unit/olympus/` — Olympus Unit Tests (~2,100 lines)
Tests for: StatefulModule, MemoryBus, ComputeRouter, all 6 kernels (FP4, gate routing, sparse expert, memory attention, Muon step), Muon optimizer.

### `tests/unit/genesis/` — Genesis Unit Tests (~500 lines)
Tests for: HLRT forward pass, Tier 1, MoE, TierGate.

---

## `agents/` — Development Agent Role Cards
Markdown files defining agent responsibilities: architect, debug, test, integration, completeness, documentation, performance.

## `docs/` — Documentation
Architecture deep dives, API reference, training guides, research notes.
