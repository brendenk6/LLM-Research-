# BUILD PLAN — Project GENESIS

Master build plan. Checkboxes track completion status. Derived from GENESIS_BLUEPRINT.md.

---

## Phase 1: Olympus Framework
> Custom training runtime — the foundation everything else sits on.

### Core Abstractions
- [x] `StatefulModule` — persistent state across forward passes with gradient flow (StateLink)
- [x] `TrainingContext` — per-step metadata dataclass
- [x] `MemoryBus` — cross-module communication (step/episode/permanent scopes)
- [x] `ComputeRouter` — dynamic routing with compiled fast paths
- [x] `TrainingOrchestrator` — multi-model, multi-objective training loop
- [x] `GrowthController` — progressive model expansion scheduling (architecture ops stubbed for subclass override)

### Optimizers
- [x] `Muon` — Newton-Schulz orthogonalized gradient descent
- [x] `MuonAdamWHybrid` — Muon for 2D, AdamW for 1D params
- [x] `WSDScheduler` — Warmup-Stable-Decay with cosine annealing

### Triton Kernels (all with PyTorch fallback)
- [x] `fp4_quantize` — INT4 quantization/dequantization with absmax scaling
- [x] `fp4_matmul` — FP4 linear layer forward/backward
- [x] `fused_gate_route` — chunk pooling + softmax-topk + gating
- [x] `sparse_expert_matmul` — sorted expert dispatch for MoE
- [x] `memory_cross_attention` — optimized cross-attention for memory reads
- [x] `muon_step` — fused Newton-Schulz orthogonalization

### Data Pipeline
- [x] `Tokenizer` — BPE wrapper with special token support
- [x] `Curriculum` — multi-phase data mixing with metric-based transitions
- [x] `MemoryAwareBatcher` — TF-IDF similarity clustering for document batching
- [x] `QualityFilter` — heuristic text quality scoring (FineWeb-style)
- [x] `FlywheelBuffer` — priority-weighted replay buffer for RL traces

### Utils
- [x] `Checkpointing` — save/load with EMA state
- [x] `Config` — dataclass-based config management
- [x] `Logging` — structured logging + metrics
- [x] `Metrics` — running stats aggregation
- [x] `Profiling` — GPU memory + wall-clock + throughput
- [x] `Seeds` — determinism utilities

### Tests
- [x] Unit tests for StatefulModule, MemoryBus, ComputeRouter
- [x] Unit tests for all 6 Triton kernels
- [x] Unit tests for Muon optimizer

---

## Phase 2: GENESIS Model (HLRT)
> Three-tier hierarchical architecture.

### Model Components
- [x] `HLRT` — top-level model orchestrating all tiers
- [x] `Tier1TokenProcessor` — lightweight fast Transformer
- [x] `Tier2SemanticPlanner` — deep Transformer on latent chunks (MoE optional)
- [x] `Tier3Deliberative` — recurrent deep reasoner
- [x] `TierGate` — learned gating classifier for tier escalation
- [x] `LatentPooling` — cross-attention pooling (tokens -> latent vectors)
- [x] `Conditioning` — top-down conditioning from higher tiers
- [x] `Embeddings` — token + position embeddings
- [x] `Attention` — multi-head attention with flash attention support + RoPE
- [x] `FFN` — SwiGLU feedforward
- [x] `MoE` — Mixture-of-Experts layer
- [x] `MoERouter` — top-k expert routing with auxiliary loss
- [x] `RMSNorm` — root-mean-square normalization
- [x] `Rotary` — RoPE precomputation

### PHMA Memory System
- [x] `WorkingMemory` — Level 1, per-sequence typed slots
- [x] `EpisodicMemory` — Level 2, cross-sequence persistent
- [x] `SemanticMemory` — Level 3, persistent key-value store
- [x] `MemoryController` — orchestrates all three levels
- [x] `MemoryCrossAttention` — cross-attention interface for reads
- [x] `MemoryConsistency` — contradiction detection auxiliary loss

### ACT-V Verifier
- [x] `VerifierModel` — classification head for correctness
- [x] `VerificationHead` — token-level classification
- [x] `NegativeGenerator` — corruption strategies (entity swap, negation, number, fact, temporal)
- [x] `Distillation` — knowledge distillation with temperature scaling
- [x] `ReplayBuffer` — priority-weighted historical outputs

### Training Pipelines
- [x] `train_phase1_bootstrap` — NTP pretraining with TrainingOrchestrator
- [x] `train_phase2_rl` — RL pretraining with GRPO
- [x] `train_phase3_flywheel` — self-improvement data loop
- [x] `train_actv` — ACT-V adversarial co-training
- [x] `GRPO` — Group Relative Policy Optimization
- [x] `RewardFunctions` — information gain, correctness, format, composite

### Tests
- [x] Unit tests for HLRT, Tier 1, MoE, TierGate

---

## Phase 3: Distributed Training (DiLoCo)
> Geo-distributed training on consumer/cloud hardware.

- [ ] `DiLoCo` — outer optimizer (Nesterov momentum)
- [ ] `Island` — single compute island abstraction
- [ ] `Coordinator` — global coordination via Hivemind DHT
- [ ] `SparseSync` — sparse gradient synchronization (MoE-aware)
- [ ] `QuantizedComm` — 4-bit gradient quantization for communication
- [ ] `FaultTolerance` — island failure detection and recovery
- [ ] Integration tests for multi-island training
- [ ] DiLoCo config files

---

## Phase 4: Vision Integration
> SigLIP vision encoder + HLRT fusion.

- [ ] `VisionEncoder` — SigLIP wrapper
- [ ] `Projection` — vision-to-HLRT projection layer
- [ ] `VisualTokenInjector` — inject visual tokens into HLRT input stream
- [ ] `ImagePreprocessing` — resize, normalize, patch extraction
- [ ] `train_vision` — vision alignment + joint fine-tuning
- [ ] Vision config files
- [ ] Unit tests for vision components

---

## Phase 5: MLX Inference Runtime
> Apple Silicon deployment.

- [ ] MLX HLRT model (inference only)
- [ ] MLX Tier 1/2/3 implementations
- [ ] MLX tier gating
- [ ] MLX attention with MLA
- [ ] MLX MoE with lazy expert loading
- [ ] MLX working memory (persistent across turns)
- [ ] MLX semantic memory (mmap from SSD)
- [ ] MLX memory manager
- [ ] MLX SigLIP vision encoder
- [ ] Generation loop (tiers + memory)
- [ ] Local HTTP server for chat interface
- [ ] Model loading with lazy weight init
- [ ] MLX quantization utilities (Q4_K, Q6_K)

---

## Phase 6: Conversion Pipeline
> PyTorch -> safetensors -> MLX.

- [ ] `export_safetensors` — PyTorch -> safetensors
- [ ] `convert_to_mlx` — safetensors -> MLX format
- [ ] `quantize` — post-training quantization
- [ ] `validate_conversion` — numerical equivalence testing

---

## Phase 7: Inference Pipeline (CUDA)
> Validation inference on training hardware.

- [x] `Generator` — text generation with HLRT cascade (batch + streaming)
- [x] `CascadeRouter` — inference-time tier routing with stats tracking
- [x] `KVCache` — KV cache management with memory estimation
- [x] `Sampling` — top-k, top-p, temperature, repetition penalty

---

## Phase 8: Integration Testing & Benchmarks
> End-to-end validation.

- [x] Full training step integration test
- [x] ACT-V adversarial loop test
- [x] Progressive growth + loss continuity test
- [x] Flywheel cycle test (generate -> filter -> retrain)
- [x] Memory persistence across sequences test
- [ ] MLX end-to-end inference test (blocked: no MLX runtime yet)
- [x] Benchmark: tier routing activation rates
- [x] Benchmark: Muon vs AdamW convergence
- [ ] Benchmark: FP4 vs BF16 accuracy (blocked: needs CUDA)
- [x] Benchmark: memory scaling overhead
- [ ] Benchmark: MLX tokens/sec on M1 (blocked: no MLX runtime yet)

---

## Phase 9: Cloud Training
> Actually train the model.

- [ ] Prepare tokenizer on training corpus
- [ ] Download and filter training data
- [ ] Phase 1 PoC: 1B params on 1x A100 (4 weeks)
- [ ] Phase 2 Scale: 7B params on 8x A100 (6 weeks)
- [ ] Phase 3 MoE: 40B total on 8x H100 (8 weeks)
- [ ] Phase 4 Vision: +400M encoder on 8x A100 (4 weeks)
- [ ] Phase 5 MLX: quantize + deploy to M1 (4 weeks)
