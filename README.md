# Project GENESIS

**Generative Engine with Networked Expert Systems, Iterative Self-improvement**

A frontier-class LLM trained from scratch using seven composable innovations, built on Olympus — a custom PyTorch training runtime.

## The Seven Innovations

1. **HLRT** — Hierarchical Latent Reasoning Transformer. 3-tier variable-compute architecture (Token Processor → Semantic Planner → Deliberative Reasoner). Easy tokens use Tier 1 alone (~60-70%), hard reasoning escalates to Tier 3 (~5-10%).

2. **Reinforcement Pretraining Flywheel** — RL-based training from day 1 with GRPO. The model generates reasoning traces, filters by quality, and retrains on its own successful thinking. Self-improving data loop.

3. **ACT-V** — Adversarial Co-Training with Verifier. A second model learns to detect hallucination, then its verification capability is distilled into the generator. Built-in bullshit detector.

4. **PHMA** — Persistent Hierarchical Memory Architecture. Three-level differentiable memory (working → episodic → semantic) replacing implicit weight storage with explicit, inspectable, editable knowledge.

5. **Muon + FP4 + Progressive Growing** — 6-8x compute savings by stacking three independent efficiency gains. Newton-Schulz optimizer, 4-bit quantized training, and function-preserving model expansion.

6. **DiLoCo Distribution** — Geo-distributed training on consumer hardware. No centralized datacenter required.

7. **Compound Inference Cascade** — Tiered inference at 5-8x less average compute. Think hard only when it matters.

## Scale Targets

| Phase | Model Size | Active Params | Hardware | Duration |
|-------|-----------|---------------|----------|----------|
| Phase 1 (PoC) | 1B total | 1B (dense) | 1x A100 | 4 weeks |
| Phase 2 (Scale) | 7B total | 7B (dense) | 8x A100 | 6 weeks |
| Phase 3 (MoE) | 40B total | 8B active | 8x H100 | 8 weeks |
| Phase 4 (Vision) | +400M encoder | 8B + 400M | 8x A100 | 4 weeks |
| Phase 5 (MLX) | Same, quantized | 8B active Q4 | M1 16GB | 4 weeks |

## Setup

```bash
# Cloud training
pip install -e ".[dev,cuda,data]"

# Local MLX inference
pip install -e ".[mlx]"

# Run tests
pytest tests/ -v
```

## Documentation

| Document | Purpose |
|----------|---------|
| [GENESIS_BLUEPRINT.md](GENESIS_BLUEPRINT.md) | Complete implementation spec |
| [PROPOSAL.md](PROPOSAL.md) | Research rationale |
| [BUILD_PLAN.md](BUILD_PLAN.md) | Phase checklists |
| [FILE_MAP.md](FILE_MAP.md) | Codebase navigation guide |
| [CHANGELOG.md](CHANGELOG.md) | What changed |
| [PROGRESS.md](PROGRESS.md) | Session-level tracking |

## Current Status

Phases 1-2 of the build plan are complete: Olympus framework + GENESIS model architecture + memory + verifier + training pipelines. ~27K lines of working code with comprehensive tests. Distributed training, vision, inference, conversion, and MLX runtime are next.
