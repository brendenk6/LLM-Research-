# CLAUDE.md — Project GENESIS

## What This Is
Project GENESIS (Generative Engine with Networked Expert Systems, Iterative Self-improvement) — a frontier-class LLM trained from scratch using seven composable innovations, built on Olympus (custom PyTorch training runtime).

## Repository Layout
- `olympus/` — Custom training framework (StatefulModule, MemoryBus, ComputeRouter, Muon optimizer, Triton kernels)
- `genesis/` — HLRT model architecture, PHMA memory, ACT-V verifier, training pipelines
- `genesis_mlx/` — MLX inference runtime (future)
- `tests/` — Unit + integration tests
- `agents/` — Development agent role cards
- `docs/` — Architecture and API documentation

## Stack
- **Training**: Python 3.11+, PyTorch 2.5+, Triton, CUDA (Lambda Labs / cloud GPU)
- **Inference**: MLX on Apple Silicon (M1 16GB target)
- **Optimizer**: Muon (2D params) + AdamW (1D params) hybrid
- **Precision**: FP4 quantized training with BF16 fallback
- **Distribution**: DiLoCo (future)

## Key Architecture
HLRT = 3-tier variable-compute Transformer:
- **Tier 1** (Token Processor): Lightweight, handles ~60-70% of tokens
- **Tier 2** (Semantic Planner): Deep, operates on latent chunks
- **Tier 3** (Deliberative Reasoner): Expensive, activated 5-10% of time

## Build & Test
```bash
# Install dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/olympus/ -v
pytest tests/unit/genesis/ -v
```

## Documentation Standards
- **CHANGELOG.md**: Keep a Changelog format. Update for EVERY feature, fix, or notable change.
- **BUGS.md**: Root cause analysis journal. Update after EACH bug fix iteration.
- **PROGRESS.md**: Session-level work tracking with phase checklists.
- **BUILD_PLAN.md**: Master build plan with checkboxes. Source of truth for what's done/next.
- **FILE_MAP.md**: File-level guide to the codebase.

## Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Metrics: slash-separated (`generator/loss`, `tier1/activation_rate`)
- Memory bus keys: `module_name/data_name`
- Checkpoints: `genesis_step{step}_phase{phase}.pt`

## Key Design Rules
1. Olympus wraps PyTorch, never forks it. All existing CUDA kernels, FlashAttention, NCCL remain untouched.
2. Every component is testable in isolation.
3. Progressive complexity — start simple, validate, then add complexity.
4. Cloud train (CUDA), local deploy (MLX).
5. All Triton kernels have PyTorch fallback paths for CPU/non-CUDA testing.

## Source of Truth
- **GENESIS_BLUEPRINT.md**: Complete implementation spec. If it's not in the blueprint, it doesn't exist.
- **PROPOSAL.md**: Research rationale and design decisions.
