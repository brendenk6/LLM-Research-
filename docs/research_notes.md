# Research Notes

## Key Papers and Decisions

### Muon Optimizer
- Paper: "Moonlight: Muon Optimizer at 100B Scale" (2025)
- Decision: Use Muon for 2D weight matrices, AdamW for 1D params
- Rationale: Muon shows 2x faster convergence at scale, but doesn't apply to vectors

### GRPO (Group Relative Policy Optimization)
- Paper: DeepSeek-R1 (2025)
- Decision: Use GRPO instead of PPO for RL pretraining
- Rationale: No value network needed, group normalization reduces variance

### DiLoCo
- Paper: "DiLoCo: Distributed Low-Communication Training" (2024)
- Decision: Implement but defer activation until multi-node scaling
- Rationale: Single-node FSDP is sufficient for Phase 1-2

### Progressive Growing
- Paper: Net2Net (2016), adapted for modern Transformers
- Decision: Function-preserving widen → deepen → dense-to-MoE schedule
- Rationale: Train small, validate, grow. Saves 3-4x compute vs training large from scratch.

### SigLIP for Vision
- Paper: "SigLIP: Sigmoid Loss for Image-Language Pretraining" (2023)
- Decision: Use SigLIP-SO400M as frozen vision encoder
- Rationale: Open source, strong performance, efficient patch-based encoding
