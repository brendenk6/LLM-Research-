# Integration Agent

You validate that all components work together correctly.

## Integration Checkpoints

### Milestone 1: Olympus Core
- [ ] StatefulModule state persists across 10 forward passes
- [ ] MemoryBus read/write works across two StatefulModules
- [ ] ComputeRouter routes tokens and gradients flow
- [ ] TrainingOrchestrator runs 10 steps with single model
- [ ] Checkpointing saves and restores full state

### Milestone 2: HLRT Architecture
- [ ] Tier 1 processes tokens and produces logits
- [ ] Tier gate activates Tier 2 for some tokens (not all, not none)
- [ ] Latent pooling compresses tokens to latent vectors
- [ ] Tier 2 processes latent vectors
- [ ] Conditioning feeds Tier 2 plan back to Tier 1
- [ ] End-to-end forward pass produces valid logits
- [ ] Backward pass computes gradients for all parameters

### Milestone 3: Training Loop
- [ ] NTP training converges on small dataset (loss decreases)
- [ ] MuonAdamWHybrid optimizer updates 2D and 1D params correctly
- [ ] WSD learning rate schedule follows expected curve
- [ ] Gradient accumulation produces same result as larger batch

### Milestone 4: Memory + Verifier
- [ ] Working memory read/write during forward pass
- [ ] Episodic memory persists across sequences in batch
- [ ] ACT-V: Verifier scores real text higher than corrupted text
- [ ] ACT-V: Generator receives gradient signal from Verifier

### Milestone 5: RL + Flywheel
- [ ] GRPO generates multiple traces per context
- [ ] Information gain reward is positive for good traces
- [ ] FlywheelBuffer stores and retrieves traces correctly

### Milestone 6: Scale + MoE
- [ ] Progressive growth: 600M -> 1B with function preservation
- [ ] MoE routing distributes tokens across experts
- [ ] Load balancing loss prevents expert collapse

### Milestone 7: Vision + MLX
- [ ] Vision encoder produces embeddings
- [ ] Visual tokens integrate with HLRT
- [ ] PyTorch -> MLX conversion preserves logits
- [ ] MLX generates text at 30+ tok/sec on M1
