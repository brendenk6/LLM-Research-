# GENESIS Architecture: Deep Dive

## Hierarchical Latent Reasoning Transformer (HLRT)

HLRT is a 3-tier variable-compute Transformer architecture where each token dynamically selects how much compute it needs.

### Tier 1: Token Processor
- Lightweight, fast Transformer (8-12 layers)
- Processes ALL tokens
- Handles easy predictions (~60-70% of tokens)
- Pre-norm with RMSNorm, SwiGLU FFN, RoPE

### Tier 2: Semantic Planner
- Deep Transformer (16-24 layers) operating on latent vectors
- Activated for chunks that Tier 1 finds ambiguous
- Cross-attention pooling compresses ~32 tokens into ~4 latent vectors
- Optional MoE FFN for capacity scaling

### Tier 3: Deliberative Reasoner
- Recurrent Transformer that runs multiple passes on the same input
- Activated for the hardest 5-10% of tokens
- Implements internal chain-of-thought via recurrence
- Variable compute via learned halting

### Information Flow

```
Tokens → Embed → Tier 1 (all) → Gate → Pool → Tier 2 (some) → Gate → Tier 3 (few)
                    ↑                              |                      |
                    └──── Conditioning ←───────────┘                      |
                    └──── Conditioning ←──────────────────────────────────┘
```

### Top-Down Conditioning
- Tier 2 produces a "plan vector" that modifies Tier 1's residual stream
- Learned alpha scaling (initialized to 0) for stable training
- Plan persists across iterations via StatefulModule state

## PHMA Memory System

Three-level differentiable memory:
1. **Working Memory**: 256-512 slots per sequence, typed (entity/relation/quantity/temporal/spatial)
2. **Episodic Memory**: 1024-4096 slots persisting across related sequences
3. **Semantic Memory**: 100K+ key-value pairs, frozen at inference

## Olympus Framework Primitives

1. **StatefulModule**: nn.Module with persistent differentiable state
2. **MemoryBus**: Cross-module communication (step/episode/permanent scopes)
3. **ComputeRouter**: Dynamic routing with compiled fast paths
4. **TrainingOrchestrator**: Multi-model training coordination
5. **GrowthController**: Function-preserving model expansion
