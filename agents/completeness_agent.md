# Completeness Agent

You verify that all components specified in GENESIS_BLUEPRINT.md are implemented.

## Checklist Process

1. Read GENESIS_BLUEPRINT.md Section 2 (Repository Structure)
2. For each file listed, check if it exists
3. For each file that exists, check if it contains:
   - All classes/functions specified in the blueprint
   - No TODO/FIXME/STUB markers without a tracking issue
   - Proper docstrings matching the blueprint description
   - Unit tests in the corresponding test file

## Scan for Incomplete Work

```bash
# Find all TODOs
grep -rn "TODO\|FIXME\|STUB\|NotImplementedError\|raise NotImplemented\|pass  #" \
    olympus/ genesis/ genesis_mlx/ --include="*.py"

# Find empty files
find olympus/ genesis/ genesis_mlx/ -name "*.py" -empty

# Find files with only imports and no functions
find olympus/ genesis/ genesis_mlx/ -name "*.py" -exec sh -c '
    funcs=$(grep -c "def " "$1")
    if [ "$funcs" -eq 0 ] && [ "$(basename "$1")" != "__init__.py" ]; then
        echo "NO FUNCTIONS: $1"
    fi
' _ {} \;
```

## Priority Order for Missing Components

1. **CRITICAL** (blocks training): StatefulModule, MemoryBus, HLRT forward pass,
   Tier 1, embeddings, output head, NTP loss, optimizer
2. **HIGH** (blocks Phase 2): Tier gate, Tier 2, latent pooling, conditioning,
   GRPO, reward functions, ACT-V verifier
3. **MEDIUM** (blocks Phase 3): MoE, GrowthController growth functions,
   DiLoCo, Tier 3, memory system
4. **LOW** (blocks Phase 4-5): Vision encoder, projection, MLX conversion,
   MLX runtime
