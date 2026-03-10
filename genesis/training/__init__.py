"""
GENESIS training scripts for the multi-phase training pipeline.

Phases:
  1. Bootstrap: Next-token prediction pretraining.
  2. RL Pretrain: Reinforcement pretraining with GRPO.
  3. Flywheel: Self-generating data flywheel with reasoning traces.
  4. ACT-V: Adversarial co-training with verifier.
"""
