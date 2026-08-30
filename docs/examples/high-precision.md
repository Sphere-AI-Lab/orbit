---
title: "High-Precision Launchers"
description: "BF16 full-finetune and PEFT RL recipes, including the one-trunk adapter-critic PPO smokes."
# Generated from examples/high_precision/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
These launchers run BF16 or high-precision Orbit training recipes. Each file is
an independent entrypoint and sources shared Orbit launcher libraries from
`scripts/lib/`.

Common smoke overrides:

```bash
NUM_ROLLOUT=1 TOTAL_EPOCHS=1 TRAIN_ROWS=1 \
ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=1 \
DISABLE_EVAL=1 ENABLE_WANDB=0
```

## PPO

`run-qwen2_5-0_5b-bf16-math-oft-ppo.sh` is the high-precision PPO starter
recipe. PPO uses a separate full-model critic, so this launcher does not use
colocation. Its default single-node layout is:

- actor: 2 GPUs
- critic: 2 GPUs
- rollout: 4 GPUs

```bash
HF_CKPT=/path/to/hf/Qwen2.5-0.5B-Instruct \
MEGATRON_LOAD=/path/to/megatron/Qwen2.5-0.5B-Instruct \
TRAIN_JSONL=/path/to/math/train.jsonl \
TEST_JSONL=/path/to/math/test.jsonl \
bash examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft-ppo.sh
```

For a CPU-free argv inspection:

```bash
ORBIT_DRY_RUN_ARGV=1 DISABLE_EVAL=1 ENABLE_WANDB=0 TRAIN_ROWS=1 \
HF_CKPT=/path/to/hf/Qwen2.5-0.5B-Instruct \
MEGATRON_LOAD=/path/to/megatron/Qwen2.5-0.5B-Instruct \
TRAIN_JSONL=/path/to/math/train.jsonl \
bash examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft-ppo.sh
```
