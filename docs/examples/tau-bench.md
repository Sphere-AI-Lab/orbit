---
title: "Tau-bench PPO"
description: "Tau-bench agentic RL recipes."
# Generated from examples/tau_bench/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
This example adds a raw `/generate` Tau-bench compatibility rollout for Orbit
PPO. It keeps Tau-bench as an optional dependency and imports it only when a
rollout starts.

The generator sets `sample.reward` directly from the Tau-bench environment, so
launchers do not need a separate reward-model path.

## Dataset

For the legacy Tau-bench package used by the Miles reference, export task-index
JSONL files with:

```bash
python examples/tau_bench/tau_tasks.py --output-dir /path/to/tau_data
```

Each row contains an `index` prompt. Launchers should use:

```bash
export TRAIN_DATA=/path/to/tau_data/retail_train_tasks.jsonl
export TEST_DATA=/path/to/tau_data/retail_dev_tasks.jsonl
```

## User Simulator

Tau-bench drives the user simulator through an external provider. Configure it
with environment variables:

```bash
export TAU_USER_MODEL_PROVIDER=gemini
export TAU_USER_MODEL=gemini-2.5-flash-lite
export GEMINI_API_KEY=...
```

For DeepSeek:

```bash
export TAU_USER_MODEL_PROVIDER=deepseek
export TAU_USER_MODEL=deepseek-chat
export DEEPSEEK_API_KEY=...
```

## Launchers

The Qwen3-4B-Instruct-2507 PPO launchers are:

```bash
bash examples/tau_bench/run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-full.sh
bash examples/tau_bench/run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-lora.sh
bash examples/tau_bench/run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-oft.sh
```

Required paths:

```bash
export HF_CKPT=/path/to/Qwen3-4B-Instruct-2507
export MEGATRON_LOAD=/path/to/Qwen3-4B-Instruct-2507-torchdist
export TRAIN_DATA=/path/to/retail_train_tasks.jsonl
export TEST_DATA=/path/to/retail_dev_tasks.jsonl
```

Useful overrides:

```bash
export TAU_BENCH_ENV=retail
export TAU_BENCH_TASK_SPLIT=train
export NUM_ROLLOUT=500
export ROLLOUT_BATCH_SIZE=32
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=256
export SGLANG_SERVER_CONCURRENCY=32
```
