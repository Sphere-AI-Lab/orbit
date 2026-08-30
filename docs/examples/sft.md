---
title: "Orbit SFT Examples"
description: "Supervised fine-tuning launchers."
# Generated from examples/sft/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Orbit is RL-first, but supervised fine-tuning is available as an explicit mode
with `--training-mode sft`. SFT uses the same Megatron training, checkpointing,
dynamic batching, PEFT, and logging stack as RL. Plain SFT runs read labeled
examples directly from the global dataset and do not start SGLang rollout
engines unless you configure generation-based evaluation.

## Data Format

Use chat-format JSONL. Each row must contain a `messages` field with the full
conversation, including assistant target turns:

```json
{"messages":[
  {"role":"user","content":"Question: Where would a person store soup?\n\nChoices:\nA. bowl\nB. shoe\n\nChoose the best answer."},
  {"role":"assistant","content":"A. bowl"}
]}
```

The launchers pass `--input-key messages`. The SFT rollout uses Orbit's
multi-turn loss-mask generator, so prompt/system tokens are masked out and loss
is computed only on assistant target tokens.

## SFT Mode

`--training-mode sft` applies SFT-safe defaults during argument validation:

- sets `--loss-type sft_loss`;
- disables advantage and return computation;
- forces `--n-samples-per-prompt 1`;
- switches the default rollout function to
  `miles.rollout.sft_rollout.generate_rollout`;
- disables rollout engines for plain training runs without generation eval.

## Convert Datasets

The launchers default to `SFT_DATA_ROOT=${ORBIT_ROOT}/data/sft` and read
`<dataset>/train.jsonl`.

```bash
python tools/convert_sft_dataset_to_orbit.py \
  --dataset numinamath \
  --output-dir data/sft/numinamath \
  --splits train test \
  --streaming \
  --force

python tools/convert_sft_dataset_to_orbit.py \
  --dataset magicoder \
  --output-dir data/sft/magicoder \
  --splits train \
  --streaming \
  --force

python tools/convert_sft_dataset_to_orbit.py \
  --dataset commonsenseqa \
  --output-dir data/sft/commonsenseqa \
  --splits train validation \
  --force

python tools/convert_sft_dataset_to_orbit.py \
  --dataset socialiqa \
  --output-dir data/sft/socialiqa \
  --splits train validation \
  --force

python tools/convert_sft_dataset_to_orbit.py \
  --dataset scienceqa-text \
  --output-dir data/sft/scienceqa-text \
  --splits train validation test \
  --force
```

Use `--max-rows N` for smoke datasets. ScienceQA rows with images are skipped
by default because these launchers are text-only.

## Split JSONL

Use `tools/split_sft_jsonl_partitions.py` when training multiple adapters on
disjoint deterministic data shards for later adapter merging.

```bash
python tools/split_sft_jsonl_partitions.py \
  --input data/sft/magicoder/train.jsonl \
  --output-dir data/sft/magicoder_partitions \
  --partitions 4 \
  --seed 1234 \
  --stratify-key metadata.lang
```

The splitter writes `P*/train.jsonl` plus a manifest with row counts, checksums,
and configured stratification counts. The default stratification key is
`metadata.dataset`.

## Launch

Qwen2.5 full-parameter SFT:

```bash
HF_CKPT=/path/to/hf/Qwen2.5-0.5B-Instruct \
MEGATRON_LOAD=/path/to/megatron/Qwen2.5-0.5B-Instruct \
ENABLE_WANDB=0 \
bash examples/sft/run-qwen2_5-0_5b-bf16-sft-numinamath.sh
```

Llama-3.1-8B OFT SFT:

```bash
HF_CKPT=/path/to/hf/Llama-3.1-8B \
MEGATRON_LOAD=/path/to/megatron/Llama-3.1-8B \
ENABLE_WANDB=0 \
bash examples/sft/run-llama3_1-8b-bf16-oft-sft-magicoder.sh
```

Override `TRAIN_JSONL` to point at any converted file, or set `SFT_DATA_ROOT` to
move all defaults at once.

## Launchers

| Launcher | Model | Training data | PEFT |
|---|---|---|---|
| `run-qwen2_5-0_5b-bf16-sft-numinamath.sh` | Qwen2.5-0.5B-Instruct | `AI-MO/NuminaMath-CoT` | full |
| `run-qwen2_5-0_5b-bf16-sft-magicoder.sh` | Qwen2.5-0.5B-Instruct | `ise-uiuc/Magicoder-OSS-Instruct-75K` | full |
| `run-qwen2_5-0_5b-bf16-sft-commonsenseqa.sh` | Qwen2.5-0.5B-Instruct | `tau/commonsense_qa` | full |
| `run-qwen2_5-0_5b-bf16-sft-socialiqa.sh` | Qwen2.5-0.5B-Instruct | `allenai/social_i_qa` | full |
| `run-qwen2_5-0_5b-bf16-sft-scienceqa-text.sh` | Qwen2.5-0.5B-Instruct | text-only `derek-thomas/ScienceQA` | full |
| `run-llama3_1-8b-bf16-oft-sft-numinamath.sh` | Llama-3.1-8B | `AI-MO/NuminaMath-CoT` | OFT |
| `run-llama3_1-8b-bf16-oft-sft-magicoder.sh` | Llama-3.1-8B | `ise-uiuc/Magicoder-OSS-Instruct-75K` | OFT |
| `run-llama3_1-8b-bf16-oft-sft-commonsenseqa.sh` | Llama-3.1-8B | `tau/commonsense_qa` | OFT |
| `run-llama3_1-8b-bf16-oft-sft-scienceqa-text.sh` | Llama-3.1-8B | text-only `derek-thomas/ScienceQA` | OFT |

Each launcher is standalone and inlines its model, dataset, and training
arguments. Qwen launchers accept `SFT_PEFT_ARGS` and `SFT_EXTRA_ARGS` for local
experiments. Llama launchers enable OFT by default.
