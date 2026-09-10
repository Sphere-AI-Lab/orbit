# Adapter-first GSM8K training

Five single-node GRPO configurations share three Python launchers. Each defaults
to 500 rollouts, four training GPUs and four rollout GPUs, with checkpointing
and GSM8K evaluation every 50 rollouts.

| Module under `examples.adapter_first` | Method | Trainer base | Rollout base |
|---|---|---|---|
| `run_qwen25_05b_gsm8k` | `lora` or `oft` | BF16 Qwen2.5-0.5B-Instruct | BF16 |
| `run_qwen3_4b_fp8_gsm8k` | `lora` or `oft` | BF16 dequantization of Qwen3-4B-FP8 | FP8 |
| `run_qwen3_4b_fp8_native_oft_gsm8k` | `oft` | Native FP8 Qwen3-4B-FP8 | FP8 |

LoRA uses rank/alpha 32 and the Ray adapter transport. OFT uses canonical OFT
with NCCL double buffering: block size 32 and epsilon 1e-5 for 0.5B, block size
128 and epsilon 6e-5 for 4B. These are synchronous training recipes; they do
not enable fully asynchronous generation.

## Prepare inputs

Activate your Orbit environment and obtain a single-node GPU allocation before
launching. Run commands from the repository root. The scripts submit through
Orbit's `command_utils`; they do not request a scheduler allocation or download
or convert checkpoints. For an existing Ray cluster, set
`ORBIT_SCRIPT_EXTERNAL_RAY=1`.

Provide these inputs, or override each path explicitly:

| Configuration | HF directory under `--model-dir` | Megatron directory under `--model-dir` |
|---|---|---|
| 0.5B | `Qwen2.5-0.5B-Instruct` | `Qwen2.5-0.5B-Instruct_torch_dist` |
| 4B FP8 serving / BF16 training | `Qwen3-4B-FP8` | `Qwen3-4B-FP8-dequant_bridge_torch_dist` |
| 4B native FP8 | `Qwen3-4B-FP8` | `Qwen3-4B-FP8_torch_dist_release` |

`--hf-checkpoint` and `--ref-load` override those inferred directories.
`--train-data` and `--eval-data` override `gsm8k/train.parquet` and
`gsm8k/test.parquet` under `--data-dir`. Both parquet files must contain
`messages` and `label` columns.

For FP8-serving/BF16-training, the Megatron checkpoint must contain a
**scale-aware BF16 dequantization of the same FP8 HF weights**. A dtype cast
without the block scales changes the base model. Use a Megatron release-layout
checkpoint (`latest_checkpointed_iteration.txt` contains `release`).

For native FP8 training, use the compatible release-layout checkpoint retaining
FP8 weights and scales. The launcher sets `ORBIT_RESTORE_MODELOPT_STATE=0` and
the native FP8 backend settings in the Ray runtime environment. Do not
substitute the BF16 checkpoint from the other route.

Canonical OFT on FP8 fused projections requires the corresponding SGLang
per-slice FP8 OFT support. Sphere SGLang `submissionv2` at `d052a322f` lacks
that support; updating Orbit alone does not make this configuration runnable.
The reviewed SGLang port stack and compatible Megatron-Bridge export/native
FP8 code must be installed before using the relevant recipes.

## Run

```bash
python -m examples.adapter_first.run_qwen25_05b_gsm8k \
  --peft-method lora --model-dir /models --data-dir /datasets \
  --output-dir /outputs/adapter_first --run-id q25-lora-01

python -m examples.adapter_first.run_qwen3_4b_fp8_gsm8k \
  --peft-method oft --model-dir /models --data-dir /datasets \
  --output-dir /outputs/adapter_first --run-id q3-bf16-oft-01

python -m examples.adapter_first.run_qwen3_4b_fp8_native_oft_gsm8k \
  --model-dir /models --data-dir /datasets \
  --output-dir /outputs/adapter_first --run-id q3-native-oft-01
```

For a launch check, add `--num-rollout 3 --save-interval 1 --eval-interval 1000`.
Use `--help` for the available options. `--actor-gpus` and `--rollout-gpus` must
sum to `--num-gpus-per-node`; the default is 4+4=8. `--tensor-parallel-size`
controls trainer TP and defaults to 1. `--megatron-path` defaults to the
repository's `thirdparty/Megatron-LM`.

Checkpoints are loaded/saved under
`<output-dir>/<model-name>_<peft-method>_<run-id>`. Reuse the same run ID to
resume; use distinct IDs for separate runs, including the two FP8 trainer
routes. No checkpoint is deleted by these recipes.

W&B uses the standard Orbit configuration helper and is enabled only when
`WANDB_API_KEY` is set. Select `WANDB_ENTITY` explicitly before launching.
Project/group defaults come from the helper; `--extra-args` can supply explicit
`--wandb-project` and `--wandb-group` overrides. Never put credentials into
command-line examples or committed files. `--extra-env-vars` supplies overrides
to the Ray runtime environment through `ExecuteTrainConfig`.

## Port scope

These launchers consolidate the five orbit-baseline 500-rollout shell recipes.
Training hyperparameters are retained. Intentional changes are configurable
paths, standard W&B configuration, generated run IDs, explicit GPU count,
and removal of the native recipe's `ORBIT_LOG_WEIGHT_SYNC=1` debug setting.
Slurm allocation/time limits now belong to the caller. Model arguments still
come from `scripts/models/`.

The old `orbit-main` adapter-first comparison harness and OPD campaign wrappers
are not included. Neither are the duplicate three-rollout recipes or the
`v2check` temporary dependency-path overrides. Historical run results do not
validate these new entrypoints or a different dependency revision.
