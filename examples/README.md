# Orbit Examples

Launchable training recipes:

- `high_precision/`: BF16 and high-precision training launchers.
- `low_precision/`: int4, fp8, and nvfp4 training launchers.

Each launcher is an independent bash entrypoint that defines its argument
arrays inline. The only shared code is `scripts/lib/` utilities for CUDA setup,
private Ray lifecycle, W&B handling, eval toggles, and checkpoint preflight.
To change a recipe value (batch size, learning rate, etc.), edit the launcher
file directly.

## Running

```bash
bash examples/low_precision/run-qwen3-4b-int4-math-oft.sh
```

Cross-cutting orchestration knobs can still be overridden inline:

```bash
ENABLE_WANDB=0 bash examples/low_precision/run-qwen3-4b-int4-math-oft.sh
```

### Providing site paths

Release launchers do not assume a specific cluster filesystem layout. Provide
the site-specific paths through environment variables when launching:

| Variable | Required | Meaning |
|---|---:|---|
| `ORBIT_VENV` | Usually | Python environment containing Orbit, Megatron-Bridge, Megatron-LM, SGLang, and CUDA Python packages. |
| `CUDA_HOME` / `ORBIT_CUDA_HOME_DEFAULT` | Usually | CUDA toolkit root. |
| `TRAIN_JSONL` | Yes | Training prompt JSONL. |
| `HF_CKPT` | Yes | HuggingFace checkpoint directory. For low-precision recipes this is the quantized HF checkpoint. |
| `MEGATRON_LOAD` | Yes | Megatron torch distributed checkpoint root. `latest_checkpointed_iteration.txt` is resolved automatically. |
| `TEST_JSONL` | If eval is enabled | Eval JSONL. Set `DISABLE_EVAL=1` to skip eval. |
| `SAVE_DIR` | No | Output checkpoint directory. Defaults under `orbit_ckpts/`. |
| `RUN_LOG` | No | Launcher stdout/stderr log path. Defaults under `logs/`. |
| `STAGE_HF_CKPT_TO` | No | Optional node-local copy destination for `HF_CKPT`. |
| `STAGE_MEGATRON_CKPT_TO` | No | Optional node-local copy destination for `MEGATRON_LOAD`. |

Example:

```bash
cd /path/to/orbit

export ORBIT_VENV=/path/to/orbit/.venv
export CUDA_HOME=/path/to/cuda-13.2
source examples/load_cuda13_2_orbit_env.sh

TRAIN_JSONL=/path/to/data/math/train.jsonl \
TEST_JSONL=/path/to/data/math/test.jsonl \
HF_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507-quantized.w4a16 \
MEGATRON_LOAD=/path/to/megatron/checkpoints/Qwen3-4B-Instruct-2507-w4a16 \
ENABLE_WANDB=0 \
bash examples/low_precision/run-qwen3-4b-int4-math-oft.sh
```

To inspect the final Python argv without starting Ray or loading the model:

```bash
ORBIT_DRY_RUN_ARGV=1 \
TRAIN_JSONL=/path/to/data/math/train.jsonl \
TEST_JSONL=/path/to/data/math/test.jsonl \
HF_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507-quantized.w4a16 \
MEGATRON_LOAD=/path/to/megatron/checkpoints/Qwen3-4B-Instruct-2507-w4a16 \
ENABLE_WANDB=0 \
bash examples/low_precision/run-qwen3-4b-int4-math-oft.sh
```

For a one-step smoke run, override the recipe schedule:

```bash
NUM_ROLLOUT=1 TOTAL_EPOCHS=1 TRAIN_ROWS=1 \
ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=1 \
DISABLE_EVAL=1 ENABLE_WANDB=0 \
TRAIN_JSONL=/path/to/data/math/train.jsonl \
HF_CKPT=/path/to/hf/Qwen3-4B-Instruct-2507-quantized.w4a16 \
MEGATRON_LOAD=/path/to/megatron/checkpoints/Qwen3-4B-Instruct-2507-w4a16 \
bash examples/low_precision/run-qwen3-4b-int4-math-oft.sh
```

Smoke-test overrides that shrink a run to a single step:

```bash
NUM_ROLLOUT=1 TOTAL_EPOCHS=1 TRAIN_ROWS=1 \
ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=1 \
DISABLE_EVAL=1 ENABLE_WANDB=0 \
    bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh
```

## Async PEFT double buffering

Async PEFT runs can opt into two-slot adapter double buffering for distributed
SGLang rollout engines. This is controlled by `ADAPTER_DOUBLE_BUFFER=1` in the
launchers, which passes `--adapter-double-buffer` to Orbit.

Requirements:

- Use an async, non-colocated launcher: `ORBIT_ENTRYPOINT=train_async.py` and
  `ORBIT_COLOCATE=0`.
- Use PEFT: `PEFT_METHOD=oft` or `PEFT_METHOD=lora`.
- Do not enable it for colocated/sync IPC runs; colocated behavior is
  intentionally unchanged.

Example:

```bash
ADAPTER_DOUBLE_BUFFER=1 \
    bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh
```

The default is off. In the current benchmark results, OFT async benefits from
double buffering on the tested 4B config, while LoRA async is roughly neutral.

## Env-knob reference

> **Note:** Recipe values (batch sizes, learning rates, parallelism degrees,
> checkpoint paths, etc.) are now inlined directly in each launcher file.
> Edit the launcher to change them. The table below documents only
> cross-cutting orchestration knobs that are read by `scripts/lib/` helpers
> or that override launcher behaviour at call-time (e.g. `ENABLE_WANDB`,
> `DISABLE_EVAL`, `ORBIT_DRY_RUN_ARGV`). Knobs like `ROLLOUT_BATCH_SIZE`,
> `LR`, `OFT_BLOCK_SIZE`, etc. are still valid environment overrides but
> their defaults now live in each launcher, not in a shared env-knob ladder.

Knobs are grouped by domain. Defaults shown are the launcher-level fallback;
each leaf launcher may pin its own values for the recipe.

### Recipe identity

| Knob | Meaning |
|---|---|
| `LAUNCHER_NAME` | Human-readable run name; used in W&B group and log path. |
| `PRECISION_PROFILE` | One of `bf16`, `fp8`, `int4`, `nvfp4`. Selects the preflight checks and quantization defaults applied by `scripts/lib/preflight.sh`. |
| `REQUIRE_MEGATRON_LOAD` | `1` = refuse to start without a valid Megatron distributed checkpoint at `MEGATRON_LOAD`. `0` = allow HF-only startup. |

### Model spec

| Knob | Meaning |
|---|---|
| `MODEL_ARGS_FILE` | Path to the per-model arg shim under `orbit_plugins/model_args/`. Sets `MODEL_ARGS=(...)` consumed by Megatron-Bridge. |
| `MODEL_ARGS_ROTARY_BASE` | Rotary base (theta). Qwen3 family: `1e4`. Qwen3-Instruct-2507: `5e6`. |

### Data

| Knob | Meaning |
|---|---|
| `DATASET` | Dataset folder name under `DATA_ROOT`. |
| `DATA_ROOT` | Base directory holding `<dataset>/{train,test}.jsonl`. |
| `TRAIN_JSONL` / `TEST_JSONL` | Direct overrides for the train/test jsonl paths. Slice syntax `path@[:N]` is supported. |
| `PROMPT_FILE` | Prompt file consumed by the runtime parity preflight. |

### Checkpoints

| Knob | Meaning |
|---|---|
| `HF_CKPT` | HuggingFace source checkpoint; the trainer reads its tokenizer and config from here. |
| `MEGATRON_LOAD` | Megatron distributed checkpoint root. The trainer resolves `latest_checkpointed_iteration.txt` to `iter_*` automatically. |
| `LOAD_CKPT` | What `--load` actually receives. Defaults to `MEGATRON_LOAD`, then falls back to `HF_CKPT`. |
| `SAVE_DIR` | Output checkpoint root. Created if missing. |
| `SAVE_INTERVAL` | Save (and retain) every N rollout steps. |
| `STAGE_HF_CKPT_TO` | If set, rsync `HF_CKPT` to this path on the local node before training. Useful for slow shared filesystems. |
| `STAGE_MEGATRON_CKPT_TO` | Same as above, for the Megatron checkpoint. |
| `FORCE_STAGE_HF_CKPT` / `FORCE_STAGE_MEGATRON_CKPT` | `1` = always re-rsync; `0` = reuse the staged copy when it looks valid. |
| `RUN_LOG` | tee target for the launcher's stdout/stderr. |

### Resources

| Knob | Meaning |
|---|---|
| `GPUS_PER_NODE` | Number of GPUs the actor occupies. |
| `RAY_NUM_CPUS` | CPU resource handed to the Ray head. |

### Parallelism

| Knob | Meaning |
|---|---|
| `TENSOR_MODEL_PARALLEL_SIZE` | TP degree. |
| `PIPELINE_MODEL_PARALLEL_SIZE` | PP degree. |
| `CONTEXT_PARALLEL_SIZE` | CP degree (sequence parallel for long contexts). |
| `EXPERT_MODEL_PARALLEL_SIZE` | EP degree for MoE models. |
| `EXPERT_TENSOR_PARALLEL_SIZE` | TP within each expert. |
| `SEQUENCE_PARALLEL` | Truthy = pass `--sequence-parallel` to Megatron. |
| `MAX_TOKENS_PER_GPU` | Dynamic micro-batch budget per GPU; controls memory headroom vs. throughput. |

### Training schedule

| Knob | Meaning |
|---|---|
| `ROLLOUT_BATCH_SIZE` | Prompts sampled per rollout step. |
| `N_SAMPLES_PER_PROMPT` | Completions generated per prompt (group size for GRPO/GSPO). |
| `ROLLOUT_MAX_PROMPT_LEN` | Hard cap on prompt token count. |
| `ROLLOUT_MAX_RESPONSE_LEN` | Max generated tokens per completion. |
| `ROLLOUT_MAX_CONTEXT_LEN` | Combined prompt+response cap. |
| `ROLLOUT_TEMPERATURE` | Sampling temperature for rollouts. |
| `GLOBAL_BATCH_SIZE` | Total prompts × samples consumed per gradient step. |
| `TOTAL_EPOCHS` | Passes over the prompt dataset. |
| `NUM_ROLLOUT` | Override the auto-computed rollout-step count. |

### Optimizer

| Knob | Meaning |
|---|---|
| `LR` | Adam learning rate. |
| `LR_DECAY_STYLE` | One of `constant`, `cosine`, `linear`, ... |
| `WEIGHT_DECAY` | AdamW weight decay. |
| `ADAM_BETA1` / `ADAM_BETA2` | Adam betas. |
| `OPTIMIZER_CPU_OFFLOAD` | Truthy = offload optimizer state to host RAM, with overlapped D2H/H2D. |
| `USE_PRECISION_AWARE_OPTIMIZER` | Truthy = use Megatron's precision-aware optimizer. |

### RL

| Knob | Meaning |
|---|---|
| `ADVANTAGE_ESTIMATOR` | One of `grpo` (default), `gspo`, `ppo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `on_policy_distillation`. |
| `USE_KL_LOSS` | Truthy = add KL-to-reference loss term. PEFT KL is computed with adapters disabled, so no separate ref-load is needed. |
| `KL_LOSS_COEF` | KL term weight. |
| `KL_LOSS_TYPE` | `low_var_kl`, `kl`, etc. |
| `ENTROPY_COEF` | Entropy bonus weight. |
| `EPS_CLIP` / `EPS_CLIP_HIGH` | PPO/GRPO clipping ranges. |
| `GAMMA` / `LAMBD` | PPO discount + GAE lambda (PPO only). |
| `DISABLE_GRPO_STD_NORMALIZATION` | Truthy = skip per-group std normalization in GRPO/GSPO. |

### Rollout backend (SGLang)

| Knob | Meaning |
|---|---|
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | GPUs per SGLang engine instance. |
| `SGLANG_MEM_FRACTION_STATIC` | Fraction of free GPU memory reserved for the static KV cache. |
| `SGLANG_CONTEXT_LENGTH` | Per-engine max context length. |
| `SGLANG_MAX_RUNNING_REQUESTS` | Concurrency cap. |
| `SGLANG_MAX_PREFILL_TOKENS` / `SGLANG_CHUNKED_PREFILL_SIZE` | Prefill scheduling knobs. |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` (default for INT4), `triton`, `fa3`, ... |
| `SGLANG_QUANTIZATION` | Engine-level quant scheme (`compressed-tensors`, etc.). |
| `SGLANG_EXPERT_PARALLEL_SIZE` | EP within the SGLang engine. |
| `SGLANG_DISABLE_CUDA_GRAPH` / `SGLANG_ENFORCE_EAGER` | Disable CUDA graph capture (debugging). |
| `SGLANG_FP8_GEMM_BACKEND` | FP8 GEMM kernel selection (FP8 recipes). |
| `SGLANG_TORCHAO_CONFIG` | Path to a torchao quant config (if applicable). |

### Eval

| Knob | Meaning |
|---|---|
| `DISABLE_EVAL` | `1` skips eval entirely. |
| `SKIP_EVAL_BEFORE_TRAIN` | `1` skips the step-0 eval (saves a cold-start pass). |
| `EVAL_INTERVAL` | Eval every N rollout steps. |
| `N_SAMPLES_PER_EVAL_PROMPT` | Completions per eval prompt. |
| `EVAL_MAX_RESPONSE_LEN` | Eval generation cap. |
| `EVAL_TOP_K` | Eval sampling top-k (`1` = greedy). |
| `EVAL_GENERATE_MAX_CONCURRENCY` | Concurrency cap for eval generation. |

### PEFT (OFT / LoRA)

| Knob | Meaning |
|---|---|
| `PEFT_METHOD` | `oft` or `lora`. |
| `TARGET_MODULES` | Comma-separated Megatron module names to wrap (e.g. `linear_qkv,linear_proj,linear_fc1,linear_fc2`) or `all-linear`. |
| `OFT_BLOCK_SIZE` | Rotation block size. Smaller = more parameters, more expressive. Must divide each linear's hidden dim. |
| `OFT_EPS` | Numerical stability constant for OFT. |
| `OFT_COFT` | `1` enables Cayley-OFT (orthogonality via Cayley transform). |
| `OFT_BLOCK_SHARE` | `1` ties the rotation matrix across blocks. |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_DROPOUT` | LoRA hyperparameters. |
| `ADAPTER_DOUBLE_BUFFER` | `1` enables `--adapter-double-buffer` for async distributed PEFT rollout engines. Default is `0`. |

### Quantization (FP8)

| Knob | Meaning |
|---|---|
| `MEGATRON_OFT_FP8_ACTIVATION_QUANT` | Activation quant scheme for OFT under FP8 (`w8a8`, `none`). |
| `MEGATRON_KEEP_NATIVE_FP8_WEIGHTS` | Keep FP8 weights in their native dtype across save/load. |
| `MEGATRON_ACTIVATION_FP8` | Forward-pass activation FP8 mode. |
| `ROLLOUT_QUANTIZATION` | Quant scheme used by the rollout engine (`fp8`, ...). |

### Quantization (INT4)

| Knob | Meaning |
|---|---|
| `OPEN_TRAINING_INT4_FAKE_QAT_FLAG` | `1` = train with fake-quant straight-through; `0` = train in full precision and requantize at save-time. |
| `OPEN_TRAINING_INT4_GROUP_SIZE` | Group size for W4A16 (must match the HF checkpoint). |

### Parity preflight gates

| Knob | Meaning |
|---|---|
| `SKIP_FP8_CHECKPOINT_PREFLIGHT` / `SKIP_INT4_CHECKPOINT_PREFLIGHT` / `SKIP_NVFP4_CHECKPOINT_PREFLIGHT` | `1` skips the checkpoint parity preflight (use only for command-path smoke checks). |
| `RUN_FP8_RUNTIME_PREFLIGHT` / `RUN_INT4_RUNTIME_PREFLIGHT` | `1` runs the step-0 logprob parity check between Megatron and SGLang. |
| `MAX_LOGPROB_ABS_DIFF` | Per-token logprob tolerance for the runtime parity check. |
| `MIN_TOPK_AGREEMENT` | Minimum top-k token agreement fraction. |

### Megatron / TE flags

| Knob | Meaning |
|---|---|
| `MEGATRON_PATH` | Path to the Megatron-LM checkout used for `PYTHONPATH`. |
| `ATTENTION_BACKEND` | `flash`, `fused`, `unfused`. |
| `ATTENTION_SOFTMAX_IN_FP32` | Truthy = pass `--attention-softmax-in-fp32`. |
| `TRUST_REMOTE_CODE` | Truthy = allow custom HF model code. |
| `OFFLOAD_TRAIN` / `OFFLOAD_TRAIN_ASYNC` / `OFFLOAD_ROLLOUT` | Memory offload toggles. Each has an explicit `--offload-...` / `--no-offload-...` form. |
| `UPDATE_WEIGHT_BUFFER_SIZE` | Optional byte size for weight-update buckets; larger values reduce adapter sync chunk count at the cost of more temporary memory. |
| `CUDA_GRAPH_IMPL` / `CUDA_GRAPH_SCOPE` | CUDA graph capture controls. |
| `USE_TE_RNG_TRACKER` | Truthy = use TransformerEngine's RNG tracker. |
| `SKIP_NAN_CHECK_IN_LOSS_AND_GRAD` | Truthy = pass `--no-check-for-nan-in-loss-and-grad`. |

### Ray lifecycle

| Knob | Meaning |
|---|---|
| `ORBIT_RAY_LIFECYCLE` | `private` (default; reserves its own port range and tmp dir) or `legacy` (clobbers any existing Ray cluster on the node). |
| `RAY_TEMP_DIR` | Override the Ray tmp dir. |
| `RAY_HEAD_PORT` | Override the head port (private lifecycle picks one in `24000-29999` if unset). |
| `RAY_WORKER_PORT_RANGE_SIZE` | Width of the worker port range. |
| `MASTER_ADDR` | Defaults to `127.0.0.1` for single-node runs. |
| `ORBIT_PORT_LOCK_DIR` | Lock-file directory used to serialize port reservation across concurrent launchers. |

### Logging / observability

| Knob | Meaning |
|---|---|
| `ENABLE_WANDB` | `auto` = enabled iff `$HOME/.wandb_key` exists; `0` disables. |
| `WANDB_PROJECT` / `WANDB_GROUP` | W&B project + group names. |
| `ORBIT_LAUNCHER_XTRACE` | `1` enables `set -x` for the launcher (very noisy; for debugging only). |
| `ORBIT_DEBUG_MODE` | `rollout` runs only the rollout phase; `train` runs only the train phase. |

### Tool-script knobs

These are read by `scripts/lib/tool_env.sh` (sourced by every leaf launcher
and standalone tool/parity script):

| Knob | Meaning |
|---|---|
| `ORBIT_LOAD_CUDA_MODULES` | `0` skips `module load cuda/13.2 nccl` (use when `module` is unavailable). |
| `CUDA_HOME` | Override the auto-detected CUDA root. |
| `MEGATRON_BRIDGE_ROOT` | If set, prepended to `PYTHONPATH` before `ORBIT_ROOT`. |
| `DEFAULT_OUTPUT_ROOT` | Base directory for `default_output_path` (used by conversion scripts). |
