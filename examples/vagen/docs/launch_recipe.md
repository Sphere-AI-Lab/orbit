# Launch-recipe design

Covers the shape of vagen experiment launchers under
`scripts/experiments/vagen-*.sh`. They all share the same skeleton: build
arg arrays (CKPT / ROLLOUT / GRPO / PERF / SGLANG / EVAL / FT / LAYOUT),
concatenate into `MILES_ARGS`, and hand off to the slurm launcher. This
doc captures the WHY behind each choice.

## VAGEN → miles knob mapping

VAGEN's `train_grpo_qwen25vl3b.sh` + `train_sokoban_vision.yaml` rewritten
as miles CLI flags. Same values verbatim:

| VAGEN | miles |
|---|---|
| `data.train_batch_size=32` | `--rollout-batch-size 32` |
| `actor_rollout_ref.rollout.n=8` | `--n-samples-per-prompt 8` |
| `ppo_mini_batch_size=32` (×rollout.n=256) | `--global-batch-size 256` |
| `actor.optim.lr=1e-6` | `--lr 1e-6` |
| `algorithm.kl_ctrl.kl_coef=0.0` | `--kl-coef 0.0` |
| `actor.entropy_coeff=0.0` | `--entropy-coef 0.0` |
| `actor.use_kl_loss=False` | `--kl-loss-coef 0.0` |
| `algorithm.adv_estimator=grpo` | `--advantage-estimator grpo` |
| (eps_clip default 0.2) | `--eps-clip 0.2` |
| `data.max_response_length=4000` | `--rollout-max-response-len 4096` |
| `trainer.total_training_steps=401` | `--num-rollout 400` |
| `gpu_memory_utilization=0.6` | `--sglang-mem-fraction-static 0.6` |
| `rollout.tensor_model_parallel_size=1` | `--rollout-num-gpus-per-engine 1` |
| `model.enable_gradient_checkpointing=True` | `--recompute-* full/uniform/1` |
| (VAGEN `base_seed=0`) | `--seed 0` **(load-bearing)** |

`--seed 0` mirrors VAGEN's `gym_agent_dataset` default. With the prebuilt
samples.jsonl this does NOT drive env reproducibility (seeds are baked into
the rows) and does NOT affect prompt shuffle / eval sampling (those use
`--rollout-seed`, default 42). It's a label-level alignment with VAGEN main.

## Deliberate deviations from VAGEN main

1. **`--save-interval 1000000`** (sokoban only): debug_dump JSONs land but
   no Megatron checkpoint hits disk in 400 rollouts. (frozenlake recipes
   use `--save-interval 100`.)
2. **Filter / dynamic-sampling (Stage 2 WIP)**: only when `MILES_VAGEN_DAPO=1`.
   Adds `--eps-clip-high 0.28` + dynamic-sampling filter +
   `--over-sampling-batch-size` (defaults to `2 × rollout_batch_size`).
3. **Layout**: 8 GPUs colocated (TP=4) vs VAGEN-main's 4-GPU FSDP. Per-update
   sample count differs (256 vs 64).
4. **Prompt format**: Sokoban yaml overrides to `free_think` (VAGEN-main env
   default is `wm`). See `configs/sokoban_train_env.yaml` for the rationale.

## Eval cadence

Mirrors VAGEN's `trainer.val_before_train=True` + `test_freq=20`:

- `--eval-interval 20` → eval every 20 rollouts.
- `--n-samples-per-eval-prompt 1` matches VAGEN's per-prompt single-sample
  eval (`val_*_vision.yaml` has no `rollout.n` override on the val path).
- miles' default fires eval at `rollout_id=0` unless `--skip-eval-before-train`.

Eval consumes a precomputed jsonl via `--eval-prompt-data`. Same
`generate()` function as train; only the env seed range differs (val seeds
[10001,10256] vs train [1,10000]).

`VAGEN_EVAL_NAME` (e.g. `sokoban_val`, `frozenlake_val`) is the wandb
namespace. By design it does NOT encode the heldout/non-heldout split —
which split a run used is recorded in `args.json` and the dataset artifact,
not in the wandb metric path, so wandb comparisons across runs share an
axis.

**Override default eval**:

```
VAGEN_EVAL_DATA=/path/to/eval.jsonl \
bash scripts/slurm/submit.sh <recipe-name>
```

Default `eval/samples.jsonl` is the heldout (map-disjoint-from-train) split
built by `examples/vagen/scripts/<dataset>-main.sh`.

**Why no `--eval-max-prompt-len`** (sokoban specifically): miles'
`filter_long_prompt` (in `data.py`) returns `False` (not the sample list)
when the sample prompt is a messages list — which it is when
`--apply-chat-template` is off and `--multimodal-keys` is on. With
`max_length=None` the filter is skipped entirely. Our prompt is overridden
inside our `generate()` anyway (rebuilt from `env.reset`), so length-
filtering on the placeholder is meaningless.

## Wandb

- `--use-wandb --wandb-project miles-imp --wandb-group "$RUN_NAME"`.
- `--disable-wandb-random-suffix` keeps the wandb UI name = `$RUN_NAME`
  exactly. Without it, miles adds a random 6-char suffix and a `-RANK_0`
  suffix (`wandb_utils.py`), which breaks grouping/diffing across re-runs.

## Performance args

VLM colocated single-node:

```
--tensor-model-parallel-size 4
--sequence-parallel
--pipeline-model-parallel-size 1
--context-parallel-size 1
--expert-model-parallel-size 1
--expert-tensor-parallel-size 1
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--qkv-format bshd
--micro-batch-size 1
```

`TP=4` matches `geo3k-vlm-multi-turn-colocate-1node`.
SGLang: 1 GPU per engine, mem fraction 0.6.

## Qwen3-VL <think> tag swap

`Qwen3-VL-2B-Instruct` has `<think>` / `</think>` as added_tokens (ids
151667/151668) with `special: false`, but they were never seen during the
Instruct training run — their embeddings are effectively random, so the
policy can't reliably emit them.

The qwen3vl2b recipe sets `export VAGEN_THINK_TAG=thinking`, which swaps
the reasoning block to `<thinking>...</thinking>` (tokenizes as multi-token
sequences the model has actually seen). The env var is task-agnostic and
read by every VAGEN env's prompt/parser module (see
`vagen/envs/frozenlake/utils/prompt.py:_think_tag`). Qwen2.5-VL recipes
don't set it, keeping the default `<think>` form.

`launch_miles.sbatch`'s `--export=ALL` propagates the var to all ranks.

## Qwen3-VL rotary_base

`Qwen3-VL-2B`'s LLM trunk = `Qwen3-1.7B` with `rotary_base` bumped to 5e6.
The qwen3vl2b recipe sets `MODEL_ARGS_ROTARY_BASE=5000000` BEFORE sourcing
`scripts/models/qwen3-1.7B.sh`, which forwards the override into the
`MODEL_ARGS` array. Without it, the 1.7B script's `rotary_base=1e6` default
silently mis-position-encodes the model.

## Fault tolerance

```
--use-fault-tolerance
--rollout-health-check-interval 30
--rollout-health-check-timeout  30
--rollout-health-check-first-wait 60
```

Watchdog probes the SGLang router every 30s after a 60s startup grace;
30s timeout per probe. Wires into the slurm launcher's bad-node detection.

## Scaling knobs (env var overrides)

| Var | Default |
|---|---|
| `MILES_SCRIPT_NUM_ROLLOUT` | 400 |
| `MILES_SCRIPT_ROLLOUT_BATCH_SIZE` | 32 |
| `MILES_SCRIPT_N_SAMPLES_PER_PROMPT` | 8 |
| `MILES_SCRIPT_GLOBAL_BATCH_SIZE` | rollout_batch × n_samples (= 256) |
| `MILES_SCRIPT_OVER_SAMPLING_BATCH_SIZE` | rollout_batch × 2 (DAPO only) |
| `MILES_VAGEN_DAPO` | 0 (set to 1 for filter recipe) |

The `vagen-sokoban-main-*-global_bsz32.sh` overlay sets
`MILES_SCRIPT_GLOBAL_BATCH_SIZE=32` (8 PPO updates per rollout vs 1).
