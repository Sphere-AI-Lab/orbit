# Fully-async experiment recipes

Recipes here run the **fully-asynchronous** rollout pattern from
[`examples/fully_async`](../../../examples/fully_async): a persistent background
worker keeps generating rollouts while training consumes already-finished
groups, instead of the per-step generate→train barrier.

## `train_async.py` is not the same thing as "fully async"

This is the easy thing to get wrong:

| Mode | Driver | Rollout function | Behavior |
|------|--------|------------------|----------|
| sync | `train.py` | default | generate a full batch, then train (barrier) |
| step-overlapped async | `train_async.py` | default | next rollout prefetched while training the current one |
| **fully async** | `train_async.py` | `generate_rollout_fully_async` | a global worker generates **continuously**; training just drains finished groups |

The "fully async" behavior comes from the **rollout function**, not the driver.
`train_async.py` on its own is only step-overlapped. Fully async needs **both**
pieces:

```bash
MILES_TRAIN_ENTRY=train_async.py   # driver
--fully-async                      # worker (miles.rollout.fully_async_rollout.FullyAsyncRolloutFn)
```

## Runnable multi-turn recipes

- `geo3k-vlm-multi-turn-fully-async-3node.sh` — Qwen3-VL-8B, geo3k multi-turn,
  3 nodes (1 trainer + 2 samplers). Baseline fully-async recipe.
- `geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh` — same 3-node topology,
  but keeps two rollout batches actively generating in the background worker.
- `geo3k-vlm-multi-turn-fully-async-prefetch2-4node.sh` — same prefetch window,
  with two trainer nodes and two sampler nodes to test whether trainer throughput
  is the bottleneck. This is the main prefetch scale-up recipe.

## Single-turn diagnostic recipe

- `geo3k-vlm-single-turn-fully-async-3node.sh` — the only single-turn Geo3K
  recipe in this folder. Keep it as a comparison path for rollout/trainer
  mismatch analysis; the main training experiments should use the multi-turn
  recipes above.

## Experimental probes

Recipes under `experimental/` are not clean comparison runs. They exist to
exercise a specific infrastructure path and may require follow-up fixes before
they are useful training experiments.

- `experimental/geo3k-vlm-multi-turn-fully-async-cp2-3node.sh` — CP2 +
  `--allgather-cp` probe. It is documented as not currently runnable as a clean
  comparison recipe for Qwen3-VL bridge init; read
  `experimental/cp2-calculate-per-token-loss.md` first.

## How the slurm harness runs the async driver

`scripts/slurm/lib/ray_lifecycle.sh` launches `python3 ${MILES_TRAIN_ENTRY:-train.py}`.
The recipe sets `export MILES_TRAIN_ENTRY=train_async.py`; everything else
(asset download, ray cluster, W&B run-id) is the same path as the sync recipes.
The default is unchanged (`train.py`), so existing recipes are unaffected.

## Submit

```bash
JOB_NAME=geo3k-async-mt-8b TIME=72:00:00 NODES=3 MILES_ENV_NAME=miles \
bash scripts/slurm/submit.sh async/geo3k-vlm-multi-turn-fully-async-3node
```

- **W&B**: `entity=M3TRL`, `project=async_envpack`, `group=$JOB_NAME`. The
  run-id is injected by `launch_miles.sbatch` as `SLURM_JOB_ID` (stable across
  requeue). `WANDB_API_KEY` comes from `~/.config/secrets.env`.
- **cudnn**: the env must have `nvidia-cudnn-cu12==9.16.0.29` (else conv3d perf
  regression in torch 2.9). Same caveat as the geo3k examples.

## Topology (3 nodes × 8 H200 = 24 GPUs)

```
node 0  trainer   8 GPUs Megatron  --actor-num-nodes 1 --actor-num-gpus-per-node 8
node 1  sampler   8 GPUs SGLang   \ --rollout-num-gpus 16  (1 GPU / engine => 16 engines)
node 2  sampler   8 GPUs SGLang   /
```

No `--colocate` (train_async.py asserts not colocate).

## Hyperparameters

Learning-relevant settings are **aligned with `examples/geo3k_vlm/multi_turn`**
so results stay comparable:

| Param | Value | Source |
|-------|-------|--------|
| `--rollout-batch-size` | 64 | example |
| `--n-samples-per-prompt` | 8 | example |
| `--global-batch-size` | 512 | example |
| `--rollout-max-response-len` | 4096 | example |
| `--rollout-temperature` | 1.0 | example |
| `--num-rollout` | 3000 | example |
| GRPO (`eps-clip` 0.2 / `eps-clip-high` 0.28, kl 0, entropy 0) | — | example |
| optimizer (adam, lr 1e-6, wd 0.1, β 0.9/0.98) | — | example |
| multi-turn (`max_turns: 3`, `env_geo3k`) | — | `geo3k_vlm_multi_turn_config.yaml` |

Async-specific: `--max-weight-staleness 2`, `--update-weights-interval 1`.
The prefetch2 recipes also set `--fully-async-prefetch-batches 2`. Since the
upstream fully-async rewrite the flag no longer drives an example worker: it
derives `--async-max-concurrent-samples` (`rollout_batch_size * prefetch *
n_samples_per_prompt`), so `rollout_batch_size * 2` prompt groups stay actively
generating while training consumes finished groups. Pass that absolute bound or
this depth knob, not both.

## Scaling for H200 (the compute knobs we opened up)

Only **compute-level** knobs were scaled — the learning math above is unchanged.

**Trainer (8× H200, 141 GB each), Qwen3-VL-8B, TP=4 → DP=2, distributed optimizer**

Rough per-GPU static memory:
- weights bf16, TP=4: 16 GB / 4 ≈ **4 GB**
- Adam fp32 master+m+v = 12 B/param = 96 GB, sharded over DP×TP=8 ≈ **12 GB**
- fp32 grads (`--accumulate-allreduce-grads-in-fp32`), sharded ≈ **~8 GB**
- static ≈ **~24 GB**, leaving **~117 GB** for activations.

With **full recompute**, only layer inputs are stored
(`tokens × hidden × 2 B × layers`, sequence-parallel /TP):
`16384 × 4096 × 2 × 36 / 4 ≈ 1.2 GB` + one recomputed layer's peak — trivial
against 117 GB. So we raise `--max-tokens-per-gpu` **4096 → 16384** (4×); fewer
grad-accum microsteps per training step ⇒ faster step until comm-bound. TP=4
(vs 8) halves tensor-parallel comm and gives DP=2. You can push
`--max-tokens-per-gpu` to ~24576 or relax recompute to `selective` for more
speed if a run stays well under memory.

**Samplers (16× H200), Qwen3-8B, TP=1, mem-fraction 0.85**

- reserved ≈ 0.85 × 141 ≈ 120 GB; weights 16 GB ⇒ **~104 GB KV / engine**.
- KV/token = `2 × layers(36) × kv_groups(8) × head_dim(128) × 2 B` ≈ **0.147 MB**.
- 104 GB / 0.147 MB ≈ **~700k tokens** of KV per engine.
- `--sglang-server-concurrency 64` ⇒ client concurrency
  `64 × 16 engines = 1024` in-flight generate requests, which keeps the worker
  queue full to feed training. SGLang admission-controls server-side, so this is
  a ceiling, not a guarantee.

Net: same objective as the example, but the trainer packs 4× the tokens/step and
the 16-engine sampler pool stays saturated, so the async pipeline rarely stalls.

## Limitations (inherited from `examples/fully_async`)

- **No eval** — `generate_rollout_fully_async` raises on `evaluation=True`, and
  `train_async.py` would call it that way on `--eval-interval`. This recipe omits
  all eval args; watch the `train/` and `rollout/` W&B panels instead.
- **No checkpoint saving** in this example run (`CKPT_ARGS` has no `--save`). Add
  `--save <dir> --save-interval N` to persist.
- Ordering is best-effort; error handling in the worker is minimal.
