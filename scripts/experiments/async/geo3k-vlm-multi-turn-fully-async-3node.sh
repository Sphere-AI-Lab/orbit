#!/bin/bash
#
# geo3k-vlm-multi-turn-fully-async-3node — Qwen3-VL-8B, geo3k multi-turn,
# fully-async rollout (examples/fully_async), 3 nodes (1 trainer + 2 samplers).
#
# Topology (24 GPUs across 3 nodes, 8 each):
#   - trainer: 1 node, 8 GPUs Megatron  (--actor-num-nodes 1 --actor-num-gpus-per-node 8)
#   - samplers: 2 nodes, 16 GPUs SGLang (--rollout-num-gpus 16, 1 GPU/engine)
#
# "Fully async" = examples/fully_async pattern: a persistent background worker
# keeps generating while training consumes already-finished groups. The async
# behavior comes from the ROLLOUT FUNCTION, not the driver: train_async.py alone
# (with the default rollout fn) is only step-overlapped async. Fully async needs
# BOTH of the following — exactly what examples/fully_async/run-*.sh pairs:
#   - the async driver train_async.py (set via ORBIT_TRAIN_ENTRY below) — NOT train.py;
#   - --fully-async;
#   - disaggregated layout (NO --colocate; train_async.py asserts not colocate);
#   - --max-weight-staleness 2 (stale groups recycled to the data buffer);
#   - No eval configured here: this recipe omits all EVAL_ARGS. (Since the
#     2026-08-18 sync, fully-async CAN evaluate — upstream #1740 added
#     shared-engine pause-the-world and dedicated eval fleets — so this is a
#     recipe choice now, not a hard limitation.)
#
# Hyperparameters (batch / samples / length / temperature / GRPO / optimizer) are
# aligned with examples/geo3k_vlm/multi_turn. Compute-level knobs are scaled up
# for 3x H200 (see PERF_ARGS / SGLANG_ARGS comments and async/README.md).
# No checkpoint saving in this example run (CKPT_ARGS has no --save).
#
# Multi-turn (geo3k) is layered on top via the per-sample custom generate fn; the
# fully-async worker calls generate_and_rm_group, which honors
# --custom-generate-function-path and multimodal inputs:
#   - --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
#   - --custom-config-path examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
#     (sets max_turns: 3 and rollout_interaction_env_path: examples.geo3k_vlm.multi_turn.env_geo3k)
#
# VLM-specific (same as the geo3k disagg recipe):
#   - MODEL_ARGS_ROTARY_BASE=5000000 before sourcing qwen3-8B.sh;
#   - --megatron-to-hf-mode bridge (VL weight export in disagg goes through AutoBridge);
#   - --multimodal-keys '{"image": "images"}';
#   - default THD qkv path with dynamic batching (no micro-batch-size=1 override);
#   - cudnn 9.16.0.29 in the env (else conv3d perf regression in torch 2.9).
#
# Submit (run id / W&B group come from JOB_NAME; see W&B notes below):
#   JOB_NAME=geo3k-async-mt-8b TIME=72:00:00 NODES=3 ORBIT_ENV_NAME=orbit \
#   bash scripts/slurm/submit.sh async/geo3k-vlm-multi-turn-fully-async-3node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# Opt into the async driver. ray_lifecycle.sh runs `python3 ${ORBIT_TRAIN_ENTRY:-train.py}`.
export ORBIT_TRAIN_ENTRY=train_async.py

# ---------------------------------------------------------------------------
# Resource metadata — read by submit.sh for sbatch
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=3
EXPERIMENT_TIME=72:00:00

# ---------------------------------------------------------------------------
# Asset metadata — read by submit.sh for the `hf download` step
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
    "VeraIsHere/geo3k_imgurl_processed"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
# HF_TORCHDIST_DIR intentionally unset — VLM loads HF directly via bridge.
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

# ---------------------------------------------------------------------------
# train_async.py args
# ---------------------------------------------------------------------------
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$ORBIT_REPO/scripts/models/qwen3-8B.sh"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --load           "$HF_MODEL_DIR"
   # No checkpoint saving for this example run (no --save / --save-interval).
   # Add them back to persist checkpoints, e.g.:
   #   --save "$ORBIT_REPO/runs/$RUN_NAME/ckpt" --save-interval 50
)

MULTIMODAL_ARGS=(
   --multimodal-keys '{"image": "images"}'
)

# Fully-async rollout driver + multi-turn per-sample generation.
ASYNC_ROLLOUT_ARGS=(
   --fully-async
   --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
   --custom-config-path           examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
   --max-weight-staleness         2
   # pre-sync worker semantics: aborted/stale groups go back to the data buffer
   # for regeneration (the class-based rollout's default is drop)
   --async-unused-samples-handler retry
   # Push fresh weights to engines every step; with staleness=2 the worker keeps
   # up to 2 versions of in-flight work before recycling.
   --update-weights-interval      1
)

# Aligned with examples/geo3k_vlm/multi_turn (run_geo3k_vlm_multi_turn.py):
# same batch/sample/length/temperature and GRPO/optimizer settings.
ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     problem
   --label-key     answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type       math
   --num-rollout   3000
   --rollout-batch-size       64
   --n-samples-per-prompt     8
   --rollout-max-response-len 4096
   --rollout-temperature      1.0
   --global-batch-size        512
   --balance-data
)

# Dynamic batching (matches the example) instead of micro-batch-size 1.
# H200 (141 GB) has headroom, so --max-tokens-per-gpu is bumped 4096 -> 16384
# (4x the example) to pack more tokens/microbatch and cut grad-accum steps.
# TP=4 on the 8-GPU trainer node => DP=2 (two replicas), halving TP comm vs TP=8.
# Full recompute is kept so the high token budget stays within memory.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# Monitoring only: compute & log rollout/entropy (policy entropy) during the
# logprob pass. Does NOT touch the loss (entropy-coef stays 0) — it just makes
# the entropy curve visible so we can watch entropy dynamics under async weight
# staleness and compare against sync.
MONITOR_ARGS=(
   --use-rollout-entropy
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # Dedicated sampler nodes (disagg), pure inference on H200 (141 GB): push the
   # static mem fraction up from the colocate/example default (0.6) to 0.85.
   # Qwen3-8B weights ~16 GB, leaving ~104 GB/engine for KV (~0.147 MB/token).
   --sglang-mem-fraction-static  0.7
   # In-flight generate concurrency. client_concurrency = server_concurrency *
   # rollout_num_gpus / per_engine = 64 * 16 = 1024 across the 16 engines, which
   # keeps the fully-async worker's queue full to feed training.
   --sglang-server-concurrency   64
   # Capture cuda graphs for a wide batch-size range (matches the example).
   --sglang-cuda-graph-bs        1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team    M3TRL
   --wandb-project async_envpack
   --wandb-group   "$RUN_NAME"
   --disable-wandb-random-suffix
   # WANDB_API_KEY comes from the env (exported by submit.sh / launch_orbit.sbatch);
   # we don't pass it on the CLI because it would leak into run.log and args.json.
   # launch_orbit.sbatch injects --wandb-run-id (= SLURM_JOB_ID) for stable resume.
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout  300
   --rollout-health-check-first-wait 60
   # Only kill an engine after 3 consecutive 300s timeouts. Run 20862 died CLUSTER_DEAD
   # from a SINGLE 30s false-positive health_generate timeout on a busy/paused engine that
   # was actually alive (its /health returned 200 just 3s before the kill), which then
   # cascaded through the broadcast collective into the trainer's update_weights().
   --rollout-health-check-max-consecutive-failures 3
)

# 1 trainer node (8 GPUs) + 2 sampler nodes (16 GPUs). NO --colocate.
LAYOUT_ARGS=(
   --actor-num-nodes        1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus       16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

ORBIT_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ASYNC_ROLLOUT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${MONITOR_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${FT_ARGS[@]}"
   "${MISC_ARGS[@]}"
)
