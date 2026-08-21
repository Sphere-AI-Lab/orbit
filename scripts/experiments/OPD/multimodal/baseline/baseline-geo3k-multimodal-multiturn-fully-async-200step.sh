#!/bin/bash
#
# Canonical 200-step Geo3K multimodal multi-turn OPD baseline.
#
#   node 1 (head): Qwen3-VL-30B-A3B-Thinking teacher, SGLang TP=8
#   node 2:        Qwen3-VL-8B-Instruct Megatron actor, TP=4 / DP=2
#   node 3:        eight single-GPU Qwen3-VL-8B-Instruct rollout engines
#
# The objective is sampled RKLD-PG plus trainer-side Top-K + Rest DAgger.
# Student SGLang emits the behavior-policy log-probabilities used as both the
# detached RKLD reference and PPO denominator. The trainer still performs its
# normal gradient-bearing actor forward, but does not run a separate no-grad
# old-policy log-probability recomputation.
#
# This is a frozen experiment-level recipe. It does not source the numbered
# milestone wrappers; only the canonical Qwen3-8B model definition is shared.
# Task reward is telemetry only, and the run never saves a checkpoint.
#
# Submit:
#   HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
#     OPD/multimodal/baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}

EXPERIMENT_NODES=3
EXPERIMENT_TIME=24:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
   "VeraIsHere/geo3k_imgurl_processed"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

# submit.sh downloads only HF_MODEL_REPO, so the fixed teacher is staged and
# owned separately on the Ray head node.
# Read-only model tree was reorganized 2026-08-10: the big VL checkpoints moved
# from hf_cache/models to /data/shared/models (same convention as the MoE recipes).
HF_MODELS_ROOT=${HF_MODELS_ROOT:-/data/shared/models}
OPD_TEACHER_MODEL_DIR=${OPD_TEACHER_MODEL_DIR:-"$HF_MODELS_ROOT/Qwen3-VL-30B-A3B-Thinking"}
OPD_TEACHER_PORT=${OPD_TEACHER_PORT:-13141}
OPD_TEACHER_TP=8
OPD_TEACHER_GPUS=0,1,2,3,4,5,6,7
OPD_TEACHER_MEM_FRACTION=0.8
OPD_TEACHER_EXTRA_ARGS=${OPD_TEACHER_EXTRA_ARGS:-}

export MILES_RAY_HEAD_NUM_GPUS=0
OPD_TEACHER_HOST=${OPD_TEACHER_HOST:-${HEAD_IP:-$(hostname -I | awk '{print $1}')}}
OPD_TEACHER_URL=${OPD_TEACHER_URL:-"http://${OPD_TEACHER_HOST}:${OPD_TEACHER_PORT}"}

if [[ ! -f "$OPD_TEACHER_MODEL_DIR/config.json" ]]; then
   echo "FATAL: OPD VLM teacher not found at $OPD_TEACHER_MODEL_DIR" >&2
   echo "  hf download Qwen/Qwen3-VL-30B-A3B-Thinking --local-dir $OPD_TEACHER_MODEL_DIR" >&2
   return 1 2>/dev/null || exit 1
fi

export ENVPACK_LOCAL_SERVER_CMD="TRITON_CACHE_DIR=/tmp/triton_${USER:-unknown}/opd_vlm_teacher \
   CUDA_VISIBLE_DEVICES=$OPD_TEACHER_GPUS python3 -m sglang.launch_server \
   --model-path $OPD_TEACHER_MODEL_DIR \
   --host 0.0.0.0 \
   --port $OPD_TEACHER_PORT \
   --tp $OPD_TEACHER_TP \
   --chunked-prefill-size 8192 \
   --mem-fraction-static $OPD_TEACHER_MEM_FRACTION \
   $OPD_TEACHER_EXTRA_ARGS"
export ENVPACK_LOCAL_SERVER_HEALTH="$OPD_TEACHER_URL/health_generate"
export ENVPACK_SERVER_WAIT_TIMEOUT=${ENVPACK_SERVER_WAIT_TIMEOUT:-1800}

# Fully async is a scheduling choice. It does not add an OPD-specific backend.
export MILES_TRAIN_ENTRY=train_async.py

MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-8B.sh"
MODEL_ARGS+=(--megatron-to-hf-mode bridge)

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL_DIR"
   --load "$HF_MODEL_DIR"
)

MULTIMODAL_ARGS=(
   --multimodal-keys '{"image": "images"}'
)

ROLLOUT_ARGS=(
   --prompt-data "$HF_TRAIN_DATA"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
   --custom-config-path examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path miles.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
   --rm-url "$OPD_TEACHER_URL/generate"
   --rm-type math
   --opd-log-task-reward
)

OPD_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1
   --opd-log-prob-top-k 0
   --opd-dagger-top-k 2
   --opd-dagger-coef 0.5
   --opd-dagger-loss cross_entropy
   # q_adv and q_den both use the SGLang behavior-policy snapshot. This skips
   # the separate trainer old-logprob forward; q_theta remains live in training.
   --use-rollout-logprobs
   # Explicitly preserve Miles' existing symmetric PPO defaults.
   --eps-clip 0.2
   --eps-clip-high 0.2
   --sglang-mm-exact-scoring-suffix
   --opd-scoring-timeout 600
   # Overridable transport knobs. The frozen canonical values stay 0/0; the
   # scaled milestone-12 topology (128 unbounded in-flight scoring requests
   # against one TP=8 teacher) exhausts the connection-level stale-socket
   # retries in one attempt (job 28156, teacher healthy throughout), so its
   # launches bound concurrency and allow request-level retries per the
   # Gate 29 failure-triage lever.
   --opd-scoring-max-inflight "${OPD_SCORING_MAX_INFLIGHT:-0}"
   --opd-scoring-retries "${OPD_SCORING_RETRIES:-0}"
   --opd-scoring-persistent-session
   --kl-coef 0
   --kl-loss-coef 0
   --entropy-coef 0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.80
)

FULLY_ASYNC_ARGS=(
   --fully-async
   --fully-async-prefetch-batches 2
   # pre-sync worker semantics: aborted/stale groups go back to the data buffer
   # for regeneration (the class-based rollout's default is drop)
   --async-unused-samples-handler retry
   --fully-async-max-completed-queue-groups 32
   --max-weight-staleness 2
   --update-weights-interval 1
)

MONITOR_ARGS=(
   --use-rollout-entropy
   --log-multi-turn
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RUN_NAME=${WANDB_RUN_NAME:-opd-mm-baseline-geo3k-mt-async-rollout-qold-200step}
WANDB_ARGS=(
   --use-wandb
   --wandb-team M3TRL
   # Override the destination project per-run via WANDB_PROJECT (e.g.
   # WANDB_PROJECT=baseline for runs that belong with the team baselines).
   --wandb-project "${WANDB_PROJECT:-OPD}"
   --wandb-group "$RUN_NAME"
   --disable-wandb-random-suffix
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 30
   --rollout-health-check-first-wait 60
)

LAYOUT_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus 8
)

MILES_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${RM_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${OPD_ARGS[@]}"
   "${MONITOR_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
   "${FULLY_ASYNC_ARGS[@]}"
)

# Freeze the baseline against accidental semantic changes from future shared
# launch infrastructure. The rollout-logprob selector must appear exactly once.
rollout_logprob_flags=0
for arg in "${MILES_ARGS[@]}"; do
   case "$arg" in
      --use-rollout-logprobs)
         rollout_logprob_flags=$((rollout_logprob_flags + 1))
         ;;
      --use-tis | --get-mismatch-metrics | --opd-optimize-task-reward)
         echo "FATAL: canonical OPD baseline forbids $arg" >&2
         return 1 2>/dev/null || exit 1
         ;;
      --save | --save-* | --async-save)
         echo "FATAL: canonical OPD baseline must not save checkpoints (found $arg)" >&2
         return 1 2>/dev/null || exit 1
         ;;
   esac
done

if ((rollout_logprob_flags != 1)); then
   echo "FATAL: canonical OPD baseline requires exactly one --use-rollout-logprobs flag" >&2
   return 1 2>/dev/null || exit 1
fi
