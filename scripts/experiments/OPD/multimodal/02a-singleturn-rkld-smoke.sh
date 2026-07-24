#!/bin/bash
#
# Milestone 02a: 5-step single-turn multimodal sampled-RKLD smoke.
#
#   node 1 (head): Qwen3-VL-30B-A3B-Thinking teacher, SGLang TP=8
#   node 2:        Qwen3-VL-8B-Instruct Megatron actor, TP=4 / DP=2
#   node 3:        eight single-GPU Qwen3-VL-8B-Instruct rollout engines
#
# The teacher scores the sampled response token at every active position. Task
# reward is logged for observation only; it remains zero in the optimization
# reward. DAgger is disabled by default and can be enabled by numbered objective
# wrappers without changing this recipe's model, data, or parallel layout. This
# recipe does not enable eval, checkpoint saving, multi-turn generation, context
# parallelism, or fully async execution.
#
# Submit:
#   HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
#     OPD/multimodal/02a-singleturn-rkld-smoke

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}

EXPERIMENT_NODES=3
EXPERIMENT_TIME=24:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
   "chenhegu/geo3k_imgurl"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/train.parquet"

# The teacher is owner-managed because submit.sh downloads only HF_MODEL_REPO.
OPD_TEACHER_MODEL_DIR=${OPD_TEACHER_MODEL_DIR:-"$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"}
OPD_TEACHER_PORT=${OPD_TEACHER_PORT:-13141}
OPD_TEACHER_TP=${OPD_TEACHER_TP:-8}
OPD_TEACHER_GPUS=${OPD_TEACHER_GPUS:-0,1,2,3,4,5,6,7}
OPD_TEACHER_MEM_FRACTION=${OPD_TEACHER_MEM_FRACTION:-0.8}
OPD_TEACHER_EXTRA_ARGS=${OPD_TEACHER_EXTRA_ARGS:-}

# Reserve every head-node GPU for the fixed teacher. The actor and rollout
# bundles consume exactly the 16 GPUs exposed by the two Ray worker nodes.
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

# Match the known-good Geo3K Qwen3-VL Megatron Bridge configuration.
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-8B.sh"
MODEL_ARGS+=(--megatron-to-hf-mode bridge)

OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
OPD_KL_COEF=${OPD_KL_COEF:-1}
OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-0}
OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-0}
OPD_DAGGER_LOSS=${OPD_DAGGER_LOSS:-cross_entropy}
RUN_NAME=${WANDB_RUN_NAME:-opd-mm-02a-rkld-smoke}

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
   --rollout-shuffle
   --num-rollout "$OPD_NUM_ROLLOUT"
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

# Keep the legacy rollout-side top-k scalar path disabled. Sampled RKLD is
# controlled independently by OPD_KL_COEF; objective wrappers add trainer-direct
# DAgger arguments through DAGGER_ARGS below.
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "$OPD_KL_COEF"
   --opd-log-prob-top-k 0
   --sglang-mm-exact-scoring-suffix
   --opd-scoring-timeout "${OPD_SCORING_TIMEOUT:-600}"
   --opd-scoring-max-inflight "${OPD_SCORING_MAX_INFLIGHT:-0}"
   --opd-scoring-retries "${OPD_SCORING_RETRIES:-0}"
   --opd-scoring-persistent-session
   --kl-coef 0
   --kl-loss-coef 0
   --entropy-coef 0
)

DAGGER_ARGS=()
if ((OPD_DAGGER_TOP_K > 0)); then
   DAGGER_ARGS+=(
      --opd-dagger-top-k "$OPD_DAGGER_TOP_K"
      --opd-dagger-coef "$OPD_DAGGER_COEF"
      --opd-dagger-loss "$OPD_DAGGER_LOSS"
   )
fi

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

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

MONITOR_ARGS=(
   --use-rollout-entropy
)

# Student rollout engines. 0.85 held through milestone 08; the 09d 200-step
# prefetch-2 run OOMed a student engine at step 65 (4.64 GiB burst against
# 2.84 GiB free + 9 GiB fragmented reserve) as responses drifted longer, so the
# fraction is now overridable per-recipe without changing the validated default.
OPD_STUDENT_MEM_FRACTION=${OPD_STUDENT_MEM_FRACTION:-0.85}
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "$OPD_STUDENT_MEM_FRACTION"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team M3TRL
   --wandb-project OPD
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

build_opd_multimodal_miles_args() {
   MILES_ARGS=(
      "${LAYOUT_ARGS[@]}"
      "${MODEL_ARGS[@]}"
      "${CKPT_ARGS[@]}"
      "${MULTIMODAL_ARGS[@]}"
      "${ROLLOUT_ARGS[@]}"
      "${RM_ARGS[@]}"
      "${OPTIMIZER_ARGS[@]}"
      "${GRPO_ARGS[@]}"
   )

   if ((OPD_DAGGER_TOP_K > 0)); then
      MILES_ARGS+=("${DAGGER_ARGS[@]}")
   fi

   MILES_ARGS+=(
      "${MONITOR_ARGS[@]}"
      "${WANDB_ARGS[@]}"
      "${PERF_ARGS[@]}"
      "${SGLANG_ARGS[@]}"
      "${MISC_ARGS[@]}"
      "${FT_ARGS[@]}"
   )
}

build_opd_multimodal_miles_args
