#!/bin/bash
#
# OPD/sglang_teacher_baseline/qwen3-8B — on-policy distillation, SGLang-served teacher.
# Port of examples/on_policy_distillation/run-qwen3-8B-opd.sh to the submit.sh
# recipe contract: student Qwen3-8B distills from a Qwen3-32B teacher whose
# token log-probs are fetched during rollout by the OPD custom reward fn.
#
# Layout (1 node x 8 GPUs, same split as the upstream example):
#   - actor:   2 GPUs Megatron (--actor-num-nodes 1 --actor-num-gpus-per-node 2)
#   - rollout: 4 GPUs SGLang student engines (--rollout-num-gpus 4)
#   - teacher: GPU 7, standalone SGLang server (GPU 6 stays idle)
#   MILES_RAY_HEAD_NUM_GPUS=6 registers only GPUs 0-5 with Ray, fencing GPU 7
#   from actor/rollout placement. GPU 6 is also outside Ray and stays idle.
#
# The teacher server rides the launcher's local-server hook
# (ENVPACK_LOCAL_SERVER_CMD): launch_miles.sbatch starts it on the head node
# AFTER the stale-process cleanup, gates t0 on /health_generate, arms the
# health watchdog, and tears it down with the job. 127.0.0.1 works because
# this is a 1-node recipe — rollout actors are colocated with the server.
#
# The teacher model is owner-managed (not auto-downloaded by submit.sh):
#   hf download Qwen/Qwen3-32B --local-dir /data/shared/models/Qwen3-32B
#
# QUICK-CHECK config: no --save/--save-interval (nothing is checkpointed;
# weights come from --ref-load via the --load fallback).
#
# Submit (HF_CACHE_DIR=/data/shared on slinky — the default /data/shared/hf_cache
# is read-only there):
#   HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/sglang_teacher_baseline/qwen3-8B

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# ---------------------------------------------------------------------------
# Resource metadata — read by orchestrator wrappers for sbatch/k8s/etc.
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=1
EXPERIMENT_TIME=24:00:00
# The actor and rollout require exactly six GPUs. Keep the teacher-owned GPU 7
# and idle GPU 6 outside Ray's resource inventory.
export MILES_RAY_HEAD_NUM_GPUS=6

# ---------------------------------------------------------------------------
# Asset metadata — read by orchestrator wrappers for `hf download` step
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-8B"
HF_DATASETS=(
    "zhuzilin/dapo-math-17k"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-8B"
HF_TORCHDIST_DIR="$HF_CACHE_DIR/models/Qwen3-8B_torch_dist"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/dapo-math-17k/dapo-math-17k.jsonl"

# ---------------------------------------------------------------------------
# Teacher server (SGLang, head-node GPU 7 — same as the upstream example)
# ---------------------------------------------------------------------------
OPD_TEACHER_MODEL_DIR=${OPD_TEACHER_MODEL_DIR:-"$HF_CACHE_DIR/models/Qwen3-32B"}
OPD_TEACHER_PORT=${OPD_TEACHER_PORT:-13141}
OPD_TEACHER_TP=${OPD_TEACHER_TP:-1}
OPD_TEACHER_GPUS=${OPD_TEACHER_GPUS:-7}
OPD_TEACHER_MEM_FRACTION=${OPD_TEACHER_MEM_FRACTION:-0.6}
OPD_TEACHER_URL="http://127.0.0.1:${OPD_TEACHER_PORT}"

if [[ ! -f "$OPD_TEACHER_MODEL_DIR/config.json" ]]; then
    echo "FATAL: OPD teacher model not found at $OPD_TEACHER_MODEL_DIR" >&2
    echo "  hf download Qwen/Qwen3-32B --local-dir $OPD_TEACHER_MODEL_DIR" >&2
    echo "  (or set OPD_TEACHER_MODEL_DIR)" >&2
    return 1 2>/dev/null || exit 1
fi

# Launcher local-server hook (see launch_miles.sbatch "envpack session
# server" — generic enough to host any HTTP sidecar on the head node).
# TRITON_CACHE_DIR pinned per-user: the sidecar bypasses miles' engine env
# plumbing, and a shared /tmp/triton is owned by whichever user touched the
# node first (killed job 23835's TP=8 teacher; cheap insurance here too).
export ENVPACK_LOCAL_SERVER_CMD="TRITON_CACHE_DIR=/tmp/triton_${USER:-unknown}/opd_teacher \
    CUDA_VISIBLE_DEVICES=$OPD_TEACHER_GPUS python3 -m sglang.launch_server \
    --model-path $OPD_TEACHER_MODEL_DIR \
    --host 0.0.0.0 \
    --port $OPD_TEACHER_PORT \
    --tp $OPD_TEACHER_TP \
    --chunked-prefill-size 4096 \
    --mem-fraction-static $OPD_TEACHER_MEM_FRACTION"
export ENVPACK_LOCAL_SERVER_HEALTH="$OPD_TEACHER_URL/health_generate"
# Default 60s is tuned for the lightweight envpack gateway; a 32B teacher
# needs minutes to load weights and warm up.
export ENVPACK_SERVER_WAIT_TIMEOUT=${ENVPACK_SERVER_WAIT_TIMEOUT:-1800}

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-8B.sh"

# Official-baseline naming: baseline-opd-<student>-<teacher mode>-t<teacher size>.
RUN_NAME=${WANDB_RUN_NAME:-baseline-opd-qwen3-8B-sglang-t32B}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --ref-load       "$HF_TORCHDIST_DIR"
)

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout   300
   --rollout-batch-size      16
   --n-samples-per-prompt    4
   --rollout-max-response-len 16384
   --rollout-temperature     1
   --global-batch-size       64
   --balance-data
)

# The OPD reward fn queries the teacher for token log-probs per sample; the
# post-process hook turns them into the per-token reverse-KL penalty.
RM_ARGS=(
   --custom-rm-path miles.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
   --rm-url "$OPD_TEACHER_URL/generate"
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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

   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   # Rethinking-OPD top-k reward (0 = sampled-token OPD). Default 2, NOT the
   # paper's 16: the teacher scoring payload grows O(len x union of per-position
   # top-k ids), and at 16k response length top-k 16 reliably wedges a TP=1 32B
   # teacher (jobs 23771/23779 both died to the resulting scoring timeout).
   # OPD_TOP_K env-overridable for experiments.
   --opd-log-prob-top-k "${OPD_TOP_K:-2}"
   # Scoring robustness: per-request timeout + in-flight cap so a whole rollout
   # batch finishing at once can't dogpile the teacher into timeout death.
   --opd-scoring-timeout "${OPD_SCORING_TIMEOUT:-600}"
   --opd-scoring-max-inflight "${OPD_SCORING_MAX_INFLIGHT:-8}"
   --opd-top-k-strategy only-student
   --opd-reward-weight-mode student_p

   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
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
   --sglang-mem-fraction-static  0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team    M3TRL
   --wandb-project OPD
   --wandb-group   "$RUN_NAME"
   --disable-wandb-random-suffix
   # WANDB_API_KEY comes from the env (exported by submit.sh / launch_miles.sbatch);
   # we don't pass it on the CLI because it would leak into run.log and args.json.
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout  30
   --rollout-health-check-first-wait 60
)

LAYOUT_ARGS=(
   --actor-num-nodes        1
   --actor-num-gpus-per-node 2
   --rollout-num-gpus       4
)

MILES_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${RM_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
