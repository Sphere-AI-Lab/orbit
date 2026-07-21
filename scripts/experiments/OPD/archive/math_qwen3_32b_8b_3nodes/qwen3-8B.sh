#!/bin/bash
#
# OPD/archive/math_qwen3_32b_8b_3nodes/qwen3-8B — archived OPD baseline on dapo-math,
# Qwen3-8B student <- Qwen3-32B SGLang teacher, 3 dedicated nodes:
#
#   node 1 (head):   teacher — whole node, SGLang TP=8 on GPUs 0-7.
#                    MILES_RAY_HEAD_NUM_GPUS=0 keeps Ray from scheduling any
#                    actor/rollout bundle onto the head, so the teacher owns
#                    its GPUs outright (no mem-fraction squeeze, no GPU-7
#                    corner like the 1-node baseline).
#   node 2+3:        16 Ray GPUs = exactly the requested bundles, so placement
#                    is deterministic: actor (Megatron student, 8 GPUs, TP=2 ->
#                    DP=4) fills one worker node, rollout (8 SGLang student
#                    engines, 1 GPU each) fills the other.
#
# The teacher URL uses the head node's routable IP (NOT 127.0.0.1): the
# RolloutManager that runs the OPD reward fn lives on a worker node here.
#
# Scoring policy (vs the archived sglang_teacher_baseline recipe):
#   - timeout kept explicit (600s; the implicit aiohttp 300s killed 23787)
#   - retries 0 — fail fast; with a whole-node teacher a failure means
#     something real, not queue pressure
#   - persistent HTTP session enabled by default; set
#     OPD_SCORING_PERSISTENT_SESSION=0 for the one-session-per-request A/B arm
#   - NO in-flight cap (0 = disabled): a TP=8 whole-node teacher absorbs the
#     full 64-request burst and sglang's continuous batching does the
#     queueing. Set OPD_SCORING_MAX_INFLIGHT (e.g. 8) if scoring-tail
#     timeouts ever reappear.
#
# QUICK-CHECK config otherwise: same data/optimizer/GRPO hyperparameters as
# the archived sglang_teacher_baseline, no --save (nothing checkpointed).
#
# The teacher model is owner-managed (not auto-downloaded by submit.sh):
#   hf download Qwen/Qwen3-32B --local-dir /data/shared/models/Qwen3-32B
#
# Submit:
#   HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/archive/math_qwen3_32b_8b_3nodes/qwen3-8B

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# ---------------------------------------------------------------------------
# Resource metadata — read by orchestrator wrappers for sbatch/k8s/etc.
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=3
EXPERIMENT_TIME=24:00:00

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
# Teacher server (SGLang, whole head node)
# ---------------------------------------------------------------------------
OPD_TEACHER_MODEL_DIR=${OPD_TEACHER_MODEL_DIR:-"$HF_CACHE_DIR/models/Qwen3-32B"}
OPD_TEACHER_PORT=${OPD_TEACHER_PORT:-13141}
OPD_TEACHER_TP=${OPD_TEACHER_TP:-8}
OPD_TEACHER_GPUS=${OPD_TEACHER_GPUS:-0,1,2,3,4,5,6,7}
OPD_TEACHER_MEM_FRACTION=${OPD_TEACHER_MEM_FRACTION:-0.8}

# Keep Ray off the head node's GPUs — the teacher owns them (see header).
export MILES_RAY_HEAD_NUM_GPUS=0

# Routable teacher URL (NOT 127.0.0.1 — the reward fn runs on a worker node in
# this layout). In-job the recipe is re-sourced by launch_miles.sbatch on the
# head node with HEAD_IP already resolved (getent, the same address workers
# use to join Ray — guaranteed worker-routable, unlike `hostname -I` whose
# first field can be any interface on a multi-homed node). The hostname
# fallback only feeds the throwaway submit-time evaluation.
OPD_TEACHER_HOST=${OPD_TEACHER_HOST:-${HEAD_IP:-$(hostname -I | awk '{print $1}')}}
OPD_TEACHER_URL="http://${OPD_TEACHER_HOST}:${OPD_TEACHER_PORT}"

if [[ ! -f "$OPD_TEACHER_MODEL_DIR/config.json" ]]; then
    echo "FATAL: OPD teacher model not found at $OPD_TEACHER_MODEL_DIR" >&2
    echo "  hf download Qwen/Qwen3-32B --local-dir $OPD_TEACHER_MODEL_DIR" >&2
    echo "  (or set OPD_TEACHER_MODEL_DIR)" >&2
    return 1 2>/dev/null || exit 1
fi

# Launcher local-server hook (see launch_miles.sbatch "envpack session
# server" — generic enough to host any HTTP sidecar on the head node).
# TRITON_CACHE_DIR is pinned to a per-user dir: the TP>1 server path ends up
# compiling into a shared /tmp/triton otherwise, which is owned by whichever
# user touched the node first (job 23835 died to another user's dir on
# slinky-36 — same landmine class as the miles engines' TRITON/deep_gemm
# fixes, but this sidecar bypasses miles' env plumbing).
export ENVPACK_LOCAL_SERVER_CMD="TRITON_CACHE_DIR=/tmp/triton_${USER:-unknown}/opd_teacher \
    CUDA_VISIBLE_DEVICES=$OPD_TEACHER_GPUS python3 -m sglang.launch_server \
    --model-path $OPD_TEACHER_MODEL_DIR \
    --host 0.0.0.0 \
    --port $OPD_TEACHER_PORT \
    --tp $OPD_TEACHER_TP \
    --chunked-prefill-size 8192 \
    --mem-fraction-static $OPD_TEACHER_MEM_FRACTION"
export ENVPACK_LOCAL_SERVER_HEALTH="$OPD_TEACHER_URL/health_generate"
# A 32B teacher needs minutes to load weights and warm up.
export ENVPACK_SERVER_WAIT_TIMEOUT=${ENVPACK_SERVER_WAIT_TIMEOUT:-1800}

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-8B.sh"

# Training-layout overrides are used by the numbered OPD parallel smokes. The
# canonical recipe remains TP=2, PP=1, hence DP=4 on the 8-GPU actor node.
TRAIN_TP_SIZE=${TRAIN_TP_SIZE:-2}
TRAIN_PP_SIZE=${TRAIN_PP_SIZE:-1}
if (( TRAIN_TP_SIZE <= 0 || TRAIN_PP_SIZE <= 0 )); then
   echo "TRAIN_TP_SIZE and TRAIN_PP_SIZE must be positive" >&2
   exit 2
fi
if (( 8 % (TRAIN_TP_SIZE * TRAIN_PP_SIZE) != 0 )); then
   echo "Actor world size 8 must be divisible by TRAIN_TP_SIZE * TRAIN_PP_SIZE" >&2
   exit 2
fi

# Sampled-token OPD by default (see GRPO_ARGS); the generated run name carries
# every optional DAgger arm plus sampled top-k and transport mode.
OPD_TOP_K=${OPD_TOP_K:-0}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-0}
OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-0}
OPD_DAGGER_LOSS=${OPD_DAGGER_LOSS:-explicit_cross_entropy}
case "${OPD_SCORING_PERSISTENT_SESSION:-1}" in
   1|true) OPD_SCORING_PERSISTENT_SESSION=1 ;;
   0|false) OPD_SCORING_PERSISTENT_SESSION=0 ;;
   *) echo "OPD_SCORING_PERSISTENT_SESSION must be 0/1 or false/true" >&2; exit 2 ;;
esac
DAGGER_RUN_SUFFIX=""
if [[ "$OPD_DAGGER_TOP_K" != "0" ]]; then
   DAGGER_RUN_SUFFIX="-dagger-topk${OPD_DAGGER_TOP_K}-${OPD_DAGGER_LOSS}-coef${OPD_DAGGER_COEF}"
fi
RUN_NAME=${WANDB_RUN_NAME:-math-opd-qwen3-8B-sglang-t32B-3nodes-topk${OPD_TOP_K}${DAGGER_RUN_SUFFIX}-persistent${OPD_SCORING_PERSISTENT_SESSION}}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --ref-load       "$HF_TORCHDIST_DIR"
)

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     prompt
   --apply-chat-template
   --rollout-shuffle
   # OPD_NUM_ROLLOUT: short cluster smokes (e.g. 5) without editing the recipe.
   --num-rollout   "${OPD_NUM_ROLLOUT:-300}"
   --rollout-batch-size      16
   --n-samples-per-prompt    4
   --rollout-max-response-len 16384
   --rollout-temperature     1
   --global-batch-size       64
   --balance-data
)

# The OPD reward fn queries the teacher once per sample. Sampled mode retains
# response-token log-probs; DAgger mode additionally retains native [T,K]
# targets from the same response.
RM_ARGS=(
   --custom-rm-path miles.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path miles.rollout.on_policy_distillation.post_process_rewards
   --rm-url "$OPD_TEACHER_URL/generate"
)

PERF_ARGS=(
   --tensor-model-parallel-size "$TRAIN_TP_SIZE"
   --pipeline-model-parallel-size "$TRAIN_PP_SIZE"
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)
if (( TRAIN_TP_SIZE > 1 )); then
   PERF_ARGS+=(--sequence-parallel)
fi

GRPO_ARGS=(
   --advantage-estimator grpo

   --use-opd
   --opd-type sglang
   --opd-kl-coef "$OPD_KL_COEF"
   # 0 = sampled-token OPD: rollout decodes at full speed (top-k>0 costs ~3x
   # decode on per-token top-logprob extraction) and teacher scoring is a
   # compact prefill-only request. top-k>0 (Rethinking-OPD) grows the scoring
   # payload O(len x union of per-position ids): top-k 16 dies at step 0
   # (jobs 23771/23779); top-k 2 survived 99 steps, then one 16.5k-token
   # sample's 1827-id union blew the 600s scoring timeout (job 23851).
   --opd-log-prob-top-k "$OPD_TOP_K"
   # Scoring policy for the dedicated-teacher layout (see header): explicit
   # timeout, fail-fast, no in-flight cap (0 = disabled).
   --opd-scoring-timeout "${OPD_SCORING_TIMEOUT:-600}"
   --opd-scoring-max-inflight "${OPD_SCORING_MAX_INFLIGHT:-0}"
   --opd-scoring-retries "${OPD_SCORING_RETRIES:-0}"

   # Pure distillation objective: no task/reward KL, reference-loss KL, entropy
   # bonus, or advantage normalization. OPD_KL_COEF and OPD_DAGGER_COEF select
   # the sampled RKLD-PG and direct DAgger contributions independently.
   --kl-coef 0
   --kl-loss-coef 0
   --entropy-coef 0
)

if [[ "$OPD_DAGGER_TOP_K" != "0" ]]; then
   GRPO_ARGS+=(
      --opd-dagger-top-k "$OPD_DAGGER_TOP_K"
      --opd-dagger-coef "$OPD_DAGGER_COEF"
      --opd-dagger-loss "$OPD_DAGGER_LOSS"
   )
fi

if [[ "$OPD_SCORING_PERSISTENT_SESSION" == "0" ]]; then
   GRPO_ARGS+=(--no-opd-scoring-persistent-session)
else
   GRPO_ARGS+=(--opd-scoring-persistent-session)
fi

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
   --sglang-mem-fraction-static  0.75
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
   --actor-num-gpus-per-node 8
   --rollout-num-gpus       8
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
