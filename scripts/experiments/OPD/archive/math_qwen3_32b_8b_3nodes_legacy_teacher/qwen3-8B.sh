#!/bin/bash
#
# OPD/archive/math_qwen3_32b_8b_3nodes_legacy_teacher/qwen3-8B — historical
# reproduction recipe for the current Miles teacher-top-k path on dapo-math,
# Qwen3-8B student <- Qwen3-32B SGLang teacher, 3 dedicated nodes.
#
# IMPORTANT: this freezes the legacy behavior before the Top-K DAgger rebuild.
# Current Miles does not yet backpropagate a [T,K]
# trainer-side loss. With only-teacher + teacher_p it instead:
#   1. gets native per-position top-k IDs/logprobs from teacher SGLang;
#   2. unions those IDs across the response and asks student SGLang to rescore;
#   3. normalizes teacher weights inside that support;
#   4. reduces K candidates to one detached scalar per position and injects it
#      through sampled-token PPO/GRPO advantage.
# Job 24749 attempted to measure this chain but died at step 0 when the student
# rescore crashed a decode-concurrent SGLang scheduler. It produced no valid
# before-distribution or training curve. The recipe is retained only to
# reproduce that legacy failure while 01 replaces the rescore path.
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
# Scoring policy:
#   - timeout kept explicit (600s; the implicit aiohttp 300s killed 23787)
#   - retries 0 — fail fast; with a whole-node teacher a failure means
#     something real, not queue pressure
#   - persistent HTTP session enabled by default (already validated separately)
#   - NO in-flight cap (0 = disabled): a TP=8 whole-node teacher absorbs the
#     full 64-request burst and sglang's continuous batching does the
#     queueing. Set OPD_SCORING_MAX_INFLIGHT (e.g. 8) if scoring-tail
#     timeouts ever reappear.
#   - only-teacher keeps rollout generation clean, but every sample performs
#     one teacher top-k request followed by one student arbitrary-ID rescore.
#
# The default remains a 5-step cap so an upstream SGLang-fix reproduction is
# bounded. Do not increase OPD_NUM_ROLLOUT for performance characterization on
# the current stack. No --save is used, so nothing is checkpointed.
#
# The teacher model is owner-managed (not auto-downloaded by submit.sh):
#   hf download Qwen/Qwen3-32B --local-dir /data/shared/models/Qwen3-32B
#
# Submit:
#   HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
#     OPD/archive/math_qwen3_32b_8b_3nodes_legacy_teacher/qwen3-8B

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

# Freeze one explicit legacy baseline. OPD_TOP_K remains overridable for later
# K sweeps, but 0 is rejected because it would silently run sampled-token OPD.
OPD_TOP_K=${OPD_TOP_K:-2}
OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
if (( OPD_TOP_K <= 0 )); then
   echo "OPD_TOP_K must be positive for the legacy teacher-top-k baseline" >&2
   exit 2
fi
case "${OPD_SCORING_PERSISTENT_SESSION:-1}" in
   1|true) OPD_SCORING_PERSISTENT_SESSION=1 ;;
   0|false) OPD_SCORING_PERSISTENT_SESSION=0 ;;
   *) echo "OPD_SCORING_PERSISTENT_SESSION must be 0/1 or false/true" >&2; exit 2 ;;
esac
RUN_NAME=${WANDB_RUN_NAME:-t-top2-legacy}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --ref-load       "$HF_TORCHDIST_DIR"
)

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout   "$OPD_NUM_ROLLOUT"
   --rollout-batch-size      16
   --n-samples-per-prompt    4
   --rollout-max-response-len 16384
   --rollout-temperature     1
   --global-batch-size       64
   --balance-data
)

# The reward fn performs the legacy teacher top-k + student rescore chain. The
# post-process hook collapses the K candidates to one per-position scalar.
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
   # only-teacher avoids student top-k extraction during rollout. The current
   # implementation still constructs a response-wide teacher-ID union and
   # sends it to student SGLang, producing the N x U rescore response whose
   # decode-concurrent failure was captured by job 24749.
   --opd-log-prob-top-k "$OPD_TOP_K"
   --opd-top-k-strategy only-teacher
   --opd-reward-weight-mode teacher_p
   # Scoring policy for the dedicated-teacher layout (see header): explicit
   # timeout, fail-fast, no in-flight cap (0 = disabled).
   --opd-scoring-timeout "${OPD_SCORING_TIMEOUT:-600}"
   --opd-scoring-max-inflight "${OPD_SCORING_MAX_INFLIGHT:-0}"
   --opd-scoring-retries "${OPD_SCORING_RETRIES:-0}"

   # Pure legacy top-k-scalar objective: no task reward, reference KL, entropy,
   # or advantage normalization. This preserves the failed operator for bounded
   # reproduction; it is not the future DAgger loss.
   --kl-coef 0
   --kl-loss-coef 0
   --entropy-coef 0
)

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
