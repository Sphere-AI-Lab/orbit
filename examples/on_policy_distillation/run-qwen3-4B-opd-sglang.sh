#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 BF16 on-policy distillation (OPD) with an external
# SGLang teacher server. Self-contained launcher.
#
# OPD objective: pure MOPD (reward-free). The advantage estimator
# `on_policy_distillation` sets adv_t = teacher_logp_t - student_logp_t, so the
# student is trained to match a (typically larger/better) teacher on its own
# sampled tokens. Unlike the Megatron-teacher recipe, the teacher here is NOT
# loaded on the training GPUs: it is scored by POSTing the student's rollout
# token sequence to a separately-hosted SGLang server for prefill-only scoring
# (max_new_tokens=0, return_logprob=True, temperature=0 -- no generation).
# Start that teacher server yourself (e.g. `python -m sglang.launch_server
# --model-path <teacher-hf-checkpoint> --port <port>`) and point
# OPD_TEACHER_URL at its /generate endpoint before launching this script.
#
# Teacher production is selected with `--opd-type sglang --opd-teacher-url`,
# wired through orbit's custom-reward hooks:
#   --custom-rm-path orbit.rollout.opd_sglang.reward_func                  (scores via the teacher server)
#   --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process (extracts teacher_log_probs)
#
# Note: pure MOPD (`--advantage-estimator on_policy_distillation`) and the
# blend (`--use-opd`) are mutually exclusive -- do not pass `--use-opd` here.
# For the blend form instead, use `--advantage-estimator grpo --use-opd
# --opd-kl-coef <λ>` on top of a reward estimator (see README). Note also that
# a reward-bearing blend still needs its own `--custom-rm-path`/`--rm-type` for
# the task reward *in addition to* the teacher-scoring hooks above -- see README.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen3_4b_opd_sglang
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path (student)}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path (student)}"
: "${OPD_TEACHER_URL:?set OPD_TEACHER_URL to the external SGLang teacher /generate endpoint, e.g. http://host:port/generate}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_opd_sglang}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Resources ===
GPUS_PER_NODE=4
RAY_NUM_CPUS=64

# === Model args ===
MODEL_ARGS_ROTARY_BASE=5000000
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-4B-Instruct-2507.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 200
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 1024
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 3e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

# On-policy distillation (pure MOPD) with an external SGLang teacher server.
RL_ARGS=(
    --advantage-estimator on_policy_distillation
    --opd-type sglang
    --opd-teacher-url "${OPD_TEACHER_URL}"
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
)

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-prompt-data math "${TEST_JSONL}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 1024
    --eval-top-k 1
    --skip-eval-before-train
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.60
    --rollout-num-gpus 0
    --sglang-max-running-requests 1024
    --sglang-chunked-prefill-size 4096
    --sglang-attention-backend flashinfer
    --router-disable-circuit-breaker
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --offload-rollout
)

DEBUG_ARGS=(
    --log-passrate
)

# The Megatron-teacher's full-fine-tuning restriction (see
# run-qwen3-4B-opd-megatron.sh) does not apply here -- the teacher is off the
# training GPUs -- but we keep `--peft-method none` for parity with that recipe.
PEFT_ARGS=(
    --peft-method none
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
