#!/usr/bin/env bash
# Dev-native parity port of orbit-develop feat/full-vocab-opd's Qwen3-1.7B <-
# Qwen3-4B full-vocab forward-KL recipe. Numerical training, rollout, PEFT,
# and evaluation settings intentionally match the source launcher; only the
# dev OPD serving/transport wiring differs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen3_17b_bf16_openmathreasoning_megatron_opd_lora_full_vocab
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen3-1.7B Hugging Face checkpoint}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Qwen3-1.7B Megatron torch_dist checkpoint}"
: "${OPD_TEACHER_CKPT:?set OPD_TEACHER_CKPT to the frozen Qwen3-4B Hugging Face checkpoint}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to the OpenReasoning training data (.jsonl or .parquet)}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-1.7B_4B_Instruct2507_openreasoning100k_full_vocab_opd_lora_fkl_rerun}"
AIME24_PATH="${AIME24_PATH:-${ORBIT_ROOT}/data/aime24/test.parquet}"
AIME25_PATH="${AIME25_PATH:-${ORBIT_ROOT}/data/aime25/test.parquet}"
HMMT25_PATH="${HMMT25_PATH:-${ORBIT_ROOT}/data/hmmt25/test.parquet}"

# === Resources ===
# Match the source's two-GPU colocated topology: actor TP2, two TP1 student
# engines, and one TP2 managed teacher time-share the same GPUs.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

# === Model args ===
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${ORBIT_ROOT}/miles_plugins/model_args/qwen3-1.7B.sh}"
source "${MODEL_ARGS_FILE}"

# === Training schedule ===
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"

COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-10}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key "${INPUT_KEY:-question}"
    --label-key "${LABEL_KEY:-answer}"
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 0.7
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path miles.orbit.opd.opd_sglang.reward_func
    --custom-reward-post-process-path miles.orbit.opd.opd_sglang.post_process
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style cosine
    --min-lr 5e-7
    --lr-warmup-fraction 0.1
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

# Dev's grpo estimator is inert here: validation disables advantages/returns
# for the direct full-vocab loss. Deferred scoring restores the source phase
# boundary (finish all student generations, then score the teacher batch).
RL_ARGS=(
    --advantage-estimator grpo
    --opd-type sglang
    --teacher-score-mode full_vocab
    --teacher-hf-checkpoint "${OPD_TEACHER_CKPT}"
    --opd-serve-teacher
    --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS:-2}"
    --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION:-0.3}"
    --opd-teacher-max-running-requests "${OPD_TEACHER_MAX_RUNNING_REQUESTS:-8}"
    --opd-teacher-max-prefill-tokens "${OPD_TEACHER_MAX_PREFILL_TOKENS:-4096}"
    --opd-defer-full-vocab-scoring
    --disable-compute-advantages-and-returns
)

LOSS_ARGS=(
    --loss-type opd_jsd_loss
    --opd-jsd-beta 0.0
    --calculate-per-token-loss
    --use-kl-loss
    --kl-loss-type low_var_kl
    --kl-loss-coef 0.0
    --opd-log-topk-overlap
    --opd-topk-overlap-ks 8 16 32 64
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data aime24 "${AIME24_PATH}" aime25 "${AIME25_PATH}" hmmt25 "${HMMT25_PATH}"
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 8192
    --eval-top-k -1
    --eval-top-p 0.95
    --eval-temperature 1.0
    --eval-pass-k-values 1 8 16
)

SGLANG_ARGS=(
    # This is distinct from --actor-num-gpus-per-node (derived from the shell
    # GPUS_PER_NODE value by driver.sh). Managed rollout/teacher placement uses
    # the generic topology value and otherwise inherits the parser default (8).
    --num-gpus-per-node "${GPUS_PER_NODE}"
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static 0.25
    --sglang-server-concurrency 4
    --sglang-max-running-requests 512
    --router-disable-circuit-breaker
    # The source recipe uses fa3. Its pinned SGLang rejects fa3 on B200/SM100,
    # so set SGLANG_ATTENTION_BACKEND=triton (or flashinfer with a clean JIT
    # cache) on Blackwell.
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND:-fa3}"
    --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND:-flashinfer}"
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
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=( --log-passrate )

PEFT_ARGS=(
    --peft-method lora
    --peft-variant standard
    --lora-type lora
    --lora-rank 64
    --lora-alpha 32
    --lora-dropout 0.0
    --target-modules all-linear
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
