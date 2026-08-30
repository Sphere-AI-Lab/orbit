#!/usr/bin/env bash
# Shared Qwen2.5 math PPO recipe for full-critic vs adapter-critic benchmarks.
# Source this file from one of the four comparison wrappers in this directory.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a PPO critic-comparison wrapper; do not run it directly." >&2
    exit 2
fi

: "${PPO_CRITIC_MODE:?PPO_CRITIC_MODE must be full or adapter}"
: "${PPO_COMPARISON_PANEL:?PPO_COMPARISON_PANEL must be controlled or budget}"
: "${GPUS_PER_NODE:?comparison wrapper must set GPUS_PER_NODE}"
: "${CRITIC_NUM_GPUS_PER_NODE:?comparison wrapper must set CRITIC_NUM_GPUS_PER_NODE}"
: "${ROLLOUT_NUM_GPUS:?comparison wrapper must set ROLLOUT_NUM_GPUS}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WRAPPER_PATH="$(realpath -m -- "${BASH_SOURCE[1]}")"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/wandb.sh"

case "${PPO_CRITIC_MODE}" in
    full | adapter) ;;
    *)
        echo "PPO_CRITIC_MODE must be full or adapter; got ${PPO_CRITIC_MODE}" >&2
        exit 2
        ;;
esac
case "${PPO_COMPARISON_PANEL}" in
    controlled | budget) ;;
    *)
        echo "PPO_COMPARISON_PANEL must be controlled or budget; got ${PPO_COMPARISON_PANEL}" >&2
        exit 2
        ;;
esac

# The wrappers own topology. Refuse environment drift that would change the
# scientific question represented by a wrapper name.
EXPECTED_CRITIC_GPUS=0
EXPECTED_ROLLOUT_GPUS=2
if [[ "${PPO_CRITIC_MODE}" == "full" ]]; then
    EXPECTED_CRITIC_GPUS=1
elif [[ "${PPO_COMPARISON_PANEL}" == "budget" ]]; then
    EXPECTED_ROLLOUT_GPUS=3
fi
if [[ "${GPUS_PER_NODE}" != "1" \
      || "${CRITIC_NUM_GPUS_PER_NODE}" != "${EXPECTED_CRITIC_GPUS}" \
      || "${ROLLOUT_NUM_GPUS}" != "${EXPECTED_ROLLOUT_GPUS}" ]]; then
    echo "invalid ${PPO_COMPARISON_PANEL}/${PPO_CRITIC_MODE} layout: " \
         "actor=${GPUS_PER_NODE}, critic=${CRITIC_NUM_GPUS_PER_NODE}, " \
         "rollout=${ROLLOUT_NUM_GPUS}; expected actor=1, " \
         "critic=${EXPECTED_CRITIC_GPUS}, rollout=${EXPECTED_ROLLOUT_GPUS}" >&2
    exit 2
fi

# Register only the GPUs used by this panel. The controlled adapter run leaves
# the fourth visible B200 idle by design (1 actor + 2 rollout = 3).
RAY_NUM_GPUS=$((GPUS_PER_NODE + CRITIC_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
PEFT_ARENA_REWARD_TIMEOUT_S="${PEFT_ARENA_REWARD_TIMEOUT_S:-${ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S:-60}}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.60}"

SMOKE="${SMOKE:-0}"
if is_true "${SMOKE}"; then
    MODEL_TAG=qwen25_05b
    MODEL_DIR_NAME=Qwen2.5-0.5B-Instruct
    MODEL_ARGS_FILE=qwen2.5-0.5B.sh
    RUN_FLAVOR=smoke
    NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-128}"
    EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-128}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
    SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
else
    MODEL_TAG=qwen25_3b
    MODEL_DIR_NAME=Qwen2.5-3B-Instruct
    MODEL_ARGS_FILE=qwen2.5-3B.sh
    RUN_FLAVOR=benchmark
    NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
    EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
    SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
fi

SEED="${SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-${SEED}}"
if [[ ! "${SEED}" =~ ^[0-9]+$ || ! "${ROLLOUT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED and ROLLOUT_SEED must be nonnegative integers" >&2
    exit 2
fi
export PYTHONHASHSEED="${SEED}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer; got ${value}" >&2
        exit 2
    fi
}
require_positive_integer NUM_ROLLOUT "${NUM_ROLLOUT}"
require_positive_integer ROLLOUT_BATCH_SIZE "${ROLLOUT_BATCH_SIZE}"
require_positive_integer N_SAMPLES_PER_PROMPT "${N_SAMPLES_PER_PROMPT}"
require_positive_integer GLOBAL_BATCH_SIZE "${GLOBAL_BATCH_SIZE}"
require_positive_integer ROLLOUT_MAX_RESPONSE_LEN "${ROLLOUT_MAX_RESPONSE_LEN}"
require_positive_integer EVAL_MAX_RESPONSE_LEN "${EVAL_MAX_RESPONSE_LEN}"
require_positive_integer MAX_TOKENS_PER_GPU "${MAX_TOKENS_PER_GPU}"
require_positive_integer SAVE_INTERVAL "${SAVE_INTERVAL}"
require_positive_integer EVAL_INTERVAL "${EVAL_INTERVAL}"
require_positive_integer RAY_NUM_CPUS "${RAY_NUM_CPUS}"
require_positive_integer PEFT_ARENA_REWARD_TIMEOUT_S "${PEFT_ARENA_REWARD_TIMEOUT_S}"
unset -f require_positive_integer
export ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S="${PEFT_ARENA_REWARD_TIMEOUT_S}"

# === Recipe identity ===
LAUNCHER_NAME="run_${MODEL_TAG}_bf16_math_oft_ppo_${PPO_COMPARISON_PANEL}_${PPO_CRITIC_MODE}_seed${SEED}_${RUN_FLAVOR}"
WANDB_PROJECT="${WANDB_PROJECT:-orbit-ppo-critic-compare}"
WANDB_GROUP="${WANDB_GROUP:-${LAUNCHER_NAME}}"
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Paths and resume contract ===
: "${HF_CKPT:?set HF_CKPT to the Qwen2.5 Hugging Face checkpoint path}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to an OpenR1-style math JSONL path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the initial Megatron torch_dist checkpoint path}"
RESUME_DIR="${RESUME_DIR:-}"
PEFT_ADAPTER_PATH=""
if [[ -n "${RESUME_DIR}" ]]; then
    SAVE_DIR="${RESUME_DIR%/}"
else
    SAVE_ROOT="${SAVE_ROOT:-${ORBIT_ROOT}/orbit_ckpts/ppo_critic_compare}"
    SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${MODEL_DIR_NAME}_${PPO_COMPARISON_PANEL}_${PPO_CRITIC_MODE}_seed${SEED}_${RUN_FLAVOR}}"
fi

# Use one canonical spelling everywhere: checkpoint paths, W&B identity, and
# the cross-node writer lock must all refer to the same directory.
if [[ -z "${SAVE_DIR}" ]]; then
    echo "SAVE_DIR must identify a benchmark run directory, not an empty path" >&2
    exit 2
fi
SAVE_DIR="$(realpath -m -- "${SAVE_DIR}")"
if [[ "${SAVE_DIR}" == "/" ]]; then
    echo "SAVE_DIR must identify a benchmark run directory, not /" >&2
    exit 2
fi

if [[ -n "${RESUME_DIR}" ]]; then
    RESUME_DIR="${SAVE_DIR}"
    CRITIC_LOAD="${SAVE_DIR}/critic"
    CRITIC_MARKER="${CRITIC_LOAD}/latest_checkpointed_iteration.txt"
    if [[ ! -f "${CRITIC_MARKER}" ]]; then
        echo "RESUME_DIR has no critic checkpoint marker: ${CRITIC_MARKER}" >&2
        exit 2
    fi
    CRITIC_RESUME_ITERATION="$(<"${CRITIC_MARKER}")"
    if [[ ! "${CRITIC_RESUME_ITERATION}" =~ ^[0-9]+$ \
          || ${#CRITIC_RESUME_ITERATION} -gt 19 \
          || ( ${#CRITIC_RESUME_ITERATION} -eq 19 \
               && "${CRITIC_RESUME_ITERATION}" > "9223372036854775807" ) ]]; then
        echo "critic checkpoint marker must contain an int64-bounded nonnegative iteration" >&2
        exit 2
    fi
    printf -v RESUME_ITERATION_PADDED '%07d' "$((10#${CRITIC_RESUME_ITERATION}))"
    PEFT_ADAPTER_PATH="${SAVE_DIR}/actor/iter_${RESUME_ITERATION_PADDED}/adapter"
    if [[ ! -f "${PEFT_ADAPTER_PATH}/adapter_megatron_tp0_pp0.pt" \
          || ! -f "${PEFT_ADAPTER_PATH}/training_state_rank0.pt" ]]; then
        echo "resumable actor adapter checkpoint is incomplete: ${PEFT_ADAPTER_PATH}" >&2
        exit 2
    fi
elif [[ "${PPO_CRITIC_MODE}" == "full" ]]; then
    CRITIC_LOAD="${MEGATRON_LOAD}"
fi

if [[ "${PPO_CRITIC_MODE}" == "adapter" && -z "${RESUME_DIR}" && -n "${CRITIC_LOAD:-}" ]]; then
    echo "CRITIC_LOAD is only valid for an adapter-critic resume; use RESUME_DIR" >&2
    exit 2
fi

# W&B's SDK reads these variables even when wandb.init() is called without
# explicit id/resume kwargs. The CLI flag is retained as provenance as well.
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    WANDB_RUN_ID="orbit$(printf '%s\0%s\0%s' "${SAVE_DIR}" "${WANDB_PROJECT}" "${LAUNCHER_NAME}" | sha256sum | cut -c1-20)"
fi
if [[ -n "${WANDB_RESUME:-}" && "${WANDB_RESUME}" != "allow" ]]; then
    echo "WANDB_RESUME is fixed to allow for benchmark continuity; got ${WANDB_RESUME}" >&2
    exit 2
fi
WANDB_RESUME=allow
export WANDB_RUN_ID WANDB_RESUME

MATH500_JSONL="${MATH500_JSONL:-${EVAL_ORBIT_DIR:+${EVAL_ORBIT_DIR%/}/math500.jsonl}}"
AIME24_JSONL="${AIME24_JSONL:-${EVAL_ORBIT_DIR:+${EVAL_ORBIT_DIR%/}/aime24.jsonl}}"
AMC23_JSONL="${AMC23_JSONL:-${EVAL_ORBIT_DIR:+${EVAL_ORBIT_DIR%/}/amc23.jsonl}}"
TEST_JSONL="${TEST_JSONL:-}"

require_local_directory() {
    local name="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "${name} must be an existing local directory; got ${path}" >&2
        exit 2
    fi
}
require_local_file() {
    local name="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "${name} must be an existing local file; got ${path}" >&2
        exit 2
    fi
}
require_local_directory HF_CKPT "${HF_CKPT}"
require_local_directory MEGATRON_LOAD "${MEGATRON_LOAD}"
require_local_file TRAIN_JSONL "${TRAIN_JSONL}"
require_local_file ORBIT_ENTRYPOINT "${ORBIT_ENTRYPOINT}"
if [[ -n "${RESUME_DIR}" ]]; then
    require_local_directory CRITIC_LOAD "${CRITIC_LOAD}"
fi
unset -f require_local_directory require_local_file

# === Model args ===
source "${ORBIT_ROOT}/miles_plugins/model_args/${MODEL_ARGS_FILE}"

# === ARGS arrays ===
COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}/actor"
    --critic-save "${SAVE_DIR}/critic"
    --save-interval "${SAVE_INTERVAL}"
    --megatron-to-hf-mode bridge
)
if [[ -n "${PEFT_ADAPTER_PATH}" ]]; then
    CKPT_ARGS+=(--peft-adapter-path "${PEFT_ADAPTER_PATH}")
fi
if [[ "${PPO_CRITIC_MODE}" == "full" ]]; then
    CKPT_ARGS+=(--critic-load "${CRITIC_LOAD:-${MEGATRON_LOAD}}")
elif [[ -n "${RESUME_DIR}" ]]; then
    CKPT_ARGS+=(--critic-load "${CRITIC_LOAD}")
fi

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rollout-seed "${ROLLOUT_SEED}"
    --rm-type custom
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path orbit.rewards.peft_arena_reward.peft_arena_reward
    --reward-key score
    --eval-reward-key score
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --critic-lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

RL_ARGS=(
    --advantage-estimator ppo
    --critic-mode "${PPO_CRITIC_MODE}"
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --value-clip 0.2
    --gamma 1.0
    --lambd 1.0
    --num-critic-only-steps 1
    --normalize-advantages
)

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-run-id "${WANDB_RUN_ID}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

if is_true "${SMOKE}"; then
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-prompt-data math "${TEST_JSONL}"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature 0.0
        --eval-top-p 1.0
        --eval-top-k 1
    )
else
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-prompt-data math500 "${MATH500_JSONL}" \
                           aime24 "${AIME24_JSONL}" \
                           amc23 "${AMC23_JSONL}"
        --eval-input-key prompt
        --eval-label-key label
        --n-samples-per-eval-prompt 4
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature 1.0
        --eval-top-p 1.0
        --eval-top-k -1
        --eval-pass-k-values 1 2 4
    )
fi
validate_eval_args

# validate_eval_args may remove EVAL_ARGS when DISABLE_EVAL=1. Validate every
# remaining local dataset before starting Ray so path typos fail immediately.
for ((EVAL_ARG_INDEX = 0; EVAL_ARG_INDEX < ${#EVAL_ARGS[@]}; EVAL_ARG_INDEX++)); do
    if [[ "${EVAL_ARGS[$EVAL_ARG_INDEX]}" != "--eval-prompt-data" ]]; then
        continue
    fi
    EVAL_ARG_INDEX=$((EVAL_ARG_INDEX + 1))
    while ((EVAL_ARG_INDEX < ${#EVAL_ARGS[@]})) && [[ "${EVAL_ARGS[$EVAL_ARG_INDEX]}" != --* ]]; do
        EVAL_DATASET_NAME="${EVAL_ARGS[$EVAL_ARG_INDEX]}"
        EVAL_DATASET_PATH="${EVAL_ARGS[$((EVAL_ARG_INDEX + 1))]}"
        if [[ ! -f "${EVAL_DATASET_PATH}" ]]; then
            echo "eval dataset ${EVAL_DATASET_NAME} must be an existing local file; got ${EVAL_DATASET_PATH}" >&2
            exit 2
        fi
        EVAL_ARG_INDEX=$((EVAL_ARG_INDEX + 2))
    done
done
unset EVAL_ARG_INDEX EVAL_DATASET_NAME EVAL_DATASET_PATH

# Fail before Ray if training records are unusable or the main evaluation
# triplet would silently fall back from dataset-specific math_alignment grading.
DATASET_VALIDATION_ARGS=("${TRAIN_JSONL}")
if ! is_true "${SMOKE}" && (( ${#EVAL_ARGS[@]} > 0 )); then
    DATASET_VALIDATION_ARGS+=(
        "${MATH500_JSONL}" math500
        "${AIME24_JSONL}" aime24
        "${AMC23_JSONL}" amc23
    )
fi
python3 - "${DATASET_VALIDATION_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path


def records(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield line_number, record


def validate_prompt_labels(path: str):
    count = 0
    for line_number, record in records(path):
        count += 1
        if record.get("prompt") is None:
            raise ValueError(f"{path}:{line_number}: prompt is missing or null")
        if record.get("label") is None:
            raise ValueError(f"{path}:{line_number}: label is missing or null")
    if count == 0:
        raise ValueError(f"{path}: contains no records")


try:
    validate_prompt_labels(sys.argv[1])
    eval_args = sys.argv[2:]
    if len(eval_args) % 2:
        raise ValueError("internal error: aligned-eval path/name arguments are unpaired")
    for path, expected_dataset in zip(eval_args[0::2], eval_args[1::2], strict=True):
        validate_prompt_labels(path)
        for line_number, record in records(path):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: metadata must be an object")
            if metadata.get("dataset_name") != expected_dataset:
                raise ValueError(
                    f"{path}:{line_number}: metadata.dataset_name must be {expected_dataset!r}"
                )
            if metadata.get("rm_type") != "math_alignment":
                raise ValueError(
                    f"{path}:{line_number}: metadata.rm_type must be 'math_alignment'"
                )
except (OSError, ValueError) as exc:
    raise SystemExit(f"PPO critic-comparison dataset preflight failed: {exc}") from exc
PY
unset DATASET_VALIDATION_ARGS

SGLANG_ARGS=(
    --num-gpus-per-node 4
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-enable-deterministic-inference
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    # sglang v0.5.16's prefill CUDA graph captures a warmup forward outside the
    # normal batch-prep path; the OFT triton backend has no batch_info there and
    # engine init dies in sgemm_oft_r_fwd. Decode graphs stay on; both critic
    # modes inherit this identically so controlled parity is unaffected.
    --sglang-cuda-graph-backend-prefill disabled
    --router-disable-circuit-breaker
    --sglang-router-policy round_robin
)

MISC_ARGS=(
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-offload-train
    --no-offload-train-async
    --no-offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)
if [[ "${PPO_CRITIC_MODE}" == "full" ]]; then
    MISC_ARGS+=(--critic-num-gpus-per-node "${CRITIC_NUM_GPUS_PER_NODE}")
fi

DEBUG_ARGS=(
    --log-passrate
    --log-reward-category acc
)

PEFT_ARGS=(
    --peft-method oft
    --peft-distributed-transport nccl
    --peft-variant standard
    --oft-type canonical_oft
    --oft-block-size 32
    --oft-eps 6e-5
    --target-modules all-linear
    --adapter-double-buffer
)

# Store the identity and schedule next to checkpoints. This prevents an
# accidental cross-mode/panel/seed resume or a fresh overwrite of a real run.
file_sha256_or_missing() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        sha256sum -- "${path}" | cut -d ' ' -f 1
    else
        printf 'missing\n'
    fi
}

directory_manifest_sha256() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        printf 'missing\n'
        return
    fi
    find "${path}" -type f -printf '%P\t%s\n' | LC_ALL=C sort | sha256sum | cut -d ' ' -f 1
}

GIT_COMMIT="$(git -C "${ORBIT_ROOT}" rev-parse --verify HEAD)"
GIT_DIFF_SHA256="$(git -C "${ORBIT_ROOT}" diff --no-ext-diff --binary HEAD | sha256sum | cut -d ' ' -f 1)"
GIT_STATUS_PORCELAIN="$(git -C "${ORBIT_ROOT}" status --porcelain --untracked-files=normal)"
GIT_STATUS_SHA256="$(printf '%s' "${GIT_STATUS_PORCELAIN}" | sha256sum | cut -d ' ' -f 1)"
GIT_DIRTY=0
if [[ -n "${GIT_STATUS_PORCELAIN}" ]]; then
    GIT_DIRTY=1
fi
COMMON_LAUNCHER_SHA256="$(file_sha256_or_missing "${BASH_SOURCE[0]}")"
WRAPPER_SHA256="$(file_sha256_or_missing "${WRAPPER_PATH}")"
ORBIT_ENTRYPOINT_SHA256="$(file_sha256_or_missing "${ORBIT_ENTRYPOINT}")"
ALLOW_DIRTY_BENCHMARK="${ALLOW_DIRTY_BENCHMARK:-0}"

render_benchmark_metadata() {
    printf '%s\t%s\n' \
        schema 2 \
        model_tag "${MODEL_TAG}" \
        model_dir_name "${MODEL_DIR_NAME}" \
        run_flavor "${RUN_FLAVOR}" \
        panel "${PPO_COMPARISON_PANEL}" \
        critic_mode "${PPO_CRITIC_MODE}" \
        seed "${SEED}" \
        rollout_seed "${ROLLOUT_SEED}" \
        git_commit "${GIT_COMMIT}" \
        git_dirty "${GIT_DIRTY}" \
        git_diff_sha256 "${GIT_DIFF_SHA256}" \
        git_status_sha256 "${GIT_STATUS_SHA256}" \
        allow_dirty_benchmark "${ALLOW_DIRTY_BENCHMARK}" \
        common_launcher_sha256 "${COMMON_LAUNCHER_SHA256}" \
        wrapper_sha256 "${WRAPPER_SHA256}" \
        orbit_entrypoint "${ORBIT_ENTRYPOINT}" \
        orbit_entrypoint_sha256 "${ORBIT_ENTRYPOINT_SHA256}" \
        hf_checkpoint "${HF_CKPT}" \
        hf_checkpoint_manifest_sha256 "$(directory_manifest_sha256 "${HF_CKPT}")" \
        megatron_base "${MEGATRON_LOAD}" \
        megatron_base_manifest_sha256 "$(directory_manifest_sha256 "${MEGATRON_LOAD}")" \
        train_jsonl "${TRAIN_JSONL}" \
        train_jsonl_sha256 "$(file_sha256_or_missing "${TRAIN_JSONL}")" \
        math500_jsonl "${MATH500_JSONL}" \
        math500_jsonl_sha256 "$(file_sha256_or_missing "${MATH500_JSONL}")" \
        aime24_jsonl "${AIME24_JSONL}" \
        aime24_jsonl_sha256 "$(file_sha256_or_missing "${AIME24_JSONL}")" \
        amc23_jsonl "${AMC23_JSONL}" \
        amc23_jsonl_sha256 "$(file_sha256_or_missing "${AMC23_JSONL}")" \
        test_jsonl "${TEST_JSONL}" \
        test_jsonl_sha256 "$(file_sha256_or_missing "${TEST_JSONL}")" \
        disable_eval "${DISABLE_EVAL:-0}" \
        reward_function orbit.rewards.peft_arena_reward.peft_arena_reward \
        reward_timeout_seconds "${PEFT_ARENA_REWARD_TIMEOUT_S}" \
        math_eval_semantics math_alignment \
        num_rollout "${NUM_ROLLOUT}" \
        rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
        samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
        global_batch_size "${GLOBAL_BATCH_SIZE}" \
        rollout_max_response_len "${ROLLOUT_MAX_RESPONSE_LEN}" \
        eval_max_response_len "${EVAL_MAX_RESPONSE_LEN}" \
        max_tokens_per_gpu "${MAX_TOKENS_PER_GPU}" \
        save_interval "${SAVE_INTERVAL}" \
        eval_interval "${EVAL_INTERVAL}" \
        actor_gpus "${GPUS_PER_NODE}" \
        critic_gpus "${CRITIC_NUM_GPUS_PER_NODE}" \
        rollout_gpus "${ROLLOUT_NUM_GPUS}" \
        ray_num_gpus "${RAY_NUM_GPUS}" \
        ray_num_cpus "${RAY_NUM_CPUS}" \
        sglang_mem_fraction_static "${SGLANG_MEM_FRACTION_STATIC}" \
        sglang_max_running_requests "${SGLANG_MAX_RUNNING_REQUESTS}" \
        sglang_deterministic_inference 1 \
        wandb_enabled "${WANDB_ENABLED}" \
        wandb_mode "${WANDB_MODE:-online}" \
        wandb_project "${WANDB_PROJECT}" \
        wandb_group "${WANDB_GROUP}" \
        wandb_run_id "${WANDB_RUN_ID}" \
        wandb_resume "${WANDB_RESUME}"
}

PPO_CRITIC_COMPARE_LOCK_DIR=""
PPO_CRITIC_COMPARE_LOCK_HELD=0

release_benchmark_lock() {
    if [[ "${PPO_CRITIC_COMPARE_LOCK_HELD:-0}" != "1" ]]; then
        return
    fi
    rm -f -- "${PPO_CRITIC_COMPARE_LOCK_DIR}/owner.tsv"
    rmdir -- "${PPO_CRITIC_COMPARE_LOCK_DIR}" 2>/dev/null || true
    PPO_CRITIC_COMPARE_LOCK_HELD=0
}

# scripts/lib/ray.sh invokes this hook from its own EXIT trap. The local EXIT
# trap below covers failures that occur before the private Ray lifecycle starts.
orbit_launcher_exit_hook() {
    release_benchmark_lock
}

prepare_benchmark_metadata() {
    local metadata_path="${SAVE_DIR}/benchmark-metadata.tsv"
    local expected_path
    local entry

    mkdir -p "${SAVE_DIR}"
    PPO_CRITIC_COMPARE_LOCK_DIR="${SAVE_DIR}.launch-lock"
    if ! mkdir -- "${PPO_CRITIC_COMPARE_LOCK_DIR}" 2>/dev/null; then
        echo "another process is already launching this benchmark run: ${SAVE_DIR}" >&2
        if [[ -f "${PPO_CRITIC_COMPARE_LOCK_DIR}/owner.tsv" ]]; then
            cat "${PPO_CRITIC_COMPARE_LOCK_DIR}/owner.tsv" >&2
        fi
        exit 2
    fi
    PPO_CRITIC_COMPARE_LOCK_HELD=1
    trap orbit_launcher_exit_hook EXIT
    printf '%s\t%s\n' \
        host "$(hostname -f 2>/dev/null || hostname)" \
        pid "$$" \
        started_utc "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
        save_dir "${SAVE_DIR}" >"${PPO_CRITIC_COMPARE_LOCK_DIR}/owner.tsv"
    expected_path="$(mktemp "${SAVE_DIR}/.benchmark-metadata.XXXXXX")"
    render_benchmark_metadata >"${expected_path}"

    if [[ -n "${RESUME_DIR}" ]]; then
        if [[ ! -f "${metadata_path}" ]]; then
            echo "resume metadata is missing: ${metadata_path}" >&2
            rm -f "${expected_path}"
            exit 2
        fi
        if ! cmp -s "${expected_path}" "${metadata_path}"; then
            echo "resume configuration does not match ${metadata_path}" >&2
            diff -u "${metadata_path}" "${expected_path}" >&2 || true
            rm -f "${expected_path}"
            exit 2
        fi
        rm -f "${expected_path}"
        return
    fi

    if [[ -f "${metadata_path}" ]] && ! cmp -s "${expected_path}" "${metadata_path}"; then
        echo "SAVE_DIR contains a different PPO critic-comparison configuration: ${metadata_path}" >&2
        diff -u "${metadata_path}" "${expected_path}" >&2 || true
        rm -f "${expected_path}"
        exit 2
    fi
    for entry in "${SAVE_DIR}"/* "${SAVE_DIR}"/.[!.]* "${SAVE_DIR}"/..?*; do
        [[ -e "${entry}" ]] || continue
        case "$(basename -- "${entry}")" in
            .launch.lock | .benchmark-metadata.* | benchmark-metadata.tsv | launch-argv.log) ;;
            *)
                echo "SAVE_DIR is not fresh; unrecognized artifact ${entry}. Set RESUME_DIR=${SAVE_DIR} to resume." >&2
                rm -f "${expected_path}"
                exit 2
                ;;
        esac
    done
    mv -f "${expected_path}" "${metadata_path}"
}

append_resolved_argv() {
    local argv_path="${SAVE_DIR}/launch-argv.log"
    {
        printf '# %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)"
        printf '%q ' \
            "${ORBIT_ENTRYPOINT}" \
            --actor-num-nodes 1 \
            --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
            "${COLOCATE_ARGS[@]}" \
            "${MODEL_ARGS[@]}" \
            "${CKPT_ARGS[@]}" \
            "${ROLLOUT_ARGS[@]}" \
            "${OPTIMIZER_ARGS[@]}" \
            "${RL_ARGS[@]}" \
            "${LOSS_ARGS[@]}" \
            "${WANDB_ARGS[@]}" \
            "${PERF_ARGS[@]}" \
            "${EVAL_ARGS[@]}" \
            "${SGLANG_ARGS[@]}" \
            "${MISC_ARGS[@]}" \
            "${DEBUG_ARGS[@]}" \
            "${PEFT_ARGS[@]}"
        printf '\n'
    } >>"${argv_path}"
    echo "Resolved benchmark argv appended to ${argv_path}"
}

# Normalize W&B before recording argv; scripts/lib/launcher.sh repeats this
# idempotently immediately before execution.
load_wandb_key
configure_wandb_args
WANDB_ENABLED=0
if (( ${#WANDB_ARGS[@]} > 0 )); then
    WANDB_ENABLED=1
fi

if ! is_true "${SMOKE}" \
   && ! is_true "${ORBIT_DRY_RUN_ARGV:-0}" \
   && [[ "${GIT_DIRTY}" == "1" ]] \
   && ! is_true "${ALLOW_DIRTY_BENCHMARK}"; then
    echo "refusing a main benchmark from a dirty/untracked worktree; commit the recipe or set ALLOW_DIRTY_BENCHMARK=1 and retain the manifest" >&2
    exit 2
fi

if ! is_true "${ORBIT_DRY_RUN_ARGV:-0}"; then
    prepare_benchmark_metadata
    append_resolved_argv
    if is_true "${PPO_CRITIC_COMPARE_PREPARE_ONLY:-0}"; then
        echo "PPO critic-comparison preparation complete; training was not launched."
        exit 0
    fi
fi
unset -f file_sha256_or_missing directory_manifest_sha256 render_benchmark_metadata \
    prepare_benchmark_metadata append_resolved_argv

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
