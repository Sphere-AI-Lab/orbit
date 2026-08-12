#!/usr/bin/env bash
# Run the Llama-3.1 OFT path at the tiny block sizes that exercise its edge
# cases.  This wrapper deliberately owns only campaign isolation and evidence;
# the configured launcher remains the production training entry point.

set -uo pipefail

ORBIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BLOCK_SIZES=(4 8 16)
GPUS_PER_NODE=2
RAY_NUM_GPUS=2
ROLLOUT_NUM_GPUS_PER_ENGINE=2
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1
PEFT_METHOD=oft
NUM_ROLLOUT=3
EVAL_INTERVAL=2
SAVE_INTERVAL=""
WANDB_MODE=offline
ORBIT_RAY_LIFECYCLE=private
ORBIT_LOG_WEIGHT_SYNC=1

RUN_ROOT=${RUN_ROOT:-}
LAUNCHER=${OFT_TINY_SMOKE_LAUNCHER:-"${ORBIT_ROOT}/examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh"}
ARM_TIMEOUT=${OFT_TINY_SMOKE_ARM_TIMEOUT-90m}
DRY_RUN=${DRY_RUN:-0}

die() {
    printf 'smoke_oft_tiny_bs_2gpu: %s\n' "$*" >&2
    exit 2
}

write_status() {
    local destination=$1
    shift
    local temporary="${destination}.tmp.$$"
    printf '%s\n' "$@" >"${temporary}"
    mv -- "${temporary}" "${destination}"
}

write_environment() {
    local destination=$1
    cat >"${destination}" <<EOF
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE}
GPUS_PER_NODE=${GPUS_PER_NODE}
RAY_NUM_GPUS=${RAY_NUM_GPUS}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE}
PEFT_METHOD=${PEFT_METHOD}
NUM_ROLLOUT=${NUM_ROLLOUT}
EVAL_INTERVAL=${EVAL_INTERVAL}
SAVE_INTERVAL=${SAVE_INTERVAL}
WANDB_MODE=${WANDB_MODE}
ORBIT_RAY_LIFECYCLE=${ORBIT_RAY_LIFECYCLE}
ORBIT_LOG_WEIGHT_SYNC=${ORBIT_LOG_WEIGHT_SYNC}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
LAUNCHER=${LAUNCHER}
OFT_TINY_SMOKE_ARM_TIMEOUT=${ARM_TIMEOUT}
RUN_LOG=${RUN_LOG}
SAVE_DIR=${SAVE_DIR}
WANDB_DIR=${WANDB_DIR}
WANDB_RUN_NAME=${WANDB_RUN_NAME}
LAUNCHER_NAME=${LAUNCHER_NAME}
RAY_TEMP_DIR=${RAY_TEMP_DIR}
EOF
}

if [[ -z "${RUN_ROOT}" || "${RUN_ROOT}" != /* || ! -d "${RUN_ROOT}" ]]; then
    die 'RUN_ROOT must be an absolute path to an existing directory'
fi

for block_size in "${BLOCK_SIZES[@]}"; do
    arm_path="${RUN_ROOT}/bs${block_size}"
    if [[ -e "${arm_path}" || -L "${arm_path}" ]]; then
        die "refusing to reuse campaign-owned path: ${arm_path}"
    fi
done
if [[ -e "${RUN_ROOT}/completion.status" || -L "${RUN_ROOT}/completion.status" ]]; then
    die "refusing to reuse campaign-owned path: ${RUN_ROOT}/completion.status"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'OFT tiny-block two-GPU smoke plan\n'
    for block_size in "${BLOCK_SIZES[@]}"; do
        arm_path="${RUN_ROOT}/bs${block_size}"
        printf 'BS%s launcher=%s tp=%s rollout_tp=%s rollouts=%s save_interval=<%s> output=%s\n' \
            "${block_size}" "${LAUNCHER}" "${TENSOR_MODEL_PARALLEL_SIZE}" \
            "${ROLLOUT_NUM_GPUS_PER_ENGINE}" "${NUM_ROLLOUT}" "${SAVE_INTERVAL}" "${arm_path}"
    done
    exit 0
fi

IFS=, read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#visible_gpus[@]} != 2 )) \
    || [[ -z "${visible_gpus[0]:-}" || -z "${visible_gpus[1]:-}" ]] \
    || [[ "${visible_gpus[0]}" == "${visible_gpus[1]}" ]]; then
    die 'CUDA_VISIBLE_DEVICES must name exactly two distinct nonempty devices'
fi

campaign_launcher_exit_code=0
campaign_console_exit_code=0
campaign_verification_exit_code=0
for block_size in "${BLOCK_SIZES[@]}"; do
    arm_path="${RUN_ROOT}/bs${block_size}"
    console_log="${arm_path}/console.log"
    RUN_LOG="${arm_path}/orbit.log"
    environment_file="${arm_path}/environment.txt"
    timings_file="${arm_path}/timings.txt"
    status_file="${arm_path}/completion.status"
    SAVE_DIR="${arm_path}/checkpoints"
    WANDB_DIR="${arm_path}/wandb"
    WANDB_RUN_NAME="oft-tiny-bs${block_size}"
    LAUNCHER_NAME="oft_tiny_bs${block_size}_2gpu"
    RAY_TEMP_DIR="${arm_path}/ray"

    export OFT_BLOCK_SIZE="${block_size}" GPUS_PER_NODE RAY_NUM_GPUS
    export ROLLOUT_NUM_GPUS_PER_ENGINE TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE
    export PEFT_METHOD NUM_ROLLOUT EVAL_INTERVAL SAVE_INTERVAL WANDB_MODE
    export ORBIT_RAY_LIFECYCLE ORBIT_LOG_WEIGHT_SYNC RUN_LOG SAVE_DIR WANDB_DIR WANDB_RUN_NAME
    export LAUNCHER_NAME RAY_TEMP_DIR
    unset ORBIT_RAY_ADDRESS RAY_ADDRESS RAY_PORT RAY_HEAD_PORT RAY_CLIENT_SERVER_PORT
    unset RAY_DASHBOARD_PORT RAY_DASHBOARD_AGENT_LISTEN_PORT RAY_DASHBOARD_AGENT_GRPC_PORT
    unset RAY_GCS_SERVER_PORT RAY_METRICS_EXPORT_PORT RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT
    unset RAY_NODE_MANAGER_PORT RAY_OBJECT_MANAGER_PORT RAY_RUNTIME_ENV_AGENT_PORT

    mkdir -p "${SAVE_DIR}" "${WANDB_DIR}" "${RAY_TEMP_DIR}"
    : >"${RUN_LOG}"
    write_environment "${environment_file}"

    started_seconds=${SECONDS}
    if [[ -n "${ARM_TIMEOUT}" ]]; then
        timeout --signal=TERM --kill-after=120s "${ARM_TIMEOUT}" bash "${LAUNCHER}" 2>&1 | tee "${console_log}"
    else
        bash "${LAUNCHER}" 2>&1 | tee "${console_log}"
    fi
    pipeline_statuses=("${PIPESTATUS[@]}")
    launcher_exit_code=${pipeline_statuses[0]}
    console_exit_code=${pipeline_statuses[1]}
    duration_seconds=$(( SECONDS - started_seconds ))

    verification_exit_code=0
    for marker in \
        'Training driver exited with code 0' \
        'progress rollout=2/2 completed=3/3 remaining=0' \
        'stage=update_weights_complete'; do
        if ! grep -Fq -- "${marker}" "${RUN_LOG}"; then
            verification_exit_code=1
        fi
    done
    grep -F 'done elapsed=' "${RUN_LOG}" >"${timings_file}" || :

    if (( launcher_exit_code != 0 )); then
        final_exit_code=${launcher_exit_code}
    elif (( console_exit_code != 0 )); then
        final_exit_code=${console_exit_code}
    else
        final_exit_code=${verification_exit_code}
    fi
    write_status "${status_file}" \
        "block_size=${block_size}" \
        "launcher_exit_code=${launcher_exit_code}" \
        "console_exit_code=${console_exit_code}" \
        "verification_exit_code=${verification_exit_code}" \
        "final_exit_code=${final_exit_code}" \
        "duration_seconds=${duration_seconds}" \
        "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if (( final_exit_code == 0 )); then
        printf 'BS%s PASSED (%ss)\n' "${block_size}" "${duration_seconds}"
    else
        printf 'BS%s FAILED launcher=%s console=%s verification=%s final=%s (%ss)\n' \
            "${block_size}" "${launcher_exit_code}" "${console_exit_code}" "${verification_exit_code}" \
            "${final_exit_code}" "${duration_seconds}" >&2
    fi
    if (( campaign_launcher_exit_code == 0 && launcher_exit_code != 0 )); then
        campaign_launcher_exit_code=${launcher_exit_code}
    fi
    if (( campaign_console_exit_code == 0 && console_exit_code != 0 )); then
        campaign_console_exit_code=${console_exit_code}
    fi
    if (( campaign_verification_exit_code == 0 && verification_exit_code != 0 )); then
        campaign_verification_exit_code=${verification_exit_code}
    fi
done

if (( campaign_launcher_exit_code != 0 )); then
    campaign_exit_code=${campaign_launcher_exit_code}
elif (( campaign_console_exit_code != 0 )); then
    campaign_exit_code=${campaign_console_exit_code}
else
    campaign_exit_code=${campaign_verification_exit_code}
fi
write_status "${RUN_ROOT}/completion.status" \
    "launcher_exit_code=${campaign_launcher_exit_code}" \
    "console_exit_code=${campaign_console_exit_code}" \
    "verification_exit_code=${campaign_verification_exit_code}" \
    "final_exit_code=${campaign_exit_code}" \
    "duration_seconds=${SECONDS}" \
    "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if (( campaign_exit_code == 0 )); then
    printf 'OFT tiny-block two-GPU smoke PASSED\n'
else
    printf 'OFT tiny-block two-GPU smoke FAILED final=%s\n' "${campaign_exit_code}" >&2
fi
exit "${campaign_exit_code}"
