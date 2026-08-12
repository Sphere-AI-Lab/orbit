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
EVIDENCE_FAILURE_CODE=74
STATUS_FAILURE_TEST_DESTINATION=${OFT_TINY_SMOKE_TEST_FAIL_STATUS:-}
SIGNAL_GRACE_SECONDS=${OFT_TINY_SMOKE_SIGNAL_GRACE_SECONDS:-10}

campaign_launcher_exit_code=0
campaign_console_exit_code=0
campaign_verification_exit_code=0
campaign_evidence_exit_code=0
campaign_exit_code=0
campaign_interrupted=none

block_size=
arm_path=
status_file=
timings_file=
RUN_LOG=
started_seconds=0
launcher_exit_code=0
console_exit_code=0
verification_exit_code=0
evidence_exit_code=0
final_exit_code=0
active_launcher_pid=
active_launcher_group=0
active_tee_pid=
active_fifo=
active_arm_reserved=0

die() {
    printf 'smoke_oft_tiny_bs_2gpu: %s\n' "$*" >&2
    exit 2
}

write_status() {
    local destination=$1
    shift
    local temporary=

    if [[ -n "${STATUS_FAILURE_TEST_DESTINATION}" \
        && "${destination}" == "${STATUS_FAILURE_TEST_DESTINATION}" ]]; then
        return "${EVIDENCE_FAILURE_CODE}"
    fi
    if ! temporary=$(mktemp "${destination}.tmp.XXXXXX"); then
        return "${EVIDENCE_FAILURE_CODE}"
    fi
    if ! printf '%s\n' "$@" >"${temporary}"; then
        rm -f -- "${temporary}" 2>/dev/null || :
        return "${EVIDENCE_FAILURE_CODE}"
    fi
    if ! mv -- "${temporary}" "${destination}"; then
        rm -f -- "${temporary}" 2>/dev/null || :
        return "${EVIDENCE_FAILURE_CODE}"
    fi
}

write_timings() {
    local source=$1
    local destination=$2
    local temporary=
    local grep_exit_code=0

    if ! temporary=$(mktemp "${destination}.tmp.XXXXXX"); then
        return "${EVIDENCE_FAILURE_CODE}"
    fi
    grep -F 'done elapsed=' "${source}" >"${temporary}"
    grep_exit_code=$?
    if (( grep_exit_code > 1 )); then
        rm -f -- "${temporary}" 2>/dev/null || :
        return "${EVIDENCE_FAILURE_CODE}"
    fi
    if ! mv -- "${temporary}" "${destination}"; then
        rm -f -- "${temporary}" 2>/dev/null || :
        return "${EVIDENCE_FAILURE_CODE}"
    fi
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
OFT_TINY_SMOKE_SIGNAL_GRACE_SECONDS=${SIGNAL_GRACE_SECONDS}
EOF
}

select_final_exit_code() {
    local selected_launcher=$1
    local selected_console=$2
    local selected_verification=$3
    local selected_evidence=$4

    if (( selected_evidence != 0 )); then
        selected_exit_code=${selected_evidence}
    elif (( selected_launcher != 0 )); then
        selected_exit_code=${selected_launcher}
    elif (( selected_console != 0 )); then
        selected_exit_code=${selected_console}
    else
        selected_exit_code=${selected_verification}
    fi
}

record_current_arm_for_campaign() {
    if (( campaign_launcher_exit_code == 0 && launcher_exit_code != 0 )); then
        campaign_launcher_exit_code=${launcher_exit_code}
    fi
    if (( campaign_console_exit_code == 0 && console_exit_code != 0 )); then
        campaign_console_exit_code=${console_exit_code}
    fi
    if (( campaign_verification_exit_code == 0 && verification_exit_code != 0 )); then
        campaign_verification_exit_code=${verification_exit_code}
    fi
    if (( campaign_evidence_exit_code == 0 && evidence_exit_code != 0 )); then
        campaign_evidence_exit_code=${evidence_exit_code}
    fi
}

publish_current_arm_status() {
    local interrupted=${1:-none}
    write_status "${status_file}" \
        "block_size=${block_size}" \
        "launcher_exit_code=${launcher_exit_code}" \
        "console_exit_code=${console_exit_code}" \
        "verification_exit_code=${verification_exit_code}" \
        "evidence_exit_code=${evidence_exit_code}" \
        "interrupted=${interrupted}" \
        "final_exit_code=${final_exit_code}" \
        "duration_seconds=$(( SECONDS - started_seconds ))" \
        "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

publish_campaign_status() {
    write_status "${RUN_ROOT}/completion.status" \
        "launcher_exit_code=${campaign_launcher_exit_code}" \
        "console_exit_code=${campaign_console_exit_code}" \
        "verification_exit_code=${campaign_verification_exit_code}" \
        "evidence_exit_code=${campaign_evidence_exit_code}" \
        "interrupted=${campaign_interrupted}" \
        "final_exit_code=${campaign_exit_code}" \
        "duration_seconds=${SECONDS}" \
        "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

stop_campaign_for_evidence_failure() {
    local description=$1
    local status_writable=${2:-1}
    printf 'smoke_oft_tiny_bs_2gpu: evidence failure: %s\n' "${description}" >&2
    campaign_evidence_exit_code=${EVIDENCE_FAILURE_CODE}
    campaign_exit_code=${EVIDENCE_FAILURE_CODE}
    if (( status_writable == 1 )) && ! publish_campaign_status; then
        printf 'smoke_oft_tiny_bs_2gpu: failed to publish campaign status\n' >&2
    fi
    exit "${EVIDENCE_FAILURE_CODE}"
}

stop_current_arm_for_evidence_failure() {
    local description=$1
    local arm_status_writable=${2:-1}
    printf 'smoke_oft_tiny_bs_2gpu: BS%s evidence failure: %s\n' \
        "${block_size}" "${description}" >&2
    evidence_exit_code=${EVIDENCE_FAILURE_CODE}
    final_exit_code=${EVIDENCE_FAILURE_CODE}
    if (( arm_status_writable == 1 )); then
        if ! publish_current_arm_status; then
            printf 'smoke_oft_tiny_bs_2gpu: failed to publish BS%s status\n' "${block_size}" >&2
        fi
    fi
    record_current_arm_for_campaign
    campaign_exit_code=${EVIDENCE_FAILURE_CODE}
    if ! publish_campaign_status; then
        printf 'smoke_oft_tiny_bs_2gpu: failed to publish campaign status\n' >&2
    fi
    exit "${EVIDENCE_FAILURE_CODE}"
}

cleanup_active_fifo() {
    if [[ -n "${active_fifo}" && ( -e "${active_fifo}" || -p "${active_fifo}" ) ]]; then
        if ! rm -f -- "${active_fifo}"; then
            return "${EVIDENCE_FAILURE_CODE}"
        fi
    fi
    active_fifo=
}

kill_active_launcher() {
    local requested_signal=$1

    if (( active_launcher_group == 1 )); then
        kill "-${requested_signal}" -- "-${active_launcher_pid}" 2>/dev/null \
            || kill "-${requested_signal}" "${active_launcher_pid}" 2>/dev/null \
            || :
    else
        kill "-${requested_signal}" "${active_launcher_pid}" 2>/dev/null || :
    fi
}

force_kill_active_launcher_after_grace() {
    local deadline=$(( SECONDS + SIGNAL_GRACE_SECONDS ))

    while kill -0 "${active_launcher_pid}" 2>/dev/null \
        && (( SECONDS < deadline )); do
        sleep 0.1
    done
    if kill -0 "${active_launcher_pid}" 2>/dev/null; then
        kill_active_launcher KILL
    fi
}

handle_signal() {
    local received_signal=$1
    local signal_exit_code=143
    local wait_exit_code=0

    trap '' INT TERM
    if [[ "${received_signal}" == INT ]]; then
        signal_exit_code=130
    fi

    if [[ -n "${active_launcher_pid}" ]]; then
        kill_active_launcher TERM
        force_kill_active_launcher_after_grace
        wait "${active_launcher_pid}" 2>/dev/null
        wait_exit_code=$?
        if (( wait_exit_code == 0 )); then
            launcher_exit_code=${signal_exit_code}
        else
            launcher_exit_code=${wait_exit_code}
        fi
    else
        launcher_exit_code=${signal_exit_code}
        if [[ -n "${active_tee_pid}" ]]; then
            kill -TERM "${active_tee_pid}" 2>/dev/null || :
        fi
    fi
    if [[ -n "${active_tee_pid}" ]]; then
        wait "${active_tee_pid}" 2>/dev/null
        console_exit_code=$?
    fi
    if ! cleanup_active_fifo; then
        evidence_exit_code=${EVIDENCE_FAILURE_CODE}
    fi

    verification_exit_code=1
    if (( active_arm_reserved == 1 )); then
        if ! write_timings "${RUN_LOG}" "${timings_file}"; then
            evidence_exit_code=${EVIDENCE_FAILURE_CODE}
        fi
        final_exit_code=${signal_exit_code}
        if ! publish_current_arm_status "${received_signal}"; then
            evidence_exit_code=${EVIDENCE_FAILURE_CODE}
            printf 'smoke_oft_tiny_bs_2gpu: failed to publish interrupted BS%s status\n' \
                "${block_size}" >&2
        fi
        record_current_arm_for_campaign
    else
        campaign_launcher_exit_code=${launcher_exit_code}
        campaign_verification_exit_code=${verification_exit_code}
        campaign_evidence_exit_code=${evidence_exit_code}
    fi
    campaign_interrupted=${received_signal}
    campaign_exit_code=${signal_exit_code}
    if ! publish_campaign_status; then
        printf 'smoke_oft_tiny_bs_2gpu: failed to publish interrupted campaign status\n' >&2
    fi
    printf 'smoke_oft_tiny_bs_2gpu: interrupted by %s\n' "${received_signal}" >&2
    exit "${signal_exit_code}"
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

if [[ -n "${ARM_TIMEOUT}" ]] && ! command -v timeout >/dev/null 2>&1; then
    die 'OFT_TINY_SMOKE_ARM_TIMEOUT requires GNU timeout on PATH; set it empty to disable'
fi
if [[ ! "${SIGNAL_GRACE_SECONDS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    die 'OFT_TINY_SMOKE_SIGNAL_GRACE_SECONDS must be a canonical nonnegative integer'
fi

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

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
    launcher_exit_code=0
    console_exit_code=0
    verification_exit_code=0
    evidence_exit_code=0
    final_exit_code=0
    active_launcher_pid=
    active_launcher_group=0
    active_tee_pid=
    active_fifo="${arm_path}/launcher.fifo"
    active_arm_reserved=0
    started_seconds=${SECONDS}

    if ! mkdir -- "${arm_path}"; then
        stop_campaign_for_evidence_failure \
            "could not reserve ${arm_path} exclusively; campaign ownership was not acquired" 0
    fi
    active_arm_reserved=1

    export OFT_BLOCK_SIZE="${block_size}" GPUS_PER_NODE RAY_NUM_GPUS
    export ROLLOUT_NUM_GPUS_PER_ENGINE TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE
    export PEFT_METHOD NUM_ROLLOUT EVAL_INTERVAL SAVE_INTERVAL WANDB_MODE
    export ORBIT_RAY_LIFECYCLE ORBIT_LOG_WEIGHT_SYNC RUN_LOG SAVE_DIR WANDB_DIR WANDB_RUN_NAME
    export LAUNCHER_NAME RAY_TEMP_DIR
    unset ORBIT_RAY_ADDRESS RAY_ADDRESS RAY_PORT RAY_HEAD_PORT RAY_CLIENT_SERVER_PORT
    unset RAY_DASHBOARD_PORT RAY_DASHBOARD_AGENT_LISTEN_PORT RAY_DASHBOARD_AGENT_GRPC_PORT
    unset RAY_GCS_SERVER_PORT RAY_METRICS_EXPORT_PORT RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT
    unset RAY_NODE_MANAGER_PORT RAY_OBJECT_MANAGER_PORT RAY_RUNTIME_ENV_AGENT_PORT
    unset MASTER_ADDR

    if ! mkdir -p -- "${SAVE_DIR}" "${WANDB_DIR}" "${RAY_TEMP_DIR}"; then
        stop_current_arm_for_evidence_failure 'could not create evidence directories'
    fi
    if ! : >"${RUN_LOG}"; then
        stop_current_arm_for_evidence_failure 'could not create orbit.log'
    fi
    if ! write_environment "${environment_file}"; then
        stop_current_arm_for_evidence_failure 'could not write environment.txt'
    fi
    if ! mkfifo -- "${active_fifo}"; then
        stop_current_arm_for_evidence_failure 'could not create launcher FIFO'
    fi

    tee "${console_log}" <"${active_fifo}" &
    active_tee_pid=$!
    launcher_command=(bash "${LAUNCHER}")
    if [[ -n "${ARM_TIMEOUT}" ]]; then
        launcher_command=(
            timeout --signal=TERM --kill-after=120s "${ARM_TIMEOUT}" bash "${LAUNCHER}"
        )
    fi
    if command -v setsid >/dev/null 2>&1; then
        setsid "${launcher_command[@]}" >"${active_fifo}" 2>&1 &
        active_launcher_group=1
    else
        "${launcher_command[@]}" >"${active_fifo}" 2>&1 &
    fi
    active_launcher_pid=$!
    wait "${active_launcher_pid}"
    launcher_exit_code=$?
    wait "${active_tee_pid}"
    console_exit_code=$?
    duration_seconds=$(( SECONDS - started_seconds ))
    active_launcher_pid=
    active_tee_pid=
    if ! cleanup_active_fifo; then
        stop_current_arm_for_evidence_failure 'could not remove launcher FIFO'
    fi

    verification_exit_code=0
    for marker in \
        'Training driver exited with code 0' \
        'progress rollout=2/2 completed=3/3 remaining=0' \
        'stage=update_weights_complete'; do
        if ! grep -Fq -- "${marker}" "${RUN_LOG}"; then
            verification_exit_code=1
        fi
    done
    if ! write_timings "${RUN_LOG}" "${timings_file}"; then
        stop_current_arm_for_evidence_failure 'could not write timings.txt'
    fi

    select_final_exit_code \
        "${launcher_exit_code}" "${console_exit_code}" "${verification_exit_code}" \
        "${evidence_exit_code}"
    final_exit_code=${selected_exit_code}
    if ! publish_current_arm_status; then
        stop_current_arm_for_evidence_failure 'could not publish completion.status' 0
    fi

    if (( final_exit_code == 0 )); then
        printf 'BS%s PASSED (%ss)\n' "${block_size}" "${duration_seconds}"
    else
        printf 'BS%s FAILED launcher=%s console=%s verification=%s final=%s (%ss)\n' \
            "${block_size}" "${launcher_exit_code}" "${console_exit_code}" "${verification_exit_code}" \
            "${final_exit_code}" "${duration_seconds}" >&2
    fi
    record_current_arm_for_campaign
    active_arm_reserved=0
done

select_final_exit_code \
    "${campaign_launcher_exit_code}" "${campaign_console_exit_code}" \
    "${campaign_verification_exit_code}" "${campaign_evidence_exit_code}"
campaign_exit_code=${selected_exit_code}
if ! publish_campaign_status; then
    campaign_evidence_exit_code=${EVIDENCE_FAILURE_CODE}
    campaign_exit_code=${EVIDENCE_FAILURE_CODE}
    printf 'smoke_oft_tiny_bs_2gpu: failed to publish campaign status\n' >&2
    exit "${EVIDENCE_FAILURE_CODE}"
fi

if (( campaign_exit_code == 0 )); then
    printf 'OFT tiny-block two-GPU smoke PASSED\n'
else
    printf 'OFT tiny-block two-GPU smoke FAILED final=%s\n' "${campaign_exit_code}" >&2
fi
exit "${campaign_exit_code}"
