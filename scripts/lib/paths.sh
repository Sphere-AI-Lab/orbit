#!/usr/bin/env bash
# Path validation, Megatron checkpoint resolution, and HF/Megatron staging
# helpers. Sourced by scripts/lib/launcher.sh after
# common.sh (relies on is_true).

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

prefer_existing_checkpoint() {
    local orbit_path="$1"
    local bridge_path="$2"
    if [[ -e "${orbit_path}" || ! -e "${bridge_path}" ]]; then
        printf '%s\n' "${orbit_path}"
    else
        printf '%s\n' "${bridge_path}"
    fi
}

resolve_local_stage_root() {
    local user_name="${USER:-example-user}"
    local candidate

    if [[ -n "${_CONDOR_SCRATCH_DIR:-}" && -w "${_CONDOR_SCRATCH_DIR}" ]]; then
        printf '%s\n' "${_CONDOR_SCRATCH_DIR}"
        return 0
    fi

    for candidate in \
        "${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage" \
        "/tmp/${user_name}/orbit_stage"
    do
        if mkdir -p "${candidate}" 2>/dev/null && [[ -w "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "No writable local checkpoint staging root found." >&2
    return 1
}

strip_generalized_dataset_path_for_shell() {
    local path_value="$1"
    if [[ "${path_value}" =~ ^(.+)@\[[^]]*\]$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        printf '%s\n' "${path_value}"
    fi
}

require_existing_path() {
    local path_name="$1"
    local path_value="$2"
    path_value="$(strip_generalized_dataset_path_for_shell "${path_value}")"
    if [[ ! -e "${path_value}" ]]; then
        echo "Missing required ${path_name}: ${path_value}" >&2
        exit 1
    fi
}

model_dir_has_hf_weights() {
    local model_dir="$1"
    [[ -d "${model_dir}" ]] || return 1
    [[ -f "${model_dir}/config.json" ]] || return 1
    [[ -f "${model_dir}/model.safetensors.index.json" ]] && return 0
    compgen -G "${model_dir}/*.safetensors" >/dev/null || return 1
}

resolve_megatron_checkpoint_dir() {
    local ckpt_dir="$1"
    local latest_iter
    local candidate

    if [[ -f "${ckpt_dir}/metadata.json" || -f "${ckpt_dir}/.metadata" ]]; then
        printf '%s\n' "${ckpt_dir}"
        return
    fi

    if [[ -f "${ckpt_dir}/latest_checkpointed_iteration.txt" ]]; then
        latest_iter="$(<"${ckpt_dir}/latest_checkpointed_iteration.txt")"
        if [[ "${latest_iter}" =~ ^[0-9]+$ ]]; then
            printf -v candidate '%s/iter_%07d' "${ckpt_dir}" "${latest_iter}"
            if [[ -f "${candidate}/metadata.json" || -f "${candidate}/.metadata" ]]; then
                printf '%s\n' "${candidate}"
                return
            fi
        elif [[ "${latest_iter}" == "release" ]]; then
            candidate="${ckpt_dir}/release"
            if [[ -f "${candidate}/metadata.json" || -f "${candidate}/.metadata" ]]; then
                printf '%s\n' "${candidate}"
                return
            fi
        fi
    fi

    printf '%s\n' "${ckpt_dir}"
}

megatron_checkpoint_tree_has_dist_checkpoint() {
    local ckpt_tree="$1"
    local resolved_ckpt_dir

    [[ -d "${ckpt_tree}" ]] || return 1
    resolved_ckpt_dir="$(resolve_megatron_checkpoint_dir "${ckpt_tree}")"
    [[ -d "${resolved_ckpt_dir}" ]] || return 1
    [[ -f "${resolved_ckpt_dir}/metadata.json" || -f "${resolved_ckpt_dir}/.metadata" ]]
}

replace_arg_value_in_array_if_present() {
    local array_name="$1"
    local flag="$2"
    local value="$3"

    if ! declare -p "${array_name}" >/dev/null 2>&1; then
        return
    fi

    local -n args_array="${array_name}"
    local i
    for ((i = 0; i < ${#args_array[@]} - 1; i++)); do
        if [[ "${args_array[$i]}" == "${flag}" ]]; then
            args_array[$((i + 1))]="${value}"
            return
        fi
    done
}

require_megatron_checkpoint_dir() {
    local path_name="$1"
    local ckpt_dir="$2"
    local resolved_ckpt_dir
    require_existing_path "${path_name}" "${ckpt_dir}"
    resolved_ckpt_dir="$(resolve_megatron_checkpoint_dir "${ckpt_dir}")"
    if [[ ! -d "${resolved_ckpt_dir}" ]]; then
        echo "${path_name} must be a directory: ${ckpt_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${resolved_ckpt_dir}/metadata.json" && ! -f "${resolved_ckpt_dir}/.metadata" ]]; then
        echo "${path_name} does not look like a Megatron checkpoint: ${ckpt_dir}" >&2
        echo "Expected metadata.json or .metadata in the resolved checkpoint directory" >&2
        exit 1
    fi
}

stage_hf_checkpoint_if_requested() {
    if [[ -z "${STAGE_HF_CKPT_TO:-}" ]]; then
        return
    fi

    require_existing_path "HF_CKPT" "${HF_CKPT}"
    local original_hf_ckpt
    local hf_ckpt_real
    local stage_hf_real
    original_hf_ckpt="${HF_CKPT}"
    hf_ckpt_real="$(realpath -m "${HF_CKPT}")"
    stage_hf_real="$(realpath -m "${STAGE_HF_CKPT_TO}")"
    if [[ "${hf_ckpt_real}" == "${stage_hf_real}" ]]; then
        echo "Skipping HF checkpoint staging: HF_CKPT already points to ${STAGE_HF_CKPT_TO}"
    elif ! is_true "${FORCE_STAGE_HF_CKPT:-0}" && model_dir_has_hf_weights "${STAGE_HF_CKPT_TO}"; then
        echo "Using existing staged HF checkpoint at ${STAGE_HF_CKPT_TO} (set FORCE_STAGE_HF_CKPT=1 to refresh)"
    else
        echo "Staging HF checkpoint from ${HF_CKPT} to ${STAGE_HF_CKPT_TO}"
        mkdir -p "${STAGE_HF_CKPT_TO}"
        rsync -ah --info=progress2 --delete "${HF_CKPT}/" "${STAGE_HF_CKPT_TO}/"
    fi
    HF_CKPT="${STAGE_HF_CKPT_TO}"
    replace_arg_value_in_array_if_present CKPT_ARGS --hf-checkpoint "${HF_CKPT}"
    if [[ "${LOAD_CKPT:-}" == "${original_hf_ckpt}" ]]; then
        LOAD_CKPT="${HF_CKPT}"
    fi
}

stage_megatron_checkpoint_if_requested() {
    if [[ -z "${STAGE_MEGATRON_CKPT_TO:-}" || -z "${MEGATRON_LOAD:-}" ]]; then
        return
    fi

    require_existing_path "MEGATRON_LOAD" "${MEGATRON_LOAD}"
    local original_megatron_load
    local megatron_load_real
    local stage_megatron_real
    original_megatron_load="${MEGATRON_LOAD}"
    megatron_load_real="$(realpath -m "${MEGATRON_LOAD}")"
    stage_megatron_real="$(realpath -m "${STAGE_MEGATRON_CKPT_TO}")"
    if [[ "${megatron_load_real}" == "${stage_megatron_real}" ]]; then
        echo "Skipping Megatron checkpoint staging: MEGATRON_LOAD already points to ${STAGE_MEGATRON_CKPT_TO}"
    elif ! is_true "${FORCE_STAGE_MEGATRON_CKPT:-0}" && megatron_checkpoint_tree_has_dist_checkpoint "${STAGE_MEGATRON_CKPT_TO}"; then
        echo "Using existing staged Megatron checkpoint at ${STAGE_MEGATRON_CKPT_TO} (set FORCE_STAGE_MEGATRON_CKPT=1 to refresh)"
    else
        echo "Staging Megatron checkpoint from ${MEGATRON_LOAD} to ${STAGE_MEGATRON_CKPT_TO}"
        mkdir -p "${STAGE_MEGATRON_CKPT_TO}"
        rsync -ah --info=progress2 --delete "${MEGATRON_LOAD}/" "${STAGE_MEGATRON_CKPT_TO}/"
    fi

    MEGATRON_LOAD="${STAGE_MEGATRON_CKPT_TO}"
    replace_arg_value_in_array_if_present CKPT_ARGS --load "${MEGATRON_LOAD}"
    if [[ "${LOAD_CKPT:-}" == "${original_megatron_load}" ]]; then
        LOAD_CKPT="${MEGATRON_LOAD}"
    fi
}

configure_save_dir_lustre_stripe() {
    local save_dir="$1"
    local stripe_count="${SAVE_DIR_STRIPE_COUNT:-}"
    local stripe_size="${SAVE_DIR_STRIPE_SIZE:-16M}"
    local -a stripe_args

    if [[ -z "${stripe_count}" ]]; then
        return 0
    fi

    if ! command -v lfs >/dev/null 2>&1; then
        echo "Skipping SAVE_DIR striping: lfs is not available" >&2
        return 0
    fi
    if ! lfs getstripe -d "${save_dir}" >/dev/null 2>&1; then
        echo "Skipping SAVE_DIR striping: ${save_dir} is not on Lustre" >&2
        return 0
    fi

    stripe_args=(-c "${stripe_count}")
    if [[ -n "${stripe_size}" ]]; then
        stripe_args+=(-S "${stripe_size}")
    fi

    echo "Configuring Lustre striping for SAVE_DIR=${save_dir}: count=${stripe_count} size=${stripe_size:-default}"
    if ! lfs setstripe "${stripe_args[@]}" "${save_dir}"; then
        echo "Warning: failed to configure Lustre striping for SAVE_DIR=${save_dir}" >&2
    fi
}

prepare_paths_and_checkpoints() {
    SAVE_DIR_STRIPE_COUNT=${SAVE_DIR_STRIPE_COUNT:-}
    SAVE_DIR_STRIPE_SIZE=${SAVE_DIR_STRIPE_SIZE:-16M}

    stage_hf_checkpoint_if_requested
    stage_megatron_checkpoint_if_requested

    if is_true "${REQUIRE_MEGATRON_LOAD:-0}"; then
        LOAD_CKPT="$(resolve_megatron_checkpoint_dir "${LOAD_CKPT}")"
    fi

    require_existing_path "HF_CKPT" "${HF_CKPT}"
    require_existing_path "LOAD_CKPT" "${LOAD_CKPT}"
    require_existing_path "TRAIN_JSONL" "${TRAIN_JSONL}"
    require_existing_path "TEST_JSONL" "${TEST_JSONL}"
    require_existing_path "MODEL_ARGS_FILE" "${MODEL_ARGS_FILE}"

    if is_true "${REQUIRE_MEGATRON_LOAD:-0}"; then
        require_megatron_checkpoint_dir "LOAD_CKPT" "${LOAD_CKPT}"
    fi

    if ! is_true "${DISABLE_SAVE:-0}"; then
        mkdir -p "${SAVE_DIR}"
        configure_save_dir_lustre_stripe "${SAVE_DIR}"
    fi

    if [[ -z "${NUM_ROLLOUT:-}" ]]; then
        TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
        NUM_ROLLOUT=$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))
    fi
}

source_model_args() {
    source "${MODEL_ARGS_FILE}"
}
