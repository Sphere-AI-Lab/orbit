#!/usr/bin/env bash
# Low-precision preflight checks (FP8 / INT4) gated on PRECISION_PROFILE.
# Sourced by scripts/lib/launcher.sh after env knobs
# and after path validation.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

apply_preflight_defaults() {
    PRECISION_PROFILE=${PRECISION_PROFILE:-bf16}
}

prepare_runtime_parity_env() {
    local verl_runtime_root="${ORBIT_RUNTIME_PARITY_VERL_ROOT:-${ORBIT_ROOT}/../verl}"
    if [[ -d "${verl_runtime_root}/verl" ]]; then
        export PYTHONPATH="${verl_runtime_root}:${PYTHONPATH:-}"
    fi

    if [[ "${ORBIT_RUNTIME_PARITY_FIRST_LAYER_ONLY:-1}" != "0" ]]; then
        export ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE="${ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE:-1}"
        export ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_LAYERS="${ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_LAYERS:-0}"
        export ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_TRUNCATE_SIZE="${ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_TRUNCATE_SIZE:-4}"
        export ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE="${ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE:-1}"
        export ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE_LAYERS="${ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE_LAYERS:-0}"
        export ORBIT_RUNTIME_PARITY_MAX_PROMPTS="${ORBIT_RUNTIME_PARITY_MAX_PROMPTS:-1}"
    fi
}

preflight_low_precision() {
    if [[ "${PRECISION_PROFILE}" == "fp8" ]]; then
        export ORBIT_LOG_WEIGHT_SYNC=${ORBIT_LOG_WEIGHT_SYNC:-1}
        # FP8 scales are loaded from regular checkpoint tensor keys; skip
        # ModelOpt extra_state restore unless a caller explicitly opts in.
        export ORBIT_RESTORE_MODELOPT_STATE=${ORBIT_RESTORE_MODELOPT_STATE:-0}
        export SGLANG_FP8_GEMM_BACKEND=${SGLANG_FP8_GEMM_BACKEND:-auto}
        export MEGATRON_OFT_FP8_ACTIVATION_QUANT=${MEGATRON_OFT_FP8_ACTIVATION_QUANT:-w8a8}
        export MEGATRON_KEEP_NATIVE_FP8_WEIGHTS=${MEGATRON_KEEP_NATIVE_FP8_WEIGHTS:-True}
        export MEGATRON_QWEN3_FP8_GEMM_BACKEND=${MEGATRON_QWEN3_FP8_GEMM_BACKEND:-sglang_native}
        export VERL_KEEP_NATIVE_FP8_WEIGHTS=${VERL_KEEP_NATIVE_FP8_WEIGHTS:-${MEGATRON_KEEP_NATIVE_FP8_WEIGHTS}}

        if [[ "${SKIP_FP8_CHECKPOINT_PREFLIGHT:-0}" != "1" ]]; then
            python "${ORBIT_ROOT}/tools/check_fp8_checkpoint_parity.py" \
                --hf-model "${HF_CKPT}" \
                --megatron-dist "${LOAD_CKPT}" \
                --report-json "${SAVE_DIR}/checkpoint_parity.json"
        fi

        if [[ "${RUN_FP8_RUNTIME_PREFLIGHT:-0}" == "1" ]]; then
            require_existing_path "PROMPT_FILE" "${PROMPT_FILE}"
            prepare_runtime_parity_env
            local fp8_runtime_preflight_args=(
                --tensor-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
                --pipeline-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
                --rollout-quantization "${ROLLOUT_QUANTIZATION:-fp8}"
                --sglang-fp8-gemm-backend "${SGLANG_FP8_GEMM_BACKEND}"
                --megatron-activation-fp8 "${MEGATRON_ACTIVATION_FP8:-none}"
                --megatron-oft-fp8-activation-quant "${MEGATRON_OFT_FP8_ACTIVATION_QUANT}"
                --max-logprob-abs-diff "${MAX_LOGPROB_ABS_DIFF:-0.20}"
                --min-topk-agreement "${MIN_TOPK_AGREEMENT:-0.50}"
                --report-json "${SAVE_DIR}/step0_parity.json"
            )
            if [[ "${FP8_RUNTIME_RESTORE_MODELOPT_STATE:-0}" == "1" ]]; then
                fp8_runtime_preflight_args+=(--restore-modelopt-state)
            else
                fp8_runtime_preflight_args+=(--no-restore-modelopt-state)
            fi
            python "${ORBIT_ROOT}/tools/check_fp8_runtime_parity.py" \
                --hf-model "${HF_CKPT}" \
                --megatron-dist "${LOAD_CKPT}" \
                --prompt-file "${PROMPT_FILE}" \
                "${fp8_runtime_preflight_args[@]}"
        fi
    elif [[ "${PRECISION_PROFILE}" == "nvfp4" ]]; then
        if is_true "${PARITY_CHECK:-0}"; then
            export ORBIT_LOG_WEIGHT_SYNC=${ORBIT_LOG_WEIGHT_SYNC:-1}
        else
            export ORBIT_LOG_WEIGHT_SYNC=${ORBIT_LOG_WEIGHT_SYNC:-0}
        fi
        export SGLANG_DISABLE_FLASHINFER_AUTOTUNE=${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}

        if [[ "${SKIP_NVFP4_CHECKPOINT_PREFLIGHT:-0}" != "1" ]]; then
            python "${ORBIT_ROOT}/tools/check_nvfp4_checkpoint_parity.py" \
                --hf-model "${HF_CKPT}" \
                --megatron-dist "${LOAD_CKPT}" \
                --report-json "${SAVE_DIR}/checkpoint_parity.json"
        fi

        if [[ "${RUN_NVFP4_RUNTIME_PREFLIGHT:-0}" == "1" ]]; then
            require_existing_path "PROMPT_FILE" "${PROMPT_FILE}"
            prepare_runtime_parity_env
            local nvfp4_runtime_preflight_args=(
                --tensor-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
                --pipeline-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
                --rollout-quantization "${ROLLOUT_QUANTIZATION:-modelopt_fp4}"
                --sglang-fp8-gemm-backend "${SGLANG_FP8_GEMM_BACKEND:-auto}"
                --max-logprob-abs-diff "${MAX_LOGPROB_ABS_DIFF:-0.20}"
                --min-topk-agreement "${MIN_TOPK_AGREEMENT:-0.50}"
                --restore-modelopt-state
                --report-json "${SAVE_DIR}/step0_parity.json"
            )
            if [[ "${NVFP4_RUNTIME_PREFLIGHT_DEQUANT:-0}" == "1" ]]; then
                nvfp4_runtime_preflight_args+=(--dequant-nvfp4-weights-for-parity)
            fi
            python "${ORBIT_ROOT}/tools/check_nvfp4_runtime_parity.py" \
                --hf-model "${HF_CKPT}" \
                --megatron-dist "${LOAD_CKPT}" \
                --prompt-file "${PROMPT_FILE}" \
                "${nvfp4_runtime_preflight_args[@]}"
        fi
    elif [[ "${PRECISION_PROFILE}" == "mxfp4" ]]; then
        # DSV4 MXFP4: experts are staged FP4 at conversion; runtime GEMMs are
        # deterministic FP8 (flags live in dsv4-common.sh). No MXFP4
        # parity tooling yet, so this arm is label + weight-sync logging only.
        if is_true "${PARITY_CHECK:-0}"; then
            export ORBIT_LOG_WEIGHT_SYNC=${ORBIT_LOG_WEIGHT_SYNC:-1}
        else
            export ORBIT_LOG_WEIGHT_SYNC=${ORBIT_LOG_WEIGHT_SYNC:-0}
        fi
    elif [[ "${PRECISION_PROFILE}" == "int4" ]]; then
        export OPEN_TRAINING_INT4_FAKE_QAT_FLAG=${OPEN_TRAINING_INT4_FAKE_QAT_FLAG:-1}
        export OPEN_TRAINING_INT4_GROUP_SIZE=${OPEN_TRAINING_INT4_GROUP_SIZE:-128}

        if [[ "${SKIP_INT4_CHECKPOINT_PREFLIGHT:-0}" != "1" ]]; then
            python "${ORBIT_ROOT}/tools/check_int4_checkpoint_parity.py" \
                --hf-model "${HF_CKPT}" \
                --megatron-dist "${LOAD_CKPT}" \
                --report-json "${SAVE_DIR}/checkpoint_parity.json"
        fi

        if [[ "${RUN_INT4_RUNTIME_PREFLIGHT:-0}" == "1" ]]; then
            require_existing_path "PROMPT_FILE" "${PROMPT_FILE}"
            prepare_runtime_parity_env
            local int4_runtime_preflight_args=(
                --tensor-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
                --pipeline-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
                --no-restore-modelopt-state
                --max-logprob-abs-diff "${MAX_LOGPROB_ABS_DIFF:-0.10}"
                --min-topk-agreement "${MIN_TOPK_AGREEMENT:-0.75}"
                --report-json "${SAVE_DIR}/step0_parity.json"
            )
            if [[ "${INT4_RUNTIME_PREFLIGHT_DEQUANT:-0}" != "1" ]]; then
                int4_runtime_preflight_args+=(--no-dequant-int4-weights-for-parity)
            fi
            if [[ -n "${ROLLOUT_QUANTIZATION:-}" ]]; then
                int4_runtime_preflight_args+=(--rollout-quantization "${ROLLOUT_QUANTIZATION}")
            fi
            python "${ORBIT_ROOT}/tools/check_int4_runtime_parity.py" \
                --hf-model "${HF_CKPT}" \
                --megatron-dist "${LOAD_CKPT}" \
                --prompt-file "${PROMPT_FILE}" \
                "${int4_runtime_preflight_args[@]}"
        fi
    fi
}
