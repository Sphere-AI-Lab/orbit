#!/usr/bin/env bash
# CUDA modules and library paths, no_proxy for local services, PYTHONPATH,
# and model-name helpers. Sourced by leaf training launchers (via
# scripts/lib/launcher.sh) and by standalone tool/parity/conversion scripts.

SCRIPTS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="${ORBIT_ROOT:-$(cd "${SCRIPTS_LIB_DIR}/../.." && pwd)}"

cd "${ORBIT_ROOT}"
export PYTHONPATH="${ORBIT_ROOT}:${PYTHONPATH:-}"

ORBIT_CUDA_MODULE=${ORBIT_CUDA_MODULE:-cuda/13.2}
if [[ "${ORBIT_LOAD_CUDA_MODULES:-1}" != "0" ]] && command -v module >/dev/null 2>&1; then
    module load "${ORBIT_CUDA_MODULE}" || true
    module load nccl || true
fi

find_cuda_home() {
    local candidate

    if [[ -n "${ORBIT_CUDA_HOME_DEFAULT:-}" && -d "${ORBIT_CUDA_HOME_DEFAULT}" ]]; then
        printf '%s\n' "${ORBIT_CUDA_HOME_DEFAULT}"
        return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        candidate="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
        if [[ -d "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    for candidate in /usr/local/cuda-13.2 /usr/local/cuda; do
        if [[ -d "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

if [[ -z "${CUDA_HOME:-}" ]]; then
    if _cuda_home_candidate="$(find_cuda_home)"; then
        export CUDA_HOME="${_cuda_home_candidate}"
    fi
    unset _cuda_home_candidate
fi

filter_conflicting_cuda_bins() {
    local entry
    local filtered=()

    IFS=':' read -r -a entries <<< "${PATH:-}"
    for entry in "${entries[@]}"; do
        [[ -z "${entry}" ]] && continue
        case "${entry}" in
            */cuda-*/bin | */cuda/bin)
                [[ -n "${CUDA_HOME:-}" && "${entry}" == "${CUDA_HOME}/bin" ]] || continue
                ;;
        esac
        filtered+=("${entry}")
    done

    local IFS=:
    export PATH="${filtered[*]}"
}

filter_conflicting_cuda_libs() {
    local entry
    local filtered=()

    IFS=':' read -r -a entries <<< "${LD_LIBRARY_PATH:-}"
    for entry in "${entries[@]}"; do
        [[ -z "${entry}" ]] && continue
        case "${entry}" in
            */cuda-*/lib64 | */cuda/lib64 | */cuda-*/targets/*/lib | */cuda/targets/*/lib)
                continue
                ;;
        esac
        filtered+=("${entry}")
    done

    local IFS=:
    export LD_LIBRARY_PATH="${filtered[*]}"
}

filter_conflicting_cudnn_libs() {
    local preferred_cudnn_lib="${1:-}"
    local entry
    local filtered=()

    IFS=':' read -r -a entries <<< "${LD_LIBRARY_PATH:-}"
    for entry in "${entries[@]}"; do
        [[ -z "${entry}" ]] && continue
        if [[ -n "${preferred_cudnn_lib}" && "${entry}" == "${preferred_cudnn_lib}" ]]; then
            continue
        fi
        case "${entry}" in
            *cudnn*/lib)
                continue
                ;;
        esac
        filtered+=("${entry}")
    done

    local IFS=:
    export LD_LIBRARY_PATH="${filtered[*]}"
}

if [[ -n "${CUDA_HOME:-}" ]]; then
    filter_conflicting_cuda_bins
    export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
    filter_conflicting_cuda_libs
    CUDA_LD_LIBRARY_PATH=""
    for cuda_lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/targets/x86_64-linux/lib"; do
        [[ -d "${cuda_lib_dir}" ]] || continue
        CUDA_LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH:+${CUDA_LD_LIBRARY_PATH}:}${cuda_lib_dir}"
    done
    if [[ -n "${CUDA_LD_LIBRARY_PATH}" ]]; then
        export LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    unset CUDA_LD_LIBRARY_PATH cuda_lib_dir
fi

PREFERRED_CUDNN_LIB=""
if [[ -n "${CUDNN_PATH:-}" && -d "${CUDNN_PATH}/lib" ]]; then
    PREFERRED_CUDNN_LIB="${CUDNN_PATH}/lib"
elif [[ -n "${CUDNN_HOME:-}" && -d "${CUDNN_HOME}/lib" ]]; then
    PREFERRED_CUDNN_LIB="${CUDNN_HOME}/lib"
fi
if [[ -n "${PREFERRED_CUDNN_LIB}" ]]; then
    filter_conflicting_cudnn_libs "${PREFERRED_CUDNN_LIB}"
    export LD_LIBRARY_PATH="${PREFERRED_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    filter_conflicting_cudnn_libs
fi
unset PREFERRED_CUDNN_LIB

ORBIT_VENV=${ORBIT_VENV:-${VIRTUAL_ENV:-}}
if [[ -n "${ORBIT_VENV}" && -z "${VIRTUAL_ENV:-}" && -x "${ORBIT_VENV}/bin/python" ]]; then
    export VIRTUAL_ENV="${ORBIT_VENV}"
    export PATH="${ORBIT_VENV}/bin:${PATH}"
elif [[ -n "${ORBIT_VENV}" ]] && ! command -v ray >/dev/null 2>&1 && [[ -x "${ORBIT_VENV}/bin/ray" ]]; then
    export PATH="${ORBIT_VENV}/bin:${PATH}"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

HOST_IP="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
LOCAL_NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1"
if [[ -n "${MASTER_ADDR:-}" ]]; then
    LOCAL_NO_PROXY="${LOCAL_NO_PROXY},${MASTER_ADDR}"
fi
if [[ -n "${HOST_IP}" ]]; then
    LOCAL_NO_PROXY="${LOCAL_NO_PROXY},${HOST_IP}"
fi
LOCAL_NO_PROXY="${LOCAL_NO_PROXY},172.22.0.0/16"
export no_proxy="${LOCAL_NO_PROXY}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

if [[ -n "${MEGATRON_BRIDGE_ROOT:-}" ]]; then
    export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}/src:${MEGATRON_BRIDGE_ROOT}/3rdparty/Megatron-LM:${PYTHONPATH}"
fi

model_name() {
    basename "$1" | tr '/' '-'
}

default_output_path() {
    local hf_model="$1"
    local suffix="${2:-}"
    local m
    m="$(model_name "${hf_model}")"
    printf '%s/%s%s\n' "${DEFAULT_OUTPUT_ROOT:-${ORBIT_ROOT}/checkpoints}" "${m}" "${suffix}"
}
