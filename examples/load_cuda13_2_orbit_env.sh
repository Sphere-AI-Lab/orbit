#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

ORBIT_VENV=${ORBIT_VENV:-${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace-cu13/.venv}
if [[ ! -x "${ORBIT_VENV}/bin/python" ]]; then
    echo "Missing Python environment: ${ORBIT_VENV}" >&2
    exit 2
fi

if command -v module >/dev/null 2>&1; then
    module load cuda/13.2
fi

export ORBIT_LOAD_CUDA_MODULES=0
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
if [[ -z "${CUDA_HOME:-}" || ! -d "${CUDA_HOME}" ]]; then
    echo "Missing CUDA 13.2 root. Set CUDA_HOME or ORBIT_CUDA_HOME_DEFAULT." >&2
    exit 2
fi

filter_conflicting_cuda_bins() {
    local entry
    local filtered=()

    IFS=':' read -r -a entries <<< "${PATH:-}"
    for entry in "${entries[@]}"; do
        [[ -z "${entry}" ]] && continue
        case "${entry}" in
            */cuda-*/bin | */cuda/bin)
                [[ "${entry}" == "${CUDA_HOME}/bin" ]] || continue
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
            *nccl*/lib | *cudnn*/lib)
                continue
                ;;
        esac
        filtered+=("${entry}")
    done

    local IFS=:
    export LD_LIBRARY_PATH="${filtered[*]}"
}

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

export ORBIT_VENV

PYTHON_SITE_VERSION="$("${ORBIT_VENV}/bin/python" - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
CUDNN_COMPAT_VERSION="$("${ORBIT_VENV}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    parts = version("nvidia-cudnn-cu13").split("+", 1)[0].split(".")
except PackageNotFoundError:
    parts = []
print(".".join(parts[:3]) if len(parts) >= 3 else "")
PY
)"
CUDNN_COMPAT_MINOR="${CUDNN_COMPAT_VERSION%.*}"
if [[ "${CUDNN_COMPAT_MINOR}" == "${CUDNN_COMPAT_VERSION}" ]]; then
    CUDNN_COMPAT_MINOR=""
fi

export CUDNN_PATH="${ORBIT_VENV}/lib/${PYTHON_SITE_VERSION}/site-packages/nvidia/cudnn"
if [[ -d "${CUDNN_PATH}/lib" ]]; then
    export CUDNN_HOME="${CUDNN_PATH}"
    export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"

    # nvidia-cudnn wheels ship versioned .so.9 files only. TransformerEngine's
    # cuDNN fused-attention path may dlopen unversioned engine libraries or
    # version-specific names; without local symlinks the loader can fall through
    # to an inherited system cuDNN path. Do not add libcudnn.so.9.* symlinks: the
    # Python cudnn package scans libcudnn.so.x and expects exactly one match.
    if [[ -f "${CUDNN_PATH}/lib/libcudnn.so.9" ]]; then
        ln -sfn libcudnn.so.9 "${CUDNN_PATH}/lib/libcudnn.so"
    fi
    for cudnn_lib in "${CUDNN_PATH}"/lib/libcudnn_*.so.9; do
        [[ -e "${cudnn_lib}" ]] || continue
        cudnn_base="${cudnn_lib%.so.9}"
        cudnn_target="$(basename "${cudnn_lib}")"
        ln -sfn "${cudnn_target}" "${cudnn_base}.so"
        if [[ -n "${CUDNN_COMPAT_MINOR}" ]]; then
            ln -sfn "${cudnn_target}" "${cudnn_base}.so.${CUDNN_COMPAT_MINOR}"
        fi
        if [[ -n "${CUDNN_COMPAT_VERSION}" ]]; then
            ln -sfn "${cudnn_target}" "${cudnn_base}.so.${CUDNN_COMPAT_VERSION}"
        fi
    done
fi

_SP_DIR="${ORBIT_VENV}/lib/${PYTHON_SITE_VERSION}/site-packages"
_REAL_LIBCUDART="$(ls "${_SP_DIR}/nvidia"/*/lib/libcudart.so.* 2>/dev/null | grep -v stub | head -1)"
if [[ -n "${_REAL_LIBCUDART}" && -e "${_SP_DIR}/tilelang/__init__.py" ]]; then
    _TL_STUB="$(dirname "$(readlink -f "${_SP_DIR}/tilelang/__init__.py")")/lib/libcudart_stub.so"
    [[ -e "${_TL_STUB}" || -L "${_TL_STUB}" ]] && ln -sfn "${_REAL_LIBCUDART}" "${_TL_STUB}"
fi
unset _SP_DIR _REAL_LIBCUDART _TL_STUB

# nvidia-modelopt (imported by megatron.bridge) dlopens libz3.so.4.15 by soname.
# The z3-solver wheel ships it under site-packages/z3/lib but does not put it on the
# loader path, so megatron.bridge import fails without this.
Z3_LIB="${ORBIT_VENV}/lib/${PYTHON_SITE_VERSION}/site-packages/z3/lib"
if [[ -d "${Z3_LIB}" ]]; then
    export LD_LIBRARY_PATH="${Z3_LIB}:${LD_LIBRARY_PATH:-}"
fi
unset Z3_LIB

export PYTHON_BIN="${PYTHON_BIN:-${ORBIT_VENV}/bin/python}"
# Keep FlashInfer JIT outputs out of shared caches. The home cache can contain
# stale kernels from other CUDA runtimes, and Lustre does not support the locks FlashInfer needs
# when all TP ranks JIT the same op at once.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer-${USER:-example-user}/orbit-workspace-cu13-cuda132}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-${CUDA_HOME}/bin/nvcc}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

unset CUDNN_COMPAT_MINOR CUDNN_COMPAT_VERSION PYTHON_SITE_VERSION
