#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PINS_FILE="${SCRIPT_DIR}/pins.env"
ORBIT_ROOT="${DEFAULT_ORBIT_ROOT}"
WORKSPACE=""
ENV_PREFIX=""
SOURCE_ROOT=""
CONDA_EXE="${CONDA_EXE:-conda}"
UV_EXE="${UV_EXE:-}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
TOOL_PYTHON="${TOOL_PYTHON:-python3}"
JOBS="${MAX_JOBS:-32}"
DRY_RUN=0
PREFLIGHT_ONLY=0
LOCK_DIR=""

usage() {
    cat <<'EOF'
Install the native Orbit CUDA 12.8 environment for NVIDIA H200 GPUs.

Usage:
  install_env.sh [options]

Options:
  --env-prefix PATH   Conda prefix (default: <workspace>/envs/orbit_cu128)
  --source-root PATH  Dedicated pinned checkouts (default: <workspace>/sources/cu128)
  --workspace PATH    miles-orbit workspace containing the main Orbit checkout
  --orbit-root PATH   Orbit checkout to install editable
  --pins PATH         Generated pins.env file
  --conda-exe PATH    Conda executable
  --uv-exe PATH       Optional pinned uv executable; otherwise bootstrap it
  --tool-python PATH  Python 3.11+ interpreter used to validate generated pins
  --jobs N            Parallel CUDA build jobs (default: 32)
  --preflight-only    Validate Slurm, H200, CUDA, pins, and tools; install nothing
  --dry-run           Print the complete plan without scheduler or hardware checks
  -h, --help          Show this help

Real installation must run inside a Slurm allocation with an H200 and CUDA 12.8.
Existing valid prefixes are resumed; unknown directories are never deleted.
EOF
}

die() {
    printf 'install_env.sh: %s\n' "$*" >&2
    exit 2
}

stage() {
    printf '\n== [%02d/12] %s ==\n' "$1" "$2"
}

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_command "$@"
    if (( ! DRY_RUN )); then
        "$@"
    fi
}

pin_any() {
    local name value
    for name in "$@"; do
        value="${!name:-}"
        if [[ -n "${value}" ]]; then
            printf '%s\n' "${value}"
            return 0
        fi
    done
    return 1
}

require_file() {
    [[ -f "$1" ]] || die "required file is missing: $1"
}

resolve_command() {
    local requested="$1" resolved
    if [[ "${requested}" == */* ]]; then
        [[ -x "${requested}" ]] || die "command is not executable: ${requested}"
        printf '%s\n' "${requested}"
        return
    fi
    resolved="$(command -v "${requested}" || true)"
    [[ -n "${resolved}" ]] || die "required command not found: ${requested}"
    printf '%s\n' "${resolved}"
}

parse_args() {
    while (( $# )); do
        case "$1" in
            --env-prefix) [[ $# -ge 2 ]] || die "$1 requires a path"; ENV_PREFIX="$2"; shift 2 ;;
            --source-root) [[ $# -ge 2 ]] || die "$1 requires a path"; SOURCE_ROOT="$2"; shift 2 ;;
            --workspace) [[ $# -ge 2 ]] || die "$1 requires a path"; WORKSPACE="$2"; shift 2 ;;
            --orbit-root) [[ $# -ge 2 ]] || die "$1 requires a path"; ORBIT_ROOT="$2"; shift 2 ;;
            --pins) [[ $# -ge 2 ]] || die "$1 requires a path"; PINS_FILE="$2"; shift 2 ;;
            --conda-exe) [[ $# -ge 2 ]] || die "$1 requires a path"; CONDA_EXE="$2"; shift 2 ;;
            --uv-exe) [[ $# -ge 2 ]] || die "$1 requires a path"; UV_EXE="$2"; shift 2 ;;
            --tool-python) [[ $# -ge 2 ]] || die "$1 requires a path"; TOOL_PYTHON="$2"; shift 2 ;;
            --jobs) [[ $# -ge 2 ]] || die "$1 requires a number"; JOBS="$2"; shift 2 ;;
            --preflight-only) PREFLIGHT_ONLY=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
}

resolve_defaults() {
    local common_dir
    if [[ -z "${WORKSPACE}" ]]; then
        common_dir="$(git -C "${ORBIT_ROOT}" rev-parse --git-common-dir 2>/dev/null)" \
            || die "cannot locate the Orbit Git common directory"
        if [[ "${common_dir}" != /* ]]; then
            common_dir="${ORBIT_ROOT}/${common_dir}"
        fi
        WORKSPACE="$(cd -- "$(dirname -- "${common_dir}")/.." && pwd)"
    fi
    ENV_PREFIX="${ENV_PREFIX:-${WORKSPACE}/envs/orbit_cu128}"
    SOURCE_ROOT="${SOURCE_ROOT:-${WORKSPACE}/sources/cu128}"
    [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
}

validate_paths() {
    [[ "${ORBIT_ROOT}" == /* ]] || die "--orbit-root must be absolute"
    [[ "${WORKSPACE}" == /* ]] || die "--workspace must be absolute"
    [[ "${ENV_PREFIX}" == /* ]] || die "--env-prefix must be absolute"
    [[ "${SOURCE_ROOT}" == /* ]] || die "--source-root must be absolute"
    case "${ENV_PREFIX}" in
        /|"${HOME}"|"${WORKSPACE}"|"${ORBIT_ROOT}")
            die "unsafe environment prefix: ${ENV_PREFIX}"
            ;;
    esac
    if [[ "${ENV_PREFIX}" == "${ORBIT_ROOT}/"* ]]; then
        die "environment prefix must not be inside the Orbit checkout"
    fi
    [[ "${SOURCE_ROOT}" != "${ENV_PREFIX}" ]] || die "source root and environment prefix must differ"
    require_file "${PINS_FILE}"
    require_file "${ORBIT_ROOT}/pyproject.toml"
}

load_profile() {
    # pins.env is generated by extract_pins.py and contains shell-quoted assignments only.
    # shellcheck disable=SC1090
    source "${PINS_FILE}"
    local required
    for required in \
        CUDA_PROFILE CUDA_TOOLKIT_VERSION PYTHON_VERSION UV_VERSION \
        TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION \
        TORCH_INDEX_URL FLASHINFER_INDEX_URL FLASHINFER_VERSION \
        CUDA_PYTHON_VERSION TRANSFORMERS_VERSION \
        NUMPY_VERSION NINJA_VERSION PYBIND11_VERSION CMAKE_VERSION \
        SCIKIT_BUILD_CORE_VERSION SETUPTOOLS_VERSION WHEEL_VERSION \
        PACKAGING_VERSION PSUTIL_VERSION FLASH_ATTN_VERSION \
        CAUSAL_CONV1D_VERSION MAMBA_SSM_VERSION FLASH_LINEAR_ATTENTION_VERSION \
        FAST_HADAMARD_VERSION FAST_HADAMARD_SOURCE_URL FAST_HADAMARD_COMMIT \
        HUMMING_KERNELS_VERSION NVIDIA_CUTLASS_DSL_VERSION TIMM_VERSION \
        SGLANG_ROUTER_VERSION SGLANG_ROUTER_WHEEL_URL \
        SGLANG_SOURCE_URL SGLANG_COMMIT MEGATRON_SOURCE_URL MEGATRON_COMMIT \
        MEGATRON_BRIDGE_SOURCE_URL MEGATRON_BRIDGE_COMMIT \
        TRANSFORMER_ENGINE_SOURCE_URL TRANSFORMER_ENGINE_COMMIT \
        APEX_SOURCE_URL APEX_COMMIT; do
        [[ -n "${!required:-}" ]] || die "required pin is missing: ${required}"
    done
    [[ "${CUDA_PROFILE}" == "cu128" ]] || die "pins select ${CUDA_PROFILE}, expected cu128"
    [[ "${CUDA_TOOLKIT_VERSION}" == "12.8" ]] \
        || die "pins select CUDA ${CUDA_TOOLKIT_VERSION}, expected 12.8"
}

preflight() {
    stage 1 "preflight: scheduler, H200, CUDA 12.8, pins, and tools"
    if (( DRY_RUN )); then
        printf 'dry-run: scheduler and hardware probes skipped\n'
        CONDA_EXE="${CONDA_EXE}"
        CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
        export CUDA_HOME
        return
    fi

    [[ -n "${SLURM_JOB_ID:-}" ]] || die "real installation must run inside a Slurm allocation"
    CONDA_EXE="$(resolve_command "${CONDA_EXE}")"
    if [[ -n "${UV_EXE}" ]]; then
        UV_EXE="$(resolve_command "${UV_EXE}")"
    fi
    NVIDIA_SMI="$(resolve_command "${NVIDIA_SMI}")"
    TOOL_PYTHON="$(resolve_command "${TOOL_PYTHON}")"
    resolve_command git >/dev/null
    "${TOOL_PYTHON}" -c 'import tomllib' >/dev/null 2>&1 \
        || die "--tool-python must provide Python 3.11+ with tomllib"

    local gpu_names nvcc_exe nvcc_output
    gpu_names="$("${NVIDIA_SMI}" --query-gpu=name --format=csv,noheader)" \
        || die "nvidia-smi GPU query failed"
    grep -qi 'H200' <<<"${gpu_names}" || die "allocated GPU is not an H200: ${gpu_names}"

    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        nvcc_exe="${CUDA_HOME}/bin/nvcc"
    else
        nvcc_exe="$(resolve_command nvcc)"
        CUDA_HOME="$(cd -- "$(dirname -- "${nvcc_exe}")/.." && pwd)"
        export CUDA_HOME
    fi
    nvcc_output="$("${nvcc_exe}" --version)" || die "nvcc version query failed"
    grep -q "release ${CUDA_TOOLKIT_VERSION}" <<<"${nvcc_output}" \
        || die "nvcc does not report CUDA ${CUDA_TOOLKIT_VERSION}"

    if [[ "${PINS_FILE}" == "${SCRIPT_DIR}/pins.env" ]]; then
        python3 "${SCRIPT_DIR}/extract_pins.py" --check \
            || die "pins.env is stale; run extract_pins.py --write"
    fi
    printf 'preflight passed for Slurm job %s on %s\n' "${SLURM_JOB_ID}" "${gpu_names//$'\n'/, }"
}

acquire_lock() {
    LOCK_DIR="${ENV_PREFIX}.install.lock"
    if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
        die "another installer may own ${LOCK_DIR}"
    fi
    trap release_lock EXIT
}

release_lock() {
    if [[ -n "${LOCK_DIR}" && -d "${LOCK_DIR}" ]]; then
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
}

ensure_environment() {
    stage 2 "create or resume Conda Python ${PYTHON_VERSION} and pinned uv"
    if [[ -e "${ENV_PREFIX}" && ! -x "${ENV_PREFIX}/bin/python" ]]; then
        die "existing prefix is not a recognizable environment: ${ENV_PREFIX}"
    fi
    if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
        run "${CONDA_EXE}" create --yes --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip
    else
        printf 'resume: %s already contains Python\n' "${ENV_PREFIX}"
    fi
    if (( DRY_RUN )); then
        run "${ENV_PREFIX}/bin/python" -m pip install "uv==${UV_VERSION}"
        UV_EXE="${ENV_PREFIX}/bin/uv"
    elif [[ -n "${UV_EXE}" ]]; then
        local uv_output
        uv_output="$("${UV_EXE}" --version)"
        [[ "${uv_output}" == "uv ${UV_VERSION} "* ]] \
            || die "uv must be ${UV_VERSION}, got: ${uv_output}"
    else
        "${ENV_PREFIX}/bin/python" -m pip install "uv==${UV_VERSION}"
        UV_EXE="${ENV_PREFIX}/bin/uv"
    fi
}

ensure_checkout() {
    local name="$1" url="$2" commit="$3" destination="$4" current
    if (( DRY_RUN )); then
        run git clone --filter=blob:none "${url}" "${destination}"
        run git -C "${destination}" fetch --depth=1 origin "${commit}"
        run git -C "${destination}" checkout --detach "${commit}"
        return
    fi
    if [[ -d "${destination}/.git" ]]; then
        [[ -z "$(git -C "${destination}" status --porcelain)" ]] \
            || die "${name} checkout has local changes: ${destination}"
    elif [[ -e "${destination}" ]]; then
        die "${name} source path exists but is not a Git checkout: ${destination}"
    else
        mkdir -p "$(dirname -- "${destination}")"
        run git clone --filter=blob:none "${url}" "${destination}"
    fi
    current="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${current}" != "${commit}" ]]; then
        run git -C "${destination}" fetch --depth=1 origin "${commit}"
        run git -C "${destination}" checkout --detach "${commit}"
    fi
}

generate_runtime_requirements() {
    local output="${ENV_PREFIX}/.orbit-cu128-requirements.txt"
    if (( DRY_RUN )); then
        printf '+ generate filtered Orbit runtime requirements at %q\n' "${output}"
        return
    fi
    "${ENV_PREFIX}/bin/python" - "${ORBIT_ROOT}/pyproject.toml" "${output}" <<'PY'
import re
import sys
import tomllib
from pathlib import Path

pyproject = Path(sys.argv[1])
output = Path(sys.argv[2])
controlled = {
    "apex",
    "cuda-python",
    "flashinfer-cubin",
    "flashinfer-python",
    "megatron-bridge",
    "megatron-core",
    "sglang",
    "torch",
    "torchaudio",
    "torchvision",
    "transformer-engine",
}
data = tomllib.loads(pyproject.read_text())
requirements = []
for requirement in data["project"].get("dependencies", []):
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match and match.group(1).lower().replace("_", "-") not in controlled:
        requirements.append(requirement)
output.write_text("\n".join(requirements) + "\n")
PY
    printf '+ generated %s\n' "${output}"
}

write_uv_overrides() {
    OVERRIDE_FILE="${ENV_PREFIX}/.orbit-cu128-overrides.txt"
    export OVERRIDE_FILE
    if (( DRY_RUN )); then
        printf '+ generate pinned uv override file at %q\n' "${OVERRIDE_FILE}"
        return
    fi
    cat > "${OVERRIDE_FILE}" <<EOF
numpy==${NUMPY_VERSION}
torch==${TORCH_VERSION}
cuda-python==${CUDA_PYTHON_VERSION}
flashinfer-python==${FLASHINFER_VERSION}
humming-kernels==${HUMMING_KERNELS_VERSION}
nvidia-cutlass-dsl==${NVIDIA_CUTLASS_DSL_VERSION}
transformers==${TRANSFORMERS_VERSION}
timm==${TIMM_VERSION}
EOF
    printf '+ generated %s\n' "${OVERRIDE_FILE}"
}

setup_cuda_build_environment() {
    local python="$1" site_packages cache_slug
    if (( DRY_RUN )); then
        site_packages="${ENV_PREFIX}/lib/python${PYTHON_VERSION}/site-packages"
    else
        site_packages="$("${python}" -c 'import site; print(site.getsitepackages()[0])')"
    fi
    cache_slug="$(basename -- "${ENV_PREFIX}")"
    export CUDA_ROOT="${CUDA_HOME}"
    export NCCL_ROOT="${site_packages}/nvidia/nccl"
    export CUDNN_PATH="${site_packages}/nvidia/cudnn"
    export CUDNN_HOME="${CUDNN_PATH}"
    export TORCH_CUDA_ARCH_LIST="9.0"
    export NVTE_CUDA_ARCHS="90"
    export FLASH_ATTN_CUDA_ARCHS="90"
    export CMAKE_CUDA_ARCHITECTURES="90a"
    export NVTE_FRAMEWORK="pytorch"
    export MAX_JOBS="${JOBS}"
    export NVCC_THREADS=2
    export NVTE_BUILD_THREADS_PER_JOB=2
    export CMAKE_BUILD_PARALLEL_LEVEL=10
    export CMAKE_PREFIX_PATH="${site_packages}/torch/share/cmake${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export MAMBA_FORCE_BUILD=TRUE
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE
    export CPATH="${CUDNN_PATH}/include:${NCCL_ROOT}/include:${CUDA_ROOT}/targets/x86_64-linux/include/cccl:${CUDA_ROOT}/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${site_packages}/torch/lib:${CUDNN_PATH}/lib:${NCCL_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${site_packages}/torch/lib:${CUDNN_PATH}/lib:${NCCL_ROOT}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer-${USER}/${cache_slug}"
    export FLASHINFER_NVCC="${CUDA_HOME}/bin/nvcc"
    export TRITON_CACHE_DIR="/tmp/triton-${USER}-${cache_slug}"
    if ! command -v cargo >/dev/null 2>&1; then
        export SGLANG_BUILD_RUST_EXTS=none
    fi
    run mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}"
}

install_environment() {
    local python="${ENV_PREFIX}/bin/python"
    local sglang_root="${SOURCE_ROOT}/sglang"
    local megatron_root="${SOURCE_ROOT}/Megatron-LM"
    local bridge_root="${SOURCE_ROOT}/Megatron-Bridge"
    local te_root="${SOURCE_ROOT}/TransformerEngine"
    local apex_root="${SOURCE_ROOT}/apex"
    local fast_hadamard_root="${SOURCE_ROOT}/fast-hadamard-transform"

    stage 3 "install pinned build tools and record tooling"
    run "${python}" -m pip install --quiet \
        "ninja==${NINJA_VERSION}" "pybind11==${PYBIND11_VERSION}" \
        "cmake==${CMAKE_VERSION}" "scikit-build-core==${SCIKIT_BUILD_CORE_VERSION}" \
        "setuptools==${SETUPTOOLS_VERSION}" "wheel==${WHEEL_VERSION}" \
        "packaging==${PACKAGING_VERSION}" "psutil==${PSUTIL_VERSION}" \
        "numpy==${NUMPY_VERSION}"
    run "${CONDA_EXE}" --version
    run "${UV_EXE}" --version
    run "${python}" --version
    write_uv_overrides

    stage 4 "install exact CUDA 12.8 PyTorch wheels"
    run "${UV_EXE}" pip install --python "${python}" --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}+${CUDA_PROFILE}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_PROFILE}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_PROFILE}"
    setup_cuda_build_environment "${python}"

    stage 5 "install pinned CUDA and inference wheels"
    run "${UV_EXE}" pip install --python "${python}" --override "${OVERRIDE_FILE}" \
        --extra-index-url "${FLASHINFER_INDEX_URL}" \
        "cuda-python==${CUDA_PYTHON_VERSION}" \
        "flashinfer-python==${FLASHINFER_VERSION}" \
        "humming-kernels==${HUMMING_KERNELS_VERSION}" \
        "nvidia-cutlass-dsl==${NVIDIA_CUTLASS_DSL_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" "timm==${TIMM_VERSION}"

    stage 6 "materialize immutable external source checkouts"
    ensure_checkout sglang "${SGLANG_SOURCE_URL}" "${SGLANG_COMMIT}" "${sglang_root}"
    ensure_checkout megatron-lm "${MEGATRON_SOURCE_URL}" "${MEGATRON_COMMIT}" "${megatron_root}"
    ensure_checkout megatron-bridge "${MEGATRON_BRIDGE_SOURCE_URL}" "${MEGATRON_BRIDGE_COMMIT}" "${bridge_root}"
    ensure_checkout transformer-engine "${TRANSFORMER_ENGINE_SOURCE_URL}" \
        "${TRANSFORMER_ENGINE_COMMIT}" "${te_root}"
    ensure_checkout apex "${APEX_SOURCE_URL}" "${APEX_COMMIT}" "${apex_root}"
    ensure_checkout fast-hadamard "${FAST_HADAMARD_SOURCE_URL}" \
        "${FAST_HADAMARD_COMMIT}" "${fast_hadamard_root}"

    stage 7 "install Orbit runtime dependencies without controlled backends"
    generate_runtime_requirements
    run "${UV_EXE}" pip install --python "${python}" --override "${OVERRIDE_FILE}" \
        --extra-index-url "${FLASHINFER_INDEX_URL}" \
        --extra-index-url "${SGLANG_WHEEL_INDEX_URL}" \
        --requirements "${ENV_PREFIX}/.orbit-cu128-requirements.txt"

    stage 8 "build pinned Hopper CUDA extension layer"
    run "${python}" -m pip install --no-build-isolation --no-deps --verbose \
        "git+${TRANSFORMER_ENGINE_SOURCE_URL}@${TRANSFORMER_ENGINE_COMMIT}"
    run "${python}" -m pip install --no-build-isolation --no-deps --verbose \
        "flash_attn==${FLASH_ATTN_VERSION}"
    run "${python}" -m pip install --no-build-isolation --no-deps \
        "causal-conv1d==${CAUSAL_CONV1D_VERSION}" \
        "mamba-ssm==${MAMBA_SSM_VERSION}" \
        "flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}"
    run "${python}" -m pip install --no-build-isolation --no-deps \
        "git+${FAST_HADAMARD_SOURCE_URL}@${FAST_HADAMARD_COMMIT}"

    stage 9 "build pinned Apex CUDA extensions"
    run env CUDA_HOME="${CUDA_HOME}" APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
        APEX_PARALLEL_BUILD="${JOBS}" "${python}" -m pip install --verbose \
        --no-build-isolation --no-deps --editable "${apex_root}"

    stage 10 "build SGLang kernel and install pinned model backends"
    run "${python}" -m pip install --no-build-isolation --no-deps --verbose \
        "git+${SGLANG_SOURCE_URL}@${SGLANG_COMMIT}#subdirectory=sgl-kernel"
    run "${UV_EXE}" pip install --python "${python}" --override "${OVERRIDE_FILE}" \
        --editable "${sglang_root}/python"
    run "${python}" -m pip install "${SGLANG_ROUTER_WHEEL_URL}"
    run "${UV_EXE}" pip install --python "${python}" --override "${OVERRIDE_FILE}" \
        --editable "${megatron_root}"
    run "${UV_EXE}" pip install --python "${python}" --override "${OVERRIDE_FILE}" \
        --editable "${bridge_root}"

    stage 11 "install Orbit editable and reassert controlled versions"
    run "${UV_EXE}" pip install --python "${python}" --no-deps --editable "${ORBIT_ROOT}"
    run "${UV_EXE}" pip install --python "${python}" --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}+${CUDA_PROFILE}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_PROFILE}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_PROFILE}"

    stage 12 "run metadata, imports, H200, BF16, cuDNN, and NCCL verification"
    run "${python}" "${SCRIPT_DIR}/verify_env.py" --pins "${PINS_FILE}" \
        --orbit-root "${ORBIT_ROOT}" --source-root "${SOURCE_ROOT}" --full-h200
}

main() {
    parse_args "$@"
    resolve_defaults
    validate_paths
    load_profile
    preflight
    if (( PREFLIGHT_ONLY )); then
        return
    fi
    if (( ! DRY_RUN )); then
        mkdir -p "$(dirname -- "${ENV_PREFIX}")"
        acquire_lock
    fi
    ensure_environment
    install_environment
}

main "$@"
