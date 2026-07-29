#!/usr/bin/env bash
# Build + runtime environment for the orbit CUDA-13.2 / torch-2.11 stack.
# These are the bits that cannot live in pyproject.toml: site CUDA paths,
# per-package build toggles, and runtime loader paths.
#
#   source env.sh && uv sync                       # build the whole env in one shot
#   source env.sh                                  # provides CUDA_HOME + the z3 runtime path
#   source examples/load_cuda13_2_orbit_env.sh     # ...then this for cudnn/flashinfer runtime setup
#                                                  #    before running an examples/ launcher
#
# Everything below is auto-detected with sensible defaults. Override any of these
# by exporting them BEFORE sourcing (e.g. `CUDA_HOME=/opt/cuda-13.2 source env.sh`):
#
#   CUDA_HOME                 CUDA 13.2 toolkit root. Auto-detected via env-modules,
#                             then `nvcc` on PATH, then common locations.
#   ORBIT_VENV                Target venv. Default: $UV_PROJECT_ENVIRONMENT or ./.venv
#   ORBIT_CUDA_MODULE         env-module to load for CUDA (default "cuda/13.2"; set
#                             empty to skip). ORBIT_NCCL_MODULE likewise (default "nccl").
#   ORBIT_PYTHON_VERSION      Python X.Y for the site-packages path before the venv
#                             exists (default "3.12", matching requires-python).
#   TORCH_CUDA_ARCH_LIST      GPU arch(s). Default "9.0 10.0" = H100 + B200 fat binary.
#                             NVTE_CUDA_ARCHS / FLASH_ATTN_CUDA_ARCHS / CMAKE_CUDA_ARCHITECTURES
#                             track it for the builds that ignore it.
#   MAX_JOBS                  ninja jobs for setup.py-style builds (default 32).
#   CMAKE_BUILD_PARALLEL_LEVEL ninja jobs for cmake builds (sgl-kernel); RAM-bound since
#                             cutlass files use 10-30 GB each. Auto: ~RAM/40 GB, capped by
#                             cores; tune the per-job budget via ORBIT_SGL_KERNEL_JOB_GB.
#   UV_CACHE_DIR              MUST be flock-capable AND persistent. Default
#                             $HOME/.cache/uv_cu13_orbit. Never /tmp — see below.
set -a

# --- CUDA toolkit: try env-modules (if present), then resolve CUDA_HOME ---
if command -v module >/dev/null 2>&1; then
    _cuda_mod="${ORBIT_CUDA_MODULE-cuda/13.2}"
    _nccl_mod="${ORBIT_NCCL_MODULE-nccl}"
    [ -n "${_cuda_mod}" ] && module load "${_cuda_mod}" >/dev/null 2>&1 || true
    [ -n "${_nccl_mod}" ] && module load "${_nccl_mod}" >/dev/null 2>&1 || true
fi
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd -P)"
    else
        for _c in /usr/local/cuda-13.2 /usr/local/cuda /opt/cuda-13.2 /opt/cuda; do
            [ -d "${_c}" ] && { CUDA_HOME="${_c}"; break; }
        done
    fi
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -d "${CUDA_HOME:-/nonexistent}" ]; then
    echo "env.sh: WARNING — CUDA 13.2 toolkit not found. Set CUDA_HOME explicitly." >&2
fi
CUDA_ROOT="${CUDA_HOME}"
PATH="${CUDA_HOME}/bin:${PATH}"
LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# --- build parallelism / OOM + file-descriptor safety ---
# uv builds source dists concurrently (default = nCPU). With 9 CUDA packages each
# running ninja -j$MAX_JOBS that explodes into hundreds of nvcc procs -> EMFILE
# ("Too many open files"). Serialize package builds; keep parallelism within each.
UV_CONCURRENT_BUILDS=1
MAX_JOBS="${MAX_JOBS:-32}"
# sgl-kernel builds via cmake->ninja, which IGNORES MAX_JOBS and defaults to nproc.
# Its cutlass template files use 10-30 GB EACH per nvcc/cicc, so nproc-wide -> OOM.
# Scale jobs to RAM (the binding constraint): ~40 GB/job budget, capped by cores.
if [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then
    _ram_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
    _jobs=$(( ${_ram_gb:-64} / ${ORBIT_SGL_KERNEL_JOB_GB:-40} ))
    _ncpu="$(nproc 2>/dev/null || echo 8)"
    [ "${_jobs}" -gt "${_ncpu}" ] && _jobs="${_ncpu}"
    [ "${_jobs}" -lt 2 ] && _jobs=2
    CMAKE_BUILD_PARALLEL_LEVEL="${_jobs}"
fi
NVCC_THREADS="${NVCC_THREADS:-2}"
NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-2}"
NVTE_FRAMEWORK=pytorch

# --- GPU arch: build fat binaries for BOTH H100 (sm_90) and B200 (sm_100) ---
# Do NOT auto-detect from nvidia-smi: that silently pins the env to whichever node
# happened to run the build, and the kernels then fail to load everywhere else.
# Override with a single arch only if you knowingly want a smaller/faster build.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0}"
# Package-specific arch lists — these builds do NOT read TORCH_CUDA_ARCH_LIST.
NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90;100}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90;100}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90a;100a}"

# --- force source builds instead of the wrong-ABI auto-downloaded prebuilts ---
FLASH_ATTENTION_FORCE_BUILD=TRUE
MAMBA_FORCE_BUILD=TRUE
CAUSAL_CONV1D_FORCE_BUILD=TRUE

# --- uv cache: must be flock-capable AND persistent ---
# Lustre (/lustre/fast) returns ENOSYS on flock, so it cannot hold the cache.
# /tmp can, but it is node-local and cleared on exit — and under uv's default
# symlink install mode that silently guts the venv (every package becomes a
# dangling link, importing as an empty namespace package). Cluster-home is NFS:
# flock-capable, persistent, and shared across nodes. Keep the default.
UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv_cu13_orbit}"
mkdir -p "${UV_CACHE_DIR}"

# --- venv + site-packages (derive python version from the venv once it exists) ---
# $VIRTUAL_ENV comes before the ./.venv fallback: the documented runtime flow is
# `source <venv>/bin/activate && source env.sh`, and without this the already-active
# venv is ignored in favour of a ./.venv that may not exist — SITE_PACKAGES then
# points at nothing and deep_ep dies with "No libnccl.so found in .../.venv/...".
ORBIT_VENV="${ORBIT_VENV:-${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-$(pwd)/.venv}}}"
if [ -x "${ORBIT_VENV}/bin/python" ]; then
    SITE_PACKAGES="$("${ORBIT_VENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
else
    SITE_PACKAGES="${ORBIT_VENV}/lib/python${ORBIT_PYTHON_VERSION:-3.12}/site-packages"
fi

# --- build needs NVSHMEM/NCCL/CUDNN from the venv (deterministic paths) ---
# DeepEP -> NVSHMEM + NCCL 2.30 ; TransformerEngine -> cudnn 9.22
NVSHMEM_ROOT="${SITE_PACKAGES}/nvidia/nvshmem"
NCCL_ROOT="${SITE_PACKAGES}/nvidia/nccl"
CUDNN_PATH="${SITE_PACKAGES}/nvidia/cudnn"
CUDNN_HOME="${CUDNN_PATH}"
EP_NVSHMEM_ROOT_DIR="${NVSHMEM_ROOT}"
EP_NCCL_ROOT_DIR="${NCCL_ROOT}"
# sgl-kernel cmake build needs Torch's cmake config (else "kineto not found").
CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch/share/cmake${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
# sgl-kernel FetchContent dep dlpack uses cmake_minimum_required<3.5 -> allow on CMake 4.x.
CMAKE_POLICY_VERSION_MINIMUM=3.5
CPATH="${CUDNN_PATH}/include:${NCCL_ROOT}/include:${NVSHMEM_ROOT}/include:${CUDA_ROOT}/targets/x86_64-linux/include/cccl:${CUDA_ROOT}/include${CPATH:+:$CPATH}"
LIBRARY_PATH="${SITE_PACKAGES}/torch/lib:${CUDNN_PATH}/lib:${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
LD_LIBRARY_PATH="${SITE_PACKAGES}/torch/lib:${CUDNN_PATH}/lib:${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib:${LD_LIBRARY_PATH}"

# --- runtime: nvidia-modelopt (via megatron.bridge) dlopens libz3.so.4.15 by soname ---
LD_LIBRARY_PATH="${SITE_PACKAGES}/z3/lib:${LD_LIBRARY_PATH}"

unset _cuda_mod _nccl_mod _c _cc _ram_gb _jobs _ncpu 2>/dev/null || true
set +a
