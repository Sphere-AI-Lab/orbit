#!/usr/bin/env bash
# Runtime environment for Orbit on the H200 Slurm cluster (conda-prefix envs
# built by install_env.sh). Source AFTER activating the env -- the conda
# activate hook must run first so the z3 library path is already exported:
#
#   source /data/shared/conda/miniconda3/etc/profile.d/conda.sh
#   conda activate <env-prefix>
#   source scripts/slurm/setup/cu130/slurm_h200_runtime.sh
#
# Launchers should additionally prepend the Orbit checkout to PYTHONPATH
# (Ray workers import orbit by path) and raise the memlock limit for RDMA:
#   ulimit -Sl "$(ulimit -Hl)" 2>/dev/null || true
#
# Every pin here was validated by the 2026-08-24 OFT smoke (job 1728) and
# 40-rollout runs on H200 nodes. No CUDA toolkit is needed on the node: the
# env is prebuilt-wheel only, and the flashinfer-jit-cache wheel covers the
# kernels these models touch.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "slurm_h200_runtime.sh: activate the orbit conda env first" >&2
    return 2
fi
PY_SITE=$("${CONDA_PREFIX}/bin/python" -c "import site; print(site.getsitepackages()[0])")

# cuDNN: compute nodes ship a system cuDNN 9.14 in /usr/lib. TransformerEngine
# can resolve its cuDNN core from there while torch loads the env's 9.22 --
# two cores in one process, and the first fused-attention call dies with
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. Pin resolution to the env copy and
# give TE's dlopen the unversioned names (the wheel ships only .so.9 files).
export CUDNN_PATH="${PY_SITE}/nvidia/cudnn"
export CUDNN_HOME="${CUDNN_PATH}"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -f "${CUDNN_PATH}/lib/libcudnn.so.9" ]]; then
    ln -sfn libcudnn.so.9 "${CUDNN_PATH}/lib/libcudnn.so"
fi
for _cudnn_lib in "${CUDNN_PATH}"/lib/libcudnn_*.so.9; do
    [[ -e "${_cudnn_lib}" ]] && ln -sfn "$(basename "${_cudnn_lib}")" "${_cudnn_lib%.so.9}.so"
done
unset _cudnn_lib

# The env carries both cudart generations (nvidia-cutlass-dsl installs its
# cu12 libs unconditionally); pin the cuDNN frontend to the cu13 runtime.
export CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13

# Node-local caches: $HOME is NFS -- Triton's cache dies with ESTALE mid-run,
# and FlashInfer's JIT locking misbehaves on shared filesystems.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}-triton-${SLURM_JOB_ID:-local}}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer-${USER}-${SLURM_JOB_ID:-local}}"
mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

# Colocate PEFT/OFT sync: this cluster's SGLang scheduler children cannot
# rebuild trainer-side CUDA IPC handles (deterministic cudaIpcOpenMemHandle
# failure -- see backends/ipc.py's docstring). Route shaped adapter payloads
# over CPU. Without this the first adapter sync kills the scheduler.
export ORBIT_PEFT_ADAPTER_TRANSPORT="${ORBIT_PEFT_ADAPTER_TRANSPORT:-cpu_gather}"

export HF_HOME="${HF_HOME:-/data/shared/hf_cache}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
