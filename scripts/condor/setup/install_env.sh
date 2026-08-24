#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)

export PATH="$HOME/.local/bin:$PATH"

# Share one warm uv cache with the Miles-IMP environment on this cluster. Both
# profiles pin from the same radixark/miles Dockerfile and install identical
# artifacts (verified bit-for-bit; see the README), so a shared cache dedupes
# the expensive downloads instead of fetching each stack separately. uv's own
# default location is used because Miles already populates it; the canonical
# installer would otherwise pick a profile-private path under home.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}

# deep_gemm resolves a CUDA home when it is imported, trying $CUDA_HOME, then
# $CUDA_PATH, then nvcc on PATH, then /usr/local/cuda, and asserting if none
# exist. The execution nodes have none of them — the toolkit is module-style —
# so `import deep_gemm` fails with a bare AssertionError and verification ends
# at 38/39 (job 17477511). This profile compiles nothing, so the toolkit is
# needed only for that runtime lookup; it is the same path the Miles-IMP
# condor wrapper uses. Export it when running the environment too, not just
# when installing it.
export CUDA_HOME=${CUDA_HOME:-/is/software/nvidia/cuda-13.2}

exec bash "$repo_dir/scripts/slurm/setup/cu130/install_env.sh" "$@"
