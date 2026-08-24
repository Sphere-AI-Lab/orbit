#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)

export PATH="$HOME/.local/bin:$PATH"

# uv needs a lock-capable cache filesystem, so the canonical installer defaults
# it to cluster home (NFS). Extraction there is slow enough to dominate a cold
# install: a clean-room run measured ~150 KB/s across uv's parallel unpack
# streams, against a ~19 GB cache. Condor gives each job a node-local scratch
# directory, which the canonical README names as the fastest cache location.
# materialize_env.py copies site-packages into the prefix at the end, so the
# finished environment does not depend on this cache or on the node.
export UV_CACHE_DIR=${UV_CACHE_DIR:-${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/orbit-uv-cache}
mkdir -p "$UV_CACHE_DIR"

exec bash "$repo_dir/scripts/slurm/setup/cu130/install_env.sh" "$@"
