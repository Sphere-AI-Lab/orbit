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

exec bash "$repo_dir/scripts/slurm/setup/cu130/install_env.sh" "$@"
