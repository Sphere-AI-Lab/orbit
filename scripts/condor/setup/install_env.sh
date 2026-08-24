#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)

export PATH="$HOME/.local/bin:$PATH"

# UV_CACHE_DIR is deliberately left to the canonical installer, which defaults
# it to cluster home so the cache persists and is reused. Only a cold cache is
# worth relocating; see "uv cache placement" in the README.

exec bash "$repo_dir/scripts/slurm/setup/cu130/install_env.sh" "$@"
