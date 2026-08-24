#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)

export PATH="$HOME/.local/bin:$PATH"

exec bash "$repo_dir/scripts/slurm/setup/cu130/install_env.sh" "$@"
