#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../.." && pwd)

ENV_PREFIX=${ENV_PREFIX:-/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1}
SOURCE_ROOT=${SOURCE_ROOT:-/fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1}

exec "$ENV_PREFIX/bin/python" \
    "$repo_dir/scripts/slurm/setup/cu130/verify_env.py" \
    --source-root "$SOURCE_ROOT" "$@"
