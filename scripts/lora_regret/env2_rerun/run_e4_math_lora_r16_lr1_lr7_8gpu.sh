#!/usr/bin/env bash
# Run all seven MATH LoRA rank-16 columns sequentially on one allocated 8-GPU node.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for column in 1 2 3 4 5 6 7; do
    bash "${HERE}/run_lora_column.sh" math 16 "${column}" "$@"
done
