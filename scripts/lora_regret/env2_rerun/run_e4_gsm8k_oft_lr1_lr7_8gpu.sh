#!/usr/bin/env bash
# Run all seven GSM8K OFT columns sequentially on one allocated 8-GPU node.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for column in 1 2 3 4 5 6 7; do
    bash "${HERE}/run_e4_gsm8k_oft_lr${column}_8gpu.sh" "$@"
done
