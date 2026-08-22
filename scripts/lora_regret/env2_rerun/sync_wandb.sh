#!/usr/bin/env bash
# Sync only this rerun's offline W&B files from a host with egress.

set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
export WANDB_SYNC_ROOT="${WANDB_DIR}"
exec bash "${ORBIT_ICLR_ROOT}/scripts/lora_regret/sync_wandb.sh" "$@"
