#!/usr/bin/env bash
#
# Upload the campaign's offline wandb runs. Run from a node WITH egress -- the
# login node -- not from a compute node, which is the whole reason the runs are
# offline in the first place (e4_protocol.sh explains).
#
#   bash scripts/lora_regret/sync_wandb.sh          # sync once
#   WATCH=300 bash scripts/lora_regret/sync_wandb.sh  # re-sync every 5 minutes
#
# Safe to run repeatedly and while arms are still training: `wandb sync` marks a
# directory done in `wandb/.sync/`, so a second pass uploads only what is new.
# A run synced mid-flight shows up as finished and is refreshed by the next pass.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No virtualenv active. Run:" >&2
    echo "  source /fast/zqiu/orbit-iclr/orbit_env/bin/activate" >&2
    exit 2
fi

# `wandb sync` must not itself be offline, whatever the shell inherited.
unset WANDB_MODE

sync_once() {
    local n
    n=$(ls -d wandb/offline-run-* 2>/dev/null | wc -l)
    echo "=== $(date +%H:%M:%S): ${n} offline run dir(s) ==="
    wandb sync --sync-all --include-offline 2>&1 | grep -vE "^wandb: (Loading|Find logs)" || true
}

if [[ -n "${WATCH:-}" ]]; then
    while true; do
        sync_once
        sleep "${WATCH}"
    done
else
    sync_once
fi
