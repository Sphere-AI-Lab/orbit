#!/usr/bin/env bash
# W&B key loading. Sourced by scripts/lib/launcher.sh after common.sh.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

load_wandb_key() {
    set +x
    if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" && "${ENABLE_WANDB:-auto}" != "0" ]]; then
        export WANDB_API_KEY
        WANDB_API_KEY="$(<"${HOME}/.wandb_key")"
    fi
    if is_true "${ORBIT_LAUNCHER_XTRACE:-0}"; then
        set -x
    fi
}

configure_wandb_args() {
    if is_false "${ENABLE_WANDB:-auto}"; then
        WANDB_ARGS=()
        return
    fi
    if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" && "${WANDB_MODE:-}" != "disabled" ]]; then
        WANDB_ARGS=()
    fi
}
