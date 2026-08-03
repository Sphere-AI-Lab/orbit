#!/usr/bin/env bash
#
# Upload the campaign's offline wandb runs. Run from a node WITH egress -- the
# login node -- not from a compute node, which is the whole reason the runs are
# offline in the first place (e4_protocol.sh explains).
#
#   bash scripts/lora_regret/sync_wandb.sh          # sync once
#   WATCH=300 bash scripts/lora_regret/sync_wandb.sh  # re-sync every 5 minutes
#
# Only STALE directories are uploaded: a directory whose run-*.wandb file is no
# newer than its .synced marker has nothing new to say, and is skipped.
#
# The marker needs help, and that is most of what this script is for. wandb
# writes `.synced` only when the run stream contains an EXIT record
# (sync.py: "Only mark synced if the run actually finished") -- so a run whose
# process was killed, which on a preempted cluster is a normal way for a run
# to end, is "unfinished" FOREVER. Measured on 2026-08-03: 49 of 63 offline
# directories came from killed or superseded invocations, none of them could
# ever be marked, and every `--sync-all` pass re-uploaded all 49 to refresh
# the two that were live. So after a successful upload, any directory whose
# .wandb file has been quiet for QUIESCE_MIN minutes -- its writer is gone,
# the file cannot grow again -- gets the marker written by us. A marked dir
# is also eligible for `wandb sync --clean`, which is correct: it is fully
# uploaded and final.
#
# Live runs stay unmarked on purpose, in BOTH mechanisms: wandb sees no exit
# record and we see a recent mtime. They re-sync on every pass, which is the
# near-live dashboard the protocol wants.
#
# A retried arm has SEVERAL offline directories sharing one run id (one per
# launcher invocation; the run id is derived from the arm name). Each replays
# into the same server run; directories sync in timestamp order, so where
# attempts overlap on a step the newest attempt's value lands last and wins.
# That is why a sync pass can legitimately print the same run id twice.
#
# Concurrent passes are excluded with a lock: two syncs replaying the same
# directory into the same run at once are convergent but racy and doubly slow.
# A second invocation (say, WATCH mode already running) exits 0 immediately.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No virtualenv active. Run:" >&2
    echo "  source /fast/zqiu/orbit-iclr/orbit_env/bin/activate" >&2
    exit 2
fi

# `wandb sync` must not itself be offline, whatever the shell inherited.
unset WANDB_MODE

# mkdir, NOT flock. `wandb/` is on Lustre, which is mounted without the flock
# option here -- `flock -n 9` fails with "Function not implemented", which is
# indistinguishable from "someone else holds it", so a flock-based guard makes
# this script refuse to sync anything, ever. mkdir is atomic on every POSIX
# filesystem including Lustre.
mkdir -p wandb
LOCK_DIR="wandb/.sync_wandb.lock.d"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    # A lock left behind by a killed pass would block every future sync, which
    # is the same failure in slower motion -- so a lock whose owner is gone is
    # taken over rather than respected. Same-host check: these all run on the
    # login node.
    owner=$(cat "${LOCK_DIR}/pid" 2>/dev/null || echo "")
    if [[ -n "${owner}" ]] && kill -0 "${owner}" 2>/dev/null; then
        echo "another sync_wandb.sh is running (pid ${owner}); nothing to do."
        exit 0
    fi
    echo "clearing a stale lock from pid ${owner:-unknown}" >&2
    rm -rf "${LOCK_DIR}"
    if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
        echo "could not take ${LOCK_DIR}; another pass just started. Nothing to do."
        exit 0
    fi
fi
echo $$ > "${LOCK_DIR}/pid"
trap 'rm -rf "${LOCK_DIR}"' EXIT

QUIESCE_MIN=${QUIESCE_MIN:-10}

sync_once() {
    local stale=() stale_files=() current=0 no_file=0
    for dir in wandb/offline-run-*; do
        [[ -d "${dir}" ]] || continue
        local wandb_file
        wandb_file=$(ls "${dir}"/run-*.wandb 2>/dev/null | head -1)
        if [[ -z "${wandb_file}" ]]; then
            no_file=$(( no_file + 1 ))   # crashed before writing anything
            continue
        fi
        # -nt is false for a missing marker too, so never-synced dirs are stale.
        if [[ -f "${wandb_file}.synced" && ! "${wandb_file}" -nt "${wandb_file}.synced" ]]; then
            current=$(( current + 1 ))
            continue
        fi
        stale+=("${dir}")
        stale_files+=("${wandb_file}")
    done

    echo "=== $(date +%H:%M:%S): ${#stale[@]} stale, ${current} already current, ${no_file} empty ==="
    (( ${#stale[@]} > 0 )) || return 0

    # No `|| true` here: it would run `true` on failure and overwrite PIPESTATUS
    # before the read below. Without `-e`, a failing pipeline doesn't exit the
    # script, and grep filtering every line (rc 1) must not read as a wandb
    # failure -- hence PIPESTATUS[0], not $?.
    wandb sync "${stale[@]}" 2>&1 | grep -vE "^wandb: (Loading|Find logs)"
    local sync_rc=${PIPESTATUS[0]}
    if (( sync_rc != 0 )); then
        echo "wandb sync exited ${sync_rc}; leaving all markers untouched." >&2
        return 0
    fi

    # Mark what wandb will not: a dir synced just now whose .wandb has been
    # quiet for QUIESCE_MIN minutes is final (see the header). Quiet is judged
    # AFTER the upload, so a run that wrote during it stays stale for the next
    # pass. wandb's own marker (exit record present) supersedes this path.
    local marked=0
    for wandb_file in "${stale_files[@]}"; do
        [[ -f "${wandb_file}.synced" ]] && continue   # wandb marked it itself
        if [[ -n "$(find "${wandb_file}" -mmin "+${QUIESCE_MIN}" 2>/dev/null)" ]]; then
            touch "${wandb_file}.synced"
            marked=$(( marked + 1 ))
        fi
    done
    if (( marked > 0 )); then
        echo "marked ${marked} quiescent dir(s) synced; they will be skipped from now on."
    fi
    return 0
}

if [[ -n "${WATCH:-}" ]]; then
    while true; do
        sync_once
        sleep "${WATCH}"
    done
else
    sync_once
fi
