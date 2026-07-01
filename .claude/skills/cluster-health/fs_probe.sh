#!/bin/bash
#
# fs_probe.sh — WekaFS DATA-PATH health + FS-residue RISK probe.
#
# WHY metadata / tiny-I/O is NOT enough: on this cluster real /data READS can hang or
# crawl while metadata ops (stat/ls) and cached/tiny I/O still look fine — this wedged
# multi-node jobs during model-weight load and explains shell/terminal slowdowns. So the
# core test is a REAL 64 MiB O_DIRECT read of a large file on /data (cache-bypassing;
# exercises the Weka backend data path) — the exact repro infra uses. Plus:
#   - stat + a tiny write (cheap metadata/write liveness; necessary, NOT sufficient),
#   - persistent D-state procs wedged in weka/commit kernel paths (a thread stuck in
#     the Weka client that survives SIGKILL and wedges new jobs) — RISK; persistent
#     D-state with any OTHER wchan is a NOTE only: on a mix node (someone's job doing
#     heavy I/O) or a shared login node, legit I/O can hold D across both samples
#     (the launcher's cleanup-gate learned this on WekaFS). Authoritative on idle only.
#
# The degradation is often TRANSIENT — one clean scan proves nothing; re-run on-demand
# (Mode 3) and watch the ledger trend. A single slow read can be a one-off backend blip,
# so a slow read is CONFIRMED with a re-read (still slow = real degradation; O_DIRECT means
# this is not a cache effect). Every op is timeout-guarded so a hung mount fails THIS probe,
# never the caller.
#
# Exit 1 if: the read hangs/times out, a warm read stays slow, stat/write hang, or
# persistent WEKA-WEDGED D-state procs exist (other persistent D-state = NOTE only). Else 0.
# Env: FS_PROBE_DIR ($HOME), FS_PROBE_TIMEOUT_S (5), FS_PROBE_READ_TIMEOUT_S (20),
#      FS_PROBE_READ_SLOW_MS (3000), FS_PROBE_READ_FILE (auto: shared-conda libtorch .so).

set -uo pipefail

DIR=${FS_PROBE_DIR:-$HOME}
T=${FS_PROBE_TIMEOUT_S:-5}
RT=${FS_PROBE_READ_TIMEOUT_S:-20}
SLOW=${FS_PROBE_READ_SLOW_MS:-3000}
host=$(hostname)
rc=0

# One 64 MiB O_DIRECT read of $READ_FILE: echoes elapsed ms, returns dd's rc.
read_ms() {
    local s r
    s=$(date +%s%N)
    timeout "$RT" dd if="$READ_FILE" of=/dev/null bs=1M count=64 iflag=direct status=none 2>/dev/null
    r=$?
    echo "$(( ($(date +%s%N) - s) / 1000000 ))"
    return $r
}

# (1) metadata + tiny-write liveness (stays fine even when the data path hangs).
if ! timeout "$T" stat "$DIR" >/dev/null 2>&1; then
    echo "FAIL: $host stat($DIR) did not return in ${T}s — FS metadata hung"
    rc=1
else
    f="$DIR/.fs_probe.${host}.$$"
    timeout "$T" bash -c "printf probe > '$f' && rm -f '$f'" 2>/dev/null \
        || { echo "FAIL: $host tiny write on $DIR did not complete in ${T}s"; timeout 2 rm -f "$f" 2>/dev/null || true; rc=1; }
fi

# (2) REAL data-path read (the point of this probe): 64 MiB O_DIRECT, cache-bypassing.
READ_FILE=${FS_PROBE_READ_FILE:-}
if [[ -z "$READ_FILE" ]]; then
    # Discovery itself is readdir (glob) + stat on /data/shared — Weka METADATA ops
    # that can wedge exactly like the data path (the commit_blocking_request D-state
    # class), and they run BEFORE any guarded dd. Run them under timeout too, so a
    # hung mount fails THIS probe, never the caller. This also doubles as the
    # metadata check on /data/shared itself — (1) above targets $DIR (default $HOME),
    # which is the same Weka mount today but would diverge if the mounts ever split.
    READ_FILE=$(timeout "$T" env ENVN="${MILES_ENV_NAME:-miles}" bash -c '
        for envn in "$ENVN" miles miles_imp; do
            for f in /data/shared/conda/miniconda3/envs/$envn/lib/python*/site-packages/torch/lib/libtorch_cuda.so; do
                [[ -f "$f" ]] && { echo "$f"; exit 0; }
            done
        done')
    disc_rc=$?
    if (( disc_rc == 124 || disc_rc == 137 )); then
        echo "FAIL: $host probe-file discovery (readdir/stat on /data/shared) did not return in ${T}s — FS metadata HUNG"
        rc=1
        READ_FILE=""
    fi
fi
if [[ -n "$READ_FILE" ]]; then
    ms1=$(read_ms); r1=$?
    if (( r1 == 0 )); then
        if (( ms1 >= SLOW )); then
            # O_DIRECT bypasses the OS page cache, so this is not a cold/warm effect: the
            # re-read confirms PERSISTENT slowness vs a one-off backend/network blip.
            ms2=$(read_ms); r2=$?
            if (( r2 == 0 )); then
                (( ms1 < ms2 )) && min=$ms1 || min=$ms2
                (( min >= SLOW )) && { echo "RISK: $host 64MiB /data O_DIRECT read slow (${ms1}/${ms2}ms >= ${SLOW}ms) — Weka data path DEGRADED"; rc=1; }
            elif (( r2 == 124 || r2 == 137 )); then
                echo "FAIL: $host 64MiB /data O_DIRECT re-read HUNG (>${RT}s; first=${ms1}ms) — Weka data path"; rc=1
            else
                echo "FAIL: $host 64MiB /data O_DIRECT re-read ERROR rc=$r2 (${ms2}ms — NOT a hang; perm/vanished file/no O_DIRECT? check FS_PROBE_READ_FILE before blaming Weka)"; rc=1
            fi
        fi
    elif (( r1 == 124 || r1 == 137 )); then
        # timeout(1) returns 124 (TERM) / 137 (KILL) — only these mean the read HUNG.
        echo "FAIL: $host 64MiB /data O_DIRECT read did NOT complete in ${RT}s — Weka data path HUNG (metadata may look fine)"
        rc=1
    else
        # dd failed with its own rc (instantly, in ${ms1}ms): permission denied, file
        # vanished between discovery and read, or EINVAL from a path without O_DIRECT
        # support. That is a probe/environment error, NOT a Weka hang — say so, or a
        # healthy node gets excluded on a mislabeled diagnosis.
        echo "FAIL: $host 64MiB /data O_DIRECT read ERROR rc=$r1 (${ms1}ms — NOT a hang; perm/vanished file/no O_DIRECT? check FS_PROBE_READ_FILE before blaming Weka)"
        rc=1
    fi
else
    [[ $rc -eq 0 ]] && echo "WARN: $host no large /data probe file found — real-read test SKIPPED (set FS_PROBE_READ_FILE)"
fi

# (3) persistent D-state procs (sample twice ~1s apart). Only weka/commit-wedged
# ones are a RISK (the Weka-client wedge class); other persistent D-state is a NOTE,
# not a fault — legit heavy I/O on a mix/login node can hold D across both samples.
d1=$(ps -eo pid,state --no-headers 2>/dev/null | awk '$2 ~ /^D/ {print $1}' | sort -u)
sleep 1
wedged=""; other=""
while read -r pid wchan comm; do
    echo "$d1" | grep -qx "$pid" || continue
    if echo "$wchan" | grep -qiE 'weka|commit'; then
        wedged="$wedged${comm}(pid=$pid,wchan=$wchan) "
    else
        other="$other${comm}(pid=$pid,wchan=$wchan) "
    fi
done < <(ps -eo pid,state,wchan:24,comm --no-headers 2>/dev/null | awk '$2 ~ /^D/ {print $1, $3, $4}')
if [[ -n "$wedged" ]]; then
    echo "RISK: $host persistent D-state procs wedged in the Weka client: $wedged"
    rc=1
fi
[[ -n "$other" ]] && echo "NOTE: $host persistent D-state, non-weka wchan (weak signal on mix/login — legit heavy I/O; authoritative on idle): $other"

[[ $rc -eq 0 ]] && echo "OK: $host /data data-path healthy (64MiB O_DIRECT read fast, metadata live, no wedged procs)"
exit $rc
