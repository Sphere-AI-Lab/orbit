#!/bin/bash
# weka_probe.sh — per-node shared-storage (WekaFS) liveness probe for the launcher
# healthcheck. The job reads its conda env, repo, and model weights from the shared
# filesystem under /data (WekaFS); a wedged Weka client makes those reads hang in
# uninterruptible D-state, which stalls engine/weight bring-up with no useful error
# (GPUs sit idle, "Load weight" never completes). See
# docs/sync-records/orbit-sync-2026-06-30/wekafs-wedge-2026-07-01.md.
#
# Reads real bytes (O_DIRECT, bypassing the page cache) from a large file on the shared
# FS. On a healthy mount this returns in ~1-2s; on a wedged one the read hangs and the
# launcher's outer `timeout` (HEALTHCHECK_TIMEOUT) marks the node BAD -> excluded ->
# requeue. It targets a WEDGE (reads that never return), not slowness: reading ~64 MiB
# finishes well within HEALTHCHECK_TIMEOUT even on a loaded-but-live FS. O_DIRECT matters
# so a probe file left warm in cache by a prior job can't mask a wedged backend.
#
# Env (with defaults):
#   CONDA_ROOT       [/data/shared/conda/miniconda3]
#   ORBIT_ENV_NAME   [orbit]
#   WEKA_PROBE_DIR   dir on the shared FS to probe  [$CONDA_ROOT/envs/$ORBIT_ENV_NAME/lib]
#   WEKA_PROBE_MB    real bytes to read (MiB)       [64]
# Exit: 0 = read OK (or nothing to probe -> skip, non-blocking); nonzero/hang = wedged.
set -uo pipefail

CONDA_ROOT="${CONDA_ROOT:-/data/shared/conda/miniconda3}"
ORBIT_ENV_NAME="${ORBIT_ENV_NAME:-orbit}"
probe_dir="${WEKA_PROBE_DIR:-$CONDA_ROOT/envs/$ORBIT_ENV_NAME/lib}"
mb="${WEKA_PROBE_MB:-64}"
min_mb=$((mb + 32))   # probe file must exceed the read so O_DIRECT never hits an unaligned EOF tail

if [[ ! -d "$probe_dir" ]]; then
    echo "weka_probe: probe dir $probe_dir not present — skipping (non-blocking)"
    exit 0
fi

# First file comfortably larger than the read (stop at the first match -> fast even on a
# huge tree). The directory traversal itself revalidates dentries against the Weka
# backend, so a wedge hangs here too, which the outer timeout catches.
f=$(find "$probe_dir" -maxdepth 6 -type f -name '*.so*' -size +"${min_mb}"M -print -quit 2>/dev/null)
[[ -z "$f" ]] && f=$(find "$probe_dir" -maxdepth 6 -type f -size +"${min_mb}"M -print -quit 2>/dev/null)
if [[ -z "$f" ]]; then
    echo "weka_probe: no file > ${min_mb}MiB under $probe_dir — skipping (non-blocking)"
    exit 0
fi

# Prefer O_DIRECT (bypasses the page cache -> always hits the Weka backend, so a wedge
# hangs here even for a warm-cached file). Fall back to a buffered read only if the mount
# rejects O_DIRECT (fast EINVAL, not a hang), so we don't false-fail where it's unsupported.
if dd if="$f" of=/dev/null bs=1M count="$mb" iflag=direct status=none 2>/dev/null; then
    echo "weka_probe: OK — read ${mb}MiB (O_DIRECT) from $f"
    exit 0
fi
if dd if="$f" of=/dev/null bs=1M count="$mb" iflag=nocache status=none 2>/dev/null; then
    echo "weka_probe: OK — read ${mb}MiB from $f (O_DIRECT unsupported)"
    exit 0
fi
echo "weka_probe: FAILED reading $f (${mb}MiB)"
exit 1
