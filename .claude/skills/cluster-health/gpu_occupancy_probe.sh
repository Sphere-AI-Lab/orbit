#!/bin/bash
#
# gpu_occupancy_probe.sh — detect an IDLE-but-OCCUPIED / RESIDUE-laden node:
#   (a) leftover GPU memory or GPU compute processes (orphaned vLLM/sglang workers
#       still holding GPU mem after a job died — the "POISONED" class); and
#   (b) orphaned heavy CPU-side procs (raylet/gcs_server/sglang/vllm/train) lingering
#       from a dead job.
# Such residue makes a node look idle to slurm while a fresh run scheduled onto it OOMs
# or wedges. The launcher's cleanup-gate catches (a) at job start; this catches both,
# standalone, BEFORE a launch. Zero-intrusion (nvidia-smi query + pgrep, no allocation).
#
# NOTE: only a FAULT on an IDLE node — on a BUSY node the GPU mem + procs belong to the
# running job (expected). Interpret per node state (the skill only fails idle nodes here).
#
# Exit 1 iff a GPU exceeds the mem threshold, a GPU compute-app is present, OR an orphan
# heavy proc is found. Threshold: GPU_OCCUPANCY_MEM_MB (default 500, matches cleanup-gate).

set -uo pipefail

THRESH=${GPU_OCCUPANCY_MEM_MB:-500}
host=$(hostname)
rc=0

if command -v nvidia-smi >/dev/null 2>&1; then
    occupied=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' -v t="$THRESH" '$2+0 > t { printf "gpu%s=%sMiB ", $1, $2 }')
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')
    if [[ -n "$occupied" || -n "$apps" ]]; then
        echo "FAIL: $host GPU occupied (>${THRESH}MiB): ${occupied:-none}"
        [[ -n "$apps" ]] && { echo "  GPU compute-apps (pid, name, used_mem):"; printf '    %s\n' "$apps"; }
        rc=1
    fi
else
    echo "WARN: $host nvidia-smi not found; GPU occupancy not checked"
fi

# Orphaned heavy procs (ray/sglang/vllm/train) lingering from a dead job — dead-job
# residue. "Orphaned" is checked STRUCTURALLY, not by name alone: when a job dies,
# surviving children get reparented to PID 1, so require PPID==1 AND a heavy cmdline.
# A bare `pgrep -af <names>` false-positives on any cmdline substring — an editor or
# `tail -f` on .../train.py, a shell cwd'd in an sglang checkout — but those run
# under a shell/tmux/sshd parent, not PID 1. (During a HEALTHY run raylet may also
# be PPID==1 — fine: occ is only a FAULT on an IDLE node. A subreaper between the
# orphan and PID 1 can hide it from this check; acceptable for a RISK signal.)
orphans=$(ps -eo pid,ppid,user:20,args --no-headers 2>/dev/null \
    | awk '$2 == 1' \
    | grep -E 'raylet|gcs_server|plasma_store|sglang|vllm|VLLM::|train_async\.py|/train\.py' \
    | grep -vE 'occupancy_probe|[[:space:]]grep ' | head -12)
if [[ -n "$orphans" ]]; then
    echo "RISK: $host orphaned heavy procs (dead-job residue — dangerous on an idle node):"
    printf '    %s\n' "$orphans"
    rc=1
fi

[[ $rc -eq 0 ]] && echo "OK: $host GPUs clear (<${THRESH}MiB, no compute apps) and no orphan procs"
exit $rc
