#!/usr/bin/env bash
#
# One run per (task, method) on a single 8xH100 node.
#
#   bash scripts/lora_regret/coverage_probe.sh
#
# Answers two questions and refuses to answer a third:
#
#   1. Does every method actually run?  Each of the 24 (task, method) pairs is
#      launched once for PROBE_ROLLOUTS rollouts. Anything that cannot start,
#      cannot wrap the model, or never reaches the eval line the parser needs
#      fails here in minutes instead of on the 40th arm of a reserved node.
#
#      Rank, OFT block size, target modules and batch size are NOT probed
#      separately: they exercise the same code -- same wrap, same launcher path,
#      same optimizer -- at different tensor shapes, so r512 after r256 re-runs a
#      path that already passed. The axes carrying distinct code are the method
#      (which adapter, or none) and the task (which launcher, dataset, metric).
#
#      PROBE_LEVEL=config launches all 61 distinct configurations instead. Reach
#      for it when hunting a shape-dependent failure -- an OOM at a batch size
#      nothing has run at -- not a code-path one.
#
#   2. How long is the real thing?  train.py logs `progress ... last=` per
#      rollout, so each probe yields a measured per-rollout time. The report
#      multiplies it by the rollout count that arm would really run and by the
#      number of arms of that method in that task.
#
#   NOT: which learning rate wins. These are three-rollout runs. Their rows carry
#   `probe_rollouts`, `analyze` exits 4 on any ledger containing one, and every
#   run goes to the `lora-regret-smoke` wandb project rather than its task's --
#   keyed off the rollout count, so a probe cannot reach a real dashboard.
#
# GPU sizing mirrors the real sweep, because a timing measured on the wrong
# number of GPUs is an estimate of nothing:
#
#   SFT LoRA/OFT  1 GPU   -- eight at a time, one per device
#   SFT FullFT    4 GPUs  -- two at a time (the registry's floor for 8.03B)
#   RL  any       8 GPUs  -- one at a time, the whole node
#
# The 1-GPU phase is contended by construction: eight arms share NVLink, host RAM
# and the filesystem. Its per-rollout times are therefore UPPER BOUNDS on the
# real arms', not estimates of them. The 4-GPU and 8-GPU phases are not contended
# and their numbers are estimates. The report says so; do not average the two.
#
# Environment (must be in this order -- megatron.core imports deep_ep, which
# asserts on an unset CUDA_HOME):
#
#   source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   export CUDA_HOME=/is/software/nvidia/cuda-13.2 && source env.sh
#
# Knobs:
#   PROBE_LEVEL=method      method (24 runs) | config (61 runs)
#   PROBE_ROLLOUTS=3        rollouts per probe run
#   PROBE_DIR=results/probe where the per-run ledgers go
#   SKIP_PREFLIGHT=0        set 1 to skip the pre-run audit
#   ONLY_PHASE=             set to 1, 4 or 8 to run just that GPU phase
#   DRY_RUN=0               set 1 to print the launches and run nothing

set -uo pipefail

ORBIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ORBIT_ROOT}"

PROBE_LEVEL=${PROBE_LEVEL:-method}
PROBE_ROLLOUTS=${PROBE_ROLLOUTS:-3}
PROBE_DIR=${PROBE_DIR:-results/probe}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
ONLY_PHASE=${ONLY_PHASE:-}
DRY_RUN=${DRY_RUN:-0}
: "${DATA_DIR:=/lustre/fast/fast/groups/ei-slm/data/lora_regret}"
export DATA_DIR

mkdir -p "${PROBE_DIR}" logs/lora_regret

say() { printf '\n=== %s ===\n' "$*"; }

# --- preflight -------------------------------------------------------------
# Cheap, and it catches the two failures that would otherwise waste the whole
# node: a venv of dangling symlinks (which imports *successfully*) and a
# truncated data split.
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    say "preflight"
    if ! python -m tools.lora_regret.preflight --stage e4; then
        echo "preflight failed -- fix it before spending the node." >&2
        exit 1
    fi
fi

# --- the plan --------------------------------------------------------------
# Built by tools/lora_regret/probe.py so the arm names, GPU counts and rollout
# targets come from the matrices themselves rather than from a list in a shell
# script that drifts the moment a matrix changes.
say "plan"
python -m tools.lora_regret.probe plan --level "${PROBE_LEVEL}" | column -t
echo "level=${PROBE_LEVEL}  runs=$(python -m tools.lora_regret.probe plan --level "${PROBE_LEVEL}" | wc -l)  rollouts each=${PROBE_ROLLOUTS}"

# One ledger per run. Two concurrent processes appending to one file interleave
# partial lines, which the reader then drops -- silently losing a probe.
# Keyed on the ARM, not on (task, method): at config level a task has several
# runs per method, and a (task, method) filename would have them overwrite
# each other -- silently keeping whichever finished last.
ledger_for() { echo "${PROBE_DIR}/$1-$2.jsonl"; }

# Runs one probe. Never `exit`s on failure: a method that dies is exactly what
# this script exists to discover, and the remaining 23 still need to run.
run_probe() {
    local matrix="$1" arm="$2" only="$3" gpus="$4" devices="$5"
    local ledger; ledger="$(ledger_for "${matrix}" "${arm}")"
    local extra=()
    # e5's OFT cell has no scouted centre yet. Any value is a valid *plumbing*
    # probe -- a learning rate does not change how long a step takes -- and the
    # real sweep still refuses to run e5 without the measured one.
    if [[ "${matrix}" == "e5" ]]; then
        extra+=(--oft-lr-centre 1e-4)
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "CUDA_VISIBLE_DEVICES=${devices} GPUS_PER_NODE=${gpus} python -m tools.lora_regret.sweep" \
             "--matrix ${matrix} --only '${only}' --probe-rollouts ${PROBE_ROLLOUTS}" \
             "--results ${ledger} ${extra[*]:-}"
        return 0
    fi
    echo "[probe] ${matrix}/${arm} on GPU(s) ${devices}"
    CUDA_VISIBLE_DEVICES="${devices}" GPUS_PER_NODE="${gpus}" \
        python -m tools.lora_regret.sweep \
            --matrix "${matrix}" \
            --only "${only}" \
            --probe-rollouts "${PROBE_ROLLOUTS}" \
            --results "${ledger}" \
            "${extra[@]}" \
        >"logs/lora_regret/probe-${matrix}-${arm}.out" 2>&1 \
        || echo "[probe] ${matrix}/${arm} FAILED -- see logs/lora_regret/probe-${matrix}-${arm}.out" >&2
}

phase_wanted() { [[ -z "${ONLY_PHASE}" || "${ONLY_PHASE}" == "$1" ]]; }

# --- phase 1: the 1-GPU arms, eight at a time ------------------------------
if phase_wanted 1; then
    say "phase 1 -- SFT LoRA/OFT, 1 GPU each, 8 concurrent"
    device=0
    while IFS=$'\t' read -r matrix method arm only gpus metric full label; do
        run_probe "${matrix}" "${arm}" "${only}" 1 "${device}" &
        device=$(( (device + 1) % 8 ))
        # Barrier every 8: eight 8B models on eight cards is the point, sixteen
        # on eight is an OOM that would be read as "the method does not work".
        if (( device == 0 )); then wait; fi
    done < <(python -m tools.lora_regret.probe plan --level "${PROBE_LEVEL}" --gpus 1)
    wait
fi

# --- phase 2: the 4-GPU FullFT arms, two at a time -------------------------
if phase_wanted 4; then
    say "phase 2 -- SFT FullFT, 4 GPUs each, 2 concurrent"
    slot=0
    while IFS=$'\t' read -r matrix method arm only gpus metric full label; do
        if (( slot == 0 )); then devices="0,1,2,3"; else devices="4,5,6,7"; fi
        run_probe "${matrix}" "${arm}" "${only}" 4 "${devices}" &
        slot=$(( (slot + 1) % 2 ))
        if (( slot == 0 )); then wait; fi
    done < <(python -m tools.lora_regret.probe plan --level "${PROBE_LEVEL}" --gpus 4)
    wait
fi

# --- phase 3: the RL arms, whole node, one at a time -----------------------
if phase_wanted 8; then
    say "phase 3 -- RL, 8 GPUs, sequential"
    while IFS=$'\t' read -r matrix method arm only gpus metric full label; do
        run_probe "${matrix}" "${arm}" "${only}" 8 "0,1,2,3,4,5,6,7"
    done < <(python -m tools.lora_regret.probe plan --level "${PROBE_LEVEL}" --gpus 8)
fi

# --- the answer ------------------------------------------------------------
say "report"
python -m tools.lora_regret.probe report --level "${PROBE_LEVEL}" --ledger "${PROBE_DIR}/*.jsonl"
echo
echo "Per-run stdout: logs/lora_regret/probe-<task>-<arm>.out"
echo "Per-arm launcher logs: logs/lora_regret/<arm>.log"
echo "wandb: every probe run is in the lora-regret-smoke project, group=<task>-<method>."
