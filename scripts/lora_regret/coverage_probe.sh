#!/usr/bin/env bash
#
# One short run per distinct code path, sequentially, on a single 8xH100 node.
#
#   bash scripts/lora_regret/coverage_probe.sh
#
# Answers two questions and refuses to answer a third:
#
#   1. Does every code path actually run?  A path is
#      (launcher, dataset, method, target modules) -- everything that is
#      genuinely different code rather than the same code at a different tensor
#      shape. Rank, OFT block size and batch size are collapsed: launching r512
#      after r256 re-runs a path that already passed. Target modules are NOT
#      collapsed, because `linear_fc1` is Orbit's fused gate+up and wrapping it
#      is not the same code as wrapping `linear_qkv`.
#
#      That is 17 runs where one-per-(task,method) was 24 -- and it covers MORE,
#      because the 24 never probed e4place's MLP placement under RL.
#
#   2. How long is the real thing?  train.py logs `progress ... last=` per
#      rollout, so each probe yields a measured per-rollout time. The report
#      still prints all 24 (task, method) rows: each reads the pace measured on
#      its own code path, which is what lets 17 runs answer 24 questions.
#
#   NOT: which learning rate wins. These are three-rollout runs. Their rows carry
#   `probe_rollouts`, `analyze` exits 4 on any ledger containing one, and every
#   run goes to the `lora-regret-smoke` wandb project rather than its task's --
#   keyed off the rollout count, so a probe cannot reach a real dashboard.
#
# SEQUENTIAL, one run at a time. The runs are three rollouts each, so packing
# them concurrently would save a fraction of an already short session while
# buying: a GPU allocator to get wrong, contended per-rollout times that are
# upper bounds rather than estimates, and interleaved failures that are harder
# to attribute. Sequential means every number below is measured on an idle node
# and every failure has exactly one candidate cause.
#
# GPU count per run still mirrors the real sweep, because a timing measured on
# the wrong number of GPUs estimates nothing:
#
#   SFT LoRA/OFT  1 GPU      SFT FullFT  4 GPUs      RL any  8 GPUs
#
# Environment (in this order -- megatron.core imports deep_ep, which asserts on
# an unset CUDA_HOME):
#
#   source scripts/lora_regret/env_v0516.sh
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   export CUDA_HOME=/is/software/nvidia/cuda-13.2 && source env.sh
#
# Knobs:
#   PROBE_LEVEL=path        path (17) | method (24) | config (61)
#   PROBE_ROLLOUTS=3        rollouts per probe run
#   PROBE_DIR=results/probe where the per-run ledgers go
#   ONLY_GPUS=              set to 1, 4 or 8 to run only the runs of that size.
#                           The three coverage_probe_<n>gpu.sh wrappers set it,
#                           so each can be booked on a differently sized node.
#   SKIP_METHODS=           comma-separated methods (full,lora,oft) to leave out.
#                           For a path whose failure is already understood, so
#                           the node is not spent re-deriving it. Skipped rows
#                           read "not run" in the report -- never "ok".
#   SKIP_PREFLIGHT=0        set 1 to skip the pre-run audit
#   DRY_RUN=0               set 1 to print the schedule and run nothing
#
# Resumable: each run writes its own ledger and the sweep skips an arm already
# recorded "ok", so re-running after an interruption picks up where it stopped.

set -uo pipefail

ORBIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ORBIT_ROOT}"

PROBE_LEVEL=${PROBE_LEVEL:-path}
PROBE_ROLLOUTS=${PROBE_ROLLOUTS:-3}
PROBE_DIR=${PROBE_DIR:-results/probe}
ONLY_GPUS=${ONLY_GPUS:-}
SKIP_METHODS=${SKIP_METHODS:-}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
DRY_RUN=${DRY_RUN:-0}
: "${DATA_DIR:=/lustre/fast/fast/groups/ei-slm/data/lora_regret}"
export DATA_DIR

mkdir -p "${PROBE_DIR}" logs/lora_regret
say() { printf '\n=== %s ===\n' "$*"; }

# --- environment ------------------------------------------------------------
# The venv cannot be entered for you -- activating it inside a script would not
# survive back to your shell -- but everything after it can be, and is.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No virtualenv active. Run:" >&2
    echo "  source scripts/lora_regret/env_v0516.sh" >&2
    echo "  cd ${ORBIT_ROOT} && bash \$0" >&2
    exit 2
fi

# env.sh sets CUDA_HOME (if unset), LD_LIBRARY_PATH and the z3 soname. Sourced
# here rather than left to the operator because forgetting it does not fail
# fast: megatron.core imports deep_ep, whose find_cuda_home() is a bare
# `assert cuda_home is not None`, so an unset CUDA_HOME surfaces as an
# AssertionError with NO message several screens into preflight -- and the
# LD_LIBRARY_PATH half fails later still, mid-run, on a missing .so.
#
# Sourcing affects this script and the runs it spawns, never your shell.
if [[ -f "${ORBIT_ROOT}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ORBIT_ROOT}/env.sh" >/dev/null 2>&1 || true
fi

if ! python -c "import megatron.core" >/dev/null 2>&1; then
    echo "megatron.core will not import even after sourcing env.sh." >&2
    echo "CUDA_HOME=${CUDA_HOME:-unset}" >&2
    echo >&2
    python -c "import megatron.core" 2>&1 | tail -6 >&2
    exit 2
fi

# --- preflight -------------------------------------------------------------
# Cheap, and it catches the two failures that would otherwise waste the node: a
# venv of dangling symlinks (which imports *successfully*) and a truncated split.
# The stage tracks ONLY_GPUS: preflight asserts the node has enough cards for
# the stage it is given, so checking `e4` (needs 8) on a one-GPU reservation
# would fail the audit for a run that was never going to use eight.
case "${ONLY_GPUS}" in
    1) PREFLIGHT_STAGE=e1-lora ;;
    4) PREFLIGHT_STAGE=e1-full ;;
    *) PREFLIGHT_STAGE=e4 ;;
esac
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    say "preflight (stage ${PREFLIGHT_STAGE})"
    if ! python -m tools.lora_regret.preflight --stage "${PREFLIGHT_STAGE}"; then
        echo "preflight failed -- fix it before spending the node." >&2
        exit 1
    fi
fi

# --- the plan --------------------------------------------------------------
# Built by tools/lora_regret/probe.py, so arm names, GPU counts and rollout
# targets come from the matrices themselves rather than from a list in a shell
# script that drifts the moment a matrix changes.
say "plan"
# Matrices whose arms cannot be built without an OFT learning-rate centre, read
# from arms.py rather than listed here so a new one cannot be missed. Resolved
# once: the run loop below consults it per arm.
CENTRE_MATRICES=$(python -c \
    'from tools.lora_regret.arms import MATRICES_REQUIRING_OFT_CENTRE as m; print(" ".join(sorted(m)))')
if [[ -z "${CENTRE_MATRICES}" ]]; then
    echo "could not read MATRICES_REQUIRING_OFT_CENTRE from tools.lora_regret.arms" >&2
    exit 1
fi
echo "matrices needing an OFT LR centre: ${CENTRE_MATRICES}"

PLAN_ARGS=(plan --level "${PROBE_LEVEL}")
# Filtered by probe.py rather than in the loop below, so the plan that is
# printed is exactly the plan that runs.
[[ -n "${ONLY_GPUS}" ]] && PLAN_ARGS+=(--gpus "${ONLY_GPUS}")
mapfile -t PLAN < <(python -m tools.lora_regret.probe "${PLAN_ARGS[@]}")
if (( ${#PLAN[@]} == 0 )); then
    echo "empty plan -- probe.py produced no runs." >&2
    exit 1
fi
printf '%s\n' "${PLAN[@]}" | column -t
echo "level=${PROBE_LEVEL}  runs=${#PLAN[@]}  rollouts each=${PROBE_ROLLOUTS}  sequential"
[[ -n "${ONLY_GPUS}" ]] && echo "restricted to ${ONLY_GPUS}-GPU runs; the other sizes are separate scripts"

# --- run, one at a time ----------------------------------------------------
say "running ${#PLAN[@]} probes sequentially"
index=0
failed=0
skipped=0
for line in "${PLAN[@]}"; do
    IFS=$'\t' read -r matrix method arm only gpus metric full label <<< "${line}"
    index=$(( index + 1 ))

    # Filtered here rather than out of the plan, so the printed schedule still
    # shows every path and the skip is visible as a line of output. A path
    # silently absent from the plan would read as coverage it never had.
    if [[ -n "${SKIP_METHODS}" ]] && [[ ",${SKIP_METHODS}," == *",${method},"* ]]; then
        printf '\n[%d/%d] %s  SKIPPED (SKIP_METHODS=%s)\n' \
            "${index}" "${#PLAN[@]}" "${label}" "${SKIP_METHODS}"
        skipped=$(( skipped + 1 ))
        continue
    fi

    # Devices 0..N-1. Nothing else is running, so which cards these are does not
    # matter -- what matters is that the count matches what the real sweep gives
    # this arm, or the measured pace estimates a machine that will never run it.
    devices="$(seq -s, 0 $(( gpus - 1 )))"
    ledger="${PROBE_DIR}/${matrix}-${arm}.jsonl"
    extra=()
    # Some matrices' OFT cells have no scouted centre yet. Any value is a valid
    # *plumbing* probe -- a learning rate does not change how long a step takes
    # -- and the real sweep still refuses to run them without the measured one.
    #
    # Which matrices those are comes from arms.py, resolved once above into
    # CENTRE_MATRICES. This line used to read `[[ "${matrix}" == "e5" ]]`, and
    # that literal is exactly why all three e5rl arms died in two seconds on
    # 2026-08-01 with `--matrix e5rl requires --oft-lr-centre`: the Python side
    # had been taught about the second such matrix and this one had not.
    if [[ " ${CENTRE_MATRICES} " == *" ${matrix} "* ]]; then
        extra+=(--oft-lr-centre 1e-4)
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry] %2d/%d  %s GPU(s) %-16s %s\n' \
            "${index}" "${#PLAN[@]}" "${gpus}" "${devices}" "${label}"
        continue
    fi

    printf '\n[%d/%d] %s  (%s GPU, %s)\n' \
        "${index}" "${#PLAN[@]}" "${label}" "${gpus}" "${arm}"
    started=${SECONDS}
    # Never `exit`s on failure: a path that dies is exactly what this script
    # exists to discover, and the remaining probes still need to run.
    CUDA_VISIBLE_DEVICES="${devices}" GPUS_PER_NODE="${gpus}" \
        python -m tools.lora_regret.sweep \
            --matrix "${matrix}" --only "${only}" \
            --probe-rollouts "${PROBE_ROLLOUTS}" \
            --results "${ledger}" "${extra[@]}" \
        >"logs/lora_regret/probe-${matrix}-${arm}.out" 2>&1
    status=$?
    elapsed=$(( SECONDS - started ))
    if (( status == 0 )); then
        printf '      ok      %dm%02ds\n' $(( elapsed / 60 )) $(( elapsed % 60 ))
    else
        failed=$(( failed + 1 ))
        printf '      FAILED  exit %d after %dm%02ds -- logs/lora_regret/probe-%s-%s.out\n' \
            "${status}" $(( elapsed / 60 )) $(( elapsed % 60 )) "${matrix}" "${arm}" >&2
    fi
done

# --- the answer ------------------------------------------------------------
say "report"
python -m tools.lora_regret.probe report --level "${PROBE_LEVEL}" --ledger "${PROBE_DIR}/*.jsonl"
echo
if (( failed > 0 )); then
    echo "${failed} probe(s) failed. Their rows read FAILED above and the campaign"
    echo "estimate is a LOWER BOUND -- it omits them rather than guessing."
fi
if (( skipped > 0 )); then
    echo "${skipped} probe(s) skipped via SKIP_METHODS=${SKIP_METHODS}. Their rows are"
    echo "unmeasured, not passing. Unset it to cover them."
fi
if [[ -n "${ONLY_GPUS}" ]]; then
    echo "This was the ${ONLY_GPUS}-GPU subset. Rows reading 'not run' belong to the"
    echo "other two scripts; the report reads every ledger in ${PROBE_DIR}, so it"
    echo "fills in as each one finishes -- in any order, on any node."
fi
echo "Per-run stdout: logs/lora_regret/probe-<task>-<arm>.out"
echo "Per-arm launcher logs: logs/lora_regret/<arm>.log"
echo "wandb: every probe run is in the lora-regret-smoke project, group=<task>-<method>."
