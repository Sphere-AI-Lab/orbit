#!/usr/bin/env bash
#
# Ten rollouts of FullFT, LoRA and OFT, then a verdict. Book a WHOLE node.
#
#   source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/smoke_e4_8gpu.sh
#
# RUN THIS BEFORE EVERY CAMPAIGN. ~30-45 minutes against a 14-node-booking
# reservation, and it is the only thing standing between a protocol change and
# the failure this exists because of.
#
# On 2026-08-03 seven gsm8k columns ran to completion -- 150 rollouts each,
# exit code 0, ~40 node-hours -- and every ledger row read `accuracy: null,
# status: "failed"`. Nothing crashed. Three defects sat downstream of anything
# the coverage probe checks:
#
#   1. train.py's generation-eval call omitted `num_rollout`, so the
#      final-rollout branch of `should_run_periodic_action` was dead. At
#      EVAL_INTERVAL=100000 -- chosen to mean "once, at the end" -- the modulo
#      never matched either, so the arms produced ZERO post-training evals. The
#      only eval line in those logs is rollout 0's: the UNTRAINED policy.
#   2. `parse_final_accuracy` demanded both math_test and gsm8k_test while
#      `arm_env` had configured gsm8k alone. It fails closed on a missing
#      dataset, so even that rollout-0 eval parsed to None.
#   3. RUN_LOG is a fixed path per arm and the launcher appends to it, so a
#      retried arm's row summarised two runs at once -- 258 rollout timings on
#      a 150-rollout row.
#
# The coverage probe passed before that campaign and would pass again: it asks
# "does this method run, and how fast", and all three defects are downstream of
# the answer. This asks the only question that protects a reservation:
#
#     does a number measured on the GPU reach the ledger, correctly labelled?
#
# WHY 10 ROLLOUTS AND AN EVAL EVERY 4 -- an interval that does NOT divide the
# rollout count, and that property IS the test. The periodic branch fires at
# rollouts 3 and 7; rollout 9 is reached only by the final-rollout branch, the
# branch defect (1) killed, so its eval present means the branch works and
# absent means the regression is back. At interval 5 the two schedules are
# IDENTICAL ([4, 9] both ways, because 10 is a multiple of 5) and the smoke
# proves nothing about the defect it exists for; a unit test pins 10 % 4 != 0.
#
# WHY A CHECKPOINT. SAVE_INTERVAL=999999 never matches the modulo, so the one
# save comes from the final-rollout branch alone, and the check wants both the
# actor's `Timer save_model` line and a non-empty orbit_ckpts/lora_regret/<arm>
# directory. The campaign runs with saves OFF, so this is the only place the
# save path executes at all -- and the day someone books a node with
# SAVE_INTERVAL set to keep a policy is the wrong day to learn it broke.
# Costs ~15 GB and ~10 min for the FullFT arm (LoRA/OFT adapters are MBs);
# SMOKE_SAVE=0 skips it, and the check then reports save UNEXERCISED, not ok.
#
# WHY THE REAL MATRIX. The three arms are read out of `e4` itself -- one per
# method, the middle learning rate of each grid -- rather than named here. A
# renamed arm or a moved cell then surfaces as a missing method instead of as a
# passing run of something else. Defects (1) and (2) both lived in code reached
# only via e4's per-dataset arms; a smoke against a stand-in matrix would have
# passed while they were live.
#
# NOT A MEASUREMENT. `--probe-rollouts` stamps every row with `probe_rollouts`,
# `analyze` exits non-zero on any ledger containing one, and the runs go to the
# `lora-regret-smoke` wandb project. Ten rollouts cannot say which learning rate
# wins and nothing here will let them try. The reward numbers it prints are
# plumbing evidence, not results.
#
# Knobs:
#   SMOKE_RESULTS=results/smoke/e4_smoke.jsonl
#   SMOKE_SAVE=1        set 0 to skip the checkpoint leg (saves ~15 GB / ~10 min)
#   SKIP_PREFLIGHT=0    set 1 to skip the pre-run audit
#   SKIP_SYNC=0         set 1 to leave wandb unsynced (the check still reports it)
#   DRY_RUN=0           set 1 to print the plan and run nothing
#
# Resumable: each arm appends `status: "ok"` and the sweep skips those, so a
# re-run picks up where it stopped. Delete the ledger to force all three.

set -uo pipefail

ORBIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ORBIT_ROOT}"

SMOKE_RESULTS=${SMOKE_RESULTS:-results/smoke/e4_smoke.jsonl}
SMOKE_SAVE=${SMOKE_SAVE:-1}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
SKIP_SYNC=${SKIP_SYNC:-0}
DRY_RUN=${DRY_RUN:-0}
: "${DATA_DIR:=/lustre/fast/fast/groups/ei-slm/data/lora_regret}"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export DATA_DIR GPUS_PER_NODE

mkdir -p "$(dirname "${SMOKE_RESULTS}")" logs/lora_regret
say() { printf '\n=== %s ===\n' "$*"; }

# --- environment ------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No virtualenv active. Run:" >&2
    echo "  source /fast/zqiu/orbit-iclr/orbit_env/bin/activate" >&2
    echo "  cd ${ORBIT_ROOT} && bash \$0" >&2
    exit 2
fi

# megatron.core imports deep_ep, whose find_cuda_home() is a bare
# `assert cuda_home is not None`, so an unset CUDA_HOME surfaces as an
# AssertionError with NO message several screens into preflight.
if [[ -f "${ORBIT_ROOT}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ORBIT_ROOT}/env.sh" >/dev/null 2>&1 || true
fi
if ! python -c "import megatron.core" >/dev/null 2>&1; then
    echo "megatron.core will not import even after sourcing env.sh." >&2
    echo "CUDA_HOME=${CUDA_HOME:-unset}" >&2
    python -c "import megatron.core" 2>&1 | tail -6 >&2
    exit 2
fi

# --- the protocol, then the two overrides -----------------------------------
#
# Sourced, not reimplemented. The point of a smoke is to exercise the
# configuration the campaign will really run -- advantage centring without std
# normalisation, clipping off, checkpoints off, wandb offline -- so a smoke that
# set its own knobs would clear a protocol nothing is going to use.
#
# The two overrides are exported FIRST because every value in the protocol is
# `: "${VAR=default}"`, which assigns only when unset. This is the documented
# override path, not a trick. Both values come from smoke.py so the schedule
# that runs and the schedule the checker assumes cannot drift apart -- the
# eval interval in particular is only diagnostic because it does not divide
# the rollout count, and two hand-copied numbers would not stay that way.
read -r SMOKE_ROLLOUTS SMOKE_EVAL_INTERVAL SMOKE_SAVE_INTERVAL < <(python -c \
    'from tools.lora_regret import smoke as s; print(s.SMOKE_ROLLOUTS, s.SMOKE_EVAL_INTERVAL, s.SMOKE_SAVE_INTERVAL)')
if [[ -z "${SMOKE_ROLLOUTS:-}" || -z "${SMOKE_EVAL_INTERVAL:-}" || -z "${SMOKE_SAVE_INTERVAL:-}" ]]; then
    echo "could not read the smoke schedule from tools.lora_regret.smoke" >&2
    exit 1
fi
export EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL}"
EXPECT_SAVES=1
if [[ "${SMOKE_SAVE}" == "1" ]]; then
    export SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL}"
else
    EXPECT_SAVES=0
fi
# shellcheck disable=SC1091
source "${ORBIT_ROOT}/scripts/lora_regret/e4_protocol.sh"

# --- preflight -------------------------------------------------------------
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    say "preflight (stage e4)"
    if ! python -m tools.lora_regret.preflight --stage e4; then
        echo "preflight failed -- fix it before spending the node." >&2
        exit 1
    fi
fi

# --- the plan --------------------------------------------------------------
say "plan"
mapfile -t PLAN < <(python -m tools.lora_regret.smoke plan)
if (( ${#PLAN[@]} != 3 )); then
    echo "expected 3 arms (full, lora, oft), got ${#PLAN[@]}." >&2
    printf '%s\n' "${PLAN[@]}" >&2
    exit 1
fi
printf '%s\n' "${PLAN[@]}" | column -t
echo "rollouts each=${SMOKE_ROLLOUTS}  eval every ${EVAL_INTERVAL}  saves=${EXPECT_SAVES}  ledger=${SMOKE_RESULTS}"

if [[ "${DRY_RUN}" == "1" ]]; then
    say "dry run -- nothing launched"
    exit 0
fi

# --- run, one at a time ----------------------------------------------------
say "running 3 arms sequentially on ${GPUS_PER_NODE} GPUs"
index=0
failed=0
for line in "${PLAN[@]}"; do
    IFS=$'\t' read -r method arm only <<< "${line}"
    index=$(( index + 1 ))
    printf '\n[%d/3] %s  (%s)\n' "${index}" "${method}" "${arm}"
    started=${SECONDS}
    # Never exits on failure: an arm that dies is what this script exists to
    # find, and the other two still have to run before the verdict is worth
    # reading. `check` distinguishes "did not run" from "ran and recorded
    # nothing", which are different defects.
    python -m tools.lora_regret.sweep \
        --matrix e4 --only "${only}" \
        --probe-rollouts "${SMOKE_ROLLOUTS}" \
        --results "${SMOKE_RESULTS}" \
        >"logs/lora_regret/smoke-${arm}.out" 2>&1
    status=$?
    elapsed=$(( SECONDS - started ))
    if (( status == 0 )); then
        printf '      ran     %dm%02ds\n' $(( elapsed / 60 )) $(( elapsed % 60 ))
    else
        failed=$(( failed + 1 ))
        printf '      FAILED  exit %d after %dm%02ds -- logs/lora_regret/smoke-%s.out\n' \
            "${status}" $(( elapsed / 60 )) $(( elapsed % 60 )) "${arm}" >&2
    fi
done

# --- sync ------------------------------------------------------------------
#
# Attempted here and NOT required to succeed. Compute nodes have no egress --
# that is why e4_protocol.sh runs wandb offline in the first place -- so this
# leg usually only passes when the smoke is driven from a node that has it. The
# check below reports the `.wandb.synced` marker either way, so an unsynced run
# is visible rather than assumed.
if [[ "${SKIP_SYNC}" != "1" ]]; then
    say "wandb sync"
    bash "${ORBIT_ROOT}/scripts/lora_regret/sync_wandb.sh" || {
        echo "sync failed (expected on a compute node -- no egress)." >&2
        echo "Run it from the login node, then re-run the check:" >&2
        echo "  bash scripts/lora_regret/sync_wandb.sh" >&2
        echo "  python -m tools.lora_regret.smoke check --ledger ${SMOKE_RESULTS} --expect-saves ${EXPECT_SAVES}" >&2
    }
fi

# --- the verdict -----------------------------------------------------------
say "verdict"
python -m tools.lora_regret.smoke check --ledger "${SMOKE_RESULTS}" --expect-saves "${EXPECT_SAVES}"
verdict=$?
if [[ "${EXPECT_SAVES}" == "0" ]]; then
    echo "NOTE: SMOKE_SAVE=0 -- the save path was NOT exercised, not passed."
fi

echo
if (( failed > 0 )); then
    echo "${failed} arm(s) exited non-zero; see logs/lora_regret/smoke-<arm>.out"
fi
if (( verdict != 0 )); then
    echo "SMOKE FAILED -- do not book the node."
    exit 1
fi
echo "SMOKE PASSED. Per-arm launcher logs: logs/lora_regret/<arm>.log"
exit 0
