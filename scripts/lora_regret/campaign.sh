#!/usr/bin/env bash
#
# Run one cell of the RL campaign: a (matrix, method) pair, on a whole node.
#
# Not called directly. The `run_<matrix>_<method>_8gpu.sh` wrappers set the
# three variables below and exec this; each is a separately bookable job with
# its own ledger, so two of them can run on two nodes without appending to the
# same file.
#
#   MATRIX=e4  METHOD_RE='^lora-'  RESULTS=results/e4_lora.jsonl \
#     bash scripts/lora_regret/campaign.sh
#
# **Every RL arm is an 8-GPU arm**, so these do not divide by GPU count the way
# the coverage probes do -- they divide by (matrix, method), and every wrapper
# is named `_8gpu` because that is what each one needs. FullFT has no choice:
# §22.2 of the runbook records that at TP=1 the standing cost is ~60 GB/GPU and
# the arm dies in the fp32 cross-entropy logits 694 MiB short, so it runs at
# TP=4/DP=2 which is eight cards. LoRA arms were *measured* at eight and would
# plausibly fit on four; nothing has run them there, so nothing here claims it.
#
# Environment (in this order -- megatron.core imports deep_ep, which asserts on
# an unset CUDA_HOME):
#
#   source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_lora_8gpu.sh
#
# Knobs:
#   MODEL=llama3.1-8b   any key in tools/lora_regret/models.py. `qwen3-1.7b`
#                       runs the same matrix as the post's own reproduction
#                       (runbook §22.6); the shapes, checkpoint and matched
#                       ranks all follow the key.
#   EXPECT_ARMS=        assert the selection is exactly this many arms and stop
#                       if it is not. The wrappers set it. A regex that silently
#                       selects 0 or 20 instead of 12 is the failure this
#                       catches, and it costs nothing to check before the node
#                       is spent.
#   SEED=0              seed replicate. Ties ROLLOUT_SEED in the launcher, so it
#                       varies problem order and sampling together.
#   GPUS_PER_NODE=8
#   SKIP_PREFLIGHT=0    set 1 to skip the pre-run audit
#   DRY_RUN=0           set 1 to print the launcher commands and run nothing
#
# Resumable: the sweep appends `status: "ok"` per finished arm and skips those on
# the next invocation, so an interrupted node picks up where it stopped. Re-run
# the same wrapper; do not start a second one against the same RESULTS.

set -uo pipefail

ORBIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ORBIT_ROOT}"

: "${MATRIX:?set MATRIX (use a run_*_8gpu.sh wrapper rather than calling this directly)}"
: "${METHOD_RE:?set METHOD_RE}"
: "${RESULTS:?set RESULTS}"
MODEL=${MODEL:-llama3.1-8b}
EXPECT_ARMS=${EXPECT_ARMS:-}
SEED=${SEED:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
DRY_RUN=${DRY_RUN:-0}
: "${DATA_DIR:=/lustre/fast/fast/groups/ei-slm/data/lora_regret}"
export DATA_DIR GPUS_PER_NODE

mkdir -p "$(dirname "${RESULTS}")" logs/lora_regret
say() { printf '\n=== %s ===\n' "$*"; }

# --- environment ------------------------------------------------------------
# The venv cannot be entered for you -- activating it inside a script would not
# survive back to your shell -- but everything after it can be, and is.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No virtualenv active. Run:" >&2
    echo "  source /fast/zqiu/orbit-iclr/orbit_env/bin/activate" >&2
    echo "  cd ${ORBIT_ROOT} && bash \$0" >&2
    exit 2
fi

# env.sh sets CUDA_HOME (if unset), LD_LIBRARY_PATH and the z3 soname. Sourced
# here rather than left to the operator because forgetting it does not fail
# fast: megatron.core imports deep_ep, whose find_cuda_home() is a bare
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

# --- preflight -------------------------------------------------------------
# Stage e4: eight cards, both checkpoints, the splits at their row counts, and
# every matrix building. Cheap, and it catches the two failures that would
# otherwise waste the node -- a venv of dangling symlinks (which imports
# *successfully*) and a truncated split.
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    say "preflight (stage e4)"
    if ! python -m tools.lora_regret.preflight --stage e4; then
        echo "preflight failed -- fix it before spending the node." >&2
        exit 1
    fi
fi

# --- the selection ---------------------------------------------------------
SWEEP_ARGS=(--model "${MODEL}" --matrix "${MATRIX}" --only "${METHOD_RE}"
            --seed "${SEED}" --results "${RESULTS}")

say "selection: ${MATRIX} ${METHOD_RE} on ${MODEL}"
# One dry run, reused for the count, the OFT check and (if DRY_RUN) the output.
# Importing the sweep pulls in megatron.core, which costs ~10s -- doing it three
# times to answer three questions about the same list is a minute of the node
# spent before anything launches.
PLAN=$(python -m tools.lora_regret.sweep "${SWEEP_ARGS[@]}" --dry-run 2>/dev/null) || {
    echo "the sweep refused to build this selection:" >&2
    python -m tools.lora_regret.sweep "${SWEEP_ARGS[@]}" --dry-run >/dev/null
    exit 1
}
SELECTED=$(printf '%s' "${PLAN}" | grep -c . )
echo "${SELECTED} arms selected -> ${RESULTS}"

# A regex is a claim about which arms run. If it drifts -- a renamed method, a
# matrix that grew a cell, a `place` tag that moved -- the sweep runs a
# different experiment and says nothing, because every arm it does run succeeds.
if [[ -n "${EXPECT_ARMS}" && "${SELECTED}" != "${EXPECT_ARMS}" ]]; then
    echo >&2
    echo "REFUSING: expected ${EXPECT_ARMS} arms, selected ${SELECTED}." >&2
    echo "The matrix or the arm names changed. Check with:" >&2
    echo "  python -m tools.lora_regret.sweep --model ${MODEL} --matrix ${MATRIX} --only '${METHOD_RE}' --dry-run" >&2
    exit 1
fi

# No OFT arm may reach a FullFT/LoRA ledger: `analyze` reads a ledger as one
# comparable set, and an `oftscout` row carries a learning rate from a different
# search. The regexes exclude them, and this is the assertion that the regexes
# did.
if printf '%s' "${PLAN}" | grep -q "PEFT_METHOD=oft"; then
    echo "REFUSING: the selection contains OFT arms; ${METHOD_RE} is wrong." >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    say "dry run -- launcher commands only"
    printf '%s\n' "${PLAN}"
    exit 0
fi

# --- run -------------------------------------------------------------------
say "running ${SELECTED} arms sequentially on ${GPUS_PER_NODE} GPUs"
echo "logs:    logs/lora_regret/<arm>.log"
echo "ledger:  ${RESULTS}"
echo "resume:  re-run this same script; finished arms are skipped"
exec python -m tools.lora_regret.sweep "${SWEEP_ARGS[@]}"
