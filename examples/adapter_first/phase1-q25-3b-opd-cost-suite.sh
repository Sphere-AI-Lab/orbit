#!/usr/bin/env bash
# R-2: the Qwen2.5-3B OPD teacher-cost suite (experiment M1, measured table).
# Five variants of examples/on_policy_distillation/opd_teacher_cost_common.sh,
# run in an order that lets `adapter` consume the OFT adapter that `base`
# saves: base -> ema -> load -> served -> adapter. Each is 1 actor + 3 rollout
# GPUs (served adds 1 teacher GPU; on a 4-GPU node the rollout pool shrinks
# to 2). NUM_ROLLOUT defaults to the recipe's 500; NUM_ROLLOUT=1 is the launch
# probe. OPD_COST_VARIANTS selects a subset.
# Launch-verified on the cu130 env, 4xB200, 2026-08-23 with NUM_ROLLOUT=1: 5/5 ok.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
adapter_first_select_model q25_3b
export TRAIN_JSONL="${OPD_COST_TRAIN_JSONL:-${Q3_4B_TRAIN_JSONL}}"   # OpenR1-style math JSONL
export EVAL_ORBIT_DIR="${EVAL_ORBIT_DIR:-/fast/groups/ei-slm/data/peft_arena_eval_math_alignment}"
export DISABLE_EVAL="${DISABLE_EVAL:-1}"
export SAVE_ROOT="${SAVE_ROOT:-${ORBIT_ROOT}/orbit_ckpts/opd_teacher_cost}"
export SEED="${SEED:-1234}"
OPD_DIR="${ORBIT_ROOT}/examples/on_policy_distillation"
BASE_SAVE_DIR="${SAVE_ROOT}/Qwen2.5-3B-Instruct_opd_cost_base_seed${SEED}"
gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
rc=0
run_variant() {  # run_variant NAME ENV...=VAL
    local name=$1; shift
    echo "[adapter_first] $(date -u +%FT%TZ) opd-cost ${name} start"
    local t0=$(date +%s) r=0
    ( cd "${ORBIT_ROOT}" && env "$@" bash "${OPD_DIR}/run-qwen2_5-3b-opd-cost-${name}.sh" ) || r=$?
    echo "[adapter_first] $(date -u +%FT%TZ) opd-cost ${name} rc=${r} elapsed=$(( $(date +%s) - t0 ))s"
    [ "$r" -eq 0 ] || rc=$r
}
for v in ${OPD_COST_VARIANTS:-base ema load served adapter}; do
    case "$v" in
        base)    run_variant base    EXTRA_TRAIN_ARGS="--save-interval 1 ${EXTRA_TRAIN_ARGS:-}" ;;
        ema)     run_variant ema ;;
        load)    run_variant load    OPD_TEACHER_LOAD="${OPD_TEACHER_LOAD:-${MEGATRON_LOAD}}" ;;
        served)  run_variant served  OPD_TEACHER_HF_CKPT="${OPD_TEACHER_HF_CKPT:-${HF_CKPT}}" OPD_TEACHER_NUM_GPUS=1 \
                                     ROLLOUT_NUM_GPUS="${SERVED_ROLLOUT_NUM_GPUS:-$([ "$gpus" -ge 5 ] && echo 3 || echo 2)}" ;;
        adapter) teacher=${OPD_TEACHER_ADAPTER:-$(adapter_first_latest_adapter "${BASE_SAVE_DIR}")} || { rc=1; continue; }
                 run_variant adapter OPD_TEACHER_ADAPTER="${teacher}" ;;
        *) echo "unknown variant: $v" >&2; rc=2 ;;
    esac
done
exit "$rc"
