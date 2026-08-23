#!/usr/bin/env bash
# Phase-0 OPD smokes at 0.5B — the five teacher realizations of experiment M1
# (teacher-cost collapse table), in the plan's order: free-teacher (`base`,
# KL-only, saves the LoRA-16 adapter the adapter-swap smoke needs), self:ema,
# mopd (Megatron teacher, --opd-teacher-load, full finetune), served full-vocab
# (self-served sglang teacher: 2+1+1 GPUs), adapter-swap (`adapter:<path>`).
# Each smoke is 2-4 GPUs and a handful of rollouts. Set OPD_SMOKES to a subset
# (e.g. OPD_SMOKES="mopd") to run fewer.
# Verified on the cu130 env, 4xB200, 2026-08-23: 5/5 ok.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
adapter_first_select_model q25_05b
export DISABLE_EVAL="${DISABLE_EVAL:-1}"
OPD_DIR="${ORBIT_ROOT}/examples/on_policy_distillation"
FREE_SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_opd_free_teacher_smoke"
rc=0
run_smoke() {  # run_smoke NAME CMD...
    local name=$1; shift
    echo "[adapter_first] $(date -u +%FT%TZ) opd smoke ${name} start"
    local t0=$(date +%s) r=0
    ( cd "${ORBIT_ROOT}" && "$@" ) || r=$?
    echo "[adapter_first] $(date -u +%FT%TZ) opd smoke ${name} rc=${r} elapsed=$(( $(date +%s) - t0 ))s"
    [ "$r" -eq 0 ] || rc=$r
}
for smoke in ${OPD_SMOKES:-free ema mopd served adapter}; do
    case "$smoke" in
        free)    run_smoke free    env EXTRA_TRAIN_ARGS="--save-interval 1" bash "${OPD_DIR}/run-qwen2_5-0_5b-opd-free-teacher-smoke.sh" ;;
        ema)     run_smoke ema     bash "${OPD_DIR}/run-qwen2_5-0_5b-opd-ema-smoke.sh" ;;
        mopd)    run_smoke mopd    env OPD_TEACHER_LOAD="${MEGATRON_LOAD}" bash "${OPD_DIR}/run-qwen2_5-0_5b-opd-mopd-smoke.sh" ;;
        served)  run_smoke served  env OPD_SERVE_TEACHER=1 OPD_TEACHER_HF_CKPT="${HF_CKPT}" ROLLOUT_NUM_GPUS=1 bash "${OPD_DIR}/run-qwen2_5-0_5b-opd-full-vocab-smoke.sh" ;;
        adapter) teacher=${OPD_TEACHER_ADAPTER:-$(adapter_first_latest_adapter "${FREE_SAVE_DIR}/actor")} || { rc=1; continue; }
                 run_smoke adapter env OPD_TEACHER_ADAPTER="${teacher}" bash "${OPD_DIR}/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh" ;;
        *) echo "unknown OPD smoke: $smoke" >&2; rc=2 ;;
    esac
done
exit "$rc"
