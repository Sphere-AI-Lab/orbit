#!/bin/bash
# Example: train with the Muon-Kimi preset (orbit's Muon configured to match
# Moonshot's Kimi-Muon; see examples/optimizers/muon-kimi.env). Runs a tiny
# 0.5B GRPO via the variance-forcing RM so a real gradient flows — success =
# rc=0, log shows optimizer=muon, use_distributed_optimizer=False, nonzero
# grad_norm. Requires the emerging-optimizers package (README "Optional: Muon").
set -o pipefail
echo "### muon-kimi example on $(hostname) at $(date)"
cd "$(dirname "$0")/../.." || exit 90
ROOT=$(pwd)
source ../uv_env_build/orbit-cu132-py312/activate.sh || exit 91
export PATH="${PATH}:/usr/local/bin:/usr/bin:/bin"
export USER="${USER:-$(id -un 2>/dev/null || echo lechen)}"; export LOGNAME="${USER}"
export HOME="${HOME:-/lustre/home/lechen}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,localhost,::1"; export NO_PROXY="${no_proxy}"
export PYTHONPATH="/home/lechen/.claude/jobs/09bf2110/tmp:${PYTHONPATH:-}"  # variance_rm
BLENDS=/fast/groups/ei-slm/data/nemotron-rl-ultra-blends

# the muon-kimi preset flags, read from the shared preset file
PRESET=$(grep -v '^#' "${ROOT}/examples/optimizers/muon-kimi.env" | grep -v '^--optimizer' | tr '\n' ' ')

if [ ! -s "${BLENDS}/orbit/muon_mini.train.jsonl" ]; then
    echo "### expected ${BLENDS}/orbit/muon_mini.train.jsonl (built by the muon smoke)"; exit 92
fi

env CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_HEAD_PORT="${RAY_HEAD_PORT:-6459}" \
    OPTIMIZER=muon \
    EXTRA_OPTIMIZER_ARGS="${PRESET}" \
    CUSTOM_RM_OVERRIDE=variance_rm.reward_func \
    HF_CKPT=/fast/groups/ei-slm/hf_models/Qwen2.5-0.5B-Instruct \
    MEGATRON_LOAD=/fast/groups/ei-slm/hf_models/Qwen2.5-0.5B-Instruct_torch_dist \
    TRAIN_JSONL="${BLENDS}/orbit/muon_mini.train.jsonl" \
    NUM_ROLLOUT=4 GPUS_PER_NODE=2 ROLLOUT_NUM_GPUS=2 \
    ROLLOUT_MAX_RESPONSE_LEN=512 ENABLE_WANDB=0 DISABLE_EVAL=1 \
    bash examples/blend_router/run-qwen2_5-0_5b-router-smoke.sh
rc=$?

LOG=$(ls -t logs/smoke_qwen25_05b_router_*.log 2>/dev/null | head -1)
echo "### RESULT muon_kimi rc=${rc} log=${LOG}"
echo "### preset applied:"; grep -E "^  (optimizer|muon_scale_mode|muon_extra_scale_factor|muon_coefficient_type|muon_nesterov|use_distributed_optimizer) " "${LOG}" 2>/dev/null | head -6
echo "### grad_norm:"; grep -o "grad_norm': [0-9.eE+-]*" "${LOG}" 2>/dev/null | awk -F': ' '{print $2}' | awk 'NR%2==1' | paste -sd' '
echo "### muon-kimi example done at $(date)"
exit ${rc}
