#!/usr/bin/env bash
# Shared environment for the adapter-first experiment program launchers
# (docs/plans/2026-08-17-adapter-first-experiments-design.md,
#  docs/superpowers/plans/2026-08-19-adapter-first-phase0-phase1.md).
# Source from a launcher; every value is overridable by exporting it first.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

ADAPTER_FIRST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${ADAPTER_FIRST_DIR}/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${ORBIT_ROOT}/.." && pwd)}"

# Workspace env: uv_env_build/activate.sh selects ORBIT_ENV (cu130 default,
# venv for the source-built one). Skip if the caller already activated one.
if [[ -z "${ORBIT_VENV:-}" ]]; then
    source "${WORKSPACE_ROOT}/uv_env_build/activate.sh"
fi

# Design-doc constraint 10: CUDA IPC is denied on this cluster; adapters move
# over cpu_gather. activate.sh deliberately does not set this.
export ORBIT_PEFT_ADAPTER_TRANSPORT="${ORBIT_PEFT_ADAPTER_TRANSPORT:-cpu_gather}"

# Harness (tools/adapter_runtime_compare/run_compare.py) branch wiring.
export ORBIT_COMPARE_RUNTIME_ROOT="${ORBIT_COMPARE_RUNTIME_ROOT:-${ORBIT_ROOT}}"
export ORBIT_COMPARE_RUNTIME_ENV="${ORBIT_COMPARE_RUNTIME_ENV:-${WORKSPACE_ROOT}/harness-env}"

export WANDB_MODE="${WANDB_MODE:-offline}"

# Checkpoints and data per rung. HF_CKPT / MEGATRON_LOAD / TRAIN_JSONL are the
# three variables every launcher requires; adapter_first_select_model sets them.
Q25_05B_HF="${Q25_05B_HF:-/lustre/fast/fast/zqiu/orbit_env_build/models/Qwen2.5-0.5B-Instruct}"
Q25_05B_TORCH_DIST="${Q25_05B_TORCH_DIST:-/lustre/fast/fast/zqiu/orbit_env_build/megatron_checkpoints/Qwen2.5-0.5B-Instruct-torchdist}"
Q25_05B_TRAIN_JSONL="${Q25_05B_TRAIN_JSONL:-/lustre/fast/fast/zqiu/orbit_env_build/data/gsm8k_agentic_train_64.jsonl}"

Q25_3B_HF="${Q25_3B_HF:-/fast/groups/ei-slm/hf_models/Qwen2.5-3B-Instruct}"
Q25_3B_TORCH_DIST="${Q25_3B_TORCH_DIST:-${WORKSPACE_ROOT}/ppo_critic_benchmark_models/Qwen2.5-3B-Instruct_torch_dist}"
Q25_3B_TRAIN_JSONL="${Q25_3B_TRAIN_JSONL:-/fast/groups/ei-slm/data/lora_regret/gsm8k_train.jsonl}"

Q3_4B_HF="${Q3_4B_HF:-/fast/groups/ei-slm/hf_models/Qwen3-4B-Instruct-2507}"
Q3_4B_TORCH_DIST="${Q3_4B_TORCH_DIST:-/fast/groups/ei-slm/hf_models/Qwen3-4B-Instruct-2507_torch_dist}"
Q3_4B_TRAIN_JSONL="${Q3_4B_TRAIN_JSONL:-${WORKSPACE_ROOT}/ppo_critic_benchmark_data/openr1_49990/train.jsonl}"

Q3_30B_HF="${Q3_30B_HF:-/fast/groups/ei-slm/hf_models/Qwen3-30B-A3B-Instruct-2507}"
Q3_30B_TORCH_DIST="${Q3_30B_TORCH_DIST:-/fast/groups/ei-slm/hf_models/Qwen3-30B-A3B-Instruct-2507_torch_dist}"
Q3_30B_TRAIN_JSONL="${Q3_30B_TRAIN_JSONL:-${WORKSPACE_ROOT}/ppo_critic_benchmark_data/openr1_49990/train.jsonl}"

adapter_first_select_model() {
    case "$1" in
        q25_05b) export HF_CKPT="$Q25_05B_HF" MEGATRON_LOAD="$Q25_05B_TORCH_DIST" TRAIN_JSONL="$Q25_05B_TRAIN_JSONL" ;;
        q25_3b)  export HF_CKPT="$Q25_3B_HF"  MEGATRON_LOAD="$Q25_3B_TORCH_DIST"  TRAIN_JSONL="$Q25_3B_TRAIN_JSONL" ;;
        q3_4b)   export HF_CKPT="$Q3_4B_HF"   MEGATRON_LOAD="$Q3_4B_TORCH_DIST"   TRAIN_JSONL="$Q3_4B_TRAIN_JSONL" ;;
        q3_30b)  export HF_CKPT="$Q3_30B_HF"  MEGATRON_LOAD="$Q3_30B_TORCH_DIST"  TRAIN_JSONL="$Q3_30B_TRAIN_JSONL" ;;
        *) echo "adapter_first_select_model: unknown rung '$1'" >&2; return 2 ;;
    esac
    for v in HF_CKPT MEGATRON_LOAD TRAIN_JSONL; do
        [[ -e "${!v}" ]] || { echo "FATAL: $v does not exist: ${!v}" >&2; return 1; }
    done
}

# run_harness CAMPAIGN ARGS... : one harness invocation, timed, exit code kept.
run_harness() {
    local campaign=$1; shift
    cd "${ORBIT_ROOT}"
    echo "[adapter_first] $(date -u +%FT%TZ) campaign=${campaign} env=$(command -v python)"
    local t0=$(date +%s) rc=0
    python tools/adapter_runtime_compare/run_compare.py run --branches runtime --campaign "${campaign}" "$@" || rc=$?
    echo "[adapter_first] $(date -u +%FT%TZ) campaign=${campaign} rc=${rc} elapsed=$(( $(date +%s) - t0 ))s"
    return "${rc}"
}
