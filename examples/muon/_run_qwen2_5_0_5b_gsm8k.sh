#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
source "${REPO_ROOT}/scripts/models/qwen2.5-0.5B.sh"

OPTIMIZER="${1:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${HOME}/models/Qwen2.5-0.5B-Instruct}"
REF_LOAD="${REF_LOAD:-${HOME}/models/Qwen2.5-0.5B-Instruct_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${HOME}/datasets/gsm8k/train.parquet}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/miles-runs/muon-smoke}"
RUN_ID="${RUN_ID:-qwen2.5-0.5b-gsm8k-${OPTIMIZER}-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
LOG_TO_STDOUT_ONLY="${LOG_TO_STDOUT_ONLY:-0}"
MILES_CONDA_ENV="${MILES_CONDA_ENV:-miles}"
MILES_CONDA_SH="${MILES_CONDA_SH:-/data/shared/conda/miniconda3/etc/profile.d/conda.sh}"

die() {
   printf 'error: %s\n' "$*" >&2
   exit 2
}

require_positive_integer() {
   local name="$1"
   local value="$2"
   [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got '${value}'"
}

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1; got '${DRY_RUN}'"
[[ "${LOG_TO_STDOUT_ONLY}" == "0" || "${LOG_TO_STDOUT_ONLY}" == "1" ]] || \
   die "LOG_TO_STDOUT_ONLY must be 0 or 1; got '${LOG_TO_STDOUT_ONLY}'"
require_positive_integer NUM_GPUS "${NUM_GPUS}"
require_positive_integer NUM_ROLLOUT "${NUM_ROLLOUT}"

case "${OPTIMIZER}" in
   adam)
      OPTIMIZER_ARGS=(
         --optimizer adam
         --lr 1e-6
         --lr-decay-style constant
         --weight-decay 0.1
         --adam-beta1 0.9
         --adam-beta2 0.98
      )
      ;;
   muon)
      OPTIMIZER_ARGS=(
         --optimizer muon
         --lr 1e-6
         --lr-decay-style constant
         --weight-decay 0.1
      )
      ;;
   *)
      die "unknown optimizer '${OPTIMIZER}'; expected 'adam' or 'muon'"
      ;;
esac

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 8
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

TRAIN_COMMAND=(
   python3 train.py
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   --colocate
   --calculate-per-token-loss
   --use-miles-router
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
   printf 'TRAIN_COMMAND='
   printf ' %q' "${TRAIN_COMMAND[@]}"
   printf '\n'
   exit 0
fi

[[ -d "${HF_CHECKPOINT}" ]] || die "HF_CHECKPOINT directory does not exist: ${HF_CHECKPOINT}"
[[ -d "${REF_LOAD}" ]] || die "REF_LOAD directory does not exist: ${REF_LOAD}"
[[ -f "${PROMPT_DATA}" ]] || die "PROMPT_DATA file does not exist: ${PROMPT_DATA}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"

[[ -f "${MILES_CONDA_SH}" ]] || die "MILES_CONDA_SH file does not exist: ${MILES_CONDA_SH}"
source "${MILES_CONDA_SH}"
conda activate "${MILES_CONDA_ENV}" || die "failed to activate Conda environment: ${MILES_CONDA_ENV}"
[[ "${CONDA_DEFAULT_ENV:-}" == "${MILES_CONDA_ENV}" ]] || \
   die "Conda activated '${CONDA_DEFAULT_ENV:-unknown}', expected '${MILES_CONDA_ENV}'"

command -v python3 >/dev/null 2>&1 || die "python3 is not available in PATH"
if [[ "${OPTIMIZER}" == "muon" ]] && ! python3 -c \
   'from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer, get_muon_scale_factor; from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp' \
   >/dev/null 2>&1; then
   die "Muon requires Emerging Optimizers v0.1.0; install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0 into ${MILES_CONDA_ENV}"
fi
command -v ray >/dev/null 2>&1 || die "ray is not available in PATH"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available in PATH"
command -v setsid >/dev/null 2>&1 || die "setsid is not available in PATH"

PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${PY_SITE}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/thirdparty/Megatron-LM}"
BRIDGE_PATH="${BRIDGE_PATH:-${REPO_ROOT}/thirdparty/Megatron-Bridge/src}"
SGLANG_PATH="${SGLANG_PATH:-${REPO_ROOT}/thirdparty/sglang/python}"
[[ -d "${MEGATRON_PATH}/megatron" ]] || \
   die "Megatron-LM is not initialized at ${MEGATRON_PATH}"
[[ -d "${BRIDGE_PATH}/megatron/bridge" ]] || \
   die "Megatron-Bridge is not initialized at ${BRIDGE_PATH}"
[[ -d "${SGLANG_PATH}/sglang" ]] || \
   die "SGLang is not initialized at ${SGLANG_PATH}"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
(( GPU_COUNT >= NUM_GPUS )) || die "requested ${NUM_GPUS} GPUs but nvidia-smi reports ${GPU_COUNT}"

if [[ "${CUDA_VISIBLE_DEVICES+x}" != "x" ]]; then
   VISIBLE_GPU_IDS=()
   for ((GPU_INDEX = 0; GPU_INDEX < NUM_GPUS; GPU_INDEX++)); do
      VISIBLE_GPU_IDS+=("${GPU_INDEX}")
   done
   CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${VISIBLE_GPU_IDS[*]}")"
else
   [[ -n "${CUDA_VISIBLE_DEVICES}" ]] || \
      die "CUDA_VISIBLE_DEVICES is set but empty; expected ${NUM_GPUS} GPU IDs"
   IFS=',' read -r -a VISIBLE_GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
   (( ${#VISIBLE_GPU_IDS[@]} >= NUM_GPUS )) || die \
      "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_GPU_IDS[@]} GPUs but NUM_GPUS=${NUM_GPUS}"
fi
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}"
mkdir "${RUN_DIR}" 2>/dev/null || die "run directory already exists: ${RUN_DIR}"
LOG_PATH="${RUN_DIR}/train.log"
if [[ "${LOG_TO_STDOUT_ONLY}" == "0" ]]; then
   exec > >(tee -a "${LOG_PATH}") 2>&1
fi
printf 'RUN_OPTIMIZER=%s\n' "${OPTIMIZER}"

RAY_PYTHONPATH="${MEGATRON_PATH}:${BRIDGE_PATH}:${SGLANG_PATH}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_ENV_JSON="$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' "${RAY_PYTHONPATH}")"

if ray status --address=127.0.0.1:6379 >/dev/null 2>&1; then
   die "a Ray runtime is already active at 127.0.0.1:6379"
fi

RAY_LAUNCHER_PID=""
cleanup() {
   local attempt

   [[ -n "${RAY_LAUNCHER_PID}" ]] || return
   if kill -0 -- "-${RAY_LAUNCHER_PID}" >/dev/null 2>&1; then
      kill -TERM -- "-${RAY_LAUNCHER_PID}" >/dev/null 2>&1 || true
      for ((attempt = 0; attempt < 10; attempt++)); do
         kill -0 -- "-${RAY_LAUNCHER_PID}" >/dev/null 2>&1 || break
         sleep 1
      done
      if kill -0 -- "-${RAY_LAUNCHER_PID}" >/dev/null 2>&1; then
         kill -KILL -- "-${RAY_LAUNCHER_PID}" >/dev/null 2>&1 || true
      fi
   fi
   wait "${RAY_LAUNCHER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${REPO_ROOT}"
setsid ray start --head --block --node-ip-address 127.0.0.1 \
   --num-gpus "${NUM_GPUS}" --disable-usage-stats &
RAY_LAUNCHER_PID="$!"

sleep 1
RAY_READY=0
for ((RAY_START_ATTEMPT = 0; RAY_START_ATTEMPT < 60; RAY_START_ATTEMPT++)); do
   if ! kill -0 "${RAY_LAUNCHER_PID}" >/dev/null 2>&1; then
      set +e
      wait "${RAY_LAUNCHER_PID}"
      RAY_START_STATUS="$?"
      set -e
      die "Ray failed to start; launcher exited with status ${RAY_START_STATUS}"
   fi
   if ray status --address=127.0.0.1:6379 >/dev/null 2>&1; then
      RAY_READY=1
      break
   fi
   sleep 1
done
(( RAY_READY == 1 )) || die "Ray failed to start within 60 seconds"

set +e
ray job submit --address=http://127.0.0.1:8265 \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_COMMAND[@]}"
JOB_STATUS="$?"
set -e
exit "${JOB_STATUS}"
