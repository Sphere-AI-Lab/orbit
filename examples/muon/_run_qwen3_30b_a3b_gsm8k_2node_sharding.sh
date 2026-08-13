#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
MODEL_ARGS_NUM_LAYERS=48
source "${REPO_ROOT}/scripts/models/qwen3-30B-A3B.sh"

TP_SIZE="${1:-}"
PP_SIZE="${2:-}"
CP_SIZE="${3:-}"
EP_SIZE="${4:-}"
ETP_SIZE="${5:-}"
NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${HOME}/models/Qwen3-30B-A3B}"
REF_LOAD="${REF_LOAD:-${HOME}/models/Qwen3-30B-A3B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${HOME}/datasets/gsm8k/train.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/miles-runs/muon-two-node-sharding-smoke}"
TOPOLOGY_ID="tp${TP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}-etp${ETP_SIZE}"
RUN_ID="${RUN_ID:-qwen3-30b-a3b-gsm8k-2node-${TOPOLOGY_ID}-muon-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
LOG_TO_STDOUT_ONLY="${LOG_TO_STDOUT_ONLY:-0}"
MILES_CONDA_ENV="${MILES_CONDA_ENV:-miles}"
MILES_CONDA_SH="${MILES_CONDA_SH:-/data/shared/conda/miniconda3/etc/profile.d/conda.sh}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-120}"

die() {
   printf 'error: %s\n' "$*" >&2
   exit 2
}

require_positive_integer() {
   local name="$1"
   local value="$2"
   [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got '${value}'"
}

require_nonnegative_decimal() {
   local name="$1"
   local value="$2"
   [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
      die "${name} must be a nonnegative decimal; got '${value}'"
}

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1; got '${DRY_RUN}'"
[[ "${LOG_TO_STDOUT_ONLY}" == "0" || "${LOG_TO_STDOUT_ONLY}" == "1" ]] || \
   die "LOG_TO_STDOUT_ONLY must be 0 or 1; got '${LOG_TO_STDOUT_ONLY}'"
require_positive_integer TP_SIZE "${TP_SIZE}"
require_positive_integer PP_SIZE "${PP_SIZE}"
require_positive_integer CP_SIZE "${CP_SIZE}"
require_positive_integer EP_SIZE "${EP_SIZE}"
require_positive_integer ETP_SIZE "${ETP_SIZE}"
require_positive_integer NUM_ROLLOUT "${NUM_ROLLOUT}"
require_nonnegative_decimal ENTROPY_COEF "${ENTROPY_COEF}"
require_positive_integer RAY_PORT "${RAY_PORT}"
require_positive_integer RAY_DASHBOARD_PORT "${RAY_DASHBOARD_PORT}"
require_positive_integer RAY_START_TIMEOUT "${RAY_START_TIMEOUT}"

case "${TP_SIZE}:${PP_SIZE}:${CP_SIZE}:${EP_SIZE}:${ETP_SIZE}" in
   2:1:1:8:1|2:2:1:8:1|2:1:2:8:1|2:1:1:16:1) ;;
   *)
      die "unsupported two-node topology '${TP_SIZE}:${PP_SIZE}:${CP_SIZE}:${EP_SIZE}:${ETP_SIZE}'; expected one of 2:1:1:8:1, 2:2:1:8:1, 2:1:2:8:1, 2:1:1:16:1"
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

OPTIMIZER_ARGS=(
   --optimizer dist_muon
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef "${ENTROPY_COEF}"
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-routing-replay
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
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
   --actor-num-nodes "${NUM_NODES}"
   --actor-num-gpus-per-node "${NUM_GPUS_PER_NODE}"
   --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
   --colocate
   --calculate-per-token-loss
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

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
   die "runtime requires an existing two-node Slurm allocation"
command -v scontrol >/dev/null 2>&1 || die "scontrol is not available"
command -v srun >/dev/null 2>&1 || die "srun is not available"

SLURM_NODES=()
while IFS= read -r node; do
   [[ -n "${node}" ]] && SLURM_NODES+=("${node}")
done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#SLURM_NODES[@]} == NUM_NODES )) || \
   die "expected exactly two allocated nodes; got ${#SLURM_NODES[@]}"

for node in "${SLURM_NODES[@]}"; do
   GPU_COUNT="$(srun --jobid="${SLURM_JOB_ID}" --overlap --mem=0 \
      -N1 -n1 -w "${node}" --gres=gpu:8 \
      nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
   [[ "${GPU_COUNT}" == "${NUM_GPUS_PER_NODE}" ]] || \
      die "node ${node} exposes ${GPU_COUNT} GPUs; expected ${NUM_GPUS_PER_NODE}"
done

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
command -v ray >/dev/null 2>&1 || die "ray is not available in PATH"
command -v setsid >/dev/null 2>&1 || die "setsid is not available in PATH"
if ! python3 -c \
   'from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer, get_muon_scale_factor; from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp' \
   >/dev/null 2>&1; then
   die "Muon requires Emerging Optimizers v0.1.0; install git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0 into ${MILES_CONDA_ENV}"
fi

PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${PY_SITE}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/thirdparty/Megatron-LM}"
BRIDGE_PATH="${BRIDGE_PATH:-${REPO_ROOT}/thirdparty/Megatron-Bridge/src}"
SGLANG_PATH="${SGLANG_PATH:-${REPO_ROOT}/thirdparty/sglang/python}"
[[ -d "${MEGATRON_PATH}/megatron" ]] || die "Megatron-LM is not initialized at ${MEGATRON_PATH}"
[[ -d "${BRIDGE_PATH}/megatron/bridge" ]] || die "Megatron-Bridge is not initialized at ${BRIDGE_PATH}"
[[ -d "${SGLANG_PATH}/sglang" ]] || die "SGLang is not initialized at ${SGLANG_PATH}"

HEAD_NODE="${SLURM_NODES[0]}"
WORKER_NODE="${SLURM_NODES[1]}"
HEAD_IP="$(srun --jobid="${SLURM_JOB_ID}" --overlap --mem=0 \
   -N1 -n1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}' || true)"
[[ -n "${HEAD_IP}" ]] || die "failed to resolve Ray head address on ${HEAD_NODE}"
if ray status --address="${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
   die "a Ray runtime is already active at ${HEAD_IP}:${RAY_PORT}"
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir "${RUN_DIR}" 2>/dev/null || die "run directory already exists: ${RUN_DIR}"
if [[ "${LOG_TO_STDOUT_ONLY}" == "0" ]]; then
   exec > >(tee -a "${RUN_DIR}/train.log") 2>&1
fi
printf 'RUN_OPTIMIZER=muon\n'
printf 'RUN_MEGATRON_OPTIMIZER=dist_muon\n'
printf 'RUN_ENTROPY_COEF=%s\n' "${ENTROPY_COEF}"
printf 'RUN_TOPOLOGY=%s\n' "${TOPOLOGY_ID}"
printf 'RUN_SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID}"
printf 'RUN_NODES=%s,%s\n' "${HEAD_NODE}" "${WORKER_NODE}"
printf 'RUN_TOTAL_GPUS=%s\n' "$((NUM_NODES * NUM_GPUS_PER_NODE))"

RAY_PYTHONPATH="${MEGATRON_PATH}:${BRIDGE_PATH}:${SGLANG_PATH}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUNTIME_ENV_JSON="$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' "${RAY_PYTHONPATH}")"
RAY_HEAD_LOG="${RUN_DIR}/ray-head.log"
RAY_WORKER_LOG="${RUN_DIR}/ray-worker.log"
CLEANUP_LOG="${RUN_DIR}/cleanup.log"
OWNED_STEP_PIDS=()

cleanup() {
   local pid attempt live

   printf 'cleanup start pids=%s\n' "${OWNED_STEP_PIDS[*]:-}" >>"${CLEANUP_LOG}"
   for pid in "${OWNED_STEP_PIDS[@]}"; do
      if kill -0 -- "-${pid}" >/dev/null 2>&1; then
         printf 'term pgid=%s\n' "${pid}" >>"${CLEANUP_LOG}"
         kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
      fi
   done
   for ((attempt = 0; attempt < 10; attempt++)); do
      live=0
      for pid in "${OWNED_STEP_PIDS[@]}"; do
         kill -0 -- "-${pid}" >/dev/null 2>&1 && live=1
      done
      printf 'poll attempt=%s live=%s\n' "${attempt}" "${live}" >>"${CLEANUP_LOG}"
      (( live == 0 )) && break
      sleep 1
   done
   for pid in "${OWNED_STEP_PIDS[@]}"; do
      if kill -0 -- "-${pid}" >/dev/null 2>&1; then
         printf 'kill pgid=%s\n' "${pid}" >>"${CLEANUP_LOG}"
         kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
      fi
   done
   printf 'cleanup complete\n' >>"${CLEANUP_LOG}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid srun --jobid="${SLURM_JOB_ID}" --overlap --mem=0 \
   -N1 -n1 -w "${HEAD_NODE}" --gres=gpu:8 \
   bash -lc '
      set -euo pipefail
      ulimit -Sl "$(ulimit -Hl)" 2>/dev/null || true
      source "$1"
      conda activate "$2"
      export LD_LIBRARY_PATH="$3/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
      export PYTHONPATH="$4"
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      exec ray start --head --block --node-ip-address "$5" --port "$6" --dashboard-host 0.0.0.0 --dashboard-port "$7" --num-gpus 8 --disable-usage-stats
   ' bash "${MILES_CONDA_SH}" "${MILES_CONDA_ENV}" "${PY_SITE}" "${RAY_PYTHONPATH}" \
      "${HEAD_IP}" "${RAY_PORT}" "${RAY_DASHBOARD_PORT}" \
   >>"${RAY_HEAD_LOG}" 2>&1 &
OWNED_STEP_PIDS+=("$!")

HEAD_READY=0
for ((attempt = 0; attempt < RAY_START_TIMEOUT; attempt++)); do
   kill -0 "${OWNED_STEP_PIDS[0]}" >/dev/null 2>&1 || \
      die "Ray head Slurm step exited before readiness; see ${RAY_HEAD_LOG}"
   if ray status --address="${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
      HEAD_READY=1
      break
   fi
   sleep 1
done
(( HEAD_READY == 1 )) || die "Ray head failed to start within ${RAY_START_TIMEOUT} seconds"

setsid srun --jobid="${SLURM_JOB_ID}" --overlap --mem=0 \
   -N1 -n1 -w "${WORKER_NODE}" --gres=gpu:8 \
   bash -lc '
      set -euo pipefail
      ulimit -Sl "$(ulimit -Hl)" 2>/dev/null || true
      source "$1"
      conda activate "$2"
      export LD_LIBRARY_PATH="$3/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
      export PYTHONPATH="$4"
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      exec ray start --block --address "$5:$6" --num-gpus 8 --disable-usage-stats
   ' bash "${MILES_CONDA_SH}" "${MILES_CONDA_ENV}" "${PY_SITE}" "${RAY_PYTHONPATH}" \
      "${HEAD_IP}" "${RAY_PORT}" \
   >>"${RAY_WORKER_LOG}" 2>&1 &
OWNED_STEP_PIDS+=("$!")

RAY_READY=0
for ((attempt = 0; attempt < RAY_START_TIMEOUT; attempt++)); do
   for pid in "${OWNED_STEP_PIDS[@]}"; do
      kill -0 "${pid}" >/dev/null 2>&1 || \
         die "owned Ray Slurm step exited before cluster readiness"
   done
   RAY_GPU_COUNT="$(srun --jobid="${SLURM_JOB_ID}" --overlap --mem=0 \
      -N1 -n1 -w "${HEAD_NODE}" \
      bash -lc '
         set -euo pipefail
         source "$1"
         conda activate "$2"
         timeout 10s python3 -c '\''import ray, sys; ray.init(address=sys.argv[1], logging_level="ERROR"); print(int(ray.cluster_resources().get("GPU", 0))); ray.shutdown()'\'' "$3"
      ' bash "${MILES_CONDA_SH}" "${MILES_CONDA_ENV}" "${HEAD_IP}:${RAY_PORT}" \
      2>/dev/null | tail -n1 || true)"
   if [[ "${RAY_GPU_COUNT}" == "$((NUM_NODES * NUM_GPUS_PER_NODE))" ]]; then
      RAY_READY=1
      break
   fi
   sleep 1
done
(( RAY_READY == 1 )) || \
   die "Ray failed to register $((NUM_NODES * NUM_GPUS_PER_NODE)) GPUs within ${RAY_START_TIMEOUT} seconds"

set +e
ray job submit --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" -- "${TRAIN_COMMAND[@]}"
JOB_STATUS="$?"
set -e
exit "${JOB_STATUS}"
