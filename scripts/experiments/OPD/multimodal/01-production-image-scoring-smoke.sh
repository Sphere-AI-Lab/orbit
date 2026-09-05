#!/bin/bash
#
# Production multimodal OPD scoring gate. This launches the exact TP=8 teacher
# used by milestone 00, then calls reward_func + post_process_rewards directly.
# It does not create trainer/rollout actors, save a checkpoint, or log to W&B.
#
# Submit:
#   bash scripts/slurm/submit.sh OPD/multimodal/01-production-image-scoring-smoke

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}

EXPERIMENT_NODES=1
EXPERIMENT_TIME=02:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
    "chenhegu/geo3k_imgurl"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
HF_TORCHDIST_DIR=""
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/train.parquet"

OPD_TEACHER_MODEL_DIR=${OPD_TEACHER_MODEL_DIR:-"$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"}
OPD_TEACHER_PORT=${OPD_TEACHER_PORT:-13141}
OPD_TEACHER_TP=${OPD_TEACHER_TP:-8}
OPD_TEACHER_GPUS=${OPD_TEACHER_GPUS:-0,1,2,3,4,5,6,7}
OPD_TEACHER_MEM_FRACTION=${OPD_TEACHER_MEM_FRACTION:-0.8}
OPD_TEACHER_LAUNCH=${OPD_TEACHER_LAUNCH:-1}
OPD_TEACHER_URL=${OPD_TEACHER_URL:-"http://127.0.0.1:${OPD_TEACHER_PORT}"}
OPD_TEACHER_EXTRA_ARGS=${OPD_TEACHER_EXTRA_ARGS:-}

case "$OPD_TEACHER_LAUNCH" in
   1|true)
      if [[ ! -f "$OPD_TEACHER_MODEL_DIR/config.json" ]]; then
         echo "FATAL: OPD VLM teacher not found at $OPD_TEACHER_MODEL_DIR" >&2
         echo "  hf download Qwen/Qwen3-VL-30B-A3B-Thinking --local-dir $OPD_TEACHER_MODEL_DIR" >&2
         return 1 2>/dev/null || exit 1
      fi
      export ENVPACK_LOCAL_SERVER_CMD="TRITON_CACHE_DIR=/tmp/triton_${USER:-unknown}/opd_vlm_teacher \
         CUDA_VISIBLE_DEVICES=$OPD_TEACHER_GPUS python3 -m sglang.launch_server \
         --model-path $OPD_TEACHER_MODEL_DIR \
         --host 0.0.0.0 \
         --port $OPD_TEACHER_PORT \
         --tp $OPD_TEACHER_TP \
         --chunked-prefill-size 8192 \
         --mem-fraction-static $OPD_TEACHER_MEM_FRACTION \
         $OPD_TEACHER_EXTRA_ARGS"
      export ENVPACK_LOCAL_SERVER_HEALTH="$OPD_TEACHER_URL/health_generate"
      ;;
   0|false)
      unset ENVPACK_LOCAL_SERVER_CMD ENVPACK_LOCAL_SERVER_HEALTH
      ;;
   *)
      echo "FATAL: OPD_TEACHER_LAUNCH must be 0/1 or false/true, got '$OPD_TEACHER_LAUNCH'" >&2
      return 1 2>/dev/null || exit 1
      ;;
esac

export ENVPACK_SERVER_WAIT_TIMEOUT=${ENVPACK_SERVER_WAIT_TIMEOUT:-1800}
export ORBIT_TRAIN_ENTRY="scripts/experiments/OPD/multimodal/production_image_scoring_smoke.py"

# launch_orbit appends --wandb-run-id to every entrypoint. The probe accepts and
# ignores it; no W&B run is created.
ORBIT_ARGS=(
   --teacher-url "$OPD_TEACHER_URL"
   --student-model-dir "$HF_MODEL_DIR"
   --dataset "$HF_TRAIN_DATA"
   --input-key problem
   --label-key answer
   --image-size "${OPD_SMOKE_IMAGE_SIZE:-512}"
   --top-k "${OPD_SMOKE_TOP_K:-2}"
   --sglang-mm-exact-scoring-suffix
   --timeout "${OPD_SCORING_TIMEOUT:-600}"
   --image-sensitivity-tolerance "${OPD_SMOKE_IMAGE_TOLERANCE:-1e-6}"
   --cache-consistency-tolerance "${OPD_SMOKE_CACHE_TOLERANCE:-1e-5}"
)
