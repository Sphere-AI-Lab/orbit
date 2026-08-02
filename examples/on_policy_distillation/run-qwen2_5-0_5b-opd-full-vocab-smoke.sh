#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 LoRA full-vocab OPD (--loss-type opd_jsd_loss,
# --teacher-score-mode full_vocab) against an external frozen sglang teacher.
# The teacher returns per-position hidden states; the trainer reconstructs the
# full teacher distribution via the teacher's LM head and trains on the GKD
# Eq.(1) generalized JSD. Pure distillation: no task reward, no advantages.
#
# Requires a running teacher server started with ALL THREE flags (each one
# missing breaks hidden-state scoring, fail-loud at the first scored sample):
#
#   python -m sglang.launch_server --model-path "${OPD_TEACHER_HF_CKPT}" \
#       --port 30001 --enable-return-hidden-states --disable-radix-cache \
#       --chunked-prefill-size -1
#
# Teacher MUST share the student's tokenizer/vocab (e.g. a larger Qwen2.5).
# The launcher preflights the endpoint unless SKIP_TEACHER_PREFLIGHT=1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=smoke_qwen25_05b_opd_full_vocab
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the student Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the student Megatron torch_dist checkpoint path}"
: "${OPD_TEACHER_URL:?set OPD_TEACHER_URL to the teacher sglang /generate endpoint}"
# Frozen teacher HF checkpoint: the SAME checkpoint the teacher server serves.
# The trainer loads its LM head (embed_tokens when tied) for logit reconstruction.
: "${OPD_TEACHER_HF_CKPT:?set OPD_TEACHER_HF_CKPT to the teacher Hugging Face checkpoint path}"
OPD_JSD_BETA="${OPD_JSD_BETA:-0.5}"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_opd_full_vocab_smoke"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Teacher preflight ===
# A tiny scoring request that fails fast if the server is missing any of the
# three required flags (a radix-cache or chunked-prefill misconfig would
# otherwise only surface mid-rollout, after Megatron init).
if ! is_true "${ORBIT_DRY_RUN_ARGV:-0}" && ! is_true "${SKIP_TEACHER_PREFLIGHT:-0}"; then
    OPD_TEACHER_URL="${OPD_TEACHER_URL}" python3 - <<'PY'
import json
import os
import urllib.request

url = os.environ["OPD_TEACHER_URL"]
payload = {
    "input_ids": [1, 2, 3, 4, 5],
    "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
    "return_hidden_states": True,
}
req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
# Bypass ambient http_proxy/https_proxy: cluster proxies intercept localhost URLs
# (orbit's own scoring client is immune via aiohttp trust_env=False).
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
body = json.loads(opener.open(req, timeout=60).read())
hidden = body.get("meta_info", {}).get("hidden_states") or []
assert len(hidden) == 1, (
    f"teacher preflight: expected 1 hidden_states batch entry, got {len(hidden)} -- "
    "start the teacher with --enable-return-hidden-states --disable-radix-cache "
    "--chunked-prefill-size -1"
)
inner = hidden[0]
positions = len(inner) if isinstance(inner, list) else "base64-buffer"
print(f"teacher preflight OK: hidden_states for {positions} positions "
      f"({'fast base64' if isinstance(inner, str) else 'legacy nested-JSON (slow; cherry-pick sglang da8625376)'})")
PY
fi

# === Resources ===
# actor=2 GPUs, rollout=2 GPUs; the teacher runs outside this allocation.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === ARGS arrays ===
COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}/actor"
    --save-interval 200
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

RL_ARGS=(
    # Estimator is inert: opd_jsd_loss disables the advantage/returns pipeline.
    --advantage-estimator grpo
    --loss-type opd_jsd_loss
    --teacher-score-mode full_vocab
    --teacher-hf-checkpoint "${OPD_TEACHER_HF_CKPT}"
    --opd-type sglang
    --opd-teacher-url "${OPD_TEACHER_URL}"
    --opd-jsd-beta "${OPD_JSD_BETA}"
    --opd-log-topk-overlap
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
)
if [[ -n "${OPD_JSD_POINTWISE_CLIP:-}" ]]; then
    RL_ARGS+=( --opd-jsd-pointwise-clip "${OPD_JSD_POINTWISE_CLIP}" )
fi

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-prompt-data math "${TEST_JSONL}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 1024
    --eval-top-k 1
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-offload-train
    --no-offload-train-async
    --no-offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=(
    --log-passrate
)

PEFT_ARGS=(
    --peft-method lora
    --lora-rank 16
    --target-modules all-linear
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
