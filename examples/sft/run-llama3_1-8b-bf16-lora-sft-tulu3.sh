#!/usr/bin/env bash
# Llama-3.1-8B base + LoRA SFT on Tulu3, for the LoRA-without-regret reproduction.
# See docs/superpowers/plans/2026-07-29-lora-without-regret-gap.md (gate G4).
#
#   Full fine-tuning:   PEFT_METHOD=none  LR=2.5e-5   GPUS_PER_NODE=4  (see P0)
#   LoRA r256 all:      PEFT_METHOD=lora  LORA_RANK=256 LR=2.5e-4
#   LoRA r256 attn:     PEFT_METHOD=lora  LORA_RANK=256 TARGET_MODULES=linear_qkv,linear_proj
#   LoRA r256 mlp:      PEFT_METHOD=lora  LORA_RANK=256 TARGET_MODULES=linear_fc1,linear_fc2
#
# Written fresh rather than ported: the source repo drove this from shared
# scripts/lib/{peft,rollout,train}.sh, which do not exist here -- this repo's
# launchers are standalone and spell out their own ARGS arrays (enforced by
# tests/test_sft_launch_scripts.py::test_sft_launchers_are_standalone). The
# knobs that lived in that shared lib therefore live here instead, and the
# "every existing launcher's command line stays byte-identical" constraint is
# satisfied trivially: no other launcher is touched.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_llama31_8b_bf16_lora_sft_tulu3}
WANDB_PROJECT=${WANDB_PROJECT:-lora-without-regret}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Data ===
# Produced by tools/lora_regret/prepare_data.py, whose SFT rows are
# {"prompt": [{"role", "content"}, ...]} -- hence --input-key prompt below.
DATA_DIR=${DATA_DIR:-/lustre/fast/fast/groups/ei-slm/data/lora_regret}
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_DIR}/tulu3_train.jsonl}
TEST_JSONL=${TEST_JSONL:-${DATA_DIR}/tulu3_test.jsonl}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a chat-format training jsonl path}"

# === Paths ===
HF_CKPT=${HF_CKPT:-/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B}
MEGATRON_LOAD=${MEGATRON_LOAD:-${ORBIT_ROOT}/checkpoints/Llama-3.1-8B_torch_dist}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Llama-3.1-8B_lora_sft_tulu3}

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-16}

# === Model args ===
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${ORBIT_ROOT}/orbit_plugins/model_args/llama3.1-8B-Instruct.sh}"
source "${MODEL_ARGS_FILE}"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === Rollout-shaped knobs, restating orbit/utils/arguments.py's own defaults ===
# LOSS_TYPE / LOSS_MASK_TYPE / APPLY_CHAT_TEMPLATE take this recipe's values
# rather than argparse's, because this launcher IS the recipe; they stay
# overridable so a sweep can vary them.
LOSS_TYPE=${LOSS_TYPE:-sft_loss}
# sft_rollout hands sample.prompt straight to MultiTurnLossMaskGenerator, which
# wants the raw messages list -- NOT a rendered chat string. Rendering here
# would make the mask generator tokenize an already-templated string.
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}
LOSS_MASK_TYPE=${LOSS_MASK_TYPE:-llama3}
# Deliberately the no-colon form. SFT rows are {"prompt": [...]} with no label
# field -- the targets are the assistant turns, located by the loss mask, not a
# label column. ${LABEL_KEY:-label} would also fire on a set-but-empty
# LABEL_KEY and silently re-default it to "label", which then crashes the
# loader looking for a column that does not exist.
LABEL_KEY=${LABEL_KEY-}

# === Reproducibility ===
# SEED restates argparse's own default (reset_arg(parser, "--seed", default=1234)).
SEED=${SEED:-1234}
# Tie data order to SEED so a seed sweep varies training dynamics, not just
# init. This tie belongs HERE and not in any shared default: --rollout-seed
# also seeds SGLang generation, so moving its 42 default would silently change
# every other RL run in the repo. HuggingFace's Trainer -- the oracle the
# step-0 NLL gate compares against -- uses one seed for both, so tying them
# keeps a measured seed-noise sigma comparable to theirs.
ROLLOUT_SEED=${ROLLOUT_SEED:-${SEED}}

# === Held-out NLL eval ===
# Forward-only through the training model, so it needs no rollout engine and is
# independent of --eval-interval.
EVAL_NLL_DATA=${EVAL_NLL_DATA-${TEST_JSONL}}
EVAL_NLL_INTERVAL=${EVAL_NLL_INTERVAL:-10}
EVAL_NLL_MICRO_BATCH_SIZE=${EVAL_NLL_MICRO_BATCH_SIZE-}

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-1000}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

# --prompt-data is NOT optional even though nothing generates: train.py calls
# create_rollout_manager() unconditionally (train.py's "create rollout manager"
# startup phase), and RolloutManager.__init__ loads the dataset. A pure-SFT run
# is not exempt from the loader's contract -- it just never generates from it.
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --rollout-shuffle
    --rollout-function-path orbit.rollout.sft_rollout.generate_rollout
    --loss-mask-type "${LOSS_MASK_TYPE}"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-seed "${ROLLOUT_SEED}"
    # Llama-3.1-8B *base* ships no chat_template, so apply_chat_template would
    # raise and MultiTurnLossMaskGenerator could not even be constructed. This
    # is consumed by load_tokenizer() in the actor and in the NLL eval.
    --chat-template-path "${ORBIT_ROOT}/orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja"
)

if is_true "${APPLY_CHAT_TEMPLATE}"; then
    ROLLOUT_ARGS+=( --apply-chat-template )
elif ! is_false "${APPLY_CHAT_TEMPLATE}"; then
    echo "Invalid APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE}; expected one of: 1 true yes y on 0 false no n off" >&2
    exit 2
fi

if [[ -n "${LABEL_KEY}" ]]; then
    ROLLOUT_ARGS+=( --label-key "${LABEL_KEY}" )
fi

# === Optimizer: constant LR, no warmup, no cooldown -- the blog's protocol ===
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-2.5e-4}"
    --lr-decay-style "${LR_DECAY_STYLE:-constant}"
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.999}"
)

RL_ARGS=()

LOSS_ARGS=(
    --training-mode sft
    --loss-type "${LOSS_TYPE}"
    --disable-compute-advantages-and-returns
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --sequence-parallel
)

# Generation-based eval is meaningless for this study; held-out NLL replaces it.
EVAL_ARGS=()
if [[ -n "${EVAL_NLL_DATA}" ]]; then
    EVAL_ARGS+=(
        --eval-nll-data "${EVAL_NLL_DATA}"
        --eval-nll-interval "${EVAL_NLL_INTERVAL}"
    )
    if [[ -n "${EVAL_NLL_MICRO_BATCH_SIZE}" ]]; then
        EVAL_ARGS+=( --eval-nll-micro-batch-size "${EVAL_NLL_MICRO_BATCH_SIZE}" )
    fi
fi

SGLANG_ARGS=()

MISC_ARGS=(
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --te-rng-tracker
    # NOTE: no --cuda-graph-scope full_iteration here, unlike the sibling
    # launchers in this directory. Megatron asserts
    # `not args.check_for_nan_in_loss_and_grad` whenever cuda_graph_impl=local
    # and full_iteration is in scope (megatron/training/arguments.py, "should be
    # set with full_iteration CUDA graph"), and the only spelling Megatron
    # offers is the negative --no-check-for-nan-in-loss-and-grad -- there is no
    # positive flag to turn the check back on. For an LR sweep that trade is
    # backwards: a silently-NaN arm reads as a bad learning rate and would
    # corrupt exactly the conclusion this study exists to draw. So the CUDA
    # graph opt-in is dropped and the NaN check is left at its default (on).
)

if ! is_true "${SFT_GRADIENT_ACCUMULATION_FUSION:-0}"; then
    MISC_ARGS+=( --no-gradient-accumulation-fusion )
fi

PEFT_METHOD=${PEFT_METHOD:-lora}
PEFT_ARGS=()
if [[ "${PEFT_METHOD}" != "none" ]]; then
    PEFT_ARGS=(
        --peft-method "${PEFT_METHOD}"
        --peft-variant standard
        --lora-rank "${LORA_RANK:-256}"
        --lora-alpha "${LORA_ALPHA:-32}"
        --lora-dropout "${LORA_DROPOUT:-0.0}"
        # PEFT-compatible init. Do NOT leave this at Orbit's xavier default:
        # kaiming_uniform_(a=sqrt(5)) and xavier_normal_ differ by ~2.4x in
        # std, which shifts the measured optimal learning rate.
        --lora-a-init-method "${LORA_A_INIT_METHOD:-kaiming}"
        --target-modules "${TARGET_MODULES:-linear_qkv,linear_proj,linear_fc1,linear_fc2}"
    )
fi

DEBUG_ARGS=()
if [[ -n "${SFT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of a flat flag string
    MISC_ARGS+=( ${SFT_EXTRA_ARGS} )
fi

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
