#!/usr/bin/env bash
# Llama-3.1-8B base, policy-gradient RL on MATH + GSM8K, for the
# LoRA-without-regret reproduction's E4 (decides claim C5: "LoRA matches FullFT
# under policy gradient even at rank 1, with a wider band of performant LRs").
#
# Prerequisite P5 of docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md.
# Runbook with every arm's command line:
# docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md
#
#   FullFT:     PEFT_METHOD=none  LR=1e-6   GPUS_PER_NODE=8
#   LoRA r256:  PEFT_METHOD=lora  LORA_RANK=256  LR=1e-5
#   LoRA r16:   PEFT_METHOD=lora  LORA_RANK=16
#   LoRA r1:    PEFT_METHOD=lora  LORA_RANK=1    <- C5's whole point; never drop
#   OFT:        PEFT_METHOD=oft   OFT_BLOCK_SIZE=<matched_oft_block_size(...)>
#
# Standalone by this repo's contract: no shared arg library, the ARGS arrays are
# spelled out here, and scripts/lib/launcher.sh assembles the command line.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_llama31_8b_bf16_rl_math_gsm8k}
WANDB_PROJECT=${WANDB_PROJECT:-lora-without-regret}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
# train.py, never train_async.py. The async loop overlaps next-rollout
# generation with current-rollout training, so "the policy at the moment of
# measurement" is undefined -- which is why train_async.py refuses
# --eval-nll-data outright. E4 compares arms by validation accuracy at matched
# step counts, so it needs the synchronous loop.
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Data ===
# RL rows from tools/lora_regret/prepare_data.py are {"prompt": str, "label":
# str, "metadata": {...}} -- a *string* prompt, unlike the SFT launcher's
# message list, because the RL path renders the prompt itself.
DATA_DIR=${DATA_DIR:-/lustre/fast/fast/groups/ei-slm/data/lora_regret}
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_DIR}/math_gsm8k_train.jsonl}
MATH_TEST_JSONL=${MATH_TEST_JSONL:-${DATA_DIR}/math_test.jsonl}
GSM8K_TEST_JSONL=${GSM8K_TEST_JSONL:-${DATA_DIR}/gsm8k_test.jsonl}
: "${TRAIN_JSONL:?set TRAIN_JSONL to an RL-format jsonl (prompt string + label)}"

# === Paths ===
HF_CKPT=${HF_CKPT:-/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B}
MEGATRON_LOAD=${MEGATRON_LOAD:-/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints/Llama-3.1-8B_torch_dist}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Llama-3.1-8B_rl_math_gsm8k}

# === Resources ===
# 8 GPUs by default: FullFT needs >=4 for optimizer state alone (P0) and the
# rollout engine wants its own share on top. LoRA arms run fine on fewer -- see
# the runbook's per-arm table.
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-16}

# === Model args ===
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${ORBIT_ROOT}/orbit_plugins/model_args/llama3.1-8B-Instruct.sh}"
source "${MODEL_ARGS_FILE}"   # provides MODEL_ARGS=(...)

# === Training schedule ===
NUM_ROLLOUT=${NUM_ROLLOUT:-500}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
# 32 samples per problem: the post's setting, and what makes the GRPO baseline a
# per-problem mean rather than noise.
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

# === Reproducibility ===
SEED=${SEED:-1234}
# Tied here, not in any shared default: --rollout-seed also seeds SGLang
# generation, so moving its 42 default would silently change other RL runs in
# this repo. Tying it means a seed sweep varies problem order and sampling
# together, which is what an RL seed replicate should vary.
ROLLOUT_SEED=${ROLLOUT_SEED:-${SEED}}

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

# `SAVE_INTERVAL=` (explicitly EMPTY) turns checkpointing off entirely.
#
# Not SAVE_INTERVAL=999999. `should_run_periodic_action` short-circuits on
# `interval is None` and only then checks the final rollout, so any non-None
# interval still writes one checkpoint at the end -- 616 s and 15 GB for a
# FullFT arm, measured. Only omitting `--save-interval` reaches the None branch,
# and a flag cannot be omitted by giving it a value.
#
# `${SAVE_INTERVAL-50}` uses `-`, not `:-`, so an explicit empty value survives
# instead of falling back to 50.
SAVE_INTERVAL=${SAVE_INTERVAL-50}
CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --megatron-to-hf-mode bridge
)
if [[ -n "${SAVE_INTERVAL}" ]]; then
    CKPT_ARGS+=(
        --save "${SAVE_DIR}"
        --save-interval "${SAVE_INTERVAL}"
        --no-save-optim
        --no-save-rng
    )
else
    echo "SAVE_INTERVAL is empty: no checkpoints will be written." >&2
    echo "  The ledger and wandb carry every number the analysis reads; what is" >&2
    echo "  lost is the ability to re-evaluate or reuse a trained policy later." >&2
fi

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --label-key label
    # NO --apply-chat-template. The policy is Llama-3.1-8B *base*, and the
    # pinned template is Llama-3.1 *Instruct*'s: it wraps every problem in
    # <|start_header_id|>system/user/assistant<|end_header_id|> turns. Those
    # tokens are in the base vocabulary but the base model was never trained to
    # condition on them as delimiters, and the 2026-07-31 probe recorded the
    # result -- "Back to Index", runs of private-use codepoints, reward 0 on
    # every one of 1,024 rollouts per step, so zero advantage and no gradient
    # for any arm at any learning rate.
    #
    # prepare_data.py's `render_prompt` now writes the Problem:/Solution: frame
    # into the jsonl instead, so the prompt string here IS the text the policy
    # sees, byte for byte, and it is identical across FullFT and every rank.
    --rollout-shuffle
    # math: grade_answer_verl extracts the final \boxed{...} from the response
    # itself and grades it against the bare answer string prepare_data.py wrote
    # (extract_boxed for MATH, the post-#### token for GSM8K).
    #
    # NOT boxed_math, which was the default until 2026-08-02 and whose range is
    # {0}. Both halves extract: rm_hub's `boxed_` prefix strips \boxed{...} down
    # to "152", and grade_answer_verl then calls extract_answer("152"), which
    # returns None for any string with no \boxed in it. A perfectly correct
    # response scored 0, so every group's advantage was 0, so no arm at any
    # learning rate received a gradient. tests/test_lora_regret_reward_grading.py
    # runs the launcher's configured RM_TYPE and fails if its range is empty.
    #
    # NOT deepscaler, which returns 0 unless the response contains "</think>"
    # or "###Response" -- a Llama-3.1 *base* policy emits neither, so every
    # rollout would score 0 and every arm would look identical.
    #
    # This requires the prompts to ask for a boxed answer. prepare_data.py's
    # --answer-instruction does that; the runbook makes it a required step.
    --rm-type "${RM_TYPE:-math}"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    # A base policy continues the pattern: after finishing its solution it
    # writes the next "Problem:" and keeps going to the token cap (10.2% of
    # probe rollouts truncated at 2,048). A truncated response has lost its
    # \boxed{...} and grades 0 however well it argued, so the stop word is part
    # of the reward path, not a throughput tweak. Must match
    # prepare_data.COMPLETION_STOP; a test pins the pair together.
    --rollout-stop "${ROLLOUT_STOP:-$'\n\nProblem:'}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-seed "${ROLLOUT_SEED}"
    # Llama-3.1-8B *base* ships no chat_template, so load_tokenizer raises
    # before training starts (prerequisite P2). Pinned byte-identical to the
    # LLAMA3_CHAT_TEMPLATE constant the loss-mask gate exercises -- do not
    # hand-write a substitute.
    --chat-template-path "${ORBIT_ROOT}/orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja"
)

# === Optimizer: constant LR, no warmup, no cooldown -- the blog's protocol ===
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style "${LR_DECAY_STYLE:-constant}"
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.999}"
)

# Policy gradient with importance sampling and GRPO-style centering.
#
# KL and entropy coefficients default to zero: both are extra forces on the
# update whose strength interacts with the learning rate, and the learning rate
# is the axis E4 sweeps. A KL penalty would also pull every arm toward the same
# reference policy, which is precisely the difference between arms that C5 is
# about. Opt in explicitly if a run diverges.
RL_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
    --kl-loss-type "${KL_LOSS_TYPE:-low_var_kl}"
    --entropy-coef "${ENTROPY_COEF:-0.0}"
    --eps-clip "${EPS_CLIP:-0.2}"
    --eps-clip-high "${EPS_CLIP_HIGH:-0.2}"
)

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    # Group and name are different things here, and both matter. The sweep sets
    # WANDB_GROUP to the METHOD, so all seven FullFT arms share a group and all
    # twenty-one LoRA arms share another -- which is what makes the dashboard
    # readable. Without an explicit run name the name IS the group, so those
    # seven arms all appear as "full" and the learning rate is only visible by
    # opening each run's config. LAUNCHER_NAME is the arm name.
    --wandb-run-name "${WANDB_RUN_NAME:-${LAUNCHER_NAME}}"
    --disable-wandb-random-suffix
)

# Full fine-tuning needs tensor parallelism to fit; PEFT does not.
#
# At TP=1 every GPU carries the whole 8B model, and the standing cost per GPU is
# (2+4)*P/TP + 12*P/N -- bf16 parameters, fp32 main_grad, DP-sharded optimizer:
#
#   TP=1   48 + 12 = 60 GB      TP=4   12 + 12 = 24 GB
#   TP=2   24 + 12 = 36 GB      TP=8    6 + 12 = 18 GB
#
# 60 GB left ~19 GB for the step, which wanted ~20: measured on 8xH100, the arm
# died in the fp32 cross-entropy logits, 694 MiB short with 660 MiB free
# (`empty_strided_cuda((s10, 1, 128256), ..., torch.float32)`; 128256 is the
# vocabulary). Recompute was already full/uniform, so activations were not the
# slack -- the unsharded logits were. `--sequence-parallel` below is what makes
# TP shard them.
#
# Half the GPUs, not all of them. TP=8 fits too, but forces DP=1 -- the
# distributed optimizer then has nothing to shard across and the gradient
# reduction becomes a no-op, the degenerate case orbit's own preflight refuses
# to test on -- and pays a per-layer all-reduce across 32 layers for headroom
# that is not needed. Rounded down to a power of two because TP must divide 32
# attention heads and 8 GQA query groups; capped at 8 for the same reason.
#
# PEFT stays at 1: LoRA and OFT carry no fp32 main_grad for the base and no full
# optimizer state, and six RL PEFT arms were measured at TP=1 on 2026-07-31.
PEFT_METHOD=${PEFT_METHOD:-lora}
if [[ "${PEFT_METHOD}" == "none" && -z "${TENSOR_MODEL_PARALLEL_SIZE:-}" ]]; then
    _fullft_tp=1
    while (( _fullft_tp * 2 <= GPUS_PER_NODE / 2 && _fullft_tp * 2 <= 8 )); do
        _fullft_tp=$(( _fullft_tp * 2 ))
    done
    TENSOR_MODEL_PARALLEL_SIZE=${_fullft_tp}
    echo "PEFT_METHOD=none: defaulting TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}" \
         "(GPUS_PER_NODE=${GPUS_PER_NODE}, DP=$(( GPUS_PER_NODE / TENSOR_MODEL_PARALLEL_SIZE )))" >&2
fi

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

# E4-3 reads validation-accuracy curves, so the eval is generation-based against
# the held-out MATH and GSM8K splits. Held-out NLL is deliberately absent: an RL
# policy's own output distribution shifts as it trains, so NLL on a fixed
# reference set stops being comparable across arms.
EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-25}"
    --eval-prompt-data math_test "${MATH_TEST_JSONL}" gsm8k_test "${GSM8K_TEST_JSONL}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-2048}"
    --eval-top-k 1
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.75}"
    --rollout-num-gpus 0
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-128}"
    --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS:-262144}"
    --router-disable-circuit-breaker
    --sglang-router-policy round_robin
)

MISC_ARGS=(
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --offload-rollout
    --te-rng-tracker
    # As in the SFT launcher: no --cuda-graph-scope full_iteration, because
    # Megatron then asserts --no-check-for-nan-in-loss-and-grad and offers no
    # positive spelling to re-enable the check. A silently-NaN arm would read as
    # a bad learning rate -- the exact conclusion this study draws.
)

DEBUG_ARGS=(
    --log-passrate
)

# === PEFT method: lora | oft | none ===
# Already resolved above, where the FullFT tensor-parallel default needs it.
TARGET_MODULES_DEFAULT=linear_qkv,linear_proj,linear_fc1,linear_fc2
PEFT_ARGS=()
case "${PEFT_METHOD}" in
    none)
        # See the SFT launcher for the arithmetic: 4*P + 12*P/N GB per GPU for
        # optimizer state alone (32 GB + 96 GB/N at 8.03B), before activations
        # and before the rollout engine's share.
        # tools/lora_regret/models.py computes the floor per model and exports
        # it; 4 is the Llama-3.1-8B value and stays the default.
        MIN_GPUS_FULLFT=${MIN_GPUS_FULLFT:-4}
        if (( GPUS_PER_NODE < MIN_GPUS_FULLFT )) && ! is_true "${ALLOW_SMALL_FULLFT:-0}"; then
            echo "PEFT_METHOD=none (full fine-tuning) needs GPUS_PER_NODE>=${MIN_GPUS_FULLFT}; got ${GPUS_PER_NODE}." >&2
            echo "Per-GPU optimizer state is 4*P+12*P/N GB. Set ALLOW_SMALL_FULLFT=1 to override." >&2
            exit 2
        fi
        # Train offload stays ON here, as it does for the PEFT arms. It used to
        # be disabled, because orbit refused --offload-train for full
        # fine-tuning outright; with that refusal removed, disabling it is what
        # breaks the arm rather than what saves it.
        #
        # In colocate mode SGLang shares these GPUs and pauses its KV cache
        # while the actor trains. With no train offload the 8B model's gradients
        # and optimizer state stay resident, and the resume fails:
        #
        #   [torch_memory_saver.cpp] cudaError error: 2 (out of memory)
        #   file=csrc/core.cpp func=resume line=182
        #
        # measured at 12.48 GB free against 16.00 GB of paused K+V. Argument
        # finalisation forces --offload-train-grad-buffers and
        # --offload-train-optimizer on for this path; parameters stay resident
        # because update_weights pushes them to the engine every rollout.
        ;;
    lora)
        PEFT_ARGS=(
            --peft-method lora
            --peft-variant standard
            --lora-rank "${LORA_RANK:-256}"
            --lora-alpha "${LORA_ALPHA:-32}"
            --lora-dropout "${LORA_DROPOUT:-0.0}"
            # kaiming, never Orbit's xavier default: the two differ by ~2.4x in
            # std, which shifts the measured optimal learning rate.
            --lora-a-init-method "${LORA_A_INIT_METHOD:-kaiming}"
            --target-modules "${TARGET_MODULES:-${TARGET_MODULES_DEFAULT}}"
        )
        ;;
    oft)
        PEFT_ARGS=(
            --peft-method oft
            --peft-variant standard
            --oft-type canonical_oft
            --oft-block-size "${OFT_BLOCK_SIZE:?set OFT_BLOCK_SIZE from orbit.utils.peft_param_match.matched_oft_block_size; there is no safe default}"
            --oft-eps "${OFT_EPS:-6e-5}"
            --target-modules "${TARGET_MODULES:-${TARGET_MODULES_DEFAULT}}"
        )
        ;;
    *)
        echo "Unsupported PEFT_METHOD=${PEFT_METHOD}; expected one of: lora oft none" >&2
        exit 2
        ;;
esac

if [[ -n "${RL_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of a flat flag string
    MISC_ARGS+=( ${RL_EXTRA_ARGS} )
fi

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
