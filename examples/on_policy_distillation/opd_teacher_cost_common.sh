#!/usr/bin/env bash
# Shared Qwen2.5-3B math recipe for the OPD teacher-cost comparison suite
# (R-2). One common recipe owns the science -- model, PEFT, optimizer,
# rollout, and training schedule, identical to
# examples/high_precision/run-qwen2_5-3b-math-oft-grpo.sh -- and the five
# thin wrappers in this directory each set OPD_COST_VARIANT and source this
# file. Only the teacher-cost factor under study changes across variants:
#
#   served  -- external sglang teacher the job serves itself, full-vocab GKD
#              loss. Ported from run-qwen2_5-0_5b-opd-full-vocab-smoke.sh.
#   load    -- in-process Megatron teacher loaded from a full checkpoint,
#              pure MOPD. Ported from run-qwen2_5-0_5b-opd-mopd-smoke.sh.
#   adapter -- teacher is an OFT adapter swapped onto the frozen base, pure
#              MOPD. Ported from Task 5's
#              run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh.
#   base    -- teacher is the student's own frozen base (adapter off), pure
#              MOPD. Ported from run-qwen2_5-0_5b-opd-free-teacher-smoke.sh.
#   ema     -- teacher is an EMA snapshot of the student's own adapter, pure
#              MOPD. Ported from run-qwen2_5-0_5b-opd-ema-smoke.sh.
#
# Every variant's OPD flag block below is ported verbatim from its 0.5B
# smoke, not from a template. In particular, none of the five smokes uses
# --use-opd/--opd-kl-coef (blend mode): load/adapter/base/ema use pure MOPD
# (--advantage-estimator on_policy_distillation), which orbit's arg validator
# forbids combining with --use-opd (they are mutually exclusive), and
# served's full-vocab --loss-type opd_jsd_loss path forbids both
# --use-opd and --advantage-estimator on_policy_distillation as well.
#
# Two variants require documented deviations from "keep PEFT unchanged" to
# satisfy orbit's own OPD validation (miles/utils/arguments.py
# _validate_opd_args), discovered by tracing that validator against the
# canonical OFT PEFT block this recipe otherwise keeps for every variant:
#   load -- PEFT_ARGS=() (full fine-tune): --opd-type megatron rejects
#           --opd-teacher load:<ckpt> (a full in-process second model)
#           whenever PEFT is enabled. Matches the mopd smoke's own
#           PEFT_ARGS=().
#   ema  -- --peft-distributed-transport ray, and no --adapter-double-buffer:
#           local --opd-type sglang self-teacher scoring rejects
#           --peft-method oft + --adapter-double-buffer (NCCL double-
#           buffering has one fixed active adapter slot, so promoting
#           orbit_teacher would clobber the student adapter instead of
#           creating an independently routable teacher). Matches the ema
#           smoke's own PEFT_ARGS.
# served also overrides the base rollout block's --custom-rm-path /
# --custom-reward-post-process-path to the OPD full-vocab reward hooks:
# _validate_opd_args requires that exact pairing whenever
# --teacher-score-mode full_vocab is set (the math task-reward path is
# inapplicable to pure distillation and is not the hook full-vocab scoring
# expects).
#
# OPD_COST_EQUIVALENCE (default 0) -- served-only m1-eq equivalence knob.
# Full-vocab scoring (orbit/opd/opd_sglang.py's post_process, ~line 1109)
# sets .teacher_hidden_states and never .teacher_log_probs, so the Task-6
# teacher-logprob dump hook (ORBIT_OPD_TEACHER_LOGPROB_DUMP) writes nothing
# for served and the M1 correctness leg's base<->served comparison cannot
# run. OPD_COST_EQUIVALENCE=1 swaps served's full-vocab flag set for the
# sampled-token external-teacher configuration that DOES set
# .teacher_log_probs (post_process's non-full-vocab branch, ~line 1135):
# drops --teacher-score-mode full_vocab and its full-vocab-only companions
# (--loss-type opd_jsd_loss, --opd-jsd-beta, --opd-log-topk-overlap,
# --opd-jsd-pointwise-clip), and adds --advantage-estimator
# on_policy_distillation plus the pure-MOPD eps-clip/gamma/lambd block the
# other four variants already use, so needs_opd_teacher() actually engages
# the OPD teacher machinery. --opd-serve-teacher, the teacher GPU/mem flags,
# the OPD reward hooks, and --rm-type math are unchanged either way. Flag
# set ported from the sampled-token external smoke
# (run-qwen2_5-0_5b-opd-sglang-smoke.sh) and cross-checked against
# miles/utils/arguments.py's _validate_opd_args (its external-sglang branch
# accepts --opd-serve-teacher + the same custom-rm-path/post-process hooks
# in sampled mode, no --teacher-score-mode required). This knob exists ONLY
# for m1-eq equivalence runs; cost-table runs (default, OPD_COST_EQUIVALENCE
# unset/0) keep the full-vocab config below unchanged.
#
# LAUNCHER_NAME (and therefore RUN_LOG/WANDB_GROUP/SAVE_DIR) is derived from
# OPD_COST_VARIANT up front, right after the variant is validated, so those
# identity strings actually reflect the variant; the RL_ARGS/PEFT_ARGS
# case-block that depends on the science blocks already being built stays at
# the bottom, immediately before source .../launcher.sh, per the porting
# plan.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Variant selection ===
: "${OPD_COST_VARIANT:?wrapper must set OPD_COST_VARIANT}"
case "${OPD_COST_VARIANT}" in
    served | load | adapter | base | ema) ;;
    *)
        echo "unknown OPD_COST_VARIANT=${OPD_COST_VARIANT}" >&2
        exit 2
        ;;
esac

# === Recipe identity ===
SEED="${SEED:-1234}"
LAUNCHER_NAME="qwen25_3b_opd_cost_${OPD_COST_VARIANT}"
WANDB_PROJECT=${WANDB_PROJECT:-orbit-opd-teacher-cost}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen2.5-3B-Instruct Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Megatron torch_dist checkpoint path}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to an OpenR1-style math JSONL path}"
: "${EVAL_ORBIT_DIR:?set EVAL_ORBIT_DIR to the math_alignment eval directory}"
SAVE_ROOT="${SAVE_ROOT:-${ORBIT_ROOT}/orbit_ckpts/opd_teacher_cost}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/Qwen2.5-3B-Instruct_opd_cost_${OPD_COST_VARIANT}_seed${SEED}}"

# Match the critic benchmark's reward-verification budget. The scorer default
# is 10s; under the eval burst's CPU-parallel grading that deflates Math500
# pass@1 by ~16 points via verification timeouts (identical generations,
# stricter grading). The benchmark recipe exports 60 and manifests it.
export ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S="${ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S:-60}"

# === Resources: 1 actor + 3 rollout (no critic) ===
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-3}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export PYTHONHASHSEED="${SEED}"

# === Model args ===
source "${ORBIT_ROOT}/miles_plugins/model_args/qwen2.5-3B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule (matches ppo_critic_compare_common.sh benchmark mode) ===
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --ckpt-format torch_dist
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL}"
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
    --rollout-seed "${SEED}"
    --rm-type custom
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path orbit.rewards.peft_arena_reward.peft_arena_reward
    --reward-key score
    --eval-reward-key score
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
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

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
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data
        math500 "${EVAL_ORBIT_DIR%/}/math500.jsonl"
        aime24 "${EVAL_ORBIT_DIR%/}/aime24.jsonl"
        amc23 "${EVAL_ORBIT_DIR%/}/amc23.jsonl"
    --eval-input-key prompt
    --eval-label-key label
    --n-samples-per-eval-prompt 4
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-temperature 1.0
    --eval-top-p 1.0
    --eval-top-k -1
    --eval-pass-k-values 1 2 4
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
    --sglang-enable-deterministic-inference
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
    --sglang-router-policy round_robin
    # Same rationale as the critic benchmark: sglang v0.5.16's prefill CUDA
    # graph is disabled for parity with the benchmark engine config.
    --sglang-cuda-graph-backend-prefill disabled
)

MISC_ARGS=(
    --seed "${SEED}"
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

# Canonical OFT actor -- identical PEFT block to the critic benchmark and to
# the 3B GRPO recipe this suite is copied from. The load and ema variants
# below override this for documented reasons; every other variant keeps it
# unchanged.
PEFT_ARGS=(
    --peft-method oft
    --peft-distributed-transport nccl
    --peft-variant standard
    --oft-type canonical_oft
    --oft-block-size 32
    --oft-eps 6e-5
    --target-modules all-linear
    --adapter-double-buffer
)

# === Teacher-cost variant dispatch ===
# Appends to RL_ARGS (and, for load/ema, overrides PEFT_ARGS -- see header).
case "${OPD_COST_VARIANT}" in
    served)
        : "${OPD_TEACHER_HF_CKPT:?set OPD_TEACHER_HF_CKPT to the teacher Hugging Face checkpoint path}"
        if is_true "${OPD_COST_EQUIVALENCE:-0}"; then
            # m1-eq equivalence config -- see OPD_COST_EQUIVALENCE note above.
            # Sampled-token external-teacher scoring: same hooks/serving as
            # the full-vocab config below, minus --teacher-score-mode
            # full_vocab (and its full-vocab-only companions), plus the pure
            # MOPD advantage-estimator/eps-clip/gamma/lambd block.
            RL_ARGS+=(
                --advantage-estimator on_policy_distillation
                --teacher-hf-checkpoint "${OPD_TEACHER_HF_CKPT}"
                --opd-type sglang
                --kl-loss-type k1
                --eps-clip 4e-4
                --eps-clip-high 4e-4
                --gamma 1.0
                --lambd 1.0
                --opd-serve-teacher
                --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS:-1}"
                --custom-rm-path orbit.opd.opd_sglang.reward_func
                --custom-reward-post-process-path orbit.opd.opd_sglang.post_process
                --rm-type math
                # The OPD reward hook returns a scalar, not the {"score": ...}
                # dict the shared ROLLOUT_ARGS' --reward-key score expects.
                --reward-key ""
                --eval-reward-key ""
            )
        else
            RL_ARGS+=(
                --loss-type opd_jsd_loss
                --teacher-score-mode full_vocab
                --teacher-hf-checkpoint "${OPD_TEACHER_HF_CKPT}"
                --opd-type sglang
                --opd-jsd-beta "${OPD_JSD_BETA:-0.5}"
                --opd-log-topk-overlap
                --kl-loss-type k1
                --opd-serve-teacher
                --opd-teacher-num-gpus "${OPD_TEACHER_NUM_GPUS:-1}"
                # Deviation: --teacher-score-mode full_vocab is validated
                # (_validate_opd_args) to require the OPD full-vocab reward
                # hooks, not the base recipe's math task-reward path.
                --custom-rm-path orbit.opd.opd_sglang.reward_func
                --custom-reward-post-process-path orbit.opd.opd_sglang.post_process
                # Deviation: opd_sglang.reward_func's full-vocab branch delegates
                # every sample to default_async_rm, which dispatches on rm_type
                # and bypasses custom_rm_path entirely -- the base recipe's
                # --rm-type custom is a NotImplementedError here. Served's task
                # reward is therefore graded by the rule-based math grader
                # (--rm-type math), not peft_arena_reward; matches the source
                # smoke (run-qwen2_5-0_5b-opd-full-vocab-smoke.sh).
                --rm-type math
                # The OPD reward hook returns a scalar, not the {"score": ...}
                # dict the shared ROLLOUT_ARGS' --reward-key score expects.
                --reward-key ""
                --eval-reward-key ""
            )
        fi
        if [[ -n "${OPD_TEACHER_MEM_FRACTION:-}" ]]; then
            RL_ARGS+=( --opd-teacher-mem-fraction "${OPD_TEACHER_MEM_FRACTION}" )
        fi
        if [[ -n "${OPD_JSD_POINTWISE_CLIP:-}" ]] && ! is_true "${OPD_COST_EQUIVALENCE:-0}"; then
            RL_ARGS+=( --opd-jsd-pointwise-clip "${OPD_JSD_POINTWISE_CLIP}" )
        fi
        ;;
    load)
        : "${OPD_TEACHER_LOAD:?set OPD_TEACHER_LOAD to a Megatron teacher ckpt}"
        RL_ARGS+=(
            --advantage-estimator on_policy_distillation
            --opd-type megatron
            --opd-teacher-load "${OPD_TEACHER_LOAD}"
            --kl-loss-type k1
            --eps-clip 4e-4
            --eps-clip-high 4e-4
            --gamma 1.0
            --lambd 1.0
        )
        # Deviation: --opd-type megatron rejects --opd-teacher load:<ckpt>
        # (a full in-process second model) whenever PEFT is enabled. Full
        # fine-tune, matching the mopd smoke's own PEFT_ARGS=().
        PEFT_ARGS=()
        ;;
    adapter)
        : "${OPD_TEACHER_ADAPTER:?set OPD_TEACHER_ADAPTER to an OFT adapter dir}"
        RL_ARGS+=(
            --advantage-estimator on_policy_distillation
            --opd-type megatron
            --opd-teacher "adapter:${OPD_TEACHER_ADAPTER}"
            --kl-loss-type k1
            --eps-clip 4e-4
            --eps-clip-high 4e-4
            --gamma 1.0
            --lambd 1.0
        )
        ;;
    base)
        RL_ARGS+=(
            --advantage-estimator on_policy_distillation
            --opd-type megatron
            --opd-teacher base
            --kl-loss-type k1
            --eps-clip 4e-4
            --eps-clip-high 4e-4
            --gamma 1.0
            --lambd 1.0
        )
        ;;
    ema)
        RL_ARGS+=(
            --advantage-estimator on_policy_distillation
            --opd-type sglang
            --opd-teacher self:ema
            --opd-ema-decay "${OPD_EMA_DECAY:-0.99}"
            --opd-promote-interval "${OPD_PROMOTE_INTERVAL:-1}"
            --kl-loss-type k1
            --eps-clip 4e-4
            --eps-clip-high 4e-4
            --gamma 1.0
            --lambd 1.0
        )
        # Deviation: local --opd-type sglang self-teacher scoring rejects
        # --peft-method oft + --adapter-double-buffer (NCCL double-buffering
        # has one fixed active adapter slot; promoting orbit_teacher would
        # clobber the student adapter). Matches the ema smoke's own
        # PEFT_ARGS: ray transport, no double-buffer.
        PEFT_ARGS=(
            --peft-method oft
            --peft-distributed-transport ray
            --oft-type canonical_oft
            --oft-block-size 32
            --target-modules all-linear
        )
        ;;
esac

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
