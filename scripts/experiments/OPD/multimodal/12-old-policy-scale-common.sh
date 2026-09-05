#!/bin/bash
#
# Shared scale contract for the milestone-12 old-policy source A/B.
#
# This file is sourced by 12a/12b and is not a standalone Slurm entry point.
# It starts from the frozen canonical multimodal baseline, scales only the
# trainer/rollout geometry and async window, then selects one internally
# consistent q_old source for both RKLD-PG and the PPO denominator.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
   echo "FATAL: source 12-old-policy-scale-common.sh through 12a or 12b" >&2
   exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

case "${M12_OLD_POLICY_SOURCE:-}" in
   rollout | trainer) ;;
   *)
      echo "FATAL: M12_OLD_POLICY_SOURCE must be rollout or trainer" >&2
      return 1
      ;;
esac

M12_NUM_ROLLOUT=200

# shellcheck disable=SC1091
source "$SCRIPT_DIR/baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step.sh"

# One teacher head node + two trainer nodes + four rollout nodes.
EXPERIMENT_NODES=7

# TP=4 over 16 actor GPUs gives DP=4. GBS=128 preserves the validated
# 32 samples per DP replica: 128 / 4 = 64 / 2 = 32.
ROLLOUT_ARGS=(
   --prompt-data "$HF_TRAIN_DATA"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
   --custom-config-path examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
   --rollout-shuffle
   --num-rollout "$M12_NUM_ROLLOUT"
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --global-batch-size 128
   --balance-data
)

LAYOUT_ARGS=(
   --actor-num-nodes 2
   --actor-num-gpus-per-node 8
   --rollout-num-gpus 32
)

FULLY_ASYNC_ARGS=(
   --fully-async
   --fully-async-prefetch-batches 4
   --fully-async-max-completed-queue-groups 32
   --max-weight-staleness 4
   # pre-sync worker semantics: aborted/stale groups go back to the data buffer
   # for regeneration (the class-based rollout's default is drop)
   --async-unused-samples-handler retry
   --update-weights-interval 1
)

# Remove the canonical baseline's selector and add it back only for the rollout
# arm. This keeps q_adv and q_den on the same snapshot in both arms.
M12_OPD_ARGS=()
for arg in "${OPD_ARGS[@]}"; do
   if [[ "$arg" != "--use-rollout-logprobs" ]]; then
      M12_OPD_ARGS+=("$arg")
   fi
done
OPD_ARGS=("${M12_OPD_ARGS[@]}")
if [[ "$M12_OLD_POLICY_SOURCE" == "rollout" ]]; then
   OPD_ARGS+=(--use-rollout-logprobs)
fi

ORBIT_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${RM_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${OPD_ARGS[@]}"
   "${MONITOR_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
   "${FULLY_ASYNC_ARGS[@]}"
)

_m12_assert_flag_value() {
   local target_flag=$1
   local expected_value=$2
   local matches=0
   local index

   for ((index = 0; index < ${#ORBIT_ARGS[@]} - 1; index++)); do
      if [[ "${ORBIT_ARGS[$index]}" == "$target_flag" ]]; then
         if [[ "${ORBIT_ARGS[$((index + 1))]}" != "$expected_value" ]]; then
            echo "FATAL: $target_flag must be $expected_value, got ${ORBIT_ARGS[$((index + 1))]}" >&2
            return 1
         fi
         matches=$((matches + 1))
      fi
   done

   if ((matches != 1)); then
      echo "FATAL: expected exactly one $target_flag, found $matches" >&2
      return 1
   fi
}

_m12_assert_flag_value --actor-num-nodes 2
_m12_assert_flag_value --actor-num-gpus-per-node 8
_m12_assert_flag_value --rollout-num-gpus 32
_m12_assert_flag_value --tensor-model-parallel-size 4
_m12_assert_flag_value --global-batch-size 128
_m12_assert_flag_value --rollout-batch-size 32
_m12_assert_flag_value --n-samples-per-prompt 4
_m12_assert_flag_value --num-rollout "$M12_NUM_ROLLOUT"
_m12_assert_flag_value --fully-async-prefetch-batches 4
_m12_assert_flag_value --max-weight-staleness 4
_m12_assert_flag_value --async-unused-samples-handler retry
_m12_assert_flag_value --eps-clip 0.2
_m12_assert_flag_value --eps-clip-high 0.2
_m12_assert_flag_value --opd-kl-coef 1
_m12_assert_flag_value --opd-log-prob-top-k 0
_m12_assert_flag_value --opd-dagger-top-k 2
_m12_assert_flag_value --opd-dagger-coef 0.5
_m12_assert_flag_value --opd-dagger-loss cross_entropy

rollout_logprob_flags=0
for arg in "${ORBIT_ARGS[@]}"; do
   case "$arg" in
      --use-rollout-logprobs)
         rollout_logprob_flags=$((rollout_logprob_flags + 1))
         ;;
      --use-tis | --get-mismatch-metrics | --opd-optimize-task-reward)
         echo "FATAL: milestone 12 forbids $arg" >&2
         return 1
         ;;
      --save | --save-* | --async-save)
         echo "FATAL: milestone 12 must not save checkpoints (found $arg)" >&2
         return 1
         ;;
   esac
done

expected_rollout_logprob_flags=0
if [[ "$M12_OLD_POLICY_SOURCE" == "rollout" ]]; then
   expected_rollout_logprob_flags=1
fi
if ((rollout_logprob_flags != expected_rollout_logprob_flags)); then
   echo "FATAL: $M12_OLD_POLICY_SOURCE arm expected $expected_rollout_logprob_flags" \
      "rollout-logprob flag(s), found $rollout_logprob_flags" >&2
   return 1
fi
