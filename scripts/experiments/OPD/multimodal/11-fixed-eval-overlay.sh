#!/bin/bash
# Shared fixed Geo3K evaluation overlay for Milestone 11.
#
# Source this only after a synchronous recipe has assembled ORBIT_ARGS. It does
# not change the model, OPD objective, optimizer, or parallel layout. It owns
# the Milestone 11 trajectory budget shared by training and fixed evaluation.

if [[ -z "${ORBIT_ARGS+x}" ]]; then
   echo "FATAL: Milestone 11 fixed-eval overlay requires an assembled ORBIT_ARGS array" >&2
   return 1 2>/dev/null || exit 1
fi

OPD_EVAL_NUM_PROMPTS=${OPD_EVAL_NUM_PROMPTS:-30}
OPD_EVAL_SEED=${OPD_EVAL_SEED:-20260720}
OPD_EVAL_INTERVAL=${OPD_EVAL_INTERVAL:-5}
OPD_EVAL_MAX_CONTEXT_LEN=${OPD_EVAL_MAX_CONTEXT_LEN:-12000}
OPD_EVAL_SOURCE_TRAIN=${OPD_EVAL_SOURCE_TRAIN:-"$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"}
OPD_EVAL_SOURCE_TEST=${OPD_EVAL_SOURCE_TEST:-"$HF_CACHE_DIR/data/geo3k_imgurl_processed/test.parquet"}
OPD_FIXED_EVAL_MANIFEST=${OPD_FIXED_EVAL_MANIFEST:-"$HF_CACHE_DIR/data/geo3k_imgurl_processed/opd_eval_seed${OPD_EVAL_SEED}_n${OPD_EVAL_NUM_PROMPTS}.parquet"}
OPD_FIXED_EVAL_CONFIG=${OPD_FIXED_EVAL_CONFIG:-"${OPD_FIXED_EVAL_MANIFEST%.parquet}.ctx${OPD_EVAL_MAX_CONTEXT_LEN}.eval.json"}
# Non-evaluated test records train (minus any record sharing an image with an
# evaluated prompt); the file is content-addressed by the manifest identity.
OPD_AUGMENTED_TRAIN_DATA=${OPD_AUGMENTED_TRAIN_DATA:-"${OPD_FIXED_EVAL_MANIFEST%.parquet}.train_augmented.parquet"}

if ! [[ "$OPD_EVAL_NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
   echo "FATAL: OPD_EVAL_NUM_PROMPTS must be a positive integer, got '$OPD_EVAL_NUM_PROMPTS'" >&2
   return 1 2>/dev/null || exit 1
fi
if ! [[ "$OPD_EVAL_SEED" =~ ^[0-9]+$ ]]; then
   echo "FATAL: OPD_EVAL_SEED must be a non-negative integer, got '$OPD_EVAL_SEED'" >&2
   return 1 2>/dev/null || exit 1
fi
if ! [[ "$OPD_EVAL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
   echo "FATAL: OPD_EVAL_INTERVAL must be a positive integer, got '$OPD_EVAL_INTERVAL'" >&2
   return 1 2>/dev/null || exit 1
fi
if ! [[ "$OPD_EVAL_MAX_CONTEXT_LEN" =~ ^[1-9][0-9]*$ ]]; then
   echo "FATAL: OPD_EVAL_MAX_CONTEXT_LEN must be a positive integer, got '$OPD_EVAL_MAX_CONTEXT_LEN'" >&2
   return 1 2>/dev/null || exit 1
fi

# The Geo3K custom rollout consumes one cumulative budget across the expanded
# multimodal prompt, every assistant generation, and every environment
# observation. Set both flags explicitly: max-context establishes that
# trajectory contract, while max-response keeps the underlying SGLang request
# ceiling and generated eval config consistent with it.
MILESTONE_11_NUM_ROLLOUT=""
MILESTONE_11_RESPONSE_LIMIT_FOUND=0
MILESTONE_11_CONTEXT_LIMIT_FOUND=0
for i in "${!ORBIT_ARGS[@]}"; do
   case "${ORBIT_ARGS[$i]}" in
      --num-rollout)
         MILESTONE_11_NUM_ROLLOUT=${ORBIT_ARGS[$((i + 1))]}
         ;;
      --rollout-max-response-len)
         ORBIT_ARGS[$((i + 1))]=$OPD_EVAL_MAX_CONTEXT_LEN
         MILESTONE_11_RESPONSE_LIMIT_FOUND=1
         ;;
      --rollout-max-context-len)
         ORBIT_ARGS[$((i + 1))]=$OPD_EVAL_MAX_CONTEXT_LEN
         MILESTONE_11_CONTEXT_LIMIT_FOUND=1
         ;;
   esac
done
if ((MILESTONE_11_RESPONSE_LIMIT_FOUND == 0)); then
   echo "FATAL: Milestone 11 recipe has no --rollout-max-response-len to bind to its trajectory budget" >&2
   return 1 2>/dev/null || exit 1
fi
if ((MILESTONE_11_CONTEXT_LIMIT_FOUND == 0)); then
   ORBIT_ARGS+=(--rollout-max-context-len "$OPD_EVAL_MAX_CONTEXT_LEN")
fi
if ((OPD_EVAL_INTERVAL == 1)) && [[ "$MILESTONE_11_NUM_ROLLOUT" != "0" ]]; then
   echo "FATAL: Milestone 11 student training requires OPD_EVAL_INTERVAL >= 2;" >&2
   echo "  interval 1 makes the pre-train and post-step-1 eval callbacks share rollout_id=0" >&2
   return 1 2>/dev/null || exit 1
fi

# submit.sh sources recipes before downloading datasets, then the batch launcher
# sources them again after assets exist. Defer only on the first pass; the second
# pass creates or validates the exact same content-addressed manifest.
if [[ -f "$OPD_EVAL_SOURCE_TRAIN" && -f "$OPD_EVAL_SOURCE_TEST" ]]; then
   # Creating the manifest/config writes into their directories; validating an
   # existing pair does not. Fail fast with an actionable message instead of a
   # PermissionError traceback: /data/shared/hf_cache is a read-only mirror that
   # also lacks both Thinking teachers — Milestone 11 submits with
   # HF_CACHE_DIR=/data/shared, the tree every completed 00-10 run used.
   if [[ ! -f "$OPD_FIXED_EVAL_MANIFEST" || ! -f "$OPD_FIXED_EVAL_CONFIG" || ! -f "$OPD_AUGMENTED_TRAIN_DATA" ]]; then
      for MILESTONE_11_EVAL_DIR in "$(dirname -- "$OPD_FIXED_EVAL_MANIFEST")" "$(dirname -- "$OPD_FIXED_EVAL_CONFIG")" "$(dirname -- "$OPD_AUGMENTED_TRAIN_DATA")"; do
         if ! mkdir -p "$MILESTONE_11_EVAL_DIR" 2>/dev/null || [[ ! -w "$MILESTONE_11_EVAL_DIR" ]]; then
            echo "FATAL: fixed-eval manifest directory is not writable: $MILESTONE_11_EVAL_DIR" >&2
            echo "  submit with HF_CACHE_DIR=/data/shared or point OPD_FIXED_EVAL_MANIFEST at a writable path" >&2
            return 1 2>/dev/null || exit 1
         fi
      done
   fi
   python3 "$ORBIT_REPO/examples/geo3k_vlm/multi_turn/fixed_eval.py" prepare \
      --train "$OPD_EVAL_SOURCE_TRAIN" \
      --test "$OPD_EVAL_SOURCE_TEST" \
      --output "$OPD_FIXED_EVAL_MANIFEST" \
      --config-output "$OPD_FIXED_EVAL_CONFIG" \
      --augmented-train-output "$OPD_AUGMENTED_TRAIN_DATA" \
      --size "$OPD_EVAL_NUM_PROMPTS" \
      --seed "$OPD_EVAL_SEED" \
      --max-response-len "$OPD_EVAL_MAX_CONTEXT_LEN"
else
   if [[ -n "${RUN_DIR:-}" ]]; then
      echo "FATAL: Geo3K train/test parquet is still missing during batch launch:" >&2
      echo "  train=$OPD_EVAL_SOURCE_TRAIN" >&2
      echo "  test=$OPD_EVAL_SOURCE_TEST" >&2
      return 1 2>/dev/null || exit 1
   fi
   echo "[fixed-eval] source parquet unavailable during recipe pre-source; preparation deferred to batch launch"
fi

# The normal training callback stays untouched. The generated per-dataset eval
# config selects a wrapper that assigns the rule-based Geo3K task reward before
# Orbit can invoke the OPD custom reward function, so eval never issues teacher-
# scoring requests.
FOUND_GENERATE_FLAG=0
if [[ -n "${ROLLOUT_ARGS+x}" ]]; then
   for i in "${!ROLLOUT_ARGS[@]}"; do
      if [[ "${ROLLOUT_ARGS[$i]}" == "--custom-generate-function-path" ]]; then
         if [[ "${ROLLOUT_ARGS[$((i + 1))]}" == "examples.geo3k_vlm.multi_turn.rollout.generate" ]]; then
            FOUND_GENERATE_FLAG=1
         fi
         break
      fi
   done
fi
if ((FOUND_GENERATE_FLAG == 0)); then
   echo "FATAL: Milestone 11 requires the Geo3K multi-turn custom generation flag" >&2
   return 1 2>/dev/null || exit 1
fi

MILESTONE_11_EVAL_ARGS=(
   --eval-interval "$OPD_EVAL_INTERVAL"
   --eval-config "$OPD_FIXED_EVAL_CONFIG"
   --custom-eval-rollout-log-function-path examples.geo3k_vlm.multi_turn.fixed_eval.log_eval_rollout_data
   --rollout-all-samples-process-path examples.geo3k_vlm.multi_turn.fixed_eval.dump_samples
)
ORBIT_ARGS+=("${MILESTONE_11_EVAL_ARGS[@]}")

# Student training arms consume the augmented prompt file (train plus the
# non-evaluated, non-eval-image test records). The eval-only teacher references
# keep the manifest itself as their inert --prompt-data.
if [[ "$MILESTONE_11_NUM_ROLLOUT" != "0" ]]; then
   MILESTONE_11_PROMPT_REWRITTEN=0
   for i in "${!ORBIT_ARGS[@]}"; do
      if [[ "${ORBIT_ARGS[$i]}" == "--prompt-data" ]]; then
         ORBIT_ARGS[$((i + 1))]=$OPD_AUGMENTED_TRAIN_DATA
         MILESTONE_11_PROMPT_REWRITTEN=1
         break
      fi
   done
   if ((MILESTONE_11_PROMPT_REWRITTEN == 0)); then
      echo "FATAL: Milestone 11 student recipe has no --prompt-data to point at the augmented train file" >&2
      return 1 2>/dev/null || exit 1
   fi
fi

# Milestone 11 is an online synchronous quality study. Fail before allocation if
# a caller accidentally mixes in the fully-async driver, reward optimization,
# rollout q_old ablation, or checkpoint saving.
if [[ "${ORBIT_TRAIN_ENTRY:-train.py}" != "train.py" ]]; then
   echo "FATAL: Milestone 11 requires synchronous train.py, got '${ORBIT_TRAIN_ENTRY:-}'" >&2
   return 1 2>/dev/null || exit 1
fi
for arg in "${ORBIT_ARGS[@]}"; do
   case "$arg" in
      examples.fully_async.* | --fully-async-* | --opd-optimize-task-reward | --use-rollout-logprobs)
         echo "FATAL: Milestone 11 forbids scheduling/objective ablation flag '$arg'" >&2
         return 1 2>/dev/null || exit 1
         ;;
      --save | --save-* | --async-save)
         echo "FATAL: Milestone 11 must not save checkpoints (found '$arg')" >&2
         return 1 2>/dev/null || exit 1
         ;;
   esac
done

export OPD_FIXED_EVAL_MANIFEST OPD_FIXED_EVAL_CONFIG OPD_AUGMENTED_TRAIN_DATA OPD_EVAL_NUM_PROMPTS OPD_EVAL_SEED OPD_EVAL_INTERVAL OPD_EVAL_MAX_CONTEXT_LEN
