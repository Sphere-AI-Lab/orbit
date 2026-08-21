#!/usr/bin/env bash
# The learning-rate columns of the clean env2 E4 rerun, in one place.
#
# Sourced by every run_*_column.sh. Each method's column N must mean the same
# learning rate whichever wrapper selects it, and the only way to guarantee
# that across separate wrappers is to read the numbers from one file.
#
# Index 0 is a placeholder so that column N is element N.
#
# FullFT keeps the original E4 grid. LoRA is the original grid shifted down by
# one column, so its column 1 is the old `lr0` point, which lives in the
# `e4lr0` matrix rather than `e4`. OFT is block 128 on all modules, centred on
# the historical MATH optimum.

FULLFT_LR=(unused 5e-08 1e-07 3e-07 7e-07 2e-06 4e-06 1e-05)
FULLFT_LR_RE=(unused '5e\-08' '1e\-07' '3e\-07' '7e\-07' '2e\-06' '4e\-06' '1e\-05')

LORA_LR=(unused 2e-06 5e-06 1e-05 3e-05 7e-05 0.0002 0.0004)
LORA_LR_RE=(unused '2e\-06' '5e\-06' '1e\-05' '3e\-05' '7e\-05' '0\.0002' '0\.0004')
LORA_MATRIX=(unused e4lr0 e4 e4 e4 e4 e4 e4)
LORA_RANKS=(1 16 256)

OFT_LR=(unused 5e-07 1e-06 3e-06 7e-06 2e-05 4e-05 0.0001)
OFT_LR_RE=(unused '5e\-07' '1e\-06' '3e\-06' '7e\-06' '2e\-05' '4e\-05' '0\.0001')

# Rollouts per dataset. Under the E4 protocol GLOBAL_BATCH_SIZE equals
# ROLLOUT_BATCH_SIZE, so one rollout is one optimizer step and this is the
# step count. GSM8K runs 200; MATH keeps the protocol's 150.
MATH_ROLLOUTS=150
GSM8K_ROLLOUTS=200

# Export NUM_ROLLOUT for the dataset unless the operator already set it.
# e4_protocol.sh assigns NUM_ROLLOUT only when unset, so the export here is
# what reaches the launcher; an explicit NUM_ROLLOUT in the calling shell still
# wins, and every runner prints the value it ended up with.
set_dataset_rollouts() {
    case "$1" in
        math) : "${NUM_ROLLOUT=${MATH_ROLLOUTS}}" ;;
        gsm8k) : "${NUM_ROLLOUT=${GSM8K_ROLLOUTS}}" ;;
        *) echo "unsupported dataset: $1" >&2; exit 2 ;;
    esac
    export NUM_ROLLOUT
}

check_dataset() {
    case "$1" in
        math|gsm8k) ;;
        *) echo "unsupported dataset: $1" >&2; exit 2 ;;
    esac
}

check_column() {
    if [[ ! "$1" =~ ^[1-7]$ ]]; then
        echo "column must be an integer from 1 through 7, got: $1" >&2
        exit 2
    fi
}

check_lora_rank() {
    case "$1" in
        1|16|256) ;;
        *) echo "LoRA rank must be one of ${LORA_RANKS[*]}, got: $1" >&2; exit 2 ;;
    esac
}
