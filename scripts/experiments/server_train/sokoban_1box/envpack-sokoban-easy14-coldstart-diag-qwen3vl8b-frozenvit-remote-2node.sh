#!/bin/bash
#
# DIAGNOSTIC (throwaway): Qwen3-VL-8B Sokoban cold-start keep-rate probe.
#
# Uses the EASY dataset envpack-sokoban-easy14 (min_solve_steps [1,4]; buckets
# solve_1 / solve_2 / solve_4 — 6x6/1box has no solve_3). GRPO (no DAPO filter,
# so the batch always fills — no cold-start deadlock), curriculum OFF (so step 0
# samples across all buckets and we can read per-bucket keep), tiny render, ViT
# frozen, only 3 rollout steps, eval effectively disabled.
#
# Goal: at step 0 (pure cold, pre-update), measure the pre-filter prompt-group
# distribution per bucket — i.e. how many solve_1 / solve_2 / solve_4 groups are
# "mixed" (= what the DAPO success-variance filter would KEEP). Tests whether a
# lower difficulty floor (1-/2-step puzzles) is solvable enough at cold start to
# give nonzero keep, unlike solve_4 (~0 at cold start).
#
# Submit (strip SLURM_* if inside an salloc):
#   JOB_NAME=diag-easy14-cold WANDB_RUN_PREFIX=diag WANDB_INIT_TIMEOUT=300 \
#   TIME=2:00:00 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 MILES_ENV_NAME=miles_imp \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_1box/envpack-sokoban-easy14-coldstart-diag-qwen3vl8b-frozenvit-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0
export ENVPACK_DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-easy14}
export ENVPACK_BUILD_TARGET=${ENVPACK_BUILD_TARGET:-sokoban_easy14}
export ENVPACK_EVAL_NAME=${ENVPACK_EVAL_NAME:-envpack_sokoban_easy14_val}
export SOKOBAN_RENDER_STYLE=${SOKOBAN_RENDER_STYLE:-tiny}
# Curriculum OFF: stage1 [3,4] would exclude solve_1/2; we want all buckets.
export SOKOBAN_CURRICULUM_ENABLED=0
# ENABLE_DAPO unset => GRPO, no dynamic-sampling filter => batch always fills.
export SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-512}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-diag}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

MODEL_ARGS+=( --freeze-vision-model )
MILES_ARGS+=( --freeze-vision-model )

# Diagnostic schedule overrides (argparse last-wins): short cold-start probe.
MILES_ARGS+=( --num-rollout 3 --eval-interval 1000 )
