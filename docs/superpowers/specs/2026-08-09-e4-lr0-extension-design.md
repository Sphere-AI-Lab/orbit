# E4 LoRA LR0 Extension Design

## Goal

Add one reproducible E4 learning-rate point below the current `5e-6` LoRA
boundary. The new point is `2e-6` for LoRA ranks 1, 16, and 256 on both GSM8K
and Math. Full fine-tuning is intentionally excluded.

## Design

Introduce a small `e4lr0` arm matrix rather than changing the existing `e4`
grid. The matrix contains six arms: three LoRA ranks for each of the two E4
datasets, all targeting every supported module at learning rate `2e-6`. It
uses the same RL launcher, accuracy metric, W&B project, eight-GPU preflight
requirement, and E4 protocol as the existing seven learning-rate columns.

Two wrappers partition the matrix by dataset:

- `run_e4_gsm8k_lr0_8gpu.sh` selects three GSM8K arms and writes
  `results/e4_gsm8k_lr0.jsonl`.
- `run_e4_math_lr0_8gpu.sh` selects three Math arms and writes
  `results/e4_math_lr0.jsonl`.

Each wrapper sets `EXPECT_ARMS=3`, so a renamed arm or incorrect regular
expression fails before training. Separate ledgers preserve the campaign's
single-writer resume behavior and are already included by the existing
`e4_<dataset>_lr*.jsonl` analysis globs.

## Verification

Unit tests will establish that `e4lr0` contains exactly six arms, two datasets,
three ranks, only LoRA, and only learning rate `2e-6`. Wrapper tests will prove
that the GSM8K and Math selectors are disjoint, each selects exactly three
arms, and together cover the matrix. A local dry run will verify the launcher
commands without starting training.

## Synchronization and Session Handoff

The local checkout is fast-forwarded to the cluster's unpublished commit
`927f4c9` before implementation. After tests pass, the implementation commit
will be pushed to `origin/feat/lora-without-regret`, then the cluster checkout
will pull it with `--ff-only`, preserving untracked result ledgers.

The idle managed session `codex-orbit-math-lr7` on mpi2 will be captured and
retired. A fresh four-login-node inventory will determine placement for
`codex-orbit-gsm8k-lr0`. The new session will be created in the remote Orbit
checkout, but no Condor allocation or training command will be started in this
change.
