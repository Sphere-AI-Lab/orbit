# Math OFT BS128 Lower-Learning-Rate Sweep Design

**Date:** 2026-08-13
**Status:** Approved for implementation planning
**Target branch:** `codex/math-oft-b128-lr-sweep`
**Base:** `feat/lora-without-regret@c1ecfdfc6a636fb49c46c3776f2a4709b957ee92`

## Context

The completed Math OFT LR3 campaign produced these successful results:

- BS8 at `3e-5`: accuracy `0.2130`;
- BS128 at `3e-5`: accuracy `0.0946`;
- BS1024 at `3e-5`: no accepted result because its evaluation ended with an
  SGLang HTTP 500.

The existing E4 OFT scout uses the LoRA-derived window
`2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4`. OFT parameterizes rotations rather
than additive low-rank updates, so the LoRA window is not a justified prior for
OFT. The weak BS128 result at `3e-5` motivates a dedicated lower-LR sweep that
holds capacity and placement fixed.

## Goal

Measure the Math accuracy curve for OFT BS128 at exactly these five learning
rates:

```text
1e-7, 3e-7, 1e-6, 3e-6, 1e-5
```

Every arm uses the existing full E4 Math protocol:

- Llama-3.1-8B;
- Math training and `math_test` evaluation;
- OFT block size 128;
- `linear_qkv,linear_proj,linear_fc1,linear_fc2` (`all`) placement;
- seed 0;
- 150 rollouts;
- 8 GPUs;
- the existing E4 optimizer, rollout, evaluation, and W&B settings.

## Non-goals

- Do not change the established E4 FullFT, LoRA, or OFT scout grids.
- Do not rerun BS8 or BS1024.
- Do not change OFT transport, kernels, optimizer behavior, or training
  protocol.
- Do not overwrite or append to `results/e4_math_oft_lr3.jsonl`.
- Do not add seeds or an adaptive second-stage search in this change.

## Design

### Dedicated matrix

Add a small matrix builder dedicated to this question. It returns exactly five
arms and accepts the standard model-shape/seed signature used by the matrix
registry. Each arm has:

- method `oft`;
- block size `128`;
- dataset `math`;
- target modules `ALL_MODULES`;
- one value from the fixed lower-LR tuple;
- the standard BS128 matched-parameter ratio computed from the selected model
  shapes.

Use the distinct matrix key `e4oftb128low` and the distinct arm label
`oftlow`. Distinct arm names prevent the `1e-5` arm from colliding with the
same numerical configuration in the original E4 scout when ledgers are later
loaded together.

The five arm names are therefore:

```text
oftlow-b128-all-math-lr1e-07-s0
oftlow-b128-all-math-lr3e-07-s0
oftlow-b128-all-math-lr1e-06-s0
oftlow-b128-all-math-lr3e-06-s0
oftlow-b128-all-math-lr1e-05-s0
```

Register the matrix with the existing RL launcher and accuracy metric. Give it
a distinct W&B project suffix so the focused scan is visibly separate from the
original broad E4 scout. Register the expected arm count as five and the stage
GPU requirement as eight.

### Dedicated launcher

Add:

```text
scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
```

The launcher sources `e4_protocol.sh`, selects only the five `oftlow` arms, and
executes the existing `campaign.sh` with:

```text
MATRIX=e4oftb128low
EXPECT_ARMS=5
ALLOW_OFT=1
RESULTS=results/e4_math_oft_b128_low_lr.jsonl
```

The launcher remains resumable through the existing campaign behavior:
successful arm names are skipped, failed arm names are retried, and exactly one
writer is allowed for the dedicated ledger.

### Data flow

1. The shell launcher fixes the matrix, arm selector, result path, and expected
   arm count.
2. `campaign.sh` invokes the existing Python sweep driver.
3. The matrix builder produces exactly five BS128 Math arms.
4. The sweep driver maps each arm to the unchanged E4 RL launcher environment.
5. Each terminal arm appends one result row to the dedicated JSONL ledger and
   writes its normal dataset-qualified log and offline W&B run.

### Failure behavior

- A malformed definition or selector that produces other than five arms fails
  before training through `EXPECT_ARMS=5`.
- A failed training arm is recorded as failed; the campaign continues according
  to the existing sequential campaign policy.
- Rerunning the launcher skips only successful rows and retries failed or
  missing rows.
- Existing ledgers are never used as the resume source for this focused sweep.

## Verification

Implementation follows test-driven development.

1. Add failing tests that require the matrix to exist and return exactly five
   arms with the precise LR tuple, BS128, Math, seed 0, `ALL_MODULES`, and
   method `oft`.
2. Require all five arm names to be unique and disjoint from existing matrix arm
   names.
3. Require sweep metadata to map the matrix to the RL launcher, accuracy metric,
   distinct W&B project, five expected arms, and eight GPUs.
4. Require the launcher to select exactly those five arms, use the dedicated
   ledger, enable OFT, and preserve the E4 protocol.
5. Run the focused fast tests covering arm construction, matrix registration,
   preflight metadata, LR-column/launcher contracts, and dry-run selection.
6. Run `bash -n` on the new launcher and `git diff --check` on the final change.

The local lightweight `.venv` at design time did not contain `pytest`, so the
baseline focused suite could not start locally. Implementation verification
must use an environment with the repository's test dependencies installed; this
is an environment limitation, not a waived test requirement.

## Acceptance criteria

- One command launches exactly the five requested Math OFT BS128 arms.
- No existing matrix, launcher, or result ledger changes meaning.
- The dedicated ledger is resumable and isolated from prior LR3 results.
- All focused tests and shell/static checks pass before the experiment is
  launched.
