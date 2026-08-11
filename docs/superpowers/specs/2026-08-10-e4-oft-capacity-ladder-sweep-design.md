# E4 OFT Capacity-Ladder Learning-Rate Sweep Design

## Goal

Add an OFT sweep to both E4 dataset panels with the same operational shape as
the LoRA portion of the existing column scripts: three capacities per script,
seven learning-rate columns per dataset, and one independently resumable
ledger per script.

Because there is no measured OFT learning-rate prior under this RL protocol,
the first sweep reuses LoRA's completed lr0–lr6 window exactly:

```text
2e-06, 5e-06, 1e-05, 3e-05, 7e-05, 2e-04, 4e-04
```

The sweep deliberately excludes lr7 (`1e-03`). It is a seven-point scout, not
a claim that OFT and LoRA share an optimum.

## Capacity ladder and arm identity

Use three fixed all-modules OFT block sizes:

```text
b8, b128, b1024
```

`b128` is an explicit experimental choice replacing the automatically nearest
middle block `b64`. On Llama-3.1-8B all-modules parameter accounting, the three
blocks correspond approximately to LoRA ranks 1, 24, and 196:

| OFT block | OFT params | implied LoRA rank | OFT / implied-LoRA params |
|--:|--:|--:|--:|
| b8 | 93,184 | 1 | 1.338 |
| b128 | 1,690,624 | 24 | 1.012 |
| b1024 | 13,618,176 | 196 | 0.998 |

The b8 mismatch is recorded rather than hidden. This ladder is a low/middle/high
OFT capacity sweep; it is not described as an exact match to the completed
LoRA r1/r16/r256 ladder.

Arms remain named `oftscout-b<block>`, because OFT's natural RL learning-rate
scale has not been measured. Every arm targets
`linear_qkv,linear_proj,linear_fc1,linear_fc2`, uses seed 0, records its realized
parameter ratio, and inherits the shared E4 protocol.

## Matrix change

Keep the OFT arms in the existing `e4` matrix rather than adding a duplicate
matrix. Add an explicit constant for the E4 OFT ladder:

```python
E4_OFT_BLOCK_LADDER = (8, 128, 1024)
```

Build one OFT cell for every block and dataset. Change the E4 OFT scout span so
the current seven-point, one-significant-figure logarithmic constructor yields
exactly the LoRA lr0–lr6 values:

```python
RL_OFT_SCOUT_SPAN = (2e-6, 4e-4)
```

With `RL_GRID_POINTS=7` and the existing rounding, this produces the requested
seven values. The general rule that an unmeasured OFT grid should not copy LoRA
will gain an explicit E4 exception: here copying the completed window is the
user-selected scout strategy made in the absence of OFT intuition.

## Launch scripts and ledgers

Create one wrapper per dataset and learning-rate point:

```text
scripts/lora_regret/run_e4_gsm8k_oft_lr0_8gpu.sh
...
scripts/lora_regret/run_e4_gsm8k_oft_lr6_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr0_8gpu.sh
...
scripts/lora_regret/run_e4_math_oft_lr6_8gpu.sh
```

Each of the fourteen wrappers:

- sources `scripts/lora_regret/e4_protocol.sh`;
- uses `MATRIX=e4`;
- selects exactly three arms, b8/b128/b1024 at one dataset and LR;
- sets `EXPECT_ARMS=3`;
- writes a unique ledger at `results/e4_<dataset>_oft_lr<column>.jsonl`;
- documents the same environment activation, checkout, whole-node, and resume
  instructions as the existing E4 column wrappers.

This yields **42 OFT runs**: 2 datasets × 7 LRs × 3 block sizes. One LR column
per wrapper preserves the current node-level distribution and prevents
concurrent writers from sharing a ledger.

## Compatibility with existing wrappers

The existing FullFT/LoRA wrappers retain their names, selectors, and ledgers.
Their coverage test will enumerate those scripts explicitly so the new
`*_oft_lr*` wrappers cannot enter the old glob accidentally.

No launcher, dataset, rollout budget, evaluation cadence, checkpoint policy,
W&B entity, or scheduler resource setting changes. This task creates local
campaign tooling only; it does not submit or start the OFT runs.

## Verification

Extend `tests/fast/utils/test_lora_regret_lr_columns.py` and the E4 arm-coverage
tests to prove:

1. The E4 OFT grid is exactly
   `{2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4}`.
2. The E4 OFT block ladder is exactly `{8, 128, 1024}`.
3. Exactly fourteen OFT wrappers exist: seven for GSM8K and seven for Math.
4. Every wrapper selects exactly three arms at its named dataset and column.
5. Every selected arm is `oftscout`, all-modules, and one of the three blocks.
6. The wrappers select all 42 E4 OFT arms exactly once, without gaps or overlap.
7. Every wrapper has `EXPECT_ARMS=3`, sources `e4_protocol.sh`, and writes a
   unique dataset/column ledger.
8. The original fourteen FullFT/LoRA wrappers still partition all 56 non-OFT
   E4 arms exactly once.
9. Every new shell script passes `bash -n`.
10. Realized parameter ratios and implied LoRA ranks remain visible and stable
    for all three blocks.

Run the focused LR-column and arm-coverage tests, then the broader LoRA-regret
fast-test subset available in the repository environment.

## Deliberate exclusions

- No OFT lr7 (`1e-03`) wrapper.
- No OFT b64 arm and no additional LoRA ranks.
- No remote synchronization, Condor allocation, training, or W&B sync.
- No claim that the three OFT blocks exactly match LoRA r1/r16/r256 or that the
  selected LR window is optimal for OFT.
