# E4 OFT b128 Learning-Rate Sweep Design

## Goal

Add a schedulable OFT learning-rate sweep to both E4 dataset panels. Because
there is no measured OFT learning-rate prior under this RL protocol, the first
sweep will reuse LoRA's completed lr0–lr6 window exactly:

```text
2e-06, 5e-06, 1e-05, 3e-05, 7e-05, 2e-04, 4e-04
```

The sweep deliberately excludes lr7 (`1e-03`). It is a seven-point scout, not
a claim that OFT and LoRA share an optimum.

## Capacity and arm identity

Use the existing E4 all-modules OFT `b128` cell for GSM8K and Math. This is one
controlled OFT capacity, approximately equivalent to LoRA rank 24 under the
Llama-3.1-8B all-modules parameter accounting. It is not presented as a match
to any one of the completed r1/r16/r256 LoRA arms.

Arms remain named `oftscout-b128`, because OFT's natural RL learning-rate scale
has not been measured. Every arm targets
`linear_qkv,linear_proj,linear_fc1,linear_fc2`, uses seed 0, and inherits the
shared E4 protocol.

## Matrix change

Keep the OFT arms in the existing `e4` matrix rather than adding a duplicate
matrix. Change the E4 OFT scout span so the current seven-point, one-significant-
figure logarithmic constructor yields exactly the LoRA lr0–lr6 values:

```python
RL_OFT_SCOUT_SPAN = (2e-6, 4e-4)
```

With `RL_GRID_POINTS=7` and the existing rounding, this produces the required
seven values. Comments and tests that currently require the OFT scout to differ
from LoRA's grid must be updated: matching LoRA's measured window is now an
explicit experimental choice made in the absence of an OFT prior.

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

Each wrapper:

- sources `scripts/lora_regret/e4_protocol.sh`;
- uses `MATRIX=e4`;
- selects exactly one `oftscout-b128-all-<dataset>-<lr>-s0` arm;
- sets `EXPECT_ARMS=1`;
- writes a unique ledger at `results/e4_<dataset>_oft_lr<column>.jsonl`;
- documents the same environment activation, checkout, whole-node, and resume
  instructions as the existing E4 column wrappers.

One arm per wrapper preserves the current one-node-per-column operating model,
makes each run independently resumable, and prevents concurrent writers from
sharing a ledger.

## Compatibility with existing wrappers

The existing FullFT/LoRA wrappers retain their names, selectors, and ledgers.
Their coverage test will enumerate those scripts explicitly so the new
`*_oft_lr*` wrappers cannot enter the old glob accidentally.

No launcher, dataset, rollout budget, evaluation cadence, checkpoint policy,
W&B entity, or scheduler resource setting changes. This task creates local
campaign tooling only; it does not submit or start the OFT runs.

## Verification

Extend `tests/fast/utils/test_lora_regret_lr_columns.py` to prove:

1. The OFT grid is exactly
   `{2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4}`.
2. Exactly fourteen OFT wrappers exist: seven for GSM8K and seven for Math.
3. Every wrapper selects exactly one arm at its named dataset and column.
4. All selected arms are `oftscout`, all-modules, and block size 128.
5. The fourteen wrappers partition all E4 OFT arms with no gaps or overlaps.
6. Every wrapper has `EXPECT_ARMS=1`, sources `e4_protocol.sh`, and writes a
   unique dataset/column ledger.
7. The original fourteen FullFT/LoRA wrappers still partition all non-OFT E4
   arms exactly once.
8. Every new shell script passes `bash -n`.

Run the focused fast tests for LR-column and arm coverage, followed by the
broader LoRA-regret fast-test subset available in the repository environment.

## Deliberate exclusions

- No OFT lr7 (`1e-03`) wrapper.
- No additional OFT block sizes or new LoRA ranks.
- No remote synchronization, Condor allocation, training, or W&B sync.
- No claim that b128 exactly matches LoRA r16 or that the selected LR window is
  optimal for OFT.
