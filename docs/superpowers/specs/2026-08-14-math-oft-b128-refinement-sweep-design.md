# Math OFT BS128 Refinement Sweep Design

**Date:** 2026-08-14  
**Status:** Approved for implementation planning  
**Target branch:** `codex/math-oft-b128-refine`  
**Base:** `feat/lora-without-regret@e8c8453d128439d492b3b3a018228287629416cc`

## Context

The completed Math OFT BS128 lower-learning-rate sweep measured:

| Learning rate | `math_test` accuracy |
|---:|---:|
| `1e-7` | `0.0554` |
| `3e-7` | `0.0776` |
| `1e-6` | `0.1528` |
| `3e-6` | `0.2098` |
| `1e-5` | `0.2684` |

An earlier BS128 arm at `3e-5` reached only `0.0946`. The best observed point is
therefore `1e-5`, but the interval immediately below it is unresolved and the
drop between `1e-5` and `3e-5` leaves the upper side of the optimum poorly
localized.

## Goal

Measure Math OFT BS128 at exactly these six additional learning rates:

```text
5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 2e-5
```

Every arm retains the established E4 protocol:

- Llama-3.1-8B;
- Math training with held-out `math_test` evaluation;
- OFT block size 128;
- `linear_qkv,linear_proj,linear_fc1,linear_fc2` placement;
- seed 0;
- 150 rollouts;
- 8 GPUs;
- the existing optimizer, rollout, evaluation, logging, and offline W&B settings.

## Chosen Approach

Add a new, disjoint matrix named `e4oftb128refine`. Do not extend or reinterpret
the completed `e4oftb128low` matrix and do not append to its ledger. The new
matrix owns six `oftrefine` arms so historical and refinement results cannot be
mistaken for one another.

Two production launchers divide the matrix into three arms each:

| Launcher | Learning rates | Ledger |
|---|---|---|
| `run_e4_math_oft_b128_refine_a_8gpu.sh` | `5e-6, 6e-6, 7e-6` | `results/e4_math_oft_b128_refine_a.jsonl` |
| `run_e4_math_oft_b128_refine_b_8gpu.sh` | `8e-6, 9e-6, 2e-5` | `results/e4_math_oft_b128_refine_b.jsonl` |

Each launcher is the only writer to its ledger and can resume independently.
Successful rows are skipped by the existing campaign machinery; failed or
missing arms remain eligible to run.

## Alternatives Rejected

### Extend `e4oftb128low`

This minimizes registry changes but mutates the meaning and expected arm count
of a completed experiment. It also makes the existing five-arm launcher select
new work unexpectedly.

### Use ad-hoc runtime overrides

This avoids tracked source changes but weakens provenance, does not register the
new arms with analysis/preflight tooling, and makes safe resume behavior depend
on operator-only commands.

## Matrix and Arm Contract

Add:

```python
E4_MATH_OFT_B128_REFINE_LRS = (5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 2e-5)
```

The matrix must produce these names in the same order:

```text
oftrefine-b128-all-math-lr5e-06-s0
oftrefine-b128-all-math-lr6e-06-s0
oftrefine-b128-all-math-lr7e-06-s0
oftrefine-b128-all-math-lr8e-06-s0
oftrefine-b128-all-math-lr9e-06-s0
oftrefine-b128-all-math-lr2e-05-s0
```

Every arm has method `oft`, block size 128, all four target modules, dataset
`math`, seed 0, and the existing BS128 matched-ratio metadata. The set of names
must be disjoint from both `e4` and `e4oftb128low`.

Register `e4oftb128refine` with:

- the existing RL launcher;
- metric `accuracy`;
- W&B project suffix `rl-b128-refine-lr`, resolving to
  `math-rl-b128-refine-lr-oft` for these arms;
- six expected matrix arms;
- an eight-GPU preflight requirement.

## Launcher Contract

Both scripts source `e4_protocol.sh` and invoke the existing `campaign.sh`.
Their selectors are exact and seed-specific:

```text
^oftrefine-b128-all-math-lr(5e-06|6e-06|7e-06)-s0$
^oftrefine-b128-all-math-lr(8e-06|9e-06|2e-05)-s0$
```

Each script sets `MATRIX=e4oftb128refine`, `EXPECT_ARMS=3`, `ALLOW_OFT=1`, and
`PREFLIGHT_STAGE=e4oftb128refine`. Extra command-line arguments pass through to
`campaign.sh` unchanged.

## Testing

Add one focused behavioral test module that proves:

1. the matrix builds the six literal learning rates and names above;
2. all scientific fields are fixed and the names are disjoint from prior matrices;
3. registry, launcher, metric, W&B project, expected-arm, and GPU requirements agree;
4. launcher A selects exactly its three lower arms and writes only ledger A;
5. launcher B selects exactly its three upper arms and writes only ledger B;
6. both real shell launchers drive the real campaign boundary under a controlled
   fake Python executable, rather than being validated by source-text matching.

The new test must be written and observed failing before production changes.
After the focused test passes, run the existing lower-LR test and the relevant
arms, sweep, preflight, and coverage suites to catch registry-wide regressions.

## Cluster Launch

Run the two scripts concurrently on two distinct whole-node H100 allocations.
Each allocation requests:

- 32 CPUs;
- 8 GPUs;
- 1000G disk;
- 1,000,000 MB memory;
- `CUDADeviceName == "NVIDIA H100 80GB HBM3"`.

The source revision must first be merged and pushed to
`feat/lora-without-regret`, then fast-forwarded into the shared cluster checkout
`/fast/zqiu/orbit-iclr/orbit`. Launch directly through the two project-native
scripts; do not add a remote qualification suite or scientific acceptance
wrapper after local qualification.

Capacity and minimum price are refreshed immediately before bidding. The user
must approve the exact numeric bid for each new allocation. Do not hard-code
the currently free node names in source or assume they remain available.

Each script keeps its own ledger, arm logs, and offline W&B runs. A failure in
one allocation does not cancel or duplicate the other. Preserve failed logs and
resume only the affected script through its dedicated ledger.

## Success Criteria

- Both launchers select exactly three disjoint arms.
- All six arms run at BS128 with the fixed E4 Math protocol.
- Both campaign-level commands terminate with exit code 0.
- Each ledger contains one accepted `status="ok"` row for each owned arm, with
  final accuracy and 150 completed rollouts.
- Offline W&B artifacts and dataset-qualified arm logs remain in the existing
  project-native locations.
- Results are reported as a single-seed LR refinement; the two physical H100
  nodes may differ in throughput and do not establish timing differences among
  learning rates.
