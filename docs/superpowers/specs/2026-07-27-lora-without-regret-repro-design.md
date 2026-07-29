# Reproducing "LoRA Without Regret" in Orbit — Design

- **Date:** 2026-07-27
- **Status:** approved, pending implementation plan
- **Sources:** [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/) ·
  [michaelbzhu/lora-without-regret](https://github.com/michaelbzhu/lora-without-regret)
- **Repo:** this one (`orbit`, branch `orbit-v0`)

## 1. Purpose

Produce a **rigorously LR-tuned LoRA and full-finetune reference line inside Orbit**, so
that Orbit's OFT / POET work can be compared against LoRA fairly. The blog is the target
because its central methodological claim is precisely the thing an unfair comparison gets
wrong: *LoRA's optimal learning rate is ~10x full fine-tuning's*, so any single-LR
LoRA-vs-X comparison is measuring LR mismatch, not method quality.

Secondary purpose: the same infrastructure (SFT launcher, held-out NLL eval, sweep driver)
is reusable for every future PEFT comparison in this repo.

This plan is a reproduction **and** a comparison: matched-parameter OFT arms are in scope
alongside the LoRA/FullFT arms.

## 2. What we are reproducing

Anchored to michaelbzhu's setup rather than the blog's own (Llama-3/Qwen3 on
Tulu3/OpenThoughts3), because his numbers are published per-arm and therefore checkable.

### 2.1 SFT target table (michaelbzhu, Qwen3-4B)

| Arm | Optimal LR | Test NLL |
|---|---|---|
| FullFT | 2.5e-5 | 1.8457 |
| LoRA r=256, all modules | 2.5e-4 | 1.8457 |
| LoRA r=256, attention-only | 3.5e-4 | 1.8548 |
| LoRA r=256, MLP-only | 3.0e-4 | 1.8491 |
| LoRA r=16, all modules | 2.2e-4 | 1.8473 |
| LoRA r=1, all modules | 1.2e-4 | 1.8489 |

Three claims are read off this table:

1. **The 10x rule** — ratio of the r256-all argmin to the FullFT argmin (published: 10.0).
2. **Rank ordering of optimal LR** — lower rank prefers lower LR (1.2e-4 → 2.2e-4 → 2.5e-4).
3. **Layer selection** — attention-only is worse than MLP-only at comparable parameter
   count, and MLP-only ≈ all-modules.

### 2.2 RL target

LoRA at ranks 256/16/1 matches FullFT under GRPO on math, qualitatively (learning curves
and validation accuracy overlap). The blog's information-theoretic argument is that policy
gradient extracts O(1) bits per episode, so even rank-1's parameter count is not the
binding constraint.

## 3. Findings from the code that shape the design

These were verified by reading the source, not assumed.

### 3.1 Pure SFT skips the SGLang engine, not the rollout manager

`--debug-train-only` gives the rollout placement group 0 GPUs, so no SGLang engine is
ever started (`orbit/ray/placement_group.py:85-88`). `--rollout-function-path
orbit.rollout.sft_rollout.generate_rollout` reduces "rollout" to tokenize + loss-mask
(`orbit/rollout/sft_rollout.py:49`), and `--loss-type sft_loss` swaps the objective
(`orbit/backends/training_utils/loss.py:1012`). No new trainer is needed.

**But:** `train.py:97` calls `create_rollout_manager` unconditionally, so the
`RolloutManager` actor is still constructed and its `__init__` still loads the dataset
through the same loader contract as every other launcher — a pure-SFT launcher is not
exempt from that contract just because no SGLang engine ever runs. **And:** `grep -ril
sft` over the repo returns zero launchers and zero e2e tests. This path has never been
driven end-to-end. That fact is the reason for the parity gate in §7.

### 3.2 LoRA scaling matches the blog; LoRA init does not

Megatron-Bridge scales by `alpha/dim`, matching `W' = W + (alpha/r) B A`
(`megatron/bridge/peft/lora.py:290`). Correct as-is.

`lora_A_init_method` defaults to `"xavier"`
(`orbit/backends/megatron_utils/lora_utils.py:59`), read via `getattr` with **no CLI
argument behind it**. The blog and HF PEFT use `kaiming_uniform_(a=sqrt(5))`, i.e.
A ~ U(+/- 1/sqrt(d_in)). Orbit's Megatron linears (`linear_qkv`, `linear_proj`,
`linear_fc1`, `linear_fc2`) never satisfy Bridge's `nn.Linear`/`te.Linear` fast-path
check (`megatron/bridge/peft/lora.py:135`), so they always land on
`ParallelLinearAdapter`, whose `_get_init_fn` implements exactly that under the name
`"kaiming"` (`megatron/bridge/peft/utils.py:651-672`) — it is simply not reachable
from Orbit. (`_get_init_fn` accepts only `{xavier, normal, kaiming, zero}`; there is
no `"uniform"` value on this path — a different Bridge branch, `LinearAdapter` /
`TELinearAdapter` in `lora_layers.py`, uses that name, but Orbit's targets never
route through it.)

This is not cosmetic. Per the blog's invariance analysis, the four nominal knobs
(alpha, LR_A, LR_B, init_A) collapse to two effective dimensions:

```
effective initial update scale  =  alpha · init_A · LR_B
effective timescale for A       =  init_A / LR_A
```

Working the two inits out, for `lora_A` of shape (r, d_in) so fan_in = d_in, fan_out = r.
`xavier_normal_` and `xavier_uniform_` target the same std (they only differ in
distribution shape), so the formula below is the one that applies on Orbit's actual
path (`xavier_normal_`, via `ParallelLinearAdapter._get_init_fn`):

```
xavier_normal_:              std = sqrt(2 / (d_in + r))
kaiming_uniform_(a=sqrt5):  gain = sqrt(2/(1+5)) = 1/sqrt(3)
                            bound = gain · sqrt(3/d_in) = 1/sqrt(d_in)
                            std = bound/sqrt(3) = 1/sqrt(3·d_in)

ratio = std_xavier / std_kaiming = sqrt(6 · d_in / (d_in + r))  ->  sqrt(6) ~= 2.45
```

Verified numerically at d_in=2560, r=16: xavier_normal_ std 0.027719, kaiming std
0.011410, **ratio 2.4293** (predicted ~2.44). Since init_A enters both effective dimensions linearly,
leaving the default in place biases the measured optimal LR by roughly **2.4x** — on the
exact axis the headline "10x" number lives on, and large enough to corrupt the result
outright. Fixing it is a one-line argument exposure.

### 3.3 Matched-parameter OFT is near-exact at low rank

`oft_r` has shape `(in_features / b, b(b-1)/2)`
(`megatron/bridge/peft/oft_layers.py:282-290`), so per wrapped linear:

```
params_OFT   =  d_in · (b - 1) / 2
params_LoRA  =  r · (d_in + d_out)
```

Setting them equal gives `b = 1 + 2r(d_in + d_out)/d_in`, which for a **square** matrix
(d_in = d_out) reduces to `b = 1 + 4r`, independent of the hidden size. The block size
auto-snaps to the nearest divisor of `d_in`
(`oft_layers.py:266`), so:

- r=1 → b=5 → exact match (when 5 divides d_in)
- r=16 → b=65 → snaps to 64 → within ~2%
- r=256 → b=1025 → nearest divisor is far away → **match is loose**

Divisibility must be checked against the actual Qwen3-4B / Qwen3-1.7B hidden size at plan
time, not assumed. Non-square matrices (`linear_qkv`, `linear_fc1`, `linear_fc2`) match
only approximately because `OFT_BLOCK_SIZE` is a single global knob that each layer snaps
independently. **The realized parameter count of every OFT arm must be logged and reported
alongside its LoRA counterpart**, rather than the arms being described as "matched" in
prose.

## 4. Architecture

Three layers, each independently runnable, separated because they fail on different
timescales — the oracle fails at install time, the trainer fails within ten steps, the
sweep fails after hours.

### 4.1 Oracle layer (throwaway)

michaelbzhu's `sft_lora.py` / `sft_full.py` vendored under
`third_party/lora-without-regret/`, run in its own uv environment. Its only job is to
answer "does Orbit's trainer produce the same number?". It is removed from the critical
path once gate G2 (§7) passes, and it is not part of the deliverable.

### 4.2 Training layer (the new artifact)

Two launcher families following the existing launcher contract — every knob as
`${VAR:-default}`, sourcing `scripts/lib/` for CUDA setup, Ray lifecycle, W&B, and
checkpoint preflight — so the sweep driver overrides rank / LR / target-modules inline
without editing files.

- `examples/sft/run-qwen3-4b-norobots-sft.sh` — new. Sets `--debug-train-only`,
  `--rollout-function-path orbit.rollout.sft_rollout.generate_rollout`,
  `--loss-type sft_loss`, `LR_DECAY_STYLE=constant`, `LORA_ALPHA=32`,
  `--lora-a-init-method kaiming`, and contains no SGLang configuration at all.
- `examples/high_precision/run-qwen3-1.7b-math-grpo-{full,lora,oft}.sh` — modelled on the
  existing `run-qwen2.5-*-bf16-math-{lora,oft}.sh` launchers.

### 4.3 Sweep layer

A driver script that emits one launcher invocation per arm with an env prefix, plus a
results table. Every arm appends one record to `results/lora_regret_sft.jsonl` keyed by
`(method, rank_or_block, target_modules, lr, seed)`, carrying final test NLL, the NLL
curve, realized adapter parameter count, and the W&B run id. Plotting reads only the
jsonl, so every figure regenerates without re-running anything. The driver records
completed arms so a crash resumes rather than restarts.

## 5. Gaps requiring new code

In dependency order — earlier items block later ones.

1. **`--lora-a-init-method` is not exposed.** Add to `add_lora_arguments`
   (`orbit/utils/arguments.py:1177`) with choices `{xavier, normal, kaiming, zero}`, thread through
   `lora_utils.py:59`, default the repro launchers to `kaiming`. See §3.2 for why.
2. **Megatron checkpoints for Qwen3-4B and Qwen3-1.7B do not exist.** Neither model is in
   `/lustre/fast/fast/zqiu/hf_models` (we have `Qwen3-4B-Instruct-2507`, not the base).
   Requires HF download plus conversion to `torch_dist` via `scripts/conversion/`.
   Mechanical, but a prerequisite for every run — do it first so it fails early.
3. **No Robots is not in the data tree.** Convert `HuggingFaceH4/no_robots` to Orbit's
   jsonl chat format (a `messages` list, since `sft_rollout` calls
   `MASK_GENERATOR.get_loss_mask(messages)`): first 6400 of train, first 100 of test.
   Same for `qwedsacf/competition_math`: first 7500 train, rows 7501-8500 as validation.
4. **Held-out NLL eval does not exist.** Orbit's eval path is generation-based
   (`math_eval` pass@k); every SFT figure in the blog is test NLL. Needs an eval hook that
   runs forward + `sft_loss` over a fixed held-out jsonl with no sampling, logging
   token-weighted mean NLL. This is the largest new component and the one most likely to
   carry a subtle bug — hence the parity gate.

## 6. Experiment matrix

### 6.1 SFT — LoRA and FullFT

Qwen3-4B base, No Robots 6400, one epoch at effective batch 32 = 200 steps, AdamW,
constant LR (no warmup, no cooldown), alpha=32.

Six arms: FullFT; r256 all; r256 attention-only (`linear_qkv,linear_proj`); r256 MLP-only
(`linear_fc1,linear_fc2`); r16 all; r1 all.

The LR grid is split in two, because the arms' optima sit a decade apart and one shared
grid would spend most of its points where nothing happens:

- **LoRA arms (5 configs):** `{5e-5, 8e-5, 1.2e-4, 2e-4, 3e-4, 5e-4, 8e-4}` — brackets
  every published LoRA optimum (1.2e-4 to 3.5e-4) with at least two points on each side.
- **FullFT:** `{5e-6, 8e-6, 1.2e-5, 2e-5, 3e-5, 5e-5, 8e-5}` — same shape, one decade
  down, brackets 2.5e-5.

**42 runs.** The 10x claim is then read as the ratio of two measured argmins, not assumed.

### 6.2 SFT — matched OFT

Block sizes from §3.3, subject to divisibility against the real hidden size.

- Three all-module arms at b matched to r1 / r16 / r256.
- Attention-only and MLP-only at the **r16-matched** block, not the r256-matched one,
  because that is where the parameter match is tight enough for "same parameter count,
  different placement" to mean anything.

OFT's optimal LR is unknown a priori — it parameterizes a rotation rather than an additive
update, so its natural scale need not resemble LoRA's. Procedure: a 5-point half-decade
scout on the b=64 all-module arm spanning `{1e-5, 3e-5, 1e-4, 3e-4, 1e-3}`, then a 7-point
grid at ~0.2-decade spacing centered on the scout's argmin, for each of the five arms.
**40 runs** (5 scout + 5 arms x 7).

### 6.3 RL

Qwen3-1.7B, competition_math 7500 train / 1000 validation, 50 GRPO steps, 32 prompts per
step, 8 rollouts per prompt, `max_new_tokens=1024`, one optimizer update per GRPO step,
constant LR, alpha=32. Reward is mathematical-equivalence grading; Orbit's vendored
`third_party/math_eval/grader.py` is the natural source.

Arms: FullFT, LoRA r256/r16/r1 (attention + MLP), OFT at the three matched block sizes.
Four LRs per arm rather than seven — the RL claim is qualitative parity, not a precise
optimum. **~28 runs.**

### 6.4 Budget

Run counts: 42 (SFT LoRA/FullFT, §6.1) + 40 (SFT OFT, §6.2) + 28 (RL, §6.3) + 3 (seed
noise, §7.1) + ~6 (gates G1/G2, §7.2) = **119 runs**.

The 50-100 GPU-hour figure was sized before OFT was added to scope. Realistic total with
OFT is **110-160 GPU-hours**. The OFT layer-ablation was already trimmed to the r16 block
(§6.2) to hold the low end. If further reduction is needed, the recommended cut is
**dropping OFT from the RL half** — RL is where LoRA-vs-FullFT parity is the interesting
claim, and the OFT comparison is better served by the SFT curves.

## 7. Validation

### 7.1 The noise floor is a hard prerequisite

The entire target table spans 1.8457 to 1.8548 — **0.009 nats**. The layer-selection claim
rests on a 0.0057 gap between attention-only and MLP-only, and MLP-only versus all-modules
is only 0.0034. If single-seed run-to-run sigma is ~0.005, the ablation is unresolvable at
one seed and the curves are noise.

Therefore: **3 seeds of LoRA r256 all-modules at lr=2.5e-4 (its published optimum), run
before the sweep launches.** Report the sample standard deviation of final test NLL as
`sigma`. This is a gating experiment, not an optional extra. If sigma turns out comparable
to the effect size, the decision — more seeds on the ablation arms only, versus reporting
error bars and declining to rank the arms — is made then, with a number in hand. (Same
discipline as the 0.0015 noise floor established for the POET work.)

### 7.2 Gates, cheapest first

Each catches a different silent failure. Run in order; do not proceed past a failure.

- **G3 — loss-mask parity (CPU, minutes).** Assert Orbit's `MultiTurnLossMaskGenerator`
  output matches HF's label mask token-for-token on a fixed batch. Catches chat-template
  drift, which would shift every NLL in the study by a constant and be invisible in the
  shape of the curves.
- **G4 — step-0 NLL (one forward pass).** Orbit FullFT's step-0 test NLL is compared against
  HF's on the identical held-out rows, both evaluating the untrained base model. Two-part
  pass condition: (1) scored-token count and sample count must match Orbit-vs-HF **exactly**
  — integer bookkeeping, so any mismatch is structural and blocks the gate (catches an
  off-by-one in the log-prob index, a masking disagreement, dropped rows, or a wrong
  reduction); (2) the NLL delta must fall within the **measured** bf16-vs-fp32 spread of the
  same computation, established by scoring the reference set at both precisions on the HF
  side. That spread is not a fixed constant — it is specific to this model, this data, and
  this hardware, and must be re-measured if any of those change. A fixed sub-1e-3-nat bar is
  not achievable: HF cannot distinguish itself from itself across bf16 and fp32 at that
  precision, so it cannot distinguish Orbit either. Catches checkpoint conversion errors that
  exceed the computation's own precision noise.

  **Measured 2026-07-28** (100-row held-out set, Qwen3-4B base, untrained): scored-token and
  sample counts matched exactly across all three runs (18472 tokens, 100 samples). HF's own
  bf16-vs-fp32 spread was 0.0072 nats (bf16 3.592773 vs fp32 3.585589) on identical code,
  masks, and rows — dtype the only difference. Orbit bf16 (3.589597) lands strictly inside
  that spread, closer to HF's own bf16 (delta 0.003176) than to HF fp32 (delta 0.004008).
  **G4 PASSED.**

  Consequence for later analysis: the attention-vs-MLP layer-selection gap this study must
  resolve (0.0057 nats, §7.1) is smaller than this cross-implementation precision spread.
  Within Orbit the offset is constant across arms and cancels, so the study's internal
  comparisons (the 10x LR ratio, rank ordering, layer selection) are unaffected — but absolute
  comparisons to michaelbzhu's published values should be quoted with roughly ±0.005 nats of
  precision-dependent slack, not to four decimals (see Task 14's write-up).
- **G1 — oracle reproduces here.** michaelbzhu's `sft_lora.py`, r256 all-modules, at
  `{1.2e-4, 2.5e-4, 5e-4}` → minimum test NLL ~= 1.8457. Confirms our environment
  reproduces the published number before we ask Orbit to.
- **G2 — Orbit parity.** Orbit SFT, r256 all-modules, `kaiming` init, same three LRs →
  the per-LR test NLL must land within **2·sigma** (sigma from §7.1) of the corresponding
  G1 value, and the argmin LR must agree. **Only after G2 passes does the 42-run sweep
  launch.** A G2 failure is a bug hunt in Orbit's SFT path, not a reason to lower the bar.

### 7.3 Unit tests (CPU)

- `--lora-a-init-method` plumbing: argument parses, reaches `create_lora_instance`,
  and `kaiming` produces `kaiming_uniform_`-distributed A.
- Matched-OFT block-size solver: given (r, d_in, d_out), returns the correct b and the
  realized parameter count, including the divisibility snap.
- Results-jsonl schema round-trip.
- G3 (loss-mask parity) as a permanent regression test.

### 7.4 Failure isolation

`REQUIRE_MEGATRON_LOAD=1` fails fast on a missing checkpoint. NaN checks stay enabled
(do not set `SKIP_NAN_CHECK_IN_LOSS_AND_GRAD`). The sweep driver's completed-arm ledger
means a crashed arm loses one arm, not the sweep.

## 8. Success criteria

1. G1-G4 all pass, with the G2 delta reported as a number against the measured sigma.
2. The measured LoRA-r256-all optimum divided by the measured FullFT optimum is ~10x
   (accepting a factor-of-2 band, since the LR grid has ~0.2-decade resolution).
3. Optimal LR increases monotonically with rank across r1 → r16 → r256.
4. Attention-only underperforms MLP-only by a margin larger than the measured sigma — or,
   if it does not, that is reported as a **failure to reproduce**, with the sigma that
   makes it unresolvable.
5. RL curves for r1 / r16 / r256 / FullFT overlap within run-to-run variation.
6. Every OFT arm's realized parameter count is reported next to its LoRA counterpart, with
   the r256-matched arm explicitly flagged as loosely matched.

A negative result on any of (2)-(5) is a valid outcome and is reported as such.

## 9. Explicitly out of scope

- The blog's own model/dataset choices (Llama-3, Tulu3, OpenThoughts3, DeepMath-103K).
- The batch-size sensitivity result (Figure 3).
- MoE-specific layer ablation (Qwen3-30B-A3B), despite the checkpoint being available.
- Merging the throwaway oracle layer into the repo's supported surface.
