# Multimodal OPD Server Validation

This runbook validates the G11 exact-action correction before advancing the
multimodal OPD trainer milestone. Run the gates in order and stop at the first
failure.

## Revisions

- Miles PR branch: `opd-mm/01-exact-action-scoring`
- SGLang revision source: the checked-out Miles commit's
  `thirdparty/sglang` gitlink
- Current `sglang-miles` and PR gitlink snapshot:
  `27d5e97c3b26127d2282900823a4abd172a3b6d5`
- Exact-suffix SGLang PR `#3` merge:
  `2b778c2da72dadcad9fcc44e89e34e91bb3967e3`
- Pre-squash SGLang PR `#3` tip:
  `25e50d4cca14e22a3e2af5b46091bc594526a1b7`

The gitlink, rather than a mutable SGLang branch name, defines the exact server
implementation under validation. After updating Miles, derive the expected
revision from that checkout and make the submodule working tree match it:

```bash
git fetch origin
git checkout opd-mm/01-exact-action-scoring
git pull --ff-only origin opd-mm/01-exact-action-scoring
EXPECTED_SGLANG_SHA="$(git rev-parse HEAD:thirdparty/sglang)"
git submodule sync --recursive
git submodule update --init --recursive thirdparty/sglang
ACTUAL_SGLANG_SHA="$(git -C thirdparty/sglang rev-parse HEAD)"
test "$ACTUAL_SGLANG_SHA" = "$EXPECTED_SGLANG_SHA"
printf 'Miles %s pins SGLang %s\n' \
  "$(git rev-parse HEAD)" "$EXPECTED_SGLANG_SHA"
```

Do not manually switch the submodule to another branch or commit after this
check. The Miles gitlink is the source of truth; validating any other SGLang
tree does not validate this PR.

## Gate 1: SGLang Contract

Run the request normalization and suffix unit coverage from the SGLang
submodule:

```bash
cd thirdparty/sglang
PYTHONPATH=python:${PYTHONPATH:-} python3 \
  test/registered/unit/managers/test_io_struct.py
PYTHONPATH=python:${PYTHONPATH:-} python3 \
  test/registered/unit/managers/test_scoring_suffix.py
```

Then run the one-GPU multimodal E2E:

```bash
PYTHONPATH=python:${PYTHONPATH:-} python3 \
  test/registered/vlm/test_mm_scoring_suffix_e2e.py
cd ../..
```

The E2E deliberately constructs a non-canonical text suffix. It passes only
when the returned token IDs are exactly the submitted suffix IDs. A request
without `scoring_suffix_ids` remains on the existing SGLang path.

Before the multi-turn gates, also run the Miles-side one-GPU generation E2E:

```bash
SGLANG_VLM_E2E_MODEL_PATH="${HF_CACHE_DIR:-/data/shared/hf_cache}/models/Qwen3-VL-8B-Instruct" \
  python -m pytest -q \
  tests/e2e/sglang/test_vlm_multiturn_exact_action.py
```

This exercises a distinct boundary from teacher scoring. It sends Student
SGLang a compact multimodal prefix followed by a deliberately non-canonical
prior-turn action and a tool observation, then verifies both the rebuilt prompt
length and the exact returned input token IDs. The training `Sample` continues
to hold the processor-expanded prefix; only the generation request uses compact
media placeholders. The rollout independently checks
`meta_info.prompt_tokens` on every multimodal turn and stops immediately if the
server rebuilt a different-length history.

## Gate 2: Miles Production Scoring

Pre-stage the target teacher if it is not already present:

```bash
export HF_CACHE_DIR=/data/shared/hf_cache
hf download Qwen/Qwen3-VL-30B-A3B-Thinking \
  --local-dir "$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"
```

Run the flagged production-path smoke:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/01-production-image-scoring-smoke
```

This gate launches no trainer and creates no checkpoint or W&B run. It must
finish without returned-ID drift, remain image-sensitive, preserve native
teacher `[T, 2]` targets, reuse the persistent HTTP session, and keep masked
rows inert. Inspect:

```text
runs/01-production-image-scoring-smoke/*/production_image_scoring_smoke.json
```

## SGLang Dependency Record

SGLang PR `impossible-inc/sglang#3` was manually squash-merged. Its final
pre-squash tip `25e50d4cca14e22a3e2af5b46091bc594526a1b7` and merged commit
`2b778c2da72dadcad9fcc44e89e34e91bb3967e3` both resolve to tree
`0fd4dc631f60a27f2735dd241ccfcfcb9d05f775`; the merge changed commit history,
not the implementation.

The PR owner confirms that the existing focused cluster validation exercised
this final implementation. The squash therefore does not require a duplicate
Gate 1 or `01` run:

| Gate | Required evidence | Status |
| --- | --- | --- |
| Gate 1 | `test_io_struct.py`, `test_scoring_suffix.py`, and the one-GPU multimodal E2E pass on the final implementation | Passed in job `26031`; accepted through tree equivalence |
| `01` | Production image-scoring smoke preserves exact IDs, image sensitivity, session reuse, inert masked rows, and native `[T,2]` targets | Passed in job `26032`; accepted through tree equivalence |

The current gitlink advances that exact-suffix merge to
`impossible-inc/sglang#4` merge commit
`27d5e97c3b26127d2282900823a4abd172a3b6d5`. PR `#4` preserves
HF-processor-produced Qwen-VL token IDs in the existing legacy multimodal
loading path. It does not replace the `scoring_suffix_ids` contract validated
by Gate 1/01; it closes the separate Student SGLang multi-turn generation
boundary described below. Before merging the multi-turn PR, run the Miles
one-GPU generation E2E and Gate 7 against this exact gitlink.

Future reproductions must still use the dynamic gitlink assertion above and
record the printed Miles and SGLang commits with their artifacts.

## Gate 3: Five-Step Trainer Smoke

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/02a-singleturn-rkld-smoke
```

Advance only when all five optimizer steps complete without token-alignment
errors, NaN, OOM or timeout. Confirm one teacher scoring request per sample, no
Student SGLang rescore, finite sampled-RKLD metrics, visible
`rollout/raw_reward`, and populated `opd_scoring/*` transport telemetry.

## Gate 4: Twenty-Step Reference

Gates 3 and 4 share no mutable state (HF direct load, separate run dirs and
W&B identities), so `02a` and `02b` may be submitted to the cluster
concurrently once Gate 2 has passed.

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/02b-singleturn-rkld-gate
```

`02b` changes only the run length and W&B identity relative to `02a`. It is the
single-turn sampled-RKLD reference required before teacher Top-K DAgger,
multi-turn or fully-async work. Milestone `00` does not need to be rerun because
it never exercises `scoring_suffix_ids`.

## Gate 5: Five-Step Top-K + Rest Smoke

Run the focused Miles contract, transport, loss, metric, and argument coverage
in the server environment before consuming the three-node allocation:

```bash
python -m pytest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py \
  tests/fast/backends/training_utils/loss/test_opd.py \
  tests/fast/backends/training_utils/loss/test_rkld_dagger.py \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/utils/test_arguments.py
```

This suite includes the composed multimodal exact-suffix DAgger round trip: one
teacher request must preserve the exact sampled suffix and produce both sampled
teacher log-probabilities and native `[T,2]` targets on the same `Sample`.

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/03a-singleturn-topk-rest-smoke
```

This is an isolated trainer-direct DAgger run: sampled RKLD-PG is disabled,
teacher K=2 targets use the Top-K + Rest cross-entropy loss, and task reward is
still telemetry only. Advance only when every sample makes one teacher request
and zero student requests, native `[T,2]` targets reach the trainer, `pg_loss`
is zero, DAgger loss and gradient norm are finite and nonzero, the CE/coarse-KL
identities hold, and all Rest-mass diagnostics are finite.

## Gate 6: Twenty-Step Top-K + Rest Reference

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/03b-singleturn-topk-rest-gate
```

`03b` changes only the run length and W&B identity relative to `03a`. Inspect
explicit CE, Rest CE, total CE, coarse KL, teacher/student Rest mass,
`rest_mass_abs_error`, detached sampled-RKLD diagnostics, raw task reward,
request counts and bytes, scoring latency, actor memory, and step time. Rest CE
decline without non-worsening Rest-mass error does not pass the algorithm gate.
Compare performance to `02b` only after normalizing for generated token count;
profile a short rerun with `MILES_PROFILE_OPD_DAGGER=1` only if the functional
gate shows more than 10% steady-step regression.

## Gate 7: Five-Step Geo3K Multi-Turn Sampled-RKLD Smoke

Run the focused rollout suite again because this gate adds a concrete
action/tool/action sequence contract:

```bash
python -m pytest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py
```

Then launch the synchronous three-node smoke:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/04a-geo3k-multiturn-rkld-smoke
```

`04a` keeps the `02a` model pair, exact-suffix scoring path, sampled-RKLD
objective and TP=4/DP=2/SP/PP=1/CP=1 trainer layout. It changes only the dataset
and rollout generator to the existing Geo3K three-turn interaction. The initial
image remains the only media item; assistant spans are active and text-tool
feedback spans are position-preserving with `loss_mask=0`. Student generation
uses compact media placeholders plus verbatim accumulated action/observation
IDs, while Megatron retains the corresponding processor-expanded sequence.

Advance only when all five optimizer steps finish and:

- teacher requests equal kept samples, Student SGLang requests are zero, and
  returned active token IDs align exactly;
- every Student SGLang generation turn reports `prompt_tokens` equal to the
  current expanded `Sample.tokens` prefix length;
- `interaction/rounds/max` exceeds one;
- `interaction/observation_tokens/mean > 0` and
  `interaction/observation_token_ratio > 0` in at least one batch, while
  sampled teacher values on masked rows remain neutral and never contribute
  to RKLD;
- `interaction/raw_tokens/max` is finite and does not exceed the configured
  response-length cap, and `interaction/length_cap_ratio` remains finite in
  `[0, 1]`;
- sampled-RKLD, gradient norm, raw task reward, transport telemetry, memory and
  step-time metrics are finite;
- no alignment, mask-length, protocol, timeout, NaN or OOM error occurs.

**Result: passed.** Job `26525` completed all five optimizer steps on
2026-07-16 after the focused fast subset passed 82/82. Every batch contained a
sample reaching `max_turns=3`, mean round count stayed between 1.81 and 2.11,
and raw response length exceeded active-only length by 40.55–54.59
tokens/sample. The run reported one teacher request per kept sample, zero
Student SGLang requests, zero alignment/mask errors, and sampled RKLD moving
from 0.40216 to 0.25557. Exact round, length, and RKLD series are retained in
`results/04a-job-26525.json`.

This historical run predates the response/context budget separation. Its
configured `4096` response limit was reduced by the processor-expanded initial
prompt, so it remains finite-training and multi-turn dataflow evidence but
does not validate the corrected response cap or `interaction/length_cap_ratio`.
Rerun the relevant gate on the current code before certifying length semantics.

The historical round extrema were emitted before multi-turn logging requested
explicit cross-rank min/max reductions. Their exact values still establish this
gate because they equal the structural bounds 1 and 3. The longer reference
must retain per-step gradient norm, request transport, scoring latency, memory,
task reward, and step-time series in addition to the current snapshot.

This gate is not a full Geo3K training run. Retain its W&B run and logs as the
multi-turn systems control. Do not introduce asynchronous scheduling while
triaging a failure here.

## Gate 8: Twenty-Step Geo3K Multi-Turn Sampled-RKLD Reference

Run the same focused rollout suite as Gate 7, then launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/04b-geo3k-multiturn-rkld-gate
```

`04b` sources `04a` and changes only the default optimizer-step count from 5
to 20 and the W&B run name to `opd-mm-04b-geo3k-mt-rkld-gate`. Models, data,
multi-turn generator, exact-suffix scoring, sampled-RKLD objective, optimizer,
task-reward observation, and TP=4/DP=2/SP/PP=1/CP=1 remain fixed. Evaluation,
checkpoint saving, Top-K + Rest, hybrid objectives, and async execution stay
disabled.

Advance only when all 20 optimizer steps finish and:

- teacher requests equal kept samples at every step, Student SGLang request
  count stays zero, retries stay zero, and active IDs align exactly;
- global `interaction/rounds/max > 1`,
  `interaction/observation_tokens/mean > 0`, and
  `interaction/observation_token_ratio > 0` at every step;
- `interaction/raw_tokens/max` is finite and does not exceed the configured
  response-length cap, and `interaction/length_cap_ratio` remains finite in
  `[0, 1]` at every step;
- sampled RKLD, gradient norm, raw task reward, scoring request count/bytes and
  latency, actor memory, rollout/train/total step time, and generated-token
  throughput are finite;
- no alignment, mask-length, protocol, timeout, NaN, or OOM error occurs.

Do not require sampled RKLD to decrease monotonically: the batches are
on-policy and this gate does not provide a fixed-prompt quality evaluation.
Retain every per-step series in a machine-readable result snapshot together
with job ID, commit, nodes, duration, exact configuration, and explicit missing
evidence. Use steps 1-19 for steady-state timing summaries unless step 0 is
separately identified as setup cost.

**Result: passed.** Job `26594` (slinky-[10,36,31], 38 min, miles `fd503f25`)
completed all twenty steps on 2026-07-16 after the focused subset passed 82/82.
Teacher requests equaled kept samples at every step (1,280 = 1,280, zero
retries, zero student requests, zero alignment/mask errors). The historical
global round maximum (now emitted as
`interaction/rounds/max`) was 2 or 3 at every step (mean 1.61–2.0) under the
corrected min/max reductions. The historical raw-versus-active length delta,
now represented directly by positive `interaction/observation_tokens/mean`,
was 27.7–52.5 tokens/sample at every step. Sampled RKLD ended lower without a
monotonicity claim (0.387 → 0.218, range 0.200–0.458). Scoring e2e mean 0.25 s
(max p95 1.01 s); over steps 1–19, median step time was 42.8 s and median
throughput was 520 tokens/GPU/s. The per-step gradient norm peaked at 51.8 on
step 1, not step 6, then settled to a last-five mean of 4.14 while remaining
finite and clip-free. The peak occurred on a short-response batch, so this run
does not establish a length-driven failure and does not justify clipping or
length bucketing.

Mean rounds moved from 1.86 over the first five steps to 1.68 over the last
five; masked tool-feedback tokens per sample moved from 43.0 to 31.9. These
metrics naturally move together when later batches contain fewer or shorter
follow-up interactions. Because prompts are shuffled and rollouts stochastic,
record this as an interaction-composition watch item rather than policy-collapse
evidence. Every step still passed the real-follow-up and nonzero-mask gates.

Per-step multi-turn, RKLD, raw reward, gradient, transport and timing series are
retained in `results/04b-job-26594.json`. Per-step actor memory is not; only
point-in-time snapshots exist in the cluster log. Raw task reward also did not
improve (`0.575 → 0.519` by first-five/last-five means), which is expected to
remain an observation rather than a pass condition for pure OPD.

## Gate 9: Five-Step Geo3K Multi-Turn Teacher Top-2 + Rest Smoke

Rerun the Gate 7 focused tests, then launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/05a-geo3k-multiturn-topk-rest-smoke
```

`05a` composes the validated 03 objective with the validated 04 sequence. It
keeps the model pair, Geo3K data, custom multi-turn generator, exact-suffix
teacher scoring, optimizer and TP=4/DP=2/SP/PP=1/CP=1 layout fixed. The only
objective change from 04 is:

```text
OPD_KL_COEF=0
OPD_DAGGER_TOP_K=2
OPD_DAGGER_COEF=1
OPD_DAGGER_LOSS=cross_entropy
```

The 04 and 05 wrappers retain fixed objective values and source the same
objective-free `geo3k-multiturn-overlay.sh`; the helper changes only data,
rollout assembly and multi-turn telemetry before rebuilding `MILES_ARGS`.

Advance only when all five optimizer steps finish and:

- `train/pg_loss` stays zero, `opd_dagger/loss` is finite and nonzero, and the
  reported total loss is explained by the DAgger branch;
- `cross_entropy = explicit_ce + rest_ce` and
  `coarse_kl = cross_entropy - teacher_entropy` hold within the established
  numerical tolerance at every step;
- active rows carry aligned finite `[T,2]` targets, observation rows remain
  inert under `loss_mask=0`, and no masked `-inf` value contaminates the loss;
- teacher requests equal kept samples, Student SGLang requests and retries stay
  zero, and the exact returned action IDs align;
- every step includes a real follow-up turn and a nonzero masked-observation
  span;
- DAgger masses, Rest-mass absolute error, gradient norm, scoring transport,
  actor memory and rollout/train/step timing are finite; and no protocol,
  timeout, NaN or OOM error occurs.

Do not interpret raw task reward or detached sampled RKLD as optimization
signals in this gate. Do not submit Gate 10 until Gate 9 passes.

**Result: passed.** Job `27087` (30 min) completed all five steps on 2026-07-17
after the focused subset passed 82/82. `pg_loss` was exactly zero with the
total loss equal to `opd_dagger/loss`; both identities held exactly at every
step; coarse KL fell 0.458 → 0.271; rest-mass error stayed at 0.058–0.069 with
`valid_position_ratio` = 1.0. One teacher request per kept sample (64 = 64),
zero student requests, zero retries, zero alignment errors. Every step had a
real follow-up turn (round max 3) and a nonzero masked span (Δ 46–56
tokens/sample). Gradient norms 7.1–15.8 — visibly smoother than the multi-turn
RKLD-PG profile on the same sequence contract.

## Gate 10: Twenty-Step Geo3K Multi-Turn Teacher Top-2 + Rest Reference

After Gate 9 passed, the matched reference was launched with:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/05b-geo3k-multiturn-topk-rest-gate
```

`05b` sources the full `05a` recipe and changes only the default optimizer-step
count from 5 to 20 and the W&B run name to
`opd-mm-05b-geo3k-mt-dagger-top2-rest-gate`. Apply every Gate 9 invariant at
all twenty steps. Retain the exact per-step multi-turn, DAgger decomposition,
teacher/student mass, detached RKLD, raw reward, gradient, transport, memory,
timing and throughput series in a machine-readable result snapshot.

Use first-five versus last-five windows for objective and interaction
composition, and exclude step 0 when reporting steady-state timing. Compare
token-normalized performance against both 03b (single-turn DAgger) and 04b
(multi-turn sampled RKLD); raw step time alone cannot identify an OPD operator
regression. Passing Gate 10 establishes a synchronous multi-turn pure-DAgger
reference, not model-quality improvement. Hybrid training remains a separate
later gate.

**Result: passed.** Job `27156` (29 min, miles `2dfc12cc`) completed all twenty
steps on 2026-07-17. Every Gate 9 invariant held at every step: identities
exact, `pg_loss` = 0, 1,280 = 1,280 teacher requests with zero retries, zero
student requests, zero alignment/mask errors, round max > 1 and raw > active
length at every step. First-five/last-five windows: coarse KL 0.431 → 0.174
(−60%); Rest CE 0.144 → 0.127 while rest-mass error moved 0.068 → 0.075 (max
0.079) and teacher top-2 mass 0.920 → 0.881 — the 03b coverage-narrowing watch
item recurs at the same magnitude and remains an observation. Token-normalized
cost 20.5 ms/token sits between 03b (22.2) and 04b (19.8) — no OPD operator
regression; steady step 46.9 s median (steps 1–19); scoring e2e mean 0.74 s
(top-2 responses are heavier than sampled-only). Gradient norms 2.2–15.8
(median 9.0) against the 51.8 RKLD-PG peak on the same sequence contract.
Full per-step series retained in `results/05b-job-27156.json`; per-step actor
memory remains the explicit missing-evidence item.

## Gate 11: Five-Step Geo3K Multi-Turn Hybrid OPD Smoke

Run the focused composition, rollout and conversion tests, then launch:

```bash
python -m pytest -q \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/06a-geo3k-multiturn-hybrid-smoke
```

`06a` holds the complete `05` model, data, multi-turn, exact-suffix scoring,
optimizer and TP=4/DP=2/SP/PP=1/CP=1 contract fixed. It changes only objective
composition:

```text
OPD_KL_COEF=1
OPD_DAGGER_TOP_K=2
OPD_DAGGER_COEF=0.5
OPD_DAGGER_LOSS=cross_entropy
```

The DAgger coefficient is deliberately lower than the earlier text-only
equal-weight hybrid. That run passed structural guards but missed its
sampled-RKLD preservation floor; `0.5` is the first conservative composition
arm, not an optimality claim.

Advance only when all five optimizer steps finish and:

- both `train/pg_loss` and `opd_dagger/loss` are finite and nonzero;
- `abs(train/loss - train/pg_loss - opd_dagger/loss)` is at most
  `1e-5 * max(1, abs(train/loss))` at every step;
- the DAgger CE decomposition and coarse-KL identity hold at every step;
- one Top-2 teacher response per kept sample supplies both the sampled-action
  score and `[T,2]` targets, with zero second teacher requests, Student SGLang
  requests, retries, malformed targets or alignment failures;
- each step contains a real follow-up assistant turn and a nonzero masked tool
  span, and masked observation rows remain inert in both objective branches;
- branch losses, gradient norm, teacher/student masses, Rest-mass error,
  scoring transport and rollout/train/step timing remain finite; and no
  protocol, timeout, NaN or OOM error occurs.

Raw task reward remains telemetry only. Do not submit Gate 12 until Gate 11
passes.

**Result: passed.** Job `27180` (31 min) completed all five steps on 2026-07-18
after the focused subset passed 90/90. Both branches were finite and nonzero at
every step (pg 0.255–0.352, DAgger 0.327–0.400) with additivity residuals at
or below 3e-8 and exact DAgger identities. One Top-2 response fed both branches
(64 = 64 requests, zero retries, zero student requests, zero alignment errors);
multi-turn criteria held at every step. Both objectives moved down together —
sampled RKLD 0.352 → 0.255 and coarse KL 0.415 → 0.271 — so the earlier
equal-weight failure mode is absent at `lambda_DAgger=0.5`. Gradient norms
7.0–16.4.

## Gate 12: Twenty-Step Geo3K Multi-Turn Hybrid OPD Reference

After Gate 11 passes, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/06b-geo3k-multiturn-hybrid-gate
```

`06b` sources the complete `06a` recipe and changes only the default optimizer
step count from 5 to 20 and the W&B run name to
`opd-mm-06b-geo3k-mt-hybrid-rkld1-dagger0p5-gate`. Apply every Gate 11
correctness invariant at all twenty steps. Retain a machine-readable per-step
snapshot covering both objective branches, loss additivity, DAgger identities,
teacher/student masses, Rest-mass error, multi-turn composition, detached raw
reward, gradients, request counts, transport, memory availability, timing and
throughput.

Report sampled RKLD and coarse KL using first-five/last-five windows, but do
not require monotonic curves from one stochastic on-policy run. Exclude step 0
from steady-state timing and compare active-token-normalized cost with both
`04b` and `05b`. Passing Gate 12 establishes the synchronous hybrid systems
reference required before async scheduling work. It does not prove model
quality or coefficient optimality.

**Result: passed.** Job `27208` (41 min, slinky-[11,44,13], miles `51f7571b`)
completed all twenty steps on 2026-07-18. Every Gate 11 invariant held at every
step; additivity max residual 5.96e-8. First-five/last-five windows: sampled
RKLD 0.322 → 0.221 and coarse KL 0.379 → 0.162. The hybrid reaches a similar
last-five coarse-KL band to the independent pure DAgger run (05b: 0.174) while
also lowering its sampled diagnostic. Coverage narrowing is milder in this
run: teacher top-2 mass 0.920 → 0.901 (05b reached 0.881) and rest-mass error
0.064 → 0.061 (max 0.070). This pattern is consistent with an RKLD anchor, but
one stochastic run per arm does not isolate objective causality.
Token-normalized 20.44 ms/token inside the
04b (19.78) / 05b (20.53) band; steady step 47.1 s median; gradient norms
3.9–27.0 (median 6.6). The late-window sample mix had longer active responses and
higher teacher-scoring latency while token-normalized total cost stayed in the
reference band; retain both as async-capacity diagnostics rather than an
operator-regression claim. Full per-step series in
`results/06b-job-27208.json`; per-step actor memory remains missing, and the
zero Student SGLang/alignment counts are retained only in the cluster log.

## Gate 13: Five-Step Geo3K Multi-Turn Hybrid Fully-Async Smoke

Run the Gate 11 focused tests, including the fully-async OPD transport lifecycle
coverage in `test_on_policy_distillation.py`, then launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/07a-geo3k-multiturn-hybrid-fully-async-smoke
```

`07a` retains the exact `06a` model pair, data, multi-turn generator, hybrid
coefficients, teacher scoring contract, optimizer and
TP=4/DP=2/SP/PP=1/CP=1 layout. It changes scheduling only:

```text
MILES_TRAIN_ENTRY=train_async.py
rollout_function=generate_rollout_fully_async
prefetch_batches=1
completed_queue_soft_cap=32 prompt groups
max_weight_staleness=2
update_weights_interval=1
```

No TIS or new importance-ratio correction is enabled. Advance only when all
five optimizer steps complete and:

- every Gate 11 hybrid objective, additivity, DAgger identity, exact-ID and
  multi-turn mask invariant still holds;
- the persistent worker continues producing generation-plus-teacher-scored
  groups while the trainer consumes completed groups;
- teacher requests equal kept samples, with zero Student SGLang rescore
  requests, malformed targets, retries or alignment failures;
- rollout weight-version statistics are present and finite; any group rejected
  by the configured staleness bound is reset and regenerated rather than
  entering the train batch;
- completed-queue observations remain bounded and there is no unbounded manager
  memory growth, collector stall or shutdown hang; and
- branch losses, gradients, masses, scoring transport and timing remain finite
  without protocol, timeout, NaN or OOM errors.

Record the worker's start/end queue sizes, average/max observed staleness and
stale recycle count from the cluster log even when those values are zero. Do
not submit Gate 14 until Gate 13 passes.

**Result: passed on the second run.** The focused suite ran `92/92` (two new
tests added with the fix below). The first submission (job `27256`,
slinky-[11,13,44], 2026-07-18) completed all five steps with every algorithmic
invariant intact (additivity residuals ≤ 3e-8, 64 = 64 teacher requests per
step, zero retries, bounded queue 0–1, clean worker shutdown, rc = 0) — but
the log contained **no `Staleness stats` line**: the collector's engine
weight-version query hit the router's `/model_info`, which the sgl-router does
not expose (it only proxies the legacy `/get_model_info`), so every query
404ed, was swallowed at debug level, and the staleness filter ran inert —
sample-side versions were recorded (`rollout/weight_version/mean` 1.0 → 3.66)
but no group could ever be rejected. This is the same silent-inert failure
class as the earlier multi-turn weight-version gap, one hop further down the
chain. Fix `6253d3f5`: the collector now falls back
`/model_info` → `/get_model_info` because it queries the sgl-router. This is
deliberately different from `sglang_engine.get_weight_version`, which contacts
the SGLang server directly and falls back to `/get_weight_version`. The
collector also logs consecutive query failures at warning level. The rerun
(job `27263`, 15 min, same nodes) passed every criterion **with the staleness
machinery observable**: per-rollout `Staleness stats` lines recycled=0 with
avg_staleness 0.0/0.0/0.8/0.7/0.8 and max_staleness 0/0/1/2/2 (touching but
not exceeding the bound of 2 — nothing was eligible for recycling under
prefetch=1), zero `Failed to query engine weight version` warnings,
`train_async.py` + persistent worker active, additivity max residual 2.98e-8,
both DAgger identities ≤ 5.96e-8, both branches nonzero, completed queue 0 at
every drain (cap 32 never approached), weight versions finite
(mean 1.0 → 3.58), zero alignment/malformed/rescore events, and a clean
`16 tasks stop → worker stopped → thread stopped` shutdown (the tail
tracebacks are wandb atexit teardown noise, not miles state).

## Gate 14: Twenty-Step Geo3K Multi-Turn Hybrid Fully-Async Reference

After Gate 13 passes, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/07b-geo3k-multiturn-hybrid-fully-async-gate
```

`07b` sources the complete `07a` recipe and changes only the optimizer-step
count from five to twenty and the W&B run name to
`opd-mm-07b-geo3k-mt-hybrid-fully-async-gate`. Apply every Gate 13 invariant at
all steps. Retain a machine-readable snapshot covering both objective branches,
loss identities, multi-turn composition, weight versions, scoring transport,
gradient, timing and throughput; retain queue and recycle evidence from the log
until it is promoted to structured W&B telemetry.

Use `06b` as the synchronous systems reference. Compare active-token-normalized
cost, generated-token throughput, trainer waiting and teacher scoring latency.
Do not interpret raw step-time boundaries as equivalent between synchronous and
fully-async drivers, and do not claim model quality or an unbiased off-policy
estimator from this short bounded-staleness run.

**Result: passed.** Job `27272` (36 min, slinky-[26,44,13], miles `6253d3f5`)
completed all twenty steps on 2026-07-18. (The first submission, job `27266`,
was killed by the preflight NCCL probe: slinky-30 memlock 8 MiB regression,
`ibv_create_cq ENOMEM` on all 8 of its ranks — the 7th node minted by the
slurmd-restart memlock class; resubmitted with it excluded.) Every Gate 13
invariant held at all steps: additivity max residual 5.96e-8, DAgger identity
residuals ≤ 1.19e-7, both branches nonzero, 64 = 64 = 64
sample/request/teacher-request per step, zero retries, zero alignment errors.
The staleness machinery stayed observable end to end: 20/20 rollouts printed
`Staleness stats` (avg 0.0–0.9; max touched the bound of 2 in 11/20 rollouts
and never exceeded it; 0 recycles; 0 version-query failures), the completed
queue stayed 0–1 against its cap of 32, and weight versions ran finite
(mean 1.0 → 18.8). Objectives landed in the same last-five bands as the
synchronous 06b reference: sampled RKLD 0.391 → 0.211 (06b 0.221) and coarse
KL 0.465 → 0.182 (06b 0.162); teacher top-2 mass 0.922 → 0.911 with
non-worsening rest-mass error (max 0.074). Scheduling A/B against 06b using
the prescribed token-normalized quantities: 13.42 vs 20.44 ms/active token
(−34%), trainer wait-ratio median 0.774 vs 0.854, generated-token throughput
644 vs 524 tokens/GPU/s (+23%), teacher scoring e2e mean median 0.38 vs
0.58 s (async spreads requests instead of bursting them per step). The
measured train/rollout mismatch grew as designed and stayed small: logprob
abs-diff median 0.0145 (max 0.0295) vs 06b's 0.0127 (max 0.0151);
train-rollout KL median 0.00104 vs 0.00055. Watch items: grad-norm 113 at
step 0 on a short-response batch (early short-batch class — steps 1–19 stayed
4.0–62.8, median 8.3, vs 06b's 3.9–27.0; no clipping enabled, both objectives
descended smoothly), and a recycle event has never been exercised on-cluster
because nothing exceeded the bound at prefetch 1. Full per-step series,
collector telemetry and explicit missing-evidence items are in
`results/07b-job-27272.json`.

## Gate 15: Five-Step Synchronous OPD + Task-RL Composition Smoke

Run the focused argument, rollout conversion and OPD tests on the cluster, then
launch:

```bash
python -m pytest -q \
  tests/fast/utils/test_arguments.py \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08a-geo3k-multiturn-opd-rl-sync-smoke
```

This gate changes only the base scalar reward relative to `06a`. Verify that
`rollout/raw_reward` is still the unmodified math score, while the optimization
reward is group-normalized and enters the policy advantage with coefficient
one. Every `06a` hybrid, scoring, sparse-target and multi-turn invariant must
remain valid. In particular, task reward must not alter teacher sampled
log-probs or native `[T,2]` targets, and masked observation positions must stay
inert in all three objective components.

Do not advance on an all-constant reward sample: at least one prompt group must
contain both correct and incorrect samples so group centering is exercised.
Retain per-step raw reward, advantage mean/min/max, sampled RKLD, DAgger loss,
coarse KL, total loss, gradient norm and request counts.

**Result: passed.** Focused suite `141/141`, then job `27415` (22 min,
slinky-[5,28,56], miles `38ad568a`) completed all five steps on 2026-07-18.
Every 06a invariant held: additivity max residual 5.96e-8, CE identity
7.45e-8, coarse-KL identity 2.98e-8, both branches nonzero, 64 = 64 = 64
sample/request/teacher-request per step, zero retries, zero alignment errors,
`valid_position_ratio` = 1.0 at every step (masked observation rows inert in
all three objective components), rounds > 1 every batch. `rollout/raw_reward`
remained the unmodified math score (0.45–0.86). The task component's presence
in the policy advantage was verified positively, not assumed, via two
independent observations: (1) `rollout/rewards` means moved from the hard 0.0
of every 02–07 run to ±1e-9/−3.7e-9 float residues — the signature of real
nonzero group-centered per-sample values being averaged; (2) a
float-fingerprint discriminator: in 06b/07b
(rewards hard-zero) `train/pg_loss` equals `train/opd_reverse_kl` bit-exactly
at all 40 steps, while 08a shows residues up to 5.96e-8 at 2/5 steps — exactly
the signature of a nonzero group-centered A_task entering per-sample advantages
and cancelling from the scalar loss by construction (the mean of centered
values is zero; the signal lives in the gradient). Note the scalar-loss
equality is therefore expected and NOT evidence of a missing task component —
`train/pg_loss` cannot show it; the residue pattern and reward telemetry can.
These fingerprints establish that at least one prompt group had reward
variation, which is the smoke criterion. The contemporaneous
`rollout/zero_std/*` series cannot establish that every group was mixed:
before the metric fix it inspected OPD's teacher-response payload rather than
`sample.metadata["raw_reward"]`. Future reruns use the corrected task-reward
source for all-zero/all-one group percentages.
Gradient norms 7.8–34.8 with the familiar short-batch step-0 peak.

## Gate 16: Twenty-Step Synchronous OPD + Task-RL Reference

After Gate 15 passes, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08b-geo3k-multiturn-opd-rl-sync-reference
```

Apply every Gate 15 invariant at all twenty steps. This is the matched
synchronous reference for separating task-objective behavior from scheduling
effects; retain a machine-readable snapshot with all three branches and the
same timing/memory evidence contract as `06b`.

**Result: passed.** Job `27423` (32 min, slinky-[5,28,56], miles `38ad568a`)
completed all twenty steps on 2026-07-18. Every Gate 15 invariant held:
additivity max residual 8.94e-8, identities ≤ 1.19e-7, 64 = 64 = 64 with zero
retries and zero alignment errors, `valid_position_ratio` = 1.0, rounds > 1
at every step. The task component stayed verifiably active (e-9 reward-mean
residues at 18/20 steps and pg-vs-`opd_reverse_kl` float residues at 8/20
steps). Per-group task-reward composition was not measured correctly in this
historical run; its pre-fix `rollout/zero_std/*` series read teacher payloads.
The OPD objectives were not distorted by the
added reward: sampled RKLD 0.320 → 0.231 and coarse KL 0.374 → 0.173 —
matching the reward-free 06b bands (0.322 → 0.221 / 0.379 → 0.162) — with
teacher mass 0.920 → 0.911 and non-worsening rest error (max 0.074).
Token-normalized 21.06 ms/token, within the 06b band (20.44). Two watch
items: (1) an isolated grad-norm spike of 216.8 at step 13 (neighbors normal,
loss kept descending; largest in the ladder — still a single event, no
clipping change justified); (2) raw task reward declined 0.584 → 0.413
first-five/last-five while `truncated_ratio` rose 0.45 → 0.70–0.75 and mean
response length grew to ~3,200. Truncated responses are still scored and can
be correct, so the observed decline is strongly correlated with the truncation
ceiling rather than mechanically caused by a zero-reward rule. This is a
sample-mix/budget diagnostic, not evidence that reward optimization degraded
the task; a longer generation budget is the lever if it persists.

## Gate 17: Five-Step Fully-Async OPD + Task-RL Prefetch-Two Smoke

After Gate 15 passes, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08c-geo3k-multiturn-opd-rl-fully-async-smoke
```

The objective is identical to Gate 15. Scheduling matches `07a` except that
`fully_async_prefetch_batches=2`, allowing 32 prompt groups / 128 sample
requests to be active. Keep `max_weight_staleness=2`, completed-queue cap 32
and update interval one. Advance only when every objective invariant remains
valid, engine versions are observable, queue growth is bounded, stale work is
recycled rather than trained, and the wider window makes forward progress
without teacher overload, OOM or shutdown failure.

**Result: passed.** The first submission (job `27424`) was killed by the
preflight NCCL probe: slinky-54's memlock cap is still 8 MiB
(`ibv_create_cq ENOMEM` on all 8 of its ranks, 2/2 attempts) — despite the
same-day fleet fix; direct probe evidence supersedes the running-job inference
that had cleared it. The resubmission (job `27425`, 23 min,
slinky-[52,36,20]) completed all five steps. Every objective invariant held
(additivity 5.96e-8, 64 = 64, zero retries/alignment errors), the task
component stayed active by the two valid observations from Gate 15; historical
per-group task-reward composition remains unavailable. In addition —
the load-bearing first — **the recycle path was exercised on-cluster for the
first time**: at prefetch 2, five prompt groups generated at weight version 1
reached staleness 3 > 2 when the engine hit version 4 and were reset and
returned to the data source rather than trained (`Recycled stale group …`
log lines retained; final-rollout stats recycled=5, avg 1.4, max 3). Forward
progress continued through the recycles, the completed queue peaked at 1
against its cap of 32, engine-version queries never failed, and the worker
shut down cleanly. The 07b watch item "a recycle event has never been
exercised" is closed.

## Gate 18: Twenty-Step Fully-Async OPD + Task-RL Prefetch-Two Gate

After Gates 16 and 17 pass, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08d-geo3k-multiturn-opd-rl-fully-async-gate
```

Compare Gate 18 against Gate 16 for objective behavior and against `07b` for
producer-window behavior. Use active-token-normalized cost rather than raw step
boundaries. Retain average/max staleness, stale recycle count, completed queue,
trainer wait ratio, teacher latency, task reward, sampled RKLD, coarse KL,
Rest-mass error, train/rollout mismatch, gradients and explicit missing
evidence. Passing this gate validates the OPD + task-RL mechanism under bounded
fully-async scheduling; it does not establish downstream quality from twenty
shuffled optimizer steps.

**Result: passed.** Job `27427` (18 min, slinky-[5,56,28], miles `38ad568a`)
completed all twenty steps on 2026-07-18. Every Gate 15/17 invariant held:
additivity ≤ 8.94e-8, identities ≤ 8.94e-8, 64 = 64 with zero retries and
zero alignment errors, `valid_position_ratio` = 1.0, and the task component
verifiably in the advantage through reward-mean residues and a pg-vs-RKLD
fingerprint at 6/20 steps. The historical pre-fix `rollout/zero_std/*` series
does not establish all-groups mixed correctness. **Recycling ran as a sustained
regime, not a one-off**: 33 prompt
groups exceeded the staleness bound across the run and were reset and
regenerated (10.3% regeneration overhead against 320 accepted groups);
accepted-sample staleness averaged 0.0–1.6 per rollout; the completed queue
peaked at 1 against its cap of 32; engine-version queries never failed; the
worker shut down cleanly. Objective behavior vs Gate 16: same last-five bands
(sampled RKLD 0.407 → 0.216 vs 08b 0.231; coarse KL 0.489 → 0.189 vs 0.173).
Producer-window behavior vs `07b`, token-normalized as prescribed:
**8.36 ms/active token vs 13.42 (pf1) and 21.06 (08b sync)** — the ladder
519 → 644 → 1,181 tokens/GPU/s with trainer wait-ratio 0.854 → 0.774 → 0.621
— while the measured train/rollout mismatch rose modestly with the wider
window (logprob abs-diff median 0.0128 sync → 0.0145 pf1 → 0.0161 pf2,
max 0.0383; train-rollout KL median 0.00111). Watch items: a 204.3 grad-norm
spike at step 4 (with 08b's 216.8 this is the second ~200-class event, both
in the reward arm — nothing above 113 in 02–07; N=2 upgrades it to a pattern
worth a dedicated look before long runs, though no clipping change is made
from these gates); the raw-reward trajectory again declined under a rising
truncation ceiling (same confound as Gate 16); and recycled generation is
unmetered work — 33 recycled at pf2 vs 5 in the pf2 smoke and 0 at pf1 —
promote recycle counts to structured telemetry before pushing prefetch
higher. Full series in `results/08d-job-27427.json`.

## Gate 19: Same-Size Fully-Async Pure-Hybrid Smoke

Download `Qwen/Qwen3-VL-8B-Thinking` under the shared model cache, then launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09a-geo3k-multiturn-hybrid-fully-async-same-size-smoke
```

Confirm that the teacher is `Qwen3-VL-8B-Thinking`, the student remains
`Qwen3-VL-8B-Instruct`, and fully-async prefetch is two. The effective objective
must match `07` exactly: sampled RKLD coefficient one, Top-2 + Rest coefficient
`0.5`, task reward logged but absent from optimization. Reject any run containing
`--opd-optimize-task-reward`.

Apply all Gate 13 hybrid, exact-suffix, multi-turn, queue and staleness
invariants. Raw task reward must be finite and present, but it need not increase:
the batches are shuffled online samples consumed in completion order.

**Result: passed.** `Qwen/Qwen3-VL-8B-Thinking` was downloaded to the shared
cache (17 GB, vision tower verified in `config.json` alongside both existing
checkpoints). Job `27430` (15.5 min, slinky-[20,36,52], miles `77bc0c79`)
completed all five steps on 2026-07-18. Teacher identity confirmed in the
serving line (`--model-path .../Qwen3-VL-8B-Thinking --tp 8`); the pure-hybrid
contract is doubly fingerprinted — `train/pg_loss` equals
`train/opd_reverse_kl` bit-exactly at every step AND `rollout/rewards` is hard
0.0 (the same discriminator that proves presence in the 08 runs proves absence
here); no `--opd-optimize-task-reward` appears in the effective command.
Additivity 5.96e-8, 64 = 64 with zero retries and zero alignment errors,
`valid_position_ratio` = 1.0, rounds > 1, raw task reward present as telemetry
(0.34–1.0). Prefetch-2 staleness behaved as in 08: 5 groups recycled at
staleness 3, queue peak 2/32, zero version-query failures, clean shutdown.
Early matrix signal worth carrying into Gate 21: the same-size teacher runs
hotter on Top-2 coverage (teacher mass 0.915–0.962 vs the 30B arm's ~0.92
band) and the smoke's hybrid loss sits lower (0.49 vs 0.67 at step 4) — the
8B-Thinking distribution is closer to the 8B-Instruct student.

## Gate 20: Big-to-Small Fully-Async Pure-Hybrid Smoke

Launch the matched prefetch-two smoke with the established 30B teacher:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09c-geo3k-multiturn-hybrid-fully-async-big-small-smoke
```

The only intended model-contract difference from Gate 19 is the teacher
checkpoint: `Qwen3-VL-30B-A3B-Thinking`. Keep the 8B-Instruct student, TP=8
teacher service, objective, data, sampler, prefetch window and age bound fixed.

**Result: passed.** Job `27429` (13 min, slinky-[5,56,28], miles `77bc0c79`)
completed all five steps on 2026-07-18 with the established 30B teacher at the
identical prefetch-2 contract. Same double fingerprint of the pure-hybrid
objective (bit-equal `pg_loss`/`opd_reverse_kl`, hard-zero `rollout/rewards`),
additivity 5.96e-8, 64 = 64, zero retries/alignment errors,
`valid_position_ratio` = 1.0, rounds > 1, raw reward telemetry present
(first batch scored 1.0 — an easy draw, noted only as telemetry). Staleness:
7 groups recycled at staleness 3, queue peak 0, zero query failures. The two
arms are now matched on code revision, prefetch window and topology; only the
teacher checkpoint differs going into Gates 21/22.

## Gate 21: Same-Size 200-Step Gate

After Gate 19 passes, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09b-geo3k-multiturn-hybrid-fully-async-same-size-gate
```

Retain a machine-readable snapshot covering reward, response length and rounds,
both OPD branches, DAgger identities, teacher transport, gradient, staleness,
queue behavior and active-token cost. Confirm `num-rollout=200` and that the
effective command contains no checkpoint-save option.

**Result: passed.** Job `27432` (73 min, slinky-[52,36,20], miles `77bc0c79`)
completed all 200 steps on 2026-07-18 — the ladder's first long window. Every
invariant held at every step: additivity 5.96e-8, identities ≤ 8.94e-8, the
pure-hybrid double fingerprint (bit-equal `pg_loss`/`opd_reverse_kl` at 200/200
steps, hard-zero `rollout/rewards`), 64 = 64 with zero retries and zero
alignment errors, `valid_position_ratio` = 1.0, no checkpoint-save option.
Staleness machinery at scale: 50 groups recycled (1.6% overhead), observed max
staleness 4 (recycled per the bound), queue peak 5/32, and 4 transient
version-query failures that recovered without cascading (the 6253d3f5
fallback + warning working in production). Objectives near-converged: sampled
RKLD 0.257 → 0.055 (mid) → 0.032 (last-five); coarse KL 0.258 → 0.040 → 0.024.
**Teacher Top-2 mass was flat 0.937 → 0.937 with rest-mass error declining
0.052 → 0.019 — the K=2 coverage-narrowing watch item does not materialize at
200 steps in this arm.** Cost: 5.71 ms/active token median, 1,674
tokens/GPU/s, steady step 18.9 s. Reward telemetry: full-run mean 0.306 with
windows 0.669 (completion-order-biased first window) → 0.327 → 0.266 as active
length grew 1,658 → 3,254 and truncation reached 0.75 — the length-drift /
truncation-ceiling pattern, now sustained over a long window. Grad norms
0.4–41.2 (median 0.8, no ~200-class spike in this reward-free arm). Full
series in `results/09b-job-27432.json`.

## Gate 22: Big-to-Small 200-Step Gate

After Gate 20 passes, launch:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09d-geo3k-multiturn-hybrid-fully-async-big-small-gate
```

Compare Gates 21 and 22 as a 200-step teacher model matrix. Do not compare only
the first/last raw-reward points: report full-run means/windows and condition
them on response length and completion order. The gate establishes compatible
fully-async hybrid training, not a causal downstream-quality ranking. Confirm
`num-rollout=200` and no checkpoint-save option before launch.

**Result: passed on the second run.** The first submission (job `27431`,
student mem fraction 0.85) failed at step 65: a student rollout engine OOMed
(4.64 GiB burst against 2.84 GiB free plus 9 GiB fragmented reserve) as
responses drifted long under prefetch 2; the engine death cascaded — router
500s on the version endpoint (loudly, thanks to the warning added in
`6253d3f5`), then the trainer died on `pause_generation` (connection
refused). Fix `e67ceed4` parameterized `OPD_STUDENT_MEM_FRACTION` (default
0.85 in the shared base); the 09d gate now defaults the successful long-window
setting to 0.80. The rerun (job `27435`, 72 min, slinky-[5,56,28], fraction
0.80) completed all 200 steps with **zero OOM**. Every invariant held:
additivity 5.96e-8, identities 7.45e-8, pure-hybrid double fingerprint at
200/200 steps, 64 = 64, zero retries/alignment errors, no checkpoint saving.
Staleness: 53 recycled (1.7%), max observed 4, queue peak 3, zero query
failures. The 200-step teacher matrix vs Gate 21, per the pre-registered
discipline (full-run windows, conditioned on length; first batch excluded as
a completion-order artifact): the student closes on the same-size teacher
roughly 3× further at every window (RKLD 0.361 → 0.117 → 0.095 vs 09b's
0.257 → 0.055 → 0.032; coarse KL 0.437 → 0.085 → 0.069 vs 0.258 → 0.040 →
0.024) — a distribution-proximity result, not a quality ranking. Teacher
Top-2 mass flat at 0.927 → 0.927 (coverage stable in BOTH arms). Both arms
double active response length (~1,660 → ~3,120) with truncation 0.66; the 30B
arm holds a higher reward band throughout (full-run mean 0.368 vs 0.306,
last-five 0.344 vs 0.266) with lower truncation. Completion order and the
different length/truncation trajectories prevent a causal teacher-quality
ranking.
Cost parity at the current topology: 5.78 vs 5.71 ms/active token. Teacher
scoring remains a small part of the overlapped window; a retopologized
GPU-seconds-per-scored-token study remains future work as scoped. Grads
0.6–79.3 (median 1.0). Full series in `results/09d-job-27435.json`.

## Failure Triage

- `scoring_suffix_ids` rejected as an unknown field: the teacher is running an
  older SGLang checkout.
- `end in a text position` mRoPE error: the processed prompt ends in a visual
  region; the current pure-text suffix contract does not apply to that prompt.
- returned token ID mismatch: stop the run; do not quarantine or train on the
  sample. Capture the expected and returned IDs plus the teacher server log.
- any `opd_scoring/student_request_count > 0` in `03` or `05`: stop the run; the recipe
  has entered a legacy student-rescore route instead of trainer-direct DAgger.
- missing or malformed `[T,2]` targets: stop before interpreting the loss; save
  the teacher response metadata and the failing Sample indices.
- `interaction/rounds/max` never exceeds one in `04a`:
  inspect environment termination and generated answer formatting before
  interpreting OPD metrics.
- `interaction/observation_tokens/mean` or
  `interaction/observation_token_ratio` never exceeds zero in `04a`: the run
  did not exercise the masked tool-feedback contract; do not pass Gate 7.
- any `04b` step lacks a real follow-up turn or masked observation span: retain
  the failing batch metrics and do not pass Gate 8 even if the loss is finite.
- nonzero `train/pg_loss` in `05`, missing DAgger identities, or a non-inert
  observation target row: stop before interpreting the optimization curves.
- zero objective branch, failed total-loss additivity, duplicate teacher
  requests or any Student SGLang request in `06`: stop before interpreting the
  hybrid curves; the intended single-response composition is not active.
- missing fully-async worker startup, unbounded completed-queue growth, repeated
  stale recycling without batch progress, or a shutdown hang in `07`: stop the
  run and retain queue/staleness logs before changing the prefetch window.
- nonzero raw task reward but zero task component in `08`, non-centered GRPO
  task rewards, or task-reward changes to teacher targets: stop before
  interpreting OPD + RL curves; the explicit composition contract is broken.
- teacher queue growth or memory pressure: record queue depth, scoring latency
  and GPU memory before setting `OPD_SCORING_MAX_INFLIGHT` to a finite value.
