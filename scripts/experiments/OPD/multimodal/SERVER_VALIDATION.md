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
RKLD 0.322 → 0.221 AND coarse KL 0.379 → 0.162 — the hybrid drives the sparse
objective as hard as pure DAgger (05b: 0.431 → 0.174) while simultaneously
lowering the sampled objective. Coverage narrowing is milder than pure DAgger:
teacher top-2 mass 0.920 → 0.901 (05b reached 0.881) and rest-mass error
0.064 → 0.061 (non-worsening, max 0.070) — the RKLD anchor appears to hold the
policy in better-covered regions. Token-normalized 20.44 ms/token inside the
04b (19.78) / 05b (20.53) band; steady step 47.1 s median; gradient norms
3.9–27.0 (median 6.6). The first 06b submission (job 27187) died pre-training
to a Ray bring-up race — the launcher's GPU-wait loop timed out at 0/16 and
proceeded instead of retrying (bring-up fix candidate); the resubmission hit
the same race but the workers joined before the trainer scheduled, so the run
proceeded normally. Full per-step series in `results/06b-job-27208.json`;
per-step actor memory remains the explicit missing-evidence item.

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
- teacher queue growth or memory pressure: record queue depth, scoring latency
  and GPU memory before setting `OPD_SCORING_MAX_INFLIGHT` to a finite value.
