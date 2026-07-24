# OPD Multimodal Bring-Up

This folder advances multimodal OPD through isolated cluster gates. Milestone
`00` validates the direct SGLang teacher-prefill contract. Milestone `01` then
exercises the same contract through Miles' production OPD scoring and parsing
path. Milestone `02` is the first end-to-end trainer gate: single-turn Geo3K
sampled RKLD on the three-node teacher/actor/rollout layout. Milestone `03`
keeps that validated layout fixed and isolates trainer-side teacher Top-K + Rest
DAgger. Milestone `04` changes only the rollout sequence and establishes the
Geo3K multi-turn sampled-RKLD reference; milestone `05` applies the sparse
Top-K + Rest objective to that same sequence. Both smoke/reference pairs passed.
Milestone `06` now prepares their synchronous hybrid composition before any
asynchronous experiment.

The G11 correction used by `02` is an exact text-suffix scoring contract. Miles
sends the rendered prompt and ordered raw images, the scoring SGLang server
builds its own multimodal prefix, and only then appends the sampled response
token IDs verbatim. It does not preserve or reuse a student-processor-expanded
visual prefix. The Miles capability flag is
`--sglang-mm-exact-scoring-suffix`; it defaults off so deployments using an
unpatched SGLang server retain the legacy request shape. The multimodal 01/02
recipes enable it explicitly.

Validate the G11 correction in increasing-cost order. From the Miles repository
root, first run the SGLang request/unit coverage and one-GPU generic VLM E2E on a
cluster node:

```bash
EXPECTED_SGLANG_SHA="$(git rev-parse HEAD:thirdparty/sglang)"
git submodule update --init --recursive thirdparty/sglang
test "$(git -C thirdparty/sglang rev-parse HEAD)" = "$EXPECTED_SGLANG_SHA"
cd thirdparty/sglang
PYTHONPATH=python:${PYTHONPATH:-} python3 \
  test/registered/unit/managers/test_io_struct.py
PYTHONPATH=python:${PYTHONPATH:-} python3 \
  test/registered/unit/managers/test_scoring_suffix.py
PYTHONPATH=python:${PYTHONPATH:-} python3 \
  test/registered/vlm/test_mm_scoring_suffix_e2e.py
cd ../..
```

The generic E2E deliberately supplies a non-canonical text suffix and proves
that SGLang returns the same token IDs. Then rerun `01` against the target
Qwen3-VL teacher, followed by the five-step `02a` trainer gate. Milestone `00`
does not need a rerun because it does not exercise `scoring_suffix_ids`.
The current gitlink pins `sglang-miles` at `27d5e97c`, which contains the
manually squash-merged exact-suffix PR `impossible-inc/sglang#3`
(`2b778c2d`) and the Qwen-VL pretokenized-ID preservation fix from
`impossible-inc/sglang#4`. The existing focused Gate 1/01 evidence remains the
validation record for the exact-suffix contract. PR `#4` closes the separate
multi-turn Student SGLang boundary below, so validate that part with the
one-GPU generation E2E and Gate 7 on the pinned tree. `SERVER_VALIDATION.md`
records the dependency history and accepted jobs.

Multi-turn generation has a second exact-action boundary. Megatron keeps the
HF-processor-expanded token stream used for training, while each Student SGLang
request receives a compact prompt with one placeholder per image plus the
verbatim sampled-action and observation IDs accumulated so far. SGLang expands
only the media placeholders. The rollout rejects the response if
`meta_info.prompt_tokens` differs from the expanded trainer prefix length,
which catches a decode/re-tokenize fallback before a mismatched history can
enter the next turn. Run the one-GPU generation E2E before milestones `04`-`06`:

```bash
SGLANG_VLM_E2E_MODEL_PATH="${HF_CACHE_DIR:-/data/shared/hf_cache}/models/Qwen3-VL-8B-Instruct" \
  python -m pytest -q \
  tests/e2e/sglang/test_vlm_multiturn_exact_action.py
```

The E2E deliberately inserts a non-canonical prior action whose token IDs
change under decode/re-encode, then verifies that the second-turn SGLang prompt
still contains the submitted IDs in order.

## 00: teacher prefill smoke

The probe uses the existing Geo3K single-turn data contract from
`scripts/experiments/disagg/geo3k-vlm-8b-disagg-thd-2node.sh`:

- student tokenizer/processor: `Qwen/Qwen3-VL-8B-Instruct`
- teacher server: `Qwen/Qwen3-VL-30B-A3B-Thinking`, TP=8
- data: `chenhegu/geo3k_imgurl`
- multimodal mapping: `image -> images`

Pre-stage the teacher once:

```bash
export HF_CACHE_DIR=/data/shared
hf download Qwen/Qwen3-VL-30B-A3B-Thinking \
  --local-dir "$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"
```

Submit the smoke through the normal Miles Slurm launcher:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/multimodal/00-teacher-prefill-smoke
```

The job does not train, save a checkpoint, or create a W&B run. It sends six
prefill requests: clean image A, image B immediately after A, and clean image B
for both sampled-token (`top_k=0`) and native teacher Top-K (`top_k=2`) modes.

The gate passes only when:

1. SGLang returns one finite sampled logprob for every response token, with
   token IDs exactly aligned after the leading placeholder is removed.
2. `top_k=2` parses into native `[T, 2]` teacher targets whose IDs fit the
   student's configured vocabulary.
3. Same prompt IDs with different same-size images change the teacher scores.
4. Image B scores agree with and without a preceding same-token/different-image
   request, ruling out an image-blind prefix-cache hit.

The machine-readable result is written to
`runs/<job-name>/<timestamp>/teacher_prefill_smoke.json`. Server startup output
is in the adjacent `envpack_server.log`.

To probe an already-running teacher instead of launching the local sidecar:

```bash
OPD_TEACHER_LAUNCH=0 \
OPD_TEACHER_URL=http://teacher-host:13141 \
HF_CACHE_DIR=/data/shared \
bash scripts/slurm/submit.sh OPD/multimodal/00-teacher-prefill-smoke
```

Useful overrides are `OPD_SMOKE_IMAGE_SIZE`, `OPD_SMOKE_TOP_K`,
`OPD_SCORING_TIMEOUT`, `OPD_SMOKE_IMAGE_TOLERANCE`, and
`OPD_SMOKE_CACHE_TOLERANCE`. Hardware-specific SGLang flags can be appended via
`OPD_TEACHER_EXTRA_ARGS`.

## 01: production image-scoring smoke

Submit the production-path gate with the same teacher sidecar setup:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/multimodal/01-production-image-scoring-smoke
```

Unlike milestone `00`, this probe does not build `/generate` payloads by hand.
It creates real `Sample` objects with ordered PIL images, then calls
`reward_func()` and `post_process_rewards()` for both sampled RKLD and native
teacher Top-K DAgger. One response position is assigned `loss_mask=0` to model
an inter-turn environment observation. The wrapper explicitly passes
`--sglang-mm-exact-scoring-suffix`.

The gate passes only when:

1. Production scoring includes the image and remains image-sensitive for both
   sampled and DAgger requests.
2. Same-token/different-image requests remain prefix-cache safe, and the
   persistent HTTP session is reused after the first request.
3. Active sampled rows remain finite and token-aligned; the masked row becomes
   a neutral sampled value of `0.0`.
4. DAgger retains native `[T, 2]` targets on active rows and maps the masked row
   to `ids=0`, `log_probs=-inf`, `valid_mask=false`.

The job still does not launch a trainer, create a W&B run, or save a checkpoint.
Its result is written to
`runs/<job-name>/<timestamp>/production_image_scoring_smoke.json`. The same
environment overrides documented for milestone `00` apply to `01`.

This is a production-boundary smoke, not the multi-turn training gate. The
existing Geo3K custom generator appends assistant and text-tool spans into one
`Sample`; milestone `04a` runs that path end to end.

## 02: single-turn sampled-RKLD trainer gate

Milestone `02` combines two already exercised configurations without adding a
new backend or parallel route:

- the dedicated three-node resource ownership from
  `scripts/experiments/OPD/math_3nodes/qwen3-8B.sh`;
- the Qwen3-VL processor, Geo3K mapping, Megatron Bridge, TP=4/SP and dynamic
  batching settings from
  `scripts/experiments/disagg/geo3k-vlm-8b-disagg-thd-2node.sh`.

The fixed layout is:

1. head node: `Qwen3-VL-30B-A3B-Thinking` teacher served by SGLang TP=8;
2. actor node: `Qwen3-VL-8B-Instruct` Megatron trainer with TP=4, DP=2,
   PP=1, SP enabled and CP=1;
3. rollout node: eight one-GPU `Qwen3-VL-8B-Instruct` SGLang engines.

The objective is sampled RKLD only. `--opd-log-prob-top-k=0` keeps rollout
decoding free of top-k extraction, one teacher prefill request supplies the
sampled-token log-probabilities, and Megatron computes
`log q_old(a_t|h_t) - log p_T(a_t|h_t)` before applying the detached coefficient
to the sampled-action policy loss. DAgger is disabled. The local `math` verifier
is exposed as `rollout/raw_reward`, but the optimization reward remains zero.

The suffix contract currently requires a non-empty rendered string prompt and
pure-text response/history tokens. The teacher-processed prefix must end at a
text position, where the terminal mRoPE coordinates are equal, before the exact
suffix can be extended linearly. Empty multimodal responses are skipped before
image encoding and without a scoring request. A structured prompt or a new image
introduced inside the suffix fails fast instead of falling back to the
decode/re-tokenize path; supporting that later turn requires a second
teacher-local processor pass over canonical messages and newly ordered media.
The current Geo3K single-turn recipe and its text-only environment observations
satisfy these constraints.
With the flag disabled, Miles emits the historical `input_ids + image_data`
payload and never sends `scoring_suffix_ids`; that compatibility mode does not
carry an exact-action guarantee and is not an acceptable G11 validation result.

The launch validator now rejects SGLang OPD before GPU work when the teacher URL,
custom reward hook, reward post-process hook, or per-sample RM mode is
misconfigured. Alternate custom hook implementations remain supported. The
validator does not run for non-OPD or Megatron-teacher configurations.

Pre-stage the owner-managed teacher at the same `HF_CACHE_DIR` used for the job:

```bash
export HF_CACHE_DIR=/data/shared/hf_cache
hf download Qwen/Qwen3-VL-30B-A3B-Thinking \
  --local-dir "$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"
```

Run the five-step smoke first:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/02a-singleturn-rkld-smoke
```

Run the matched 20-step gate alongside or after `02a` (the recipes share no
mutable state — HF direct load, separate run dirs and W&B identities — so the
two submissions can run concurrently):

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/02b-singleturn-rkld-gate
```

`02b` sources the complete `02a` recipe and changes only `OPD_NUM_ROLLOUT=20`
and the W&B run name. Neither recipe enables eval or checkpoint saving.

`--opd-scoring-max-inflight` remains `0` (uncapped) in the matched baseline.
The prior production smoke observed requests of about 104 KB at batch 64, but
image-prefill concurrency has not yet been memory-profiled on the teacher. Watch
teacher queue depth, latency and GPU memory during the first rerun; use
`OPD_SCORING_MAX_INFLIGHT` only if the uncapped run shows pressure.

The `02a` gate passes when:

1. all five optimizer steps complete without NaN, OOM, timeout, token-alignment
   error or missing multimodal trainer input;
2. each sample produces one teacher scoring request and no Student SGLang
   rescore request;
3. `rollout/opd_kl/k{1,2,3}/{mean,min,max}` and the train-side
   `opd_reverse_kl` metric remain finite;
4. `rollout/raw_reward` is present while the optimization reward and base GRPO
   advantage remain zero before the RKLD contribution;
5. `opd_scoring/*` records request counts, latency, bytes, retry counts and
   persistent-session reuse, and train/rollout step times remain finite.

The `02b` gate applies the same checks over 20 steps and establishes the
single-turn sampled-RKLD reference before teacher Top-K DAgger, multi-turn, or
fully async work begins.

## 03: single-turn teacher Top-K + Rest trainer gate

Milestone `03` changes only the objective selected by the `02a` base recipe.
The model pair, Geo3K data, exact multimodal suffix contract, teacher ownership,
Megatron TP=4/DP=2/SP/PP=1/CP=1 layout, rollout engines, optimizer, and task
reward telemetry remain fixed. The `02a` defaults still expand to sampled RKLD
with DAgger disabled; the new DAgger arguments are appended only when
`OPD_DAGGER_TOP_K > 0`.

Run the five-step isolated smoke first:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/03a-singleturn-topk-rest-smoke
```

Then run the matched 20-step gate:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/03b-singleturn-topk-rest-gate
```

Both wrappers set:

```text
--opd-kl-coef 0
--opd-log-prob-top-k 0
--opd-dagger-top-k 2
--opd-dagger-coef 1
--opd-dagger-loss cross_entropy
```

This is pure trainer-direct teacher Top-K distillation. One exact-suffix teacher
request returns both the sampled-token log-probabilities used for diagnostics
and native per-position `[T, 2]` teacher targets. `Sample` retains those sparse
IDs/log-probs/masks through rollout conversion and DP split; the Megatron loss
uses current student logits to compute Top-K + Rest cross entropy directly.
There is no response-wide candidate union, no arbitrary-ID request, no Student
SGLang rescore, and no detached Top-K scalar entering sampled-token PPO.

The `03a` gate passes only when:

1. all five optimizer steps finish without alignment, protocol, timeout, NaN,
   OOM, or missing-target errors;
2. every sample produces exactly one teacher request and zero student requests;
3. active targets are native `[T, 2]`, finite, unique per row, and inside the
   real student vocabulary; masked rows remain inert;
4. `train/pg_loss` is numerically zero, `opd_dagger/loss` is finite and nonzero,
   and `train/loss` is explained by the DAgger branch because sampled RKLD and
   task-reward optimization are both disabled;
5. `opd_dagger/cross_entropy` equals `explicit_ce + rest_ce`, and
   `opd_dagger/coarse_kl` equals `cross_entropy - teacher_entropy`, within the
   existing numerical tolerance;
6. teacher/student Top-K and Rest masses, `rest_mass_abs_error`, gradient norm,
   request/response bytes, scoring latency, and step time are finite.

The `03b` gate applies the same invariants over 20 steps. Its decision metrics
are explicit CE, Rest CE, total cross entropy, coarse KL, teacher/student Rest
mass, Rest-mass absolute error, sampled RKLD as a detached diagnostic, raw task
reward, scoring transport, actor memory, and steady step time. A falling Rest CE
alone is not an alignment result: the Rest-mass error must also be inspected.

Compare `03b` with the `02b` steady-step median of 38.94 seconds only after
normalizing for response-token count. Up to 10% regression is acceptable for
this correctness gate; 10–20% triggers a dedicated profile, and more than 20%
blocks the hybrid until diagnosed. Do not enable profiling in the canonical
run. If needed, rerun a short smoke with `MILES_PROFILE_OPD_DAGGER=1` so the
existing Stable-TP operator ranges appear in the cluster profiler.

## 04a: synchronous Geo3K multi-turn sampled-RKLD gate

Milestone `04a` sources the validated `02a` recipe and replaces only its data
and rollout-sequence arguments:

- data: `VeraIsHere/geo3k_imgurl_processed`;
- generator: `examples.geo3k_vlm_multi_turn.rollout.generate`;
- interaction config: `max_turns: 3` with the Geo3K feedback environment;
- objective: sampled RKLD only (`opd_kl_coef=1`, DAgger disabled);
- execution: synchronous, CP=1, five optimizer steps, no eval or checkpoint.

The initial Geo3K problem supplies the image. The current environment returns
text-only feedback, so later turns do not add media. The generator retains two
aligned representations: an expanded sequence for Megatron and a compact
media-prefix sequence for Student SGLang. Every sampled action ID is appended
verbatim to both; only SGLang expands the compact image placeholder. Assistant
spans use `loss_mask=1`, while tool spans use `loss_mask=0`. Miles sends the
complete position-preserving response suffix in one exact-suffix teacher
request per sample. The teacher scores the full sequence for autoregressive
context, then Miles makes masked tool rows inert before sampled RKLD reaches
the trainer.

Run the smoke after the focused server-side tests:

```bash
python -m pytest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py

SGLANG_VLM_E2E_MODEL_PATH="${HF_CACHE_DIR:-/data/shared/hf_cache}/models/Qwen3-VL-8B-Instruct" \
  python -m pytest -q \
  tests/e2e/sglang/test_vlm_multiturn_exact_action.py

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/04a-geo3k-multiturn-rkld-smoke
```

The gate passes only when:

1. all five optimizer steps finish without alignment, mask-length, protocol,
   timeout, NaN, OOM, or missing multimodal trainer-input errors;
2. every kept sample makes one teacher request and zero Student SGLang rescore
   requests, with zero retries under the canonical recipe;
3. `interaction/rounds/max` exceeds one, proving that at least one sample
   reaches a real follow-up turn rather than degenerating to single-turn
   generation;
4. `interaction/observation_tokens/mean > 0` and
   `interaction/observation_token_ratio > 0` in at least one batch, proving
   that masked tool feedback is present, while all active returned IDs remain
   exactly aligned; `interaction/raw_tokens/max` must be finite and no greater
   than the configured response-length cap, and
   `interaction/length_cap_ratio` must remain finite in `[0, 1]`;
5. sampled-RKLD, gradient norm, task-reward telemetry, scoring latency/bytes,
   rollout time, train time, and total step time are finite.

Job `26525` passed the gate on 2026-07-16 after the fixture-corrected fast subset
completed 82/82:

- all five optimizer steps completed without alignment, mask-length, NaN, or
  OOM errors;
- every batch contained both single-turn episodes and a sample reaching
  `max_turns=3`; the per-step mean round count was 1.81–2.11;
- raw response length exceeded active-only length by 40.55–54.59 tokens/sample
  in every batch (mean 49.07), proving that masked feedback spans were exercised;
- sampled RKLD moved from 0.40216 to 0.25557 over the smoke;
- the retained run log reported one teacher request per kept sample and zero
  Student SGLang requests.

The exact plotted sequences are retained in
`results/04a-job-26525.json`. Request transport, scoring latency, memory,
task-reward, gradient-norm, and step-time series were not committed for this
smoke and are mandatory evidence for the longer synchronous reference. The
historical 04–06 jobs predate the response/context budget separation: their
configured response limit was reduced by the processor-expanded initial
prompt. They remain systems and objective evidence, but a current-code rerun is
required before using them to certify the response cap or
`interaction/length_cap_ratio`. The historical round extrema were logged
before the cross-rank extrema reducer was
fixed; because they equal the structural bounds 1 and 3, the gate conclusion is
unchanged.

`04a` is a sequence-contract smoke, not a complete Geo3K training result. Do
not infer convergence or model quality from five steps.

## 04b: 20-step synchronous Geo3K multi-turn sampled-RKLD gate

`04b` is the matched longer reference run after `04a` passed. Reproduce it
with:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/04b-geo3k-multiturn-rkld-gate
```

The wrapper sources the complete `04a` recipe and changes only
`OPD_NUM_ROLLOUT=20` and the W&B identity
`opd-mm-04b-geo3k-mt-rkld-gate`. It therefore keeps the teacher and student,
Geo3K data and generator, synchronous sampled-RKLD objective, exact-suffix
scoring contract, task-reward observation, and TP=4/DP=2/SP/PP=1/CP=1 trainer
layout fixed. It does not enable evaluation, checkpoint saving, Top-K + Rest,
hybrid training, or asynchronous scheduling.

Before launch, rerun the focused rollout tests listed for `04a`. The declared
gate requires all 20 optimizer steps to complete and:

1. teacher requests equal kept samples, Student SGLang requests remain zero,
   and active returned token IDs stay exactly aligned;
2. every step contains a real follow-up turn and masked observation span:
   global `interaction/rounds/max > 1`,
   `interaction/observation_tokens/mean > 0`, and
   `interaction/observation_token_ratio > 0`; additionally,
   `interaction/raw_tokens/max` is finite and no greater than the configured
   response-length cap, and `interaction/length_cap_ratio` remains finite in
   `[0, 1]`;
3. sampled RKLD, gradient norm, raw task reward, request count/bytes, scoring
   latency, actor memory, rollout/train/total step time, and generated-token
   throughput remain finite;
4. no alignment, mask-length, protocol, timeout, retry, NaN, or OOM error
   occurs.

Job `26594` passed the functional gate on 2026-07-16 after the focused subset
passed 82/82. It completed all 20 steps on `slinky-[10,36,31]` in 38 minutes:

- teacher requests equaled kept samples (`1,280/1,280`) with zero retries; the
  run log reported zero Student SGLang requests and zero alignment/mask errors;
- global round max was 2 or 3 and raw response length exceeded active-only
  length at every step;
- sampled RKLD moved `0.387 → 0.218`; the first-five and last-five means were
  `0.350 → 0.223`, so the lower endpoint is not a single-point artifact;
- raw task reward was telemetry only and did not improve by the same windows
  (`0.575 → 0.519`), so this run makes no model-quality claim;
- scoring e2e latency averaged 0.25 s, its maximum step-level p95 was 1.01 s;
  excluding the warm-up step, median step time was 42.8 s and median throughput
  was 520 tokens/GPU/s;
- gradient norm peaked at 51.75 on step 1, then settled to a last-five mean of
  4.14. The peak occurred on a short-response batch and does not establish a
  length-driven failure or justify clipping.

The historical round mean, now emitted as `interaction/rounds/mean`, is the
average number of assistant generation rounds per sample. The historical
raw-minus-active response gap, now emitted directly as
`interaction/observation_tokens/mean`, is the number of masked tool-feedback
tokens per sample. Their first-five to last-five means moved together:
`1.86 → 1.68` rounds and `43.0 → 31.9` masked tokens. This is consistent with
fewer or shorter follow-up interactions in the later batches, but prompt
shuffling and stochastic sampling prevent attributing it to policy behavior.
Every step still exercised both a real follow-up turn and a nonzero masked
span.

The exact machine-readable series are retained in
`results/04b-job-26594.json`. Multi-turn, RKLD, raw reward, gradient, transport,
and timing series are present. Per-step actor memory is not; only point-in-time
snapshots remain in the cluster log, so the evidence package carries that
explicit limitation.

New runs use `interaction/rounds/max`,
`interaction/observation_tokens/mean`,
`interaction/observation_token_ratio`, `interaction/raw_tokens/max`, and
`interaction/length_cap_ratio` for the gate contract. The removed
`multi_turn/*` series are not expected to appear.

## 05: synchronous Geo3K multi-turn teacher Top-K + Rest gates

Milestone `05` composes the two independently validated mechanisms without
introducing new training or inference code:

- `03` supplies trainer-direct teacher Top-2 + Rest DAgger;
- `04` supplies the Geo3K action/tool/action sequence and masked observations.

The numbered recipes retain fixed objectives: `04a` selects sampled RKLD and
`05a` selects isolated DAgger. Both source the complete `02a` base and then the
same objective-free `geo3k-multiturn-overlay.sh`, which owns only the processed
dataset, custom generator, rollout arguments and multi-turn telemetry. This
keeps `04a`/`04b` behavior immune to stray objective environment variables
while data, rollout assembly, exact-suffix scoring, optimizer and
TP=4/DP=2/SP/PP=1/CP=1 remain shared.

Run the focused rollout/data-conversion tests, then launch the five-step smoke:

```bash
python -m pytest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/05a-geo3k-multiturn-topk-rest-smoke
```

The canonical `05a` objective is:

```text
--opd-kl-coef 0
--opd-log-prob-top-k 0
--opd-dagger-top-k 2
--opd-dagger-coef 1
--opd-dagger-loss cross_entropy
```

One exact-suffix teacher request scores the complete position-preserving
multi-turn sequence and returns native `[T,2]` sparse targets. Tool-feedback
rows remain in the autoregressive context but become inert through
`loss_mask=0`; active assistant rows receive the differentiable trainer-direct
Top-K + Rest loss. The recipe performs no response-wide union, arbitrary-ID
request, or Student SGLang rescore. Raw task reward and sampled RKLD remain
detached diagnostics and do not contribute to the optimizer objective.

`05a` passes only when all five steps complete and:

1. `train/pg_loss` remains numerically zero while `opd_dagger/loss` and the
   total training loss are finite;
2. `CE = explicit_ce + rest_ce` and
   `coarse_kl = cross_entropy - teacher_entropy` hold within the established
   floating-point tolerance;
3. every active position carries aligned, finite, unique `[T,2]` targets, and
   every `loss_mask=0` observation row is inert;
4. each kept sample makes one teacher request, zero Student SGLang requests and
   zero retries under the canonical recipe;
5. every step contains a real follow-up turn and a nonzero masked observation
   span, and no alignment, mask-length, protocol, timeout, NaN or OOM error
   occurs;
6. teacher/student Top-K and Rest masses, Rest-mass absolute error, gradient
   norm, request bytes/latency and rollout/train/step timing remain finite.

Only after `05a` passes, launch the matched twenty-step gate:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/05b-geo3k-multiturn-topk-rest-gate
```

`05b` changes only `OPD_NUM_ROLLOUT=20` and the W&B identity. Retain an exact
machine-readable per-step snapshot covering the complete `05a` invariant set,
multi-turn composition, DAgger decomposition and masses, detached sampled RKLD,
raw task reward, gradients, scoring transport, actor memory and steady-state
timing/throughput. Compare performance with both `03b` and `04b` after
normalizing for generated and active response tokens; a raw step-time
difference alone is not an operator regression.

Both gates passed on 2026-07-17. Job `27087` completed `05a` in five steps;
job `27156` completed `05b` in twenty. Every step preserved both loss
identities, `train/pg_loss=0`, one teacher request per kept sample, zero
Student SGLang requests, zero retries and zero alignment errors. Across the
`05b` first-five/last-five windows, coarse KL moved `0.431 -> 0.174`; active
token-normalized cost was 20.5 ms/token, between the matched `03b` and `04b`
references. Teacher Top-2 coverage (`0.920 -> 0.881`) and Rest-mass error
(`0.068 -> 0.075`) remain diagnostics rather than pass/fail quality claims.

Neither recipe enables evaluation, checkpoint saving, hybrid sampled-RKLD +
DAgger, asynchronous scheduling, or context parallelism. The synchronous
hybrid is isolated in milestone `06` below.

## 06: synchronous Geo3K multi-turn hybrid OPD gates

Milestone `06` combines the two objectives that `04b` and `05b` established
independently, without adding a trainer, SGLang, Megatron or parallel-layout
path. The canonical coefficients are fixed in the wrapper:

```text
--opd-kl-coef 1
--opd-log-prob-top-k 0
--opd-dagger-top-k 2
--opd-dagger-coef 0.5
--opd-dagger-loss cross_entropy
```

`beta_RKLD=1` preserves the sampled-action anchor. `lambda_DAgger=0.5` is the
first conservative coefficient from the predeclared `{0.25, 0.5}` follow-up to
the text-only equal-weight hybrid: that run was finite and structurally
correct, but missed its sampled-RKLD preservation floor. This is a composition
choice, not a claim that `0.5` is optimal for Geo3K.

One exact-suffix teacher Top-2 response serves both objectives. Its sampled
token log-probs form the detached RKLD advantage, while its native `[T,2]`
targets form the differentiable trainer-direct Top-K + Rest loss. The trainer
therefore computes:

```text
total_loss = sampled_rkld_pg + 0.5 * topk_rest_cross_entropy
```

There is no second teacher request, response-wide candidate union or Student
SGLang rescore. The existing fast composition regression already verifies
loss and gradient additivity and verifies that neither sampled-teacher nor
sparse-teacher targets receive gradients.

Run the focused composition and rollout tests on the cluster, then launch the
five-step smoke:

```bash
python -m pytest -q \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/06a-geo3k-multiturn-hybrid-smoke
```

`06a` passes only when all five steps complete and:

1. `train/pg_loss` and `opd_dagger/loss` are both finite and nonzero;
2. the reported total loss equals their sum within
   `1e-5 * max(1, abs(total_loss))` at every step;
3. `CE = explicit_ce + rest_ce` and
   `coarse_kl = cross_entropy - teacher_entropy` retain the established
   floating-point identities;
4. teacher requests equal kept samples, with zero Student SGLang requests,
   retries, alignment failures and malformed sparse targets;
5. each step has a real follow-up turn and a nonzero masked observation span,
   while every observation row stays inert under `loss_mask=0`; and
6. both branch gradients, teacher/student masses, Rest-mass error, scoring
   latency/bytes and rollout/train/step timing remain finite.

Only after `06a` passes, launch the matched twenty-step gate:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/06b-geo3k-multiturn-hybrid-gate
```

`06b` sources `06a` and changes only `OPD_NUM_ROLLOUT=20` and the W&B identity.
Retain first-five/last-five windows for sampled RKLD, coarse KL, explicit/Rest
CE, teacher/student masses and Rest-mass error. Compare active-token-normalized
cost with `04b` and `05b`; raw step time alone is not sufficient. Because
on-policy batches are shuffled and stochastic, objective monotonicity is not a
hard gate. Passing `06b` establishes a synchronous systems reference before
fully async work; it does not establish downstream quality or an optimal loss
coefficient. Both `06` scripts are prepared and have not yet run.
