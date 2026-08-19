# OPD Multimodal Bring-Up

This folder advances multimodal OPD through isolated cluster gates. Milestone
`00` validates the direct SGLang teacher-prefill contract. Milestone `01` then
exercises the same contract through Miles' production OPD scoring and parsing
path. Milestone `02` is the first end-to-end trainer gate: single-turn Geo3K
sampled RKLD on the three-node teacher/actor/rollout layout. Milestone `03`
keeps that validated layout fixed and isolates trainer-side teacher Top-K + Rest
DAgger. Milestone `04` changes only the rollout sequence and establishes the
Geo3K multi-turn sampled-RKLD reference; milestone `05` applies the sparse
Top-K + Rest objective to that same sequence. Milestone `06` composes both
objectives from one teacher response. All synchronous smoke/reference pairs
through `06` passed. Milestone `07` is now prepared as a scheduling-only
comparison: it reuses the complete `06` hybrid contract in Miles' existing
fully-async rollout worker, without adding another OPD algorithm path.

## Canonical 200-step baseline

Use
`baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step.sh` as the frozen
end-to-end baseline. Unlike the numbered milestone wrappers, it is a complete
experiment recipe: model pair, processed Geo3K multi-turn data, teacher
sidecar, hybrid objective, fully-async scheduler, parallel layout, optimizer,
monitoring, and no-save guards are visible in one file. It imports only the
canonical `scripts/models/qwen3-8B.sh` model definition.

The baseline fixes the following training contract:

- Qwen3-VL-30B-A3B-Thinking SGLang teacher and Qwen3-VL-8B-Instruct student;
- 200 rollout steps on `VeraIsHere/geo3k_imgurl_processed`;
- sampled RKLD-PG with coefficient `1`;
- trainer-side Top-K + Rest DAgger with `K=2`, coefficient `0.5`, and
  `cross_entropy`;
- `--use-rollout-logprobs`, so both the detached RKLD reference and PPO
  denominator use the Student SGLang behavior-policy snapshot;
- explicit symmetric PPO clipping at `0.2/0.2`, with no TIS or dual clip;
- fully async prefetch `2`, maximum accepted weight staleness `2`, and no
  separate trainer old-logprob recomputation;
- task reward telemetry only and no checkpoint saving.

Submit it directly:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step
```

The numbered `00`-`11` files remain as development gates and historical A/B
evidence. New work should compare against the canonical baseline rather than
reconstructing a launch through the milestone inheritance chain.

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
- generator: `examples.geo3k_vlm.multi_turn.rollout.generate`;
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
coefficient.

Both gates passed on 2026-07-18. Job `27180` completed the five-step smoke and
job `27208` completed all twenty reference steps. The focused fast subset was
`90/90`; both branches stayed finite and nonzero; total-loss additivity held to
a maximum residual of `5.96e-8`; one teacher request served each kept sample;
and all DAgger identities and multi-turn mask invariants held. First-five to
last-five windows moved sampled RKLD `0.322 -> 0.221` and coarse KL
`0.379 -> 0.162`. These are within-run finite-training diagnostics, not model
quality measurements.

The hybrid reached a similar last-five coarse-KL band to the independent pure
DAgger reference (`0.162` versus `0.174`) while its teacher Top-2 mass ended in
a less narrowed band (`0.901` versus `0.881`). This pattern is consistent with
the sampled RKLD branch acting as an anchor, but one stochastic run per arm
does not isolate objective causality. Active-token-normalized cost was
`20.44 ms/token`, inside the `04b`/`05b` reference band.

Two evidence limits remain explicit. The retained JSON lacks per-step actor GPU
memory, and zero Student SGLang requests/alignment errors are retained in the
cluster log rather than as per-step JSON series. The next implementation gate
holds this synchronous hybrid contract fixed and changes scheduling only.

## 07: Geo3K multi-turn hybrid OPD under fully-async scheduling

Milestone `07` does not change the objective established by `06`. It switches
the training entry point to `train_async.py` and selects `--fully-async`
(`miles.rollout.fully_async_rollout.FullyAsyncRolloutFn` since the 2026-08 sync). The
persistent background worker continuously performs the existing per-sample
sequence:

```text
Student SGLang multi-turn generation
  -> one fixed-teacher exact-suffix Top-2 request
  -> completed hybrid Sample queue
  -> Megatron trainer consumes the next batch
```

Generation plus teacher scoring is therefore one producer pipeline, overlapped
with the trainer consumer. This first gate does not split teacher scoring into
a separate queue, change EnvPack, or add a new SGLang/Megatron backend path.

The two student probability roles remain deliberately different. Sampled
RKLD-PG uses a detached old-policy sampled-token score as its coefficient and
applies that coefficient to the differentiable sampled-action policy loss.
Top-K + Rest DAgger never uses that old sampled score: its fixed teacher
`[T,2]` targets reach the trainer, where the current student logits reconstruct
the explicit-token and Rest probabilities. Fully async scheduling may age the
sampled state, but it does not turn trainer-current DAgger logits into stale
rollout-side logits. The canonical `07` recipes do not enable TIS or any new
importance-ratio correction; they bound rollout-engine age and measure the
existing train/rollout mismatch instead.

The canonical scheduling settings are intentionally conservative:

```text
MILES_TRAIN_ENTRY=train_async.py
--fully-async-prefetch-batches 1
--fully-async-max-completed-queue-groups 32
--max-weight-staleness 2
--async-unused-samples-handler retry
--update-weights-interval 1
```

With `rollout_batch_size=16`, the worker keeps at most sixteen prompt groups
actively generating. The completed-queue soft cap allows two additional
batches to wait in CPU memory; already-running groups can finish beyond the
cap. Stale groups are reset and returned to the data source, which clears their
response, rollout log-probs, teacher sampled log-probs and sparse teacher
targets before regeneration.

Run the focused async lifecycle and hybrid tests before the five-step smoke:

```bash
python -m pytest -q \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/07a-geo3k-multiturn-hybrid-fully-async-smoke
```

`07a` advances only when all five optimizer steps preserve every `06a`
algorithmic invariant and additionally show:

1. `train_async.py` and the persistent fully-async worker are both active;
2. one teacher request still supplies sampled RKLD and native `[T,2]` targets,
   with zero Student SGLang rescore requests;
3. exact-suffix alignment, multi-turn masks, total-loss additivity and both
   DAgger identities remain valid;
4. accepted sample weight versions are finite and bounded, stale groups are
   recycled rather than trained, and the completed queue remains bounded;
5. persistent OPD transport closes on the worker's owner event loop; and
6. gradients, masses, scoring transport and rollout/train/total timing remain
   finite without NaN, OOM, timeout or deadlock.

After `07a` passes, launch the matched twenty-step gate:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/07b-geo3k-multiturn-hybrid-fully-async-gate
```

`07b` changes only the optimizer-step count and W&B identity. Compare it with
`06b` using active-token-normalized cost and trainer waiting, not raw step time
alone: an async step drains already-completed groups and therefore has a
different timing boundary. Retain sampled RKLD, coarse KL, Rest-mass error,
weight-version statistics, stale recycle count, queue observations, teacher
latency, active-token throughput and explicit missing-evidence notes. This is a
scheduling/staleness systems gate, not a full-training or model-quality claim.

Both gates passed on 2026-07-18: `07a` on its second run (job `27263`) and
`07b` on job `27272` (92/92 focused tests, including two added with the fix
below). The first smoke (job `27256`) completed all five steps with every
algorithmic invariant intact but exposed a real defect: the collector's
engine-weight-version query targeted the router's `/model_info`, which the
sgl-router does not expose (it only proxies the legacy `/get_model_info`), so
every query 404ed silently and the staleness filter ran inert — the same
silent-failure class as the earlier missing per-turn weight versions, one hop
further down the chain. Commit `6253d3f5` adds the endpoint fallback and logs
consecutive query failures at warning level; staleness is now an observed
quantity (20/20 rollouts printed stats in `07b`: avg 0.0–0.9, max touching
but never exceeding the bound of 2, zero recycles, zero query failures).
This router-specific fallback intentionally differs from
`sglang_engine.get_weight_version`, which contacts the SGLang server directly
and uses the legacy `/get_weight_version` route.

Under the identical hybrid contract, `07b` landed in the same last-five bands
as the synchronous `06b` reference (sampled RKLD `0.391 -> 0.211` vs `0.221`;
coarse KL `0.465 -> 0.182` vs `0.162`) while active-token-normalized cost fell
`20.44 -> 13.42 ms/token`, trainer wait-ratio fell `0.854 -> 0.774`, and
generated-token throughput rose `524 -> 644 tokens/GPU/s` — at the most
conservative window (prefetch 1). The measured train/rollout mismatch grew as
designed and stayed small (logprob abs-diff median `0.0127 -> 0.0145`;
train-rollout KL median `0.00055 -> 0.00104`). Two watch items are recorded in
the snapshot: a step-0 grad-norm spike (113, short-response batch, early
short-batch class) and the fact that no recycle event has yet been exercised
on-cluster. Evidence: `results/07b-job-27272.json`. These remain systems
measurements, not model-quality claims.

## 08: Geo3K multi-turn OPD + task RL

Milestone `08` promotes the task score that earlier recipes logged only as
`rollout/raw_reward` into an explicit, opt-in base RL objective. The logging
contract does not change: without `--opd-optimize-task-reward`, optimization
reward remains exactly zero. With the flag enabled, Miles applies the same
group reward transformation as the standard GRPO path and then composes the
three already-separated signals:

```text
A_task  = group_normalize(r_task)
A_total = task_reward_coef * A_task
          + opd_kl_coef * (log p_teacher(y | h) - log q_old(y | h))

L_total = L_PG(q_current, stop_gradient(A_total))
          + dagger_coef * L_TopK+Rest(q_current, teacher_targets)
```

The task coefficient is applied after group normalization, so it remains a
real objective weight rather than being divided back out by standardization.
The sampled RKLD term still uses the trainer's detached pre-update `q_old` for
the accepted batch; these recipes do not enable `--use-rollout-logprobs`.
Sampler staleness therefore changes which states/actions arrive, while the
reported train/rollout mismatch measures the gap between sampler logprobs and
that trainer recomputation. Top-K + Rest uses the differentiable logits from
the policy-loss forward. No TIS, new importance ratio, teacher request or
Student SGLang rescore is introduced.

The gate is staged so a new objective and a wider async producer window are not
first tested at the same time:

```bash
# 1. New task-RL + OPD composition, synchronous, five steps.
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08a-geo3k-multiturn-opd-rl-sync-smoke

# 2. Matched synchronous twenty-step reference.
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08b-geo3k-multiturn-opd-rl-sync-reference

# 3. Same objective under fully async scheduling, five steps, prefetch two.
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08c-geo3k-multiturn-opd-rl-fully-async-smoke

# 4. Matched fully async twenty-step gate.
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/08d-geo3k-multiturn-opd-rl-fully-async-gate
```

`08a/08b` source the completed synchronous `06` contract and add only:

```text
--opd-optimize-task-reward
--opd-task-reward-coef 1
```

`08c/08d` source the completed async `07` contract, add the same task-reward
flags, and set `--fully-async-prefetch-batches 2`. With rollout batch size 16
and four samples per prompt, this allows 32 prompt groups / 128 sample requests
to be active. The completed queue cap remains 32 groups and the accepted age
bound remains two weight versions. The existing warning contract permits this
window because `prefetch=2 <= max_weight_staleness+1=3`.

Advance `08a` only when task rewards are non-constant in at least one prompt
group and the resulting base GRPO advantages are finite and group-centered.
Every `06` invariant must still hold: both OPD branches nonzero, loss
additivity, both DAgger identities, one teacher response per sample, exact-ID
alignment and multi-turn masks. Also retain task reward, total advantages,
gradient norm and each objective branch so coefficient-scale failures are
distinguishable.

Advance `08c` only after `08a`; additionally require observable staleness,
bounded queue growth and forward progress at prefetch two. Run `08b/08d` only
after their matching smokes pass. Compare those twenty-step references using
raw task reward windows, sampled RKLD, coarse KL, Rest-mass error, gradient
scale, staleness, stale recycle count, trainer waiting and active-token cost.
Short shuffled runs remain mechanism evidence, not downstream quality proof.

The completed `07b` run is the OPD-only async control. A topology-matched pure
RL control remains a separate follow-up: it should not launch or score against
the fixed teacher, so it must not be faked by setting OPD coefficients to zero
inside the three-node OPD recipe.

All four `08` gates passed on 2026-07-18 (08a job `27415`, 08b job `27423`,
08c job `27425` after a preflight resubmission, 08d job `27427`; focused suite
`141/141`). The synchronous pair proved the objective composition: every 06
invariant held while the group-normalized math score verifiably entered the
GRPO base advantage — verified by two valid observations rather than assumed,
because group centering cancels the task term from the scalar loss by
construction (`rollout/rewards` float residues vs the hard 0.0 of every 02–07
run; a pg-loss vs `opd_reverse_kl` float fingerprint present only in the 08
runs). The historical `rollout/zero_std/*` values cannot establish that every
prompt group was mixed: before the metric fix they compared OPD teacher-response
payloads instead of `sample.metadata["raw_reward"]`. The fingerprints establish
the smoke requirement that at least one group had nonzero centered reward; a
future rerun must use the corrected metric for per-group composition. The OPD
objectives were not distorted: 08b landed in the reward-free 06b bands
(sampled RKLD `0.320 -> 0.231` vs `0.322 -> 0.221`; coarse KL
`0.374 -> 0.173` vs `0.379 -> 0.162`) at `21.06 ms/active token`.

The async pair proved the wider window: prefetch 2 exercised the recycle path
on-cluster for the first time (08c: 5 groups at staleness 3 reset and
regenerated; 08d: 33 across twenty steps, 10.3% regeneration overhead, none
trained) and closed the scheduling ladder at `8.36 ms/active token`
(sync `21.06` -> prefetch-1 `13.42` -> prefetch-2 `8.36`;
`519 -> 644 -> 1,181 tokens/GPU/s`; trainer wait-ratio
`0.854 -> 0.774 -> 0.621`), with the measured train/rollout mismatch rising
modestly (abs-diff median `0.0128 -> 0.0145 -> 0.0161`). Objective bands
matched the synchronous 08b reference.

Three watch items are recorded in the snapshots
(`results/08b-job-27423.json`, `results/08d-job-27427.json`): two ~200-class
grad-norm spikes, one in each 08 reference (216.8 and 204.3 — nothing above
113 in 02–07; both in the reward arm, so N=2 justifies a dedicated look
before long runs, but not a clipping change from these gates); the raw-reward
trajectory declining under a truncation ratio that rises to ~0.6–0.75 as
responses drift toward the Thinking teacher's length (a generation-budget
confound, not a quality claim in either direction); and recycled generation
being unmetered work that should become structured telemetry before prefetch
is pushed higher. Ops note: the first 08c submission (job `27424`) was
preflight-killed by slinky-54's persistent memlock cap — direct probe
evidence supersedes the running-job inference that had cleared that node.

## 09: Fully-async teacher behavior and capacity matrix

Milestone `09` branches from the completed pure-hybrid `07` contract, not from
the task-RL `08` recipes. Both arms remain multimodal, multi-turn and fully
async with prefetch two. They retain `--opd-log-task-reward` for observation but
do not add `--opd-optimize-task-reward`; the base GRPO reward is therefore zero
and all gradients still come from sampled RKLD-PG plus trainer-direct Top-K +
Rest:

```text
sampled RKLD coefficient = 1
teacher Top-K            = 2
Top-K + Rest coefficient = 0.5
task reward               = rollout/raw_reward telemetry only
```

The two model arms are:

1. same size: `Qwen3-VL-8B-Thinking` teacher to
   `Qwen3-VL-8B-Instruct` student (`09a/09b`);
2. big to small: `Qwen3-VL-30B-A3B-Thinking` teacher to the same
   `Qwen3-VL-8B-Instruct` student (`09c/09d`).

The big-to-small pair is the model pair already validated by `07`, but `09c/09d`
rerun it at prefetch two and on the same code revision as the same-size arm.
Both recipes pin the teacher to the existing TP=8 head-node layout. This first
matrix therefore changes the teacher checkpoint/capacity without introducing a
serving-topology change; a later cost study may retopologize each teacher and
must then report GPU-seconds per scored token rather than comparing latency
alone.

Download the additional same-size teacher before launching `09a`:

```bash
hf download Qwen/Qwen3-VL-8B-Thinking \
  --local-dir /data/shared/hf_cache/models/Qwen3-VL-8B-Thinking
```

Run the smoke for each arm before either 200-step gate:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09a-geo3k-multiturn-hybrid-fully-async-same-size-smoke

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09c-geo3k-multiturn-hybrid-fully-async-big-small-smoke

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09b-geo3k-multiturn-hybrid-fully-async-same-size-gate

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/09d-geo3k-multiturn-hybrid-fully-async-big-small-gate
```

`rollout/raw_reward` is the fraction of the current 64 sampled responses whose
last boxed answer matches the Geo3K label. It is an online training-batch
diagnostic, not a fixed evaluation set. Each step contains 16 shuffled prompts
with four correlated samples per prompt, uses temperature one, and the
fully-async collector consumes groups in completion order. Short/easy groups
can therefore enter earlier batches while long/hard groups finish later.

That distinction is visible in `07b`: step zero had raw reward `0.9375` but was
also an unusually short batch (mean raw/active response lengths `1021/552`),
whereas later batches commonly contained `2300-3360` raw tokens. First-five to
last-five reward moved `0.613 -> 0.450`; excluding the step-zero outlier, steps
1-5 averaged `0.534`. The synchronous references also moved down, but by
different amounts (`04b`: `0.575 -> 0.519`; `06b`: `0.616 -> 0.425`). Because
none of those objectives optimized task reward and the prompt batches differ at
every step, the repeated direction is a watch item, not evidence of a measured
quality regression.

Neither arm enables checkpoint saving: the inherited checkpoint arguments load
the 8B-Instruct student only, and the 09 wrappers reject `--save`, `--save-*` or
`--async-save` if a shared recipe introduces one later.

Do not require monotonic raw reward to pass `09`. Require finite reward logging,
then compare the two arms using the full-run reward distribution together with
response length, rounds, sampled RKLD, coarse KL, Rest-mass error, teacher
latency/bytes, staleness and active-token-normalized cost. A causal model-quality
claim requires a fixed held-out prompt set or matched per-prompt evaluation;
these 200-step online gates establish model-pair compatibility and training
behavior only.

All four `09` gates passed on 2026-07-18 (09a job `27430`, 09b job `27432`,
09c job `27429`, 09d job `27435` after an OOM resubmission; the smokes ran
13–15 min, the 200-step gates ~72 min each). The pure-hybrid contract is
doubly fingerprinted in every run (bit-equal `pg_loss`/`opd_reverse_kl` and
hard-zero `rollout/rewards`); every invariant held across 410 total optimizer
steps; no checkpoints were written.

The 200-step matrix, per the pre-registered discipline (full-run windows
conditioned on length; the first async batch excluded as a completion-order
artifact — it is the fastest/easiest generations and scores ~1.0 in every
async run): the student closes on the same-size `8B-Thinking` teacher roughly
3x further at every window (sampled RKLD `0.257 -> 0.032` vs `0.361 -> 0.095`;
coarse KL `0.258 -> 0.024` vs `0.437 -> 0.069`) — distribution proximity, not
a quality ranking. Teacher Top-2 mass is flat in BOTH arms (0.937 / 0.927
first-five = last-five) with declining rest error: the K=2 coverage-narrowing
watch item does not materialize at this horizon. Both arms roughly double
active response length (~1,660 -> ~3,100–3,250); under the fixed generation
budget, truncation reaches 0.66–0.75 while reward and completed-round telemetry
move down (full-run reward means 0.368 for the 30B arm vs 0.306 for the 8B arm).
Truncated samples are still scored and can be correct, so this is a strong
budget-pressure correlation rather than a deterministic reward rule or a
teacher-quality ranking. Cost is near-identical at this window: 5.71 vs 5.78
ms/active token; the retopologized
GPU-seconds-per-scored-token study remains future work.

Ops: the first 09d run (job `27431`) OOMed a student engine at step 65 under
prefetch-2 length drift; commit `e67ceed4` made the student mem fraction
overridable (`OPD_STUDENT_MEM_FRACTION`; shared-base default remains 0.85) and
the rerun at 0.80 completed clean. The 09d gate now defaults that override to
0.80 so the checked-in recipe reproduces the successful long-window setting.
Engine death still cascades to job death in the async
path (no engine-level recovery) — recorded as a launcher/manager follow-up.
Evidence: `results/09b-job-27432.json`, `results/09d-job-27435.json`.

## 10: Rollout log-probs as the old-policy reference

Milestone `10` is a paired old-policy-source experiment over the two completed
`09` long-window arms. It keeps the model pair, pure-hybrid objective, Geo3K
multi-turn data, prefetch-two scheduler, staleness bound, memory fractions,
teacher topology, 200-step horizon, task-reward telemetry, and no-checkpoint
contract fixed. The treatment adds exactly one existing Miles flag:

```text
--use-rollout-logprobs
```

The default `09` path accepts actions sampled by Student SGLang at weight
version `v`, then asks the trainer actor at its pre-update version `v+k` to
recompute those sampled-action log-probs. That detached recomputation supplies
both `q_old` in sampled RKLD and the old-policy side of sampled-token PPO. It
does not keep a second actor checkpoint, but it does run one additional
forward-only pass over the accepted batch.

The `10` treatment instead reuses the log-probs emitted when each action was
sampled. It therefore makes `q_old = q_rollout`, skips the trainer pre-update
log-prob forward, and exposes the behavior/trainer gap directly in the PPO
ratio. Teacher sampled log-probs, native `[T,2]` targets, current trainer
logits, Top-K + Rest DAgger, optimizer settings, and all parallel groups remain
unchanged. This is neither TIS nor a new clipping rule.

The two possibilities have different, legitimate biases:

- trainer-recomputed `q_old` makes the ratio close to one and can damp the
  numerical effect of asynchronous age and SGLang/Megatron log-prob mismatch;
  it is a stable local surrogate, but not the exact behavior-policy reference;
- rollout `q_old` is faithful to the policy that generated the action and saves
  a trainer forward, but stale samples and backend mismatch now enter the PPO
  ratio and RKLD coefficient. That may improve estimator fidelity or increase
  variance. The direction is empirical, not assumed.

Launch the two 200-step treatments:

```bash
HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/10a-geo3k-multiturn-hybrid-fully-async-rollout-qold-same-size-gate

HF_CACHE_DIR=/data/shared/hf_cache bash scripts/slurm/submit.sh \
  OPD/multimodal/10b-geo3k-multiturn-hybrid-fully-async-rollout-qold-big-small-gate
```

Compare `10a` with `09b` and `10b` with `09d`; do not compare 10a directly
with 10b to infer the old-policy-source effect. Preserve the existing
first/middle/last windows and exclude the fully-async step-zero completion-order
artifact from trend claims.

The metric semantics matter:

- `train/train_rollout_logprob_abs_diff` and `train/train_rollout_kl` compare
  the policy-loss reference with rollout log-probs. They are zero by definition
  in `10` and must not be reported as improved mismatch.
- `train/current_rollout_logprob_abs_diff` and
  `train/current_rollout_kl` compare the live training forward with rollout
  log-probs without another forward or any gradient contribution. These are the
  non-degenerate mismatch diagnostics for `10`.
- `train/ppo_kl`, `train/pg_clipfrac`, and `train/ess_ratio` show how much of
  that mismatch actually enters the sampled-token policy update. A stable run
  requires finite values, no sustained clip-fraction saturation, and no ESS
  collapse.
- `perf/log_probs_time` should disappear or become zero because the detached
  pre-update actor forward is skipped. Also compare active-token cost,
  `perf/train_time`, wait ratio, and tokens/GPU/s; wall-clock alone is confounded
  by response-length and completion-order changes.

Retain every `09` correctness guard: one teacher request per kept sample, zero
Student SGLang rescoring, exact suffix alignment, finite/additive hybrid losses,
valid DAgger identities, hard-zero optimization reward, bounded accepted
staleness, and no checkpoint saving. Also compare sampled RKLD, coarse KL,
Rest-mass error, gradients, response length, truncation, rounds, and observed
task reward. The experiment passes as a mechanism study if both treatments
finish 200 steps without violating those guards; choosing a default requires
the paired stability, objective, and systems evidence rather than a single
lower curve.

**Result: both gates passed (2026-07-19; 10a job 27455, 77 min; 10b job
27456, 81 min; commit `65ae6baa`).** Both treatments finished 200/200 steps
with every 09 guard intact and objective trajectories indistinguishable from
their controls (sampled RKLD last-window 0.033 vs 0.033 same-size and 0.096
vs 0.095 big-small; coarse KL, Rest error and Top-2 mass likewise — the 09
teacher-matrix gap is preserved unchanged). The mechanism is now measured
instead of assumed. In the 09 controls the trainer-recomputed `q_old` equals
the training forward bit-exactly (one optimizer step per rollout step), so
`ppo_kl`/`pg_clipfrac` are exact 0.0 and `ess_ratio` exact 1.0 at all 400
control steps — the PPO clipping machinery was structurally inert throughout
02–09, a property this milestone makes explicit for the first time. Under
rollout `q_old` the ~0.016 nats/token SGLang↔Megatron mismatch enters the
ratio and stays harmless: median `ppo_kl` ~1e-3, clip fraction 0.3–0.6% (max
2.8%), ESS never below 0.989. The mismatch band itself is unchanged
(`current_rollout_abs_diff` 0.016–0.021 vs the controls' diagnostic
0.015–0.019); `train_rollout_*` reads 0.0 at all 400 treatment steps exactly
as the definition check predicts, and the `pg_loss`/`opd_reverse_kl` float
fingerprint flips from bit-equal (400/400 control steps) to unequal (400/400
treatment steps, median |Δ| ~1.3e-3). The cost story has two halves: the
~2 s/step detached forward disappears (`perf/log_probs_time` absent; trainer
time −2 s) but median step time is flat because this topology is
generation-bound — the saving is absorbed as rollout wait (0.53 → 0.65).
Verdict: mechanism study passed; either old-policy reference is defensible at
this scale, and selecting a production default needs a trainer-bound or
wider-staleness regime where the two can actually separate. Watch items:
three early treatment grad spikes with immediate recovery (10a 146.5/355.2 at
steps 3/25; step 25 mismatch 0.0311 was near, but not equal to, the run maximum
0.0320 at step 19; 10b 231.8 at
step 7; no control step exceeds 100). Evidence:
`results/10a-job-27455.json`, `results/10b-job-27456.json`.

## 11: Fixed held-out Geo3K quality evaluation (12k rerun)

Milestone `11` answers the quality question that the shuffled
`rollout/raw_reward` series in `09` cannot answer. It evaluates every model on
one content-addressed subset of `VeraIsHere/geo3k_imgurl_processed/test.parquet`.
The default contract (revised 2026-07-21) is 30 unique prompts selected by a
deterministic content hash with seed `20260720` from the *image-clean* test
population: a test record is eligible only if neither its exact record
fingerprint nor its media fingerprint appears in the training split. Prompt-text
collisions do not exclude a record — the processed Geo3K `problem` field is a
fixed protocol wrapper around generic titles such as "Find x.", so identical
text over a different diagram is a different problem, not leakage. The exclusion
is explicit and counted in the manifest metadata, never silent.

The revision was forced by measurement, not preference: of the 601 unique test
records, 299 share an exact record (1) or bit-identical image bytes (298) with
the training split — the upstream Geo3K splits are by problem, not by diagram —
leaving 302 image-clean candidates, which can never satisfy the original strict
256-prompt/all-fingerprints contract. The 2026-07-21 decision keeps a small
30-prompt fixed manifest and folds every non-evaluated test record into the
Milestone-11 student training prompts, excluding the 6 records that share an
image with an evaluated prompt. The generated
`opd_eval_seed20260720_n30.train_augmented.parquet` therefore holds
`2101 train + 565 test = 2666` rows, and the overlay rewrites the student
recipes' `--prompt-data` to it (`11a`/`11b` teacher references are untouched).
Note two consequences: 11c/11d train on a larger prompt pool than 04–10 did,
and the n=30 Wilson 95% interval is wide (±≈0.18 at p≈0.5) — raise
`OPD_EVAL_NUM_PROMPTS` (≤302 eligible) if a tighter interval is worth the eval
wall-clock.

The generated parquet carries a stable `opd_eval_id`, source-row index, manifest
size, seed, and manifest fingerprint for every prompt. The same fingerprint must
appear in all four runs. Re-sourcing a recipe validates the existing manifest,
eval config, and augmented train parquet against the source-file hashes, ordered
prompt IDs, and every row value, so a stale, corrupted, or silently changed
split fails before Ray starts. The data manifest remains independent of decoding
length. The generated eval-config identity includes the trajectory budget, for
example `opd_eval_seed20260720_n30.ctx12000.eval.json`, so the archived 4k config
cannot be silently reused by the 12k rerun.

Evaluation uses the same Geo3K multimodal multi-turn protocol as training:

```text
fixed prompts             = 30 image-clean held-out test records
student training prompts  = train + non-evaluated test records (2666 rows)
generation                = greedy, one trajectory per prompt
trajectory budget         = 12000 total tokens, including multimodal prompt,
                            every assistant generation and every observation
max turns                 = 3 rounds
quality metric            = rule-based Geo3K answer accuracy + Wilson 95% CI
student eval checkpoints  = model steps 0, 5, 10, ..., 200
optimization reward       = unchanged: task reward remains telemetry-only
checkpoint saving         = disabled
```

The eval dataset selects `examples.geo3k_vlm.multi_turn.fixed_eval.generate`
through a per-dataset `--eval-config`. Training keeps the original
`examples.geo3k_vlm.multi_turn.rollout.generate` callback and metadata contract.
On eval only, the wrapper assigns the built-in `math` task reward before Miles
can invoke the OPD custom RM. Student evaluation therefore generates with the
current Student SGLang weights but issues no teacher-scoring request and cannot
enter the training batch or loss.

The budget wording is deliberate. Geo3K's custom rollout is not using
`--rollout-max-response-len` as a fresh allowance on each turn: it initializes
one remaining budget after subtracting the processor-expanded prompt, then
subtracts every generated and observation token. Milestone 11 therefore sets
both `--rollout-max-context-len 12000` and `--rollout-max-response-len 12000`.
The first flag makes the cumulative context contract explicit; the second keeps
the underlying SGLang request ceiling and generated eval config consistent.
`--max-tokens-per-gpu 16384` remains above the 12k trajectory cap, so this does
not introduce a new trainer parallelism or packing path.

The two reference jobs run each teacher as the evaluated actor, not as an OPD
score-only sidecar:

1. `11a`: `Qwen3-VL-8B-Thinking`, eval-only on two nodes;
2. `11b`: `Qwen3-VL-30B-A3B-Thinking`, eval-only on two nodes.

The student jobs are synchronous three-node runs because the current fully
async rollout callback rejects evaluation. They inherit the completed `06`
hybrid contract unchanged: `sampled RKLD coefficient = 1`, teacher Top-K `= 2`,
Top-K + Rest coefficient `= 0.5`, trainer-recomputed pre-update `q_old`, and
task reward excluded from optimization. `11c` uses the 8B-Thinking teacher;
`11d` uses the 30B-A3B-Thinking teacher and retains the proven 0.80 Student
SGLang memory fraction.

Milestone 11 submits with `HF_CACHE_DIR=/data/shared`. That is the tree every
completed 00-10 run actually used (see any retained `args.json`): the models
live under `/data/shared/models` — including both Thinking teachers — and the
Geo3K parquet plus the generated eval manifest live under the user-writable
`/data/shared/data/geo3k_imgurl_processed`. The historically documented
`/data/shared/hf_cache` is a read-only mirror without the Thinking teachers;
the fixed-eval overlay now rejects such a location before allocation instead
of failing on a manifest-write traceback.

Run the references once, then smoke each training arm before the full horizon:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/multimodal/11a-geo3k-fixed-eval-teacher8b-reference

HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/multimodal/11b-geo3k-fixed-eval-teacher30b-reference

OPD_NUM_ROLLOUT=5 WANDB_RUN_NAME=opd-mm-11c-teacher8b-fixed-eval-ctx12000-smoke \
  HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/multimodal/11c-geo3k-multiturn-hybrid-sync-eval-teacher8b-200step

OPD_NUM_ROLLOUT=5 WANDB_RUN_NAME=opd-mm-11d-teacher30b-fixed-eval-ctx12000-smoke \
  HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/multimodal/11d-geo3k-multiturn-hybrid-sync-eval-teacher30b-200step
```

Inspect the two teacher references before allocating the student runs. The 12k
protocol is intended to remove length as the dominant confound. The original
gate required each reference to truncate at most 3/30 trajectories; measurement
amended it (decision 2026-07-21). At 12k the references truncate 11/30 and
9/30, but the tail is not budget-driven: 10 of the 11 are the same prompts
that truncated at 4k (tripling the budget only halved the rate), truncated
trajectories score zero, and completed trajectories top out at p90 ~7.8k, far
below the cap. The binding criterion is therefore that completed-trajectory
p90 sits well below the budget (met at 12k); the residual truncation is a
Thinking-decoding property of a fixed hard-prompt subset and is reported, not
budgeted away. After the references pass and both
smokes record eval points at exactly 0 and 5, launch the wrappers
without overrides for 200 steps. Each full student arm performs 41 evaluations,
or 1,230 held-out trajectories. An eval blocks the synchronous driver and uses
the same Student SGLang engines, but the 30 prompt tasks are submitted
concurrently and bounded by the existing SGLang semaphore; only the turns within
one trajectory are sequential. Include this cost in wall-clock reporting rather
than comparing directly with the eval-free `09` runs. Exploratory launches may
override `OPD_EVAL_INTERVAL` (minimum 2) and `OPD_EVAL_NUM_PROMPTS` (≤302); Gate 27
smokes keep interval 5 to preserve the required step-0/step-5 pair, and the
canonical quality curves keep 5 steps and 30 prompts. The generated W&B name
encodes interval, prompt count and context budget so an override cannot
masquerade as the canonical arm. `OPD_EVAL_MAX_CONTEXT_LEN` is available for an
explicit noncanonical length study; the canonical rerun is 12,000.

Primary W&B metrics live under `eval/geo3k_fixed/*`: `accuracy`,
`accuracy_ci95_low/high`, `num_prompts`, `truncated_rate`, and mean/min/max for
response, active, observation tokens, and rounds. Compact per-prompt JSONL
evidence is written to `$MILES_RUN_DIR/fixed_eval/geo3k_fixed/step_XXXX.jsonl`.
The standard Slurm launcher injects `MILES_RUN_DIR`; a nonstandard direct launch
without it still logs aggregate metrics but warns and skips JSONL. Gates 25-28
continue to require the JSONL evidence. All rerun references share the same
12,000-token, three-round contract; always interpret accuracy together with
`truncated_rate`, response length, and rounds even after the larger budget.
Accuracy and the raw student-to-teacher gap are primary. A normalized gap-closure
view may be computed after the run as
`(student_t - student_0) / (teacher - student_0)` only when the teacher baseline
is above the initial student; it must never replace the raw accuracy and CI.

Status: **the 12k rerun ran on 2026-07-21: references complete (0.6333 /
0.7000, jobs 27848/27849, amended gate), smokes exact (27850/27851), and the
200-step arms were deliberately stopped at 58/64 optimizer steps after 3 h 16 m
(jobs 27854/27855, ~12 h projected wall-clock at the 12k budget).** The partial
12k window already answers two questions: the early trough is not a 4k budget
artifact (both arms crater to 0.067 at step 25 with 28-29/30 held-out
trajectories truncated — a truncation blowout, students overshoot even 12k
during the distribution shock) and both arms recover to roughly their starting
level by steps 40-60 (11c 0.50@55, 11d 0.47@60) while the 12k teachers sit
above at 0.633 / 0.700. Whether the recovery crosses the teacher lines over
the remaining 140 steps is unmeasured. Partial-window training invariants all
verified from `results/11a`-`11d-ctx12000-job-*.json`; 12k curves are Figures
14a-14b in the HTML report. The archived 4k history follows.

Archived 4k status: **the 4k mechanism/evidence pass completed on 2026-07-21;
the quality interpretation is superseded.** Archived
references 27803/27811, smokes 27814/27815 and 200-step arms 27816/27817 remain
useful diagnostics. Teacher references landed at 0.367
(8B-Thinking) and 0.500 (30B-A3B) — both below the untouched 0.57-0.63
Instruct student, so gap closure is undefined and raw accuracy is the only
legal reading. Both student arms fell through an early trough (minima 0.033 /
0.133) and settled at their matched teacher's budget-constrained level
(last-five 0.313 vs 0.367; 0.420 vs 0.500) while held-out truncation doubled:
under the old 4,096-total-token protocol the hybrid objective transferred the
teachers' long-form behavior faster than task accuracy. That protocol counted
the expanded multimodal prompt and observations against the nominal response
limit, producing 50-63% teacher truncation; it cannot carry the final quality
claim. Per-step series live in
`results/11a`-`11d-job-*.json`; curves are Figures 13a-13b in the HTML report.
The current 11a-11d wrappers use a new `ctx12000` run/config identity. The server
gate and exact acceptance checks are in `SERVER_VALIDATION.md` Gates 25-28.
