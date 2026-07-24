# OPD Multimodal Bring-Up

This folder advances multimodal OPD through isolated cluster gates. Milestone
`00` validates the direct SGLang teacher-prefill contract. Milestone `01` then
exercises the same contract through Miles' production OPD scoring and parsing
path. Milestone `02` is the first end-to-end trainer gate: single-turn Geo3K
sampled RKLD on the three-node teacher/actor/rollout layout. Milestone `03`
keeps that validated layout fixed and isolates trainer-side teacher Top-K + Rest
DAgger before any hybrid, multi-turn, or asynchronous experiment.

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
EnvPack adapter appends assistant and environment spans into one `Sample`; a
later gate must run that complete rollout path end to end with multiple images.

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
