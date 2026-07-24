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

## Failure Triage

- `scoring_suffix_ids` rejected as an unknown field: the teacher is running an
  older SGLang checkout.
- `end in a text position` mRoPE error: the processed prompt ends in a visual
  region; the current pure-text suffix contract does not apply to that prompt.
- returned token ID mismatch: stop the run; do not quarantine or train on the
  sample. Capture the expected and returned IDs plus the teacher server log.
- any `opd_scoring/student_request_count > 0` in `03`: stop the run; the recipe
  has entered a legacy student-rescore route instead of trainer-direct DAgger.
- missing or malformed `[T,2]` targets: stop before interpreting the loss; save
  the teacher response metadata and the failing Sample indices.
- teacher queue growth or memory pressure: record queue depth, scoring latency
  and GPU memory before setting `OPD_SCORING_MAX_INFLIGHT` to a finite value.
