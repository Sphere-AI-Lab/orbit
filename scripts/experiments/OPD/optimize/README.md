# Teacher Top-K Optimization Sequence

Scripts in this directory are numbered controls and treatments for the
teacher-top-k rebuild. A new script is added only when the corresponding code
path is runnable; planned but unimplemented treatments are documented here
instead of shipping placeholder launchers.

| Order | Script | Purpose | Status |
| --- | --- | --- | --- |
| 00 | `00-t-top2-legacy.sh` | Current `only-teacher + teacher_p`, including student SGLang rescore, response-wide union, support normalization, scalar reduction, and sampled-token PPO/GRPO | Ran (24749): student rescore crashes decode-concurrent engines; the legacy leg is inoperable on SGLang v0.5.13 |
| 01 | `01-topk-explicit-ablation.sh` | Use `opd-dagger-loss=explicit_cross_entropy` to validate `[T,K]` transport and candidate gradients; no Rest | Passed (24890): 50/50 steps, 3,200 teacher / 0 student requests, finite CE 0.365→0.303, step time approximately matched the sampled baseline |
| 02a | `02a-topk-rest-tp2-smoke.sh` | Canonical 5-step complete Top-K + Rest DAgger smoke with TP=2, PP=1, DP=4 | Passed (25079): 5/5, CE 0.48377→0.43084, 320/0 requests |
| 02b | `02b-topk-rest-tp4-smoke.sh` | Verify the Stable TP operator generalizes to TP=4, PP=1, DP=2 | Passed (25080): 5/5; displayed step-0/4 CE stays within the 0.0028 three-layout spread |
| 02c | `02c-topk-rest-pp2-smoke.sh` | Verify last-pipeline-stage target ownership with TP=2, PP=2, DP=2 | Passed (25085; first attempt 25081 was an IB preflight refusal, not code): 5/5; no PP ownership failure |
| 02d | `02d-topk-rest-tp2-gate.sh` | Canonical 50-step objective/stability gate after all 5-step smokes pass | Passed (25137): 50/50, CE 0.487→0.407, coarse KL -58.5%, 3,200/0 requests, 149.7 s/step (+1.6% descriptive versus sampled baseline) |
| 03a | `03a-rkld-topk-rest-smoke.sh` | Five-step TP2 composition smoke: sampled RKLD-PG + Stable-TP Top-K + Rest | Passed (25267): 5/5, both branches active and finite |
| 03p | `03p-rkld-topk-rest-profile.sh` | Dedicated eight-step hybrid profile with three active post-warmup steps; capture per-rank/per-step operator, NCCL, shape, and CUDA-memory evidence | Trace captured (25278); the original 4.9% / 6x summary is withdrawn because CPU and GPU annotations were double-counted. A reviewer CPU-only rank-0 check is approximately 3.1%; regenerate the corrected CSV before using cross-rank ratios. |
| 03b | `03b-sampled-rkld-control.sh` | Fresh 50-step sampled-RKLD control from the same commit as the hybrid | Done (25279): 50/50, `I_rkld=0.01584`, median step 143.4 s |
| 03c | `03c-topk-rest-control.sh` | Fresh 50-step pure Top-K + Rest control from the same commit as the hybrid | Done (25280): 50/50; rerun floors calibrated against job 25137 |
| 03d | `03d-rkld-topk-rest-gate.sh` | Matched 50-step hybrid decision gate | Inconclusive (25281): guards 1/3/4 passed, RKLD preservation missed by 4.7x its floor, and coarse KL missed by 1.3x; G5 held for a coefficient sweep |

The 01 treatment will isolate the candidate-level term with:

```text
--opd-log-prob-top-k 0
--opd-kl-coef 0
--opd-dagger-top-k 2
--opd-dagger-coef 1
--opd-dagger-loss explicit_cross_entropy
```

Task reward, reference KL, entropy, advantage normalization, model placement,
batching, and response limits remain identical to the sampled-RKLD control.

## 01 validation result

Use the following focused tests to reproduce the numerical and contract checks:

```bash
python -m pytest -q \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/backends/training_utils/loss/test_rkld_dagger.py \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/backends/training_utils/test_ulysses_cp_utils.py \
  tests/fast/utils/test_arguments.py
```

These tests must prove that teacher targets reject non-finite values,
out-of-vocabulary IDs, duplicate IDs, and Top-K mass above 1; target tensors
are detached; masked `-inf` rows remain finite; and Megatron dummy vocabulary
logits are excluded before the existing fused target-logprob operator.

The cluster gate used the single 50-step treatment:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/01-topk-explicit-ablation
```

The wrapper uses `01-teacher-top2-ce` as both the W&B run name and group.

Job 24890 completed 50/50. Startup masked 128 Megatron dummy vocabulary logits
(`padded_vocab_size=152064`, real `vocab_size=151936`) under TP=2. Every step made
64 teacher and zero student scoring requests: 3,200 total, with zero retries,
alignment errors, or protocol failures. Direct CE and gradient series remained
finite; the aggregate teacher Top-K mass mean stayed in [0.927, 0.947], while the
parser enforced the per-position mass protocol. Median end-to-end step time was
145.4 s versus 147.4 s for the sampled-token control. This passes G2 and advances
to milestone 02; no 100-step extension is required. Peak-memory and dedicated
DAgger-loss timing were not exported and remain optional systems telemetry before
any one-pass sparse-kernel work.

## 02 implementation and validation sequence

Milestone 02 keeps the proven 01 teacher/Sample/Ray/batch contract and replaces
only the trainer objective selected by:

```text
--opd-log-prob-top-k 0
--opd-kl-coef 0
--opd-dagger-top-k 2
--opd-dagger-coef 1
--opd-dagger-loss cross_entropy
```

The production operator is
`vocab_parallel_topk_rest_cross_entropy` in
`miles/backends/training_utils/loss_hub/math_utils.py`. It evaluates the PDF
Stable TP form

```text
L_D = logZ - sum_i(p_i * z_i) - p_R * logZ_R
```

without gathering the full vocabulary. Each rank computes local full/Rest
log-sum-exp values; TP combines those `[T]` values and the selected weighted
logits. Both full-partition accumulation and Rest masking are row-chunked, so a
BF16 input never creates a full `[T,V_local]` FP32 conversion, and neither
forward nor backward saves full `[T,V_local]` softmax buffers. Backward
recomputes local probabilities and returns only the local vocabulary-shard
gradient. Every collective receives the existing TP group explicitly; the
operator never defines or changes the Megatron topology.

### Numerical protocol: guards are not objective clipping

The PDF shows `(1 - sum(p)).clamp(0, 1)` in its dense reference pseudocode,
then recommends the Stable TP form for production. The production path follows
the latter and uses these narrower rules:

| Quantity | Production computation | Why this does not change the objective |
| --- | --- | --- |
| Teacher Top-K mass | `logP = logsumexp(teacher_topk_logp)` in FP32 | Raw full-softmax log-probs are never normalized inside Top-K. A row with `logP > log1p(1e-5)` is rejected as a protocol error. |
| Teacher Rest mass | `p_R = -expm1(clamp_max(logP, 0))` | `expm1` preserves precision near `P=1`. The clamp runs only after the mass check and maps tolerated positive roundoff to `p_R=0`; every valid `logP <= 0` is unchanged. |
| Student Rest log-prob | `log q_R = logZ_R - logZ` | The training loss never forms `1-sum(q_i)`, never inserts epsilon, and never clamps `q_R`. This remains finite for arbitrarily small positive Rest mass when logits are finite. |
| Zero teacher Rest | Select a finite placeholder for `logZ_R` before multiplying by `p_R=0` | This prevents `0 * -inf = NaN` when candidates span the vocabulary; it does not alter any positive Rest term. |
| `student_topk_mass` | `(1-student_rest_mass).clamp(0,1)` | Telemetry only. Neither loss nor backward reads this value. |
| Candidate IDs | Clamp local gather indices to the shard range, then apply the owner/valid mask | Index-safety only; masked or non-owner values contribute exactly zero. |
| Coarse KL metric | `cross_entropy - teacher_entropy` | Telemetry only and intentionally not clamped. Tiny negative roundoff is diagnosable; a material negative value is a correctness failure. |
| Backward mass coefficient | Use the same represented `P+p_R` as forward | In exact arithmetic it is 1. Using the represented value makes custom backward exactly differentiate forward under FP32 roundoff without renormalizing teacher Top-K probabilities. |

There is no DAgger advantage clamp, probability renormalization, importance
ratio, or PPO clipping. Those mechanisms remain part of Miles' independent
sampled-policy branch; `opd_dagger/cross_entropy` is added directly from current
trainer logits. Full/Rest LSE, teacher mass, loss accumulation, and backward
probabilities use FP32 for BF16/FP16 input logits.
`log1mexp(logP)` would be needed only if the implementation required
`log(1-exp(logP))`. Production needs teacher `p_R`, so `-expm1(logP)` is the
direct stable operation; student `log q_R` already comes from `logZ_R-logZ`.

The independent FP64 Oracle lives only under
`tests/fast/backends/training_utils/loss/rkld_dagger_test_utils.py`. It builds
the dense K+1 distribution directly in probability space, so it shares no
`logZ`, Rest-LSE, owner-shard, or collective helper with production. It is a
correctness reference, not a training fallback.

Run the focused CPU/unit contract first:

```bash
python -m pytest -q \
  tests/fast/backends/training_utils/loss/test_rkld_dagger.py \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/utils/test_arguments.py
```

The tests cover the hand-computed K=2 loss and PDF gradient, FP64 gradcheck,
TP=1 production/Oracle parity, two independent CPU/Gloo TP2 groups in one
four-process test (including shard and collective-group isolation), one gradient
step reducing the coarse CE, raw and near-unit
teacher mass, tolerated mass overshoot, structural full-vocabulary Rest removal,
zero-K/empty/all-masked graphs, near-zero student Rest under FP64/BF16/FP16,
detached targets, protocol failures, padded-vocabulary zero gradients, response
alignment, additive loss dispatch, and the `coef=0` argument contract.

Then run the real process-group validator on a node with the Miles environment:

```bash
# TP=1 local degeneration.
torchrun --standalone --nproc-per-node 1 \
  scripts/experiments/OPD/optimize/02_validate_stable_tp.py --tp-size 1

# One TP=2 group.
torchrun --standalone --nproc-per-node 2 \
  scripts/experiments/OPD/optimize/02_validate_stable_tp.py --tp-size 2

# Two independent TP=2 groups: catches accidental WORLD-group collectives.
torchrun --standalone --nproc-per-node 4 \
  scripts/experiments/OPD/optimize/02_validate_stable_tp.py --tp-size 2

# One TP=4 group.
torchrun --standalone --nproc-per-node 4 \
  scripts/experiments/OPD/optimize/02_validate_stable_tp.py --tp-size 4
```

For every rank the validator compares the replicated per-token loss and local
vocabulary-shard gradient with the corresponding slice of the dense FP64
Oracle. It uses a non-divisible real vocabulary plus high dummy logits to prove
padding is excluded, candidates owned by different TP ranks, a partial-K row,
and different inputs in separate DP groups.

Only after all operator checks pass, submit cluster experiments in order:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/02a-topk-rest-tp2-smoke

HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/02b-topk-rest-tp4-smoke

HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/02c-topk-rest-pp2-smoke

HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/02d-topk-rest-tp2-gate
```

The three smokes finished 5/5 before the 50-step gate was allocated. All runs
retained 64 teacher and zero student scoring requests per step, finite loss and
gradient series, zero protocol/alignment failures, and stable
`opd_dagger/{explicit_ce,rest_ce,cross_entropy,teacher_entropy,coarse_kl,teacher_rest_mass,student_rest_mass,rest_mass_abs_error}`.
Use `coarse_kl = cross_entropy - teacher_entropy`, not raw CE magnitude, when
judging distribution mismatch.
The displayed step-0 and step-4 CE values have maximum cross-layout spreads of
0.00027 and 0.00280 respectively. Together with the independent Oracle tests,
that is evidence against an obvious `xTP` scale or PP last-stage ownership bug;
it is not a claim that independent runs are bitwise identical.

Job 25137 completed 50/50, but the embedded W&B export contains 49 trainer-loss
records at steps 0-48. On those records, first-10 to last-10 means are
0.4397→0.4231 for total CE, 0.3374→0.3398 for explicit CE, and 0.1022→0.0833
for Rest CE. The observed window improvement is therefore Rest-driven; the
explicit component is flat/noisy. Sampled reverse-KL, which has coefficient zero
in 02, moves 0.1131→0.1145 across the same first/last ten-point diagnostic
windows and does not support a "comparable decline" claim. This does not reject
the coarse forward-KL objective; it makes the matched 03 composition experiment
necessary.

Median end-to-end step time was 149.7 s versus 147.4 s for the sampled-token
control (+1.6 %). Dedicated Stable-TP operator timing and peak-memory counters
were not exported, so the cross-run delta must not be presented as the causal
cost of the two vocabulary scans. CP and FSDP are intentionally outside the 02
cluster gate and must not be advertised as validated by these scripts.

### PDF parity boundary

The current 02 claim is deliberately narrower than the PDF's final hybrid
acceptance checklist:

| PDF contract | Current state |
| --- | --- |
| Teacher-selected raw `[T,K]` targets plus sampled teacher log-prob from one request | Implemented and validated by rollout/Sample/Ray tests and job 24890; no Student SGLang rescore. |
| Current trainer logits provide candidate and Rest gradients | Implemented by the Stable TP loss; teacher tensors detach at the loss boundary. |
| No Top-K renormalization; stable `p_R`, `logZ`, and `logZ_R` | Implemented with the numerical protocol above and dense FP64 Oracle parity. |
| TP=1/2/4 and independent process groups | TP1 gradcheck plus real Gloo TP2/DP2 and TP4 parity passed; NCCL TP2 and TP4 smokes passed as jobs 25079 and 25080. |
| PP=2 | Runtime validation passed in job 25085 after an unrelated node-IB preflight refusal in job 25081. VPP remains untested. |
| Zigzag CP=2 / allgather CP | Candidate-axis slicing has a unit test, but no distributed loss/gradient gate; allgather CP is explicitly rejected. |
| THD/dynamic packing | The canonical 3-node recipe exercises THD packing at CP=1, but no dedicated mixed-length packed-versus-unpacked loss, local-logit-gradient, mask, metric, and per-sample-reducer parity gate exists. |
| Multimodal/agentic alignment | Multi-turn observation rows are masked. Prompt-only media expansion preserves response targets; response-side media expansion is not hardened and will fail the response-length guard. |
| Shared tokenizer/vocabulary | The loss enforces IDs inside the student vocabulary. Semantic tokenizer equality is a recipe/deployment contract, not yet verified from the remote teacher endpoint; the Qwen3-32B/8B recipe assumes the shared Qwen3 vocabulary. |
| FSDP | Not supported: its micro-batch key path is not wired. |
| `RKLD-PG + lambda * DAgger` hybrid | Composition regression and 03a-03d recipes are implemented; server fast tests, the five-step smoke, and matched 50-step arms remain unrun. The 02 scripts intentionally isolate pure DAgger with sampled RKLD coefficient zero. |

One systems caveat is separate from mathematical correctness: the current loss
invokes the Stable TP primitive once per response sample. That is five TP
reductions per sample (full MAX/SUM, Rest MAX/SUM, selected SUM), with total
payload `Theta(T)` but latency proportional to the number of samples in the
micro-batch. The duplicate CPU/GPU protocol checks also synchronize per sample.
The completed 02a-02d build did not export dedicated loss time or peak memory,
so those metrics remain prerequisites before considering a packed row-index
interface; this is not a reason to change the objective.

“CP/packing parity” therefore remains incomplete in the strict PDF sense. For
zigzag CP=2, the missing test must run Stable-TP over the CP-local response rows,
reconstruct the global result, and compare scalar loss plus every TP rank's local
logit gradient against the dense/unsharded reference, including padded and empty
local spans. For packing, the same unequal-length samples must match between the
current THD-packed layout and an unpacked reference while preserving masks,
metrics, and the existing sum-of-sample-mean reduction. A global token mean is
not equivalent because it would overweight longer responses. This gap does not
block the current CP=1 03 recipes; it limits any claim of general CP/packing
support. CP is a future milestone: no CP launcher or CP implementation change is
part of 03.

## 03 implementation and validation sequence

Milestone 03 composes the two already-isolated objectives without adding a new
teacher request, student forward, trainer operator, or distributed topology:

```text
A_RKLD,t = beta * (log p_T(a_t | h_t) - log q_old(a_t | h_t))

L_03 = L_policy(log q_theta(a_t | h_t), stop_gradient(A_RKLD,t))
       + lambda * L_TopK+Rest(current trainer logits, teacher [T,K] targets)
```

The same teacher prefill returns the sampled-action log-probability required by
RKLD-PG and the native `[T,K]` IDs/raw log-probabilities/valid mask required by
DAgger. `Sample`, multi-turn merge, train-data conversion, and DP splitting now
have regression coverage that retains both contracts together. The trainer's
single current-policy forward supplies both the sampled-action log-probability
and the vocabulary-sharded logits consumed by Stable TP.

`apply_opd_kl_to_advantages` explicitly detaches `q_old`, sampled teacher
log-probabilities, and any precomputed reverse-KL tensor at the RKLD boundary.
This makes the PDF's `stop_gradient` contract independent of how a custom data
source constructed those tensors. It does not detach current trainer logits or
change the numerical value produced by the normal HTTP/Ray path.
Milestone 03 introduces no extra importance ratio: RKLD-PG continues through
Miles' existing sampled policy-loss machinery, while Top-K + Rest continues to
bypass advantages, PPO clipping, TIS, and sampled-action reduction.

Run the server fast-test gate first:

```bash
python -m pytest -q \
  tests/fast/backends/training_utils/loss/test_opd.py \
  tests/fast/backends/training_utils/loss/test_rkld_dagger.py \
  tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py \
  tests/fast/rollout/test_on_policy_distillation.py \
  tests/fast/ray/rollout/test_train_data_conversion.py \
  tests/fast/utils/test_arguments.py
```

The dedicated 03 fixture checks a stronger invariant than finite execution. It
runs hybrid, RKLD-only, and DAgger-only branches from identical logits and
requires both `L_hybrid = L_RKLD + L_DAgger` and
`grad(L_hybrid) = grad(L_RKLD) + grad(L_DAgger)`. It also requires nonzero
branch gradients, independent `pg_loss`, `opd_reverse_kl`, and
`opd_dagger/*` metrics, and no gradients on `q_old` or teacher targets. The
sampled-token lookup uses the single-rank reference implementation so the fast
fixture is not coupled to Megatron's optional fused-CE import; the DAgger term
still executes the production Stable-TP operator. The cluster smoke is the
real Megatron/NCCL gate.

After the fast tests pass, run the five-step composition smoke:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/03a-rkld-topk-rest-smoke
```

03a must show 64 teacher and zero student-rescore requests per step; finite,
nonzero `train/pg_loss`, `train/opd_reverse_kl`, and `opd_dagger/loss`; finite
total loss and gradient norm; and no protocol, alignment, retry, or timeout
failure. It uses `opd_kl_coef=1`, `opd_dagger_top_k=2`, and
`opd_dagger_coef=1` unless explicitly overridden, with distinct coefficients
encoded in the W&B run name.

After 03a passes, collect the missing operator evidence with the dedicated
profile arm:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/03p-rkld-topk-rest-profile
```

`03p` uses Miles' existing `TrainProfiler` with `record_shapes`, stacks,
flops, and `profile_memory=True`. The default run has eight rollout steps: the
profiler waits for one step, warms up for one, then records three complete
post-warmup `train_overall` steps (`start=2`, `end=5`). This is deliberately
bounded; profiling all 50 decision steps would create large traces and could
change queueing or memory behavior.

The environment flag `MILES_PROFILE_OPD_DAGGER=1` activates ranges that are
absent from normal runs:

- `operator_forward|rows=...|vlocal=...|k=...` and `operator_backward`;
- local full-vocabulary and Rest-vocabulary LSE scans;
- distributed full/Rest LSE, each containing MAX + SUM TP collectives;
- the selected-candidate TP SUM.

When the flag is absent, the phase symbols alias the original functions at
module import and the forward/backward decorators return the original callables;
normal 03b-03d runs do not execute profiler ranges or no-op context managers.

Raw per-rank traces are written under
`${OPD_PROFILE_DIR:-$HF_CACHE_DIR/opd_profiles/<run-name>}`. Summarize every
profiled step with the companion tool in this folder:

```bash
python scripts/experiments/OPD/optimize/summarize_03p_trace.py \
  /data/shared/opd_profiles/03p-rkld1-top2-rest1-profile \
  --strict
```

The generated `03p-step-profile.csv` contains one row per trainer rank and CPU
`ProfilerStep`. CPU `user_annotation` ranges define logical steps, call counts,
shapes, and wall time; Kineto's duplicate `gpu_user_annotation` ranges are
reported separately and never added to CPU wall time. The CSV also contains
response rows, local vocabulary/K, full/Rest local-scan time, three TP-reduction
scope times, marker-derived expected collective count, independently observed
top-level `c10d::allreduce_` counts, GPU-annotation time, NCCL-kernel
counts/times, whole-step NCCL time, and step/operator CUDA allocated/reserved
peaks and deltas.

`--strict` checks that every forward call has all CPU phase contracts, a
backward, memory samples, and independently observed c10d all-reduces in the
exact per-scope pattern: two inside full TP LSE, two inside Rest TP LSE, and one
inside selected-candidate TP SUM. Phase markers alone cannot satisfy strict
mode; a trace with five expected markers and zero observed c10d calls fails.

Archive the raw rank traces, the CSV, commit SHA, run name, profile window, and
normal 03a/03d W&B links together. Interpret synchronous latency from the slowest
rank as well as rank spread; report each active step before any aggregate. The
operator memory peak is total process memory observed during its range, while
the delta is only a workspace proxy because the CUDA allocator may reuse cached
blocks. NCCL kernels are timestamp-attributed to the three labeled collective
scopes and can under-count asynchronous work; they remain diagnostic and are
not the strict-mode source of truth. Retain raw traces whenever kernel counts
differ from the one-per-call c10d model.

Historical correction for job 25278: the original summarizer emitted 48 rows
instead of the expected 8 ranks x 3 CPU steps = 24 because it treated each CPU
and GPU annotation as a separate logical range. The published 4.9% Stable-TP
share and 6x Rest/full scan ratio therefore combine incompatible clocks and are
withdrawn. A reviewer recomputation of rank-0 CPU ranges gives approximately
14.36 s / 468.87 s = 3.1%. The raw traces are not stored in this repository, so
the corrected tool must regenerate the CSV before reporting a cross-rank share,
scan ratio, or collective conclusion from that historical run.

Use this evidence to choose the next systems change:

1. If per-sample TP scope count/time is material, pack micro-batch response rows
   first (`5B_m -> 5`) while preserving sample offsets and the reducer.
2. If TP launch/wait time remains material after packing, fuse MAX/SUM payloads
   (`5 -> 2`).
3. If local full/Rest scans dominate instead, optimize chunking/kernel work;
   do not change collectives based only on end-to-end step time.
4. If operator memory delta or last-rank peak is limiting, sweep row chunk size
   in a separate profile arm before changing the mathematical objective.

The profiler itself adds synchronization, stack collection, and trace I/O, so
neither its step time nor its memory peak is an A/B result. Keep `03b`-`03d`
unprofiled; compare their normal end-to-end step time separately.

Only after 03a passes, launch the same-commit 50-step arms:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/03b-sampled-rkld-control

HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/03c-topk-rest-control

HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/03d-rkld-topk-rest-gate
```

The 03 decision rule is fixed before launch. For any lower-is-better metric
`m`, define `I_m(run) = mean_first10(m) - mean_last10(m)`. After 03c finishes,
use the same-recipe rerun pair 03c versus historical job 25137 to define a
metric-specific empirical noise floor: the absolute difference between their
matching window statistics. This N=1 rerun gap is a practical threshold, not a
confidence interval or a claim of statistical significance. `rollout_seed=42`
fixes prompt shuffling, but this recipe does not enable deterministic SGLang
inference, so completion sampling remains stochastic across runs.

Advance 03d only when all of the following hold:

1. It completes 50/50 rollout steps with 64 teacher and zero student-rescore
   requests per step, zero retry/protocol/alignment failures, finite loss and
   gradient series, and nonzero RKLD-PG and DAgger branches.
2. Sampled-RKLD improvement satisfies
   `I_rkld(03d) >= I_rkld(03b) - epsilon_rkld`.
3. The last-10 mean of `opd_dagger/rest_mass_abs_error` in 03d is no larger
   than 03c plus its rerun noise floor. Track the associated
   `teacher_rest_mass` and `student_rest_mass`; Rest CE falling by itself does
   not pass this guard.
4. The last-10 mean of `opd_dagger/explicit_ce` in 03d is no larger than 03c
   plus its rerun noise floor. Apply the same no-material-regression guard to
   `opd_dagger/coarse_kl`.

Also report Rest CE, total loss, gradient norm, request counts, retries, and
normal end-to-end step time. Total loss is an integrity signal, not a
cross-objective efficacy comparison, because the hybrid sums two terms. Jobs
24374 and 25137 remain historical references rather than substitutes for the
same-commit controls; 25137 is used only to calibrate the pure-DAgger rerun
floor. If any hard guard misses by more than that floor, do not advance from a
single run: classify the result as inconclusive/regressed and rerun or retune.
Fully async and staleness remain out of G5 until the synchronous composition is
stable.

## 00 characterization outcome

Job 24749 launched the intended 5-step characterization smoke with:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/00-t-top2-legacy
```

The intended control would make 64 teacher requests and 64 student rescore
requests per completed rollout step. Its training curve was characterization
data only; the candidate-to-scalar operator is a known correctness defect.

The intended `opd_scoring/*` W&B measurement contract was:

- `teacher_request_count` and `student_request_count`
- `teacher/candidate_logprob_cells/{p95,max}` (approximately `N×K`)
- `student/requested_token_ids/{p50,p95,max}` (the response-wide union `U`)
- `student/returned_positions/{p50,p95,max}` (the returned-position count `N`)
- `student/response_tokens/{p50,p95,max}` (the sampled response length `T`)
- `student/candidate_logprob_cells/{p50,p95,max}` (approximately `N×U`)
- target-split `request_body_bytes`, `response_body_bytes`, `e2e_latency_s`,
  `http_s`, `semaphore_wait_s`, `body_read_s`, and `json_decode_s`

Job 24749 (2026-07-11) died at step 0: the `token_ids_logprob` rescore crashes
student engines that are concurrently decoding (`_normalize_decode_outputs`
calls `.tolist()` on plain lists — sglang scheduler dies; router returns 500).
No `candidate_logprob_cells` before-values exist, so no payload baseline is
claimed. The legacy leg is inoperable on this stack; the 01+ DAgger path
removes the rescore leg entirely.

Do not spend another 3-node allocation rerunning 00 for throughput or payload
percentiles: it cannot complete the measurement on this stack. Reproduce it
only when diagnosing or validating an upstream SGLang fix. The replacement
path must establish its own O(BTK) payload, zero-student-RPC, loss, and stability
evidence under the 01 gates.

## Numbering rule

Each treatment inherits the same Qwen3-8B student, Qwen3-32B TP=8 teacher,
3-node placement, data, batch size, optimizer, response limit, and task reward
of zero. A script should change only the mechanism named in its filename.
