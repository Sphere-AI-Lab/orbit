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
| 03 | `03-rkld-topk-dagger.sh` | Add the existing sampled RKLD-PG branch to Top-K + Rest DAgger | Planned after pure DAgger is stable |

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

The three smokes must finish 5/5 before allocating the 50-step gate. All runs
must retain 64 teacher and zero student scoring requests per step, finite loss
and gradient series, zero protocol/alignment failures, and stable
`opd_dagger/{explicit_ce,rest_ce,cross_entropy,teacher_entropy,coarse_kl,teacher_rest_mass,student_rest_mass,rest_mass_abs_error}`.
Use `coarse_kl = cross_entropy - teacher_entropy`, not raw CE magnitude, when
judging distribution mismatch.
TP2/TP4 runs additionally check for `xTP` loss/gradient scaling; PP2 checks that
the final pipeline stage receives all three sparse target tensors. CP and FSDP
are intentionally outside the 02 cluster gate and must not be advertised as
validated by these scripts.

### PDF parity boundary

The current 02 claim is deliberately narrower than the PDF's final hybrid
acceptance checklist:

| PDF contract | Current state |
| --- | --- |
| Teacher-selected raw `[T,K]` targets plus sampled teacher log-prob from one request | Implemented and validated by rollout/Sample/Ray tests and job 24890; no Student SGLang rescore. |
| Current trainer logits provide candidate and Rest gradients | Implemented by the Stable TP loss; teacher tensors detach at the loss boundary. |
| No Top-K renormalization; stable `p_R`, `logZ`, and `logZ_R` | Implemented with the numerical protocol above and dense FP64 Oracle parity. |
| TP=1/2/4 and independent process groups | TP1 gradcheck plus real Gloo TP2/DP2 and TP4 parity passed; NCCL TP2/TP4 smokes remain. |
| PP=2 | Batch fields are wired and `02c` is prepared; runtime validation remains. |
| Zigzag CP=2 / allgather CP | Candidate-axis slicing has a unit test, but no distributed loss/gradient gate; allgather CP is explicitly rejected. |
| Multimodal/agentic alignment | Multi-turn observation rows are masked. Prompt-only media expansion preserves response targets; response-side media expansion is not hardened and will fail the response-length guard. |
| Shared tokenizer/vocabulary | The loss enforces IDs inside the student vocabulary. Semantic tokenizer equality is a recipe/deployment contract, not yet verified from the remote teacher endpoint; the Qwen3-32B/8B recipe assumes the shared Qwen3 vocabulary. |
| FSDP | Not supported: its micro-batch key path is not wired. |
| `RKLD-PG + lambda * DAgger` hybrid | Data and loss branches are composable, but the matched hybrid experiment is milestone 03. The 02 scripts intentionally isolate pure DAgger with sampled RKLD coefficient zero. |

One systems caveat is separate from mathematical correctness: the current loss
invokes the Stable TP primitive once per response sample. That is five TP
reductions per sample (full MAX/SUM, Rest MAX/SUM, selected SUM), with total
payload `Theta(T)` but latency proportional to the number of samples in the
micro-batch. The duplicate CPU/GPU protocol checks also synchronize per sample.
02a-02d must therefore measure loss time and peak memory before considering a
packed row-index interface; this is not a reason to change the objective.

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
