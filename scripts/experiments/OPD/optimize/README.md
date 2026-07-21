# Teacher Top-K Optimization Sequence

Scripts in this directory are numbered controls and treatments for the
teacher-top-k rebuild. A new script is added only when the corresponding code
path is runnable; planned but unimplemented treatments are documented here
instead of shipping placeholder launchers.

| Order | Script | Purpose | Status |
| --- | --- | --- | --- |
| 00 | `00-t-top2-legacy.sh` | Current `only-teacher + teacher_p`, including student SGLang rescore, response-wide union, support normalization, scalar reduction, and sampled-token PPO/GRPO | Ran (24749): student rescore crashes decode-concurrent engines; the legacy leg is inoperable on SGLang v0.5.13 |
| 01 | `01-topk-explicit-ablation.sh` | Use `opd-dagger-loss=explicit_cross_entropy` to validate `[T,K]` transport and candidate gradients; no Rest | Passed (24890): 50/50 steps, 3,200 teacher / 0 student requests, finite CE 0.365→0.303, step time approximately matched the sampled baseline |
| 02 | `02-topk-dagger.sh` | Use `opd-dagger-loss=cross_entropy` to add Rest and validate complete Top-K + Rest DAgger with sampled RKLD disabled | Planned after 01 is stable |
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
