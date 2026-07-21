# On-Policy Distillation (OPD) Recipes

The active baseline is `math_qwen3_32b_8b_3nodes/`: Qwen3-8B student,
Qwen3-32B SGLang teacher, and one dedicated node each for the teacher, actor,
and rollout workers. It runs pure sampled-token distillation by default:
`--opd-kl-coef 1.0`, task reward 0, reward/loss KL 0, entropy 0, and no
`--normalize-advantages`.

```text
math_qwen3_32b_8b_3nodes/   Active 3-node baseline. Whole-node TP=8 teacher
                            on the head node, 8-GPU Megatron actor, and 8-GPU
                            SGLang rollout worker. Also contains the persistent
                            HTTP treatment described below.
math_qwen3_32b_8b_3nodes_legacy_teacher/
                            Frozen reproduction recipe for the legacy
                            `only-teacher + teacher_p` top-k scalar path. Job
                            24749 showed that its student-rescore leg crashes
                            decode-concurrent SGLang v0.5.13 engines at step 0.
optimize/                   Numbered teacher-top-k controls and treatments.
                            `00-t-top2-legacy.sh` records the failed legacy
                            characterization; 01+ scripts are added only after
                            their DAgger code paths are runnable.
archive/                    Historical 1-node smoke recipes; not active
                            baselines or current validation targets.
```

The active recipe defaults `OPD_TOP_K=0`. This is sampled-token OPD: teacher
scoring is a compact prefill request and the distillation advantage depends on
the sampled token. `OPD_TOP_K>0` remains experimental. The current top-k path
precomputes one divergence value per response position and injects it through
sampled-token PPO/GRPO advantage; it is not a differentiable teacher-top-k
`[T,K]` distillation loss. It also incurs slower rollout decoding and an
`O(length x response-wide candidate union)` scoring payload.
On the current SGLang v0.5.13 stack, the student arbitrary-ID rescore can also
crash an engine that is concurrently serving rollout decode requests. This is
why the legacy top-k recipe is retained for audit/reproduction, not as a
performance baseline.

The recipe is a quick-check configuration and intentionally omits checkpoint
saving. W&B uses entity `M3TRL`, project `OPD`.

Launch (on the slinky cluster, use `HF_CACHE_DIR=/data/shared`; the default
`/data/shared/hf_cache` is read-only):

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/math_qwen3_32b_8b_3nodes/qwen3-8B
```

## Legacy teacher-top-k characterization result

The 00 recipe attempted to measure the starting point for the teacher-top-k
rebuild. It is not trainer-side Top-K DAgger: current Miles gets teacher
top-k targets, unions their IDs across the response, asks student SGLang to
rescore the union, normalizes `teacher_p` inside the selected support, and
reduces the candidate dimension to one detached scalar per position.

Job 24749 submitted the intended 5-step smoke with:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/00-t-top2-legacy
```

The run died at step 0 after approximately 21 minutes. A student SGLang
scheduler called `.tolist()` on an already-list-valued
`next_token_token_ids_logprobs_val`, the router returned HTTP 500, and the
fail-fast scoring policy stopped the job. No completed rollout means there are
no legacy `candidate_logprob_cells` p95/max values to compare; none should be
invented. Do not rerun 00 as a performance baseline. Retain it only to reproduce
the upstream SGLang failure; the numbered sequence now measures the replacement
Top-K DAgger `[T,K]` path on its own gates.

The new branch does not overload legacy `OPD_TOP_K`. It uses independent Miles
arguments: `--opd-dagger-top-k`, `--opd-dagger-coef`, and
`--opd-dagger-loss`. Milestone 01 uses `explicit_cross_entropy`; the complete
Top-K + Rest objective uses `cross_entropy`. One teacher prefill returns both sampled-token
log-probs for RKLD-PG and native Top-K targets for DAgger. The defaults
`dagger_top_k=0` and `dagger_coef=0` preserve the existing sampled RKLD path.

Milestone 01 passed its 50-step, no-checkpoint validation as job 24890. Its
default W&B run/group is `01-teacher-top2-ce`:

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/optimize/01-topk-explicit-ablation
```

It disables sampled RKLD contribution, keeps the native teacher `K=2` targets,
and adds only the raw explicit candidate term in the trainer. Direct-loss
metrics use an independent `opd_dagger/*` W&B section on `train/step`; generic
sampled KL diagnostics remain under `rollout/kl/*`, and scoring transport stays
under `opd_scoring/*`. The run completed 50/50 with 3,200 teacher and zero
student-rescore requests, finite direct CE, no protocol failures, and no
detectable end-to-end step-time regression versus the sampled-token control.
Milestone 02 is next; no 100-step extension is required for the G2 gate.

## Persistent HTTP transport experiment

The control run `origin-topk0-response-http` has already completed. The new
recipe preserves its `top_k=0` setup and the response-window/T+1 scoring change
from `b88d7cf`; it only changes the aiohttp `ClientSession` lifecycle.

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/math_qwen3_32b_8b_3nodes/persistent-topk0-response-http
```

The new W&B run name is `persistent-topk0-response-http2`; compare it with the
existing `origin-topk0-response-http` run. First verify that the treatment
reports a high `opd_scoring/client_session_reuse_rate`; then compare
`opd_scoring/http_s`, `opd_scoring/e2e_latency_s`, overall rollout time, and
timeout/retry behavior. Session reuse enables connection pooling but does not
by itself prove that a TCP connection was reused.

The teacher model must already exist on disk:

```bash
hf download Qwen/Qwen3-32B --local-dir $HF_CACHE_DIR/models/Qwen3-32B
```

The launcher converts the student torch_dist artifact on first use if it is
missing. See `docs/advanced/on-policy-distillation.md` for the core OPD path.
