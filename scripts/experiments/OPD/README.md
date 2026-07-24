# On-Policy Distillation (OPD) Recipes

The active baseline is `math_3nodes/`: Qwen3-8B student,
Qwen3-32B SGLang teacher, and one dedicated node each for the teacher, actor,
and rollout workers. It runs pure sampled-token distillation by default:
`--opd-kl-coef 1.0`, optimization reward 0, reward/loss KL 0, entropy 0, and
no `--normalize-advantages`. Deepscaler also scores every training response,
but that score is telemetry only: W&B receives it as `rollout/raw_reward`,
while `rollout/rewards` stays zero and therefore cannot enter advantages. This
contract is supported only by the canonical SGLang OPD reward/post-process
hooks. The observed scorer always uses the configured built-in `--rm-type` and
ignores per-sample `metadata.rm_type` overrides; Megatron OPD rejects the flag.

```text
math_3nodes/                Active 3-node baseline. Whole-node TP=8 teacher on
                            the head node, 8-GPU Megatron actor, and 8-GPU
                            SGLang rollout worker. Logs training-set math
                            correctness without optimizing it.
optimize/                   Numbered teacher-top-k controls and treatments.
                            `00-t-top2-legacy.sh` records the failed legacy
                            characterization; 01+ scripts are added only after
                            their DAgger code paths are runnable.
multimodal/                 Staged Qwen3-VL OPD ladder. `00`–`06` establish
                            image-conditioned exact-suffix scoring, sampled
                            RKLD, trainer-direct Top-K + Rest, and synchronous
                            Geo3K multi-turn references for each objective and
                            their hybrid composition. `07` runs the same fixed
                            hybrid under fully-async scheduling. All gates
                            through `07b` passed; staleness is observed, not
                            assumed (see the `6253d3f5` collector fix). `08`
                            explicitly composes normalized task reward with
                            hybrid OPD; its sync and prefetch-two async gates
                            all passed (jobs 27415–27427) — reward enters the
                            GRPO advantage with distillation invariants
                            untouched, recycling enforcement-tested, 8.4
                            ms/active token at prefetch 2.
                            `09` runs a 200-step two-teacher matrix
                            (8B-Thinking vs 30B-A3B → 8B-Instruct, jobs
                            27429–27435): invariants exact across 410 steps,
                            K=2 coverage stable in both arms, students drift
                            toward Thinking-length responses (budget lever
                            recorded for full runs). `10` closes the roadmap
                            with a paired rollout-q_old A/B over both 09 arms
                            (jobs 27455/27456): objective parity, the PPO
                            ratio measured non-degenerate for the first time
                            (the 02–09 default made it identically 1), and the
                            saved trainer forward absorbed as generation wait
                            at this topology.
archive/                    Historical 1-node smokes, the former canonical
                            3-node recipe and HTTP A/B wrapper, and the frozen
                            legacy teacher-top-k reproduction recipe.
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
saving and held-out eval. `--opd-log-task-reward --rm-type deepscaler` uses
the labels already present in DAPO-Math-17K; it adds no extra rollout and no
teacher/student model request. It remains telemetry-only unless paired with
`--opd-optimize-task-reward`; `--opd-task-reward-coef` then scales the reward
after the standard group normalization. W&B uses entity `M3TRL`, project
`OPD`.

Launch (on the slinky cluster, use `HF_CACHE_DIR=/data/shared`; the default
`/data/shared/hf_cache` is read-only):

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/math_3nodes/qwen3-8B
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
metrics use an independent `opd_dagger/*` W&B section on `train/step`;
policy/reference sampled KL diagnostics remain under `rollout/kl/*`, sampled
student/teacher OPD diagnostics use `rollout/opd_kl/*`, and scoring transport
stays under `opd_scoring/*`. The run completed 50/50 with 3,200 teacher and zero
student-rescore requests, finite direct CE, no protocol failures, and no
detectable end-to-end step-time regression versus the sampled-token control.
Milestone 02 is implemented locally; run the `optimize/02a` through `02d`
validation sequence before composing it with RKLD-PG. No 100-step extension is
required for the G2 gate. The optimize README is also the normative reference
for Stable TP `expm1`/clamp semantics and the remaining PDF parity boundaries.

## Archived persistent HTTP transport experiment

The control run `origin-topk0-response-http` has already completed. The new
recipe preserves its `top_k=0` setup and the response-window/T+1 scoring change
from `b88d7cf`; it only changes the aiohttp `ClientSession` lifecycle.

```bash
HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh \
  OPD/archive/math_qwen3_32b_8b_3nodes/persistent-topk0-response-http
```

The treatment W&B run name is `persistent-topk0-response-http2`; compare it with the
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
