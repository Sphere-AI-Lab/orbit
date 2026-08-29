---
title: Orbit's delta over miles, and the cost of moving it to latest miles
kind: investigation
subtitle: Fork-base recovery, full modification inventory, and a measured transfer dry-run
tags: miles, fork, lineage, graft, delta, transfer
fork_base: radixark/miles ef7481ae3 (2026-04-13, "switch model to actor")
miles_upstream: dbbab156 (2026-08-27)
graft_branch: miles-graft (rename 18e96ad, graft 7c5f463)
---

Orbit's public history is squashed (root `dd94c38`, 2026-05-28), so what we inherited from
miles and what we built was not recoverable from git alone. This record answers three
questions: **where orbit forked from miles**, **exactly what orbit adds or changes**, and
**what it would cost to move that layer onto the latest miles**.

## Fork base: miles `ef7481ae3` (2026-04-13)

Blob-SHA matching of orbit's root tree against all 1,761 first-parent miles commits peaks at
112/418 shared blobs on a plateau whose edges are pinned by content evidence: orbit contains
the post-image of every code commit up to and including `ef7481ae3` (exact blob match at
`ef228e648`; orbit's `actor.py` carries `ef7481ae3`'s own hunk) and the pre-image of the next
commit `85fe6519a`. Package rename at fork: `miles/` -> `orbit/`, `miles_plugins/` ->
`orbit_plugins/`.

Branch **`miles-graft`** rebuilds ancestry non-destructively: miles history (`refs/miles/base`)
-> rename commit `18e96ad` (shared subtrees) -> graft commit `7c5f463` (release tree) -> all
orbit commits reparented byte-identically. `git merge-base`, `blame`, `log --follow`, and
merges now work across the fork boundary. `orbit-main` itself is untouched.

<div class="callout">
The full delta is one native command: <code>git diff 18e96ad orbit-main</code> —
rename-free, since <code>18e96ad</code> is the miles base already renamed into orbit's
namespaces.
</div>

## What orbit adds, as features

**Orbit is an adapter-first RL infra.** On top of miles — a general RL trainer
that syncs full model weights — its new features span the whole stack: an
adapter system that trains and serves PEFT adapters on (possibly quantized)
frozen base models, **new training designs for async RL, PPO, and MOPD**,
verified-reward backends, and a numerical-verification discipline. The
argument surface makes the census objective: orbit adds **83 CLI arguments**
to the base's 230 and removes none; the clusters map one-to-one onto what
follows.

### New training designs

1. **Async RL.** A fully-asynchronous rollout driver
   (`fully_async_rollout.py`) paired with **double-buffered adapter slots**
   (`--adapter-double-buffer`, `peft_transport/slots.py`): rollout keeps
   serving adapter version N from one slot while N+1 streams in over NCCL,
   then activates atomically — generation never stops for a weight sync. The
   `--offload-*` suite (rollout/train adapter, frozen-base mode, grad
   buffers, optimizer, async) manages memory around it, and the
   **true-on-policy contract** (`--true-on-policy-contract`,
   `--recompute-logprobs-via-prefill`) enforces, rather than assumes, that
   async rollouts and the trainer agree.
2. **PPO.** The **one-trunk adapter critic** (`--critic-mode`): the critic
   shares the actor's frozen trunk by parameter aliasing — no second trunk
   copy in memory; only adapters and the value head are critic-owned.
   Benchmarked indistinguishable from a full critic in learning at ~23%
   faster per step. This design is why `loss.py`/`ppo_utils.py` carry ~1.6k
   changed lines: the rewrite is algorithmic, not wiring.
3. **MOPD — managed on-policy distillation** (33 `--opd-*` args, the largest
   cluster; near-zero OPD at the fork base): managed teacher serving and
   teacher pools with placement-group integration, sampled-token and
   full-vocab teacher score modes, JSD/KL loss variants with pointwise
   clipping, EMA teachers, and **self-teacher distillation** — promoting the
   student's own adapter versions as teacher through the adapter transport.
   Supporting modules: `teacher_lm_head.py`, `vocab_parallel.py`,
   `prefill_logprobs.py`.

### The adapter stack

- **OFT as a first-class RL method** (6 `--oft-*` args; miles has LoRA only —
  no OFT anywhere at the fork base): Megatron-side training, serving via the
  sglang fork, COFT/block variants, examples as full-FT vs LoRA vs OFT arms.
- **Adapter-delta weight sync** (`peft_transport/`,
  `--peft-distributed-transport`): only adapter deltas move to rollout
  engines, over pluggable NCCL/IPC/Ray backends behind a registry — what
  makes RL on a Kimi-1T-class frozen base tractable.
- **RL on quantized base models**: direct INT4/NVFP4/FP8 checkpoint
  converters and bridges, recipes for Kimi-K2.5/K2.6 INT4 and NVFP4, DSv4
  MXFP4 (`--dsv4-*`), Qwen3-30B FP8 — OFT adapters on the quantized weights.
  Miles had FP8 *export* quantizers, not a train-on-quantized path.
- **Adapter-aware serving**: engine-level adapter staging/activation and
  PEFT-safe radix caching (disabled or keyed per adapter, so cached prefixes
  cannot leak stale adapter activations).

### Rewards, verification, evaluation, operations

**Verified-reward backends and routing** (14 args): LLM-judge (`--judge-*`),
containerized SWE-agent rewards with SIF cache (`--swe-*`), code-execution
rewards with test/memory/timeout budgets (`--code-*`), a Lean prover-server
reward (`--lean-*`), and a reward router. **Verification discipline**: ~30
parity checkers (checkpoint and runtime per precision, cross-repo DeepGEMM,
step-0, adapter runtime compare). **Evaluation**: PEFT-Arena (vendored
math_eval: AIME24/AMC23/MATH500, arena reward), NLL and pass@k extensions.
**Operations**: 50 per-model launch profiles in `orbit_plugins/model_args`
(DSv3/v4, GLM4–5, Kimi K2–K2.6, gpt-oss, Qwen), MTP-under-RL patches, pinned
cu128/cu130 install contracts with ratchet tests.

Equally telling is what orbit **drops** (406 files): VLM and tool-use examples,
experimental FSDP, AMD support, docker/CI — breadth traded for depth on the
adapter-first thesis.

## The delta, in three layers

Against the fork base, `orbit-main` is: **789 added files** (the additive layer), **133 modified files, 14,912 changed lines** (the entangled layer), **406 dropped files**, and 87 files still byte-identical.

| Layer | Size | How it transfers to a new miles |
|:--|:--|:--|
| Orbit-only files | 789 files | Copies as-is — no miles counterpart exists |
| Modified miles files | 133 files, ~14.9k lines | Merge work, re-done per miles version |
| Dropped miles files | 406 files | Keep-or-drop decision list (tests, docker, old launchers) |

### Layer 1 — orbit-only files (ports as-is)

The adapter-first identity lives here: `peft_transport/` (NCCL/IPC adapter weight sync), OFT support, the FP8/INT4/NVFP4 conversion + parity-checking stack, `model_args` catalog, PEFT-Arena, and the high/low-precision example suites.

<details>
<summary>All 789 orbit-only files, grouped by subsystem</summary>


**(root)** (5)

```
CUDA-13-install.md
INSTALL.md
SETUP.md
env.sh
uv.lock
```

**assets/.gitkeep** (1)

```
assets/.gitkeep
```

**assets/license-138438922-4722503.pdf** (1)

```
assets/license-138438922-4722503.pdf
```

**assets/orbit-logo.png** (1)

```
assets/orbit-logo.png
```

**assets/orbit_logo.png** (1)

```
assets/orbit_logo.png
```

**assets/orbital.png** (1)

```
assets/orbital.png
```

**assets/orbital.svg** (1)

```
assets/orbital.svg
```

**docs/assets** (10)

```
docs/assets/kimi-memory.png
docs/assets/kimi-rl-curves.png
docs/assets/memory-scaling-lines.png
docs/assets/memory-scaling.png
docs/assets/orbit-logo.png
docs/assets/peftarena-results.png
docs/assets/qwen3-loravsoft.png
docs/assets/v4flash-memory.png
docs/assets/v4flash-rl-curves.png
docs/assets/v4pro-validation.png
```

**docs/css** (6)

```
docs/css/blog-post.css
docs/css/common.css
docs/css/fonts.css
docs/css/syntax.css
docs/css/typography.css
docs/css/ui.css
```

**docs/index.html** (1)

```
docs/index.html
```

**docs/orbit-adapter-async-db.html** (1)

```
docs/orbit-adapter-async-db.html
```

**docs/orbit_icon.png** (1)

```
docs/orbit_icon.png
```

**docs/plans** (2)

```
docs/plans/2026-08-06-ppo-critic-comparison-design.md
docs/plans/2026-08-17-adapter-first-experiments-design.md
```

**docs/reports** (15)

```
docs/reports/2026-08-10-e4-gsm8k-math-panel.html
docs/reports/2026-08-10-ppo-critic-comparison.html
docs/reports/2026-08-19-merged-stack-numerical-equivalence.html
docs/reports/2026-08-27-restructure-numerical-equivalence.html
docs/reports/_src/2026-08-10-e4-gsm8k-math-panel.md
docs/reports/_src/2026-08-10-ppo-critic-comparison.md
docs/reports/_src/2026-08-19-merged-stack-numerical-equivalence.md
docs/reports/_src/2026-08-21-phase0-qualification.md
docs/reports/_src/2026-08-27-restructure-numerical-equivalence.md
docs/reports/_src/figs/fig1_panels.png
docs/reports/_src/figs/fig2_matrix.png
docs/reports/_src/figs/fig3_lr.png
docs/reports/_src/figs/fig4_systems.png
docs/reports/_src/figs/fig5_followups.png
docs/reports/index.html
```

**docs/superpowers** (3)

```
docs/superpowers/plans/2026-08-19-adapter-first-phase0-phase1.md
docs/superpowers/plans/2026-08-19-orbit-cu128-install-pipeline.md
docs/superpowers/specs/2026-08-19-orbit-cu128-install-pipeline-design.md
```

**examples/adapter_first** (9)

```
examples/adapter_first/README.md
examples/adapter_first/env.sh
examples/adapter_first/phase0-opd-smokes.sh
examples/adapter_first/phase0-q25-lora-async.sh
examples/adapter_first/phase0-q25-oft-arms.sh
examples/adapter_first/phase1-q25-3b-oft-arms.sh
examples/adapter_first/phase1-q25-3b-opd-cost-suite.sh
examples/adapter_first/phase1-q3-30b-arms.sh
examples/adapter_first/phase1-q3-4b-arms.sh
```

**examples/blend_router** (1)

```
examples/blend_router/run-qwen2_5-0_5b-router-smoke.sh
```

**examples/genrm** (1)

```
examples/genrm/run-qwen2_5-0_5b-genrm-smoke.sh
```

**examples/high_precision** (49)

```
examples/high_precision/README.md
examples/high_precision/ppo_critic_compare_common.sh
examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh
examples/high_precision/run-qwen2_5-0_5b-bf16-math-fullft-async.sh
examples/high_precision/run-qwen2_5-0_5b-bf16-math-lora-ppo-adapter-critic-smoke.sh
examples/high_precision/run-qwen2_5-0_5b-bf16-math-lora.sh
examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft-ppo-adapter-critic.sh
examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft-ppo.sh
examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft.sh
examples/high_precision/run-qwen2_5-0_5b-fullft-head-critic-smoke.sh
examples/high_precision/run-qwen2_5-3b-bf16-math-fullft-async.sh
examples/high_precision/run-qwen2_5-3b-bf16-math-lora.sh
examples/high_precision/run-qwen2_5-3b-bf16-math-oft.sh
examples/high_precision/run-qwen2_5-3b-math-fullft-grpo.sh
examples/high_precision/run-qwen2_5-3b-math-fullft-head-critic.sh
examples/high_precision/run-qwen2_5-3b-math-fullft-ppo.sh
examples/high_precision/run-qwen2_5-3b-math-oft-adapter-critic-tune.sh
examples/high_precision/run-qwen2_5-3b-math-oft-grpo.sh
examples/high_precision/run-qwen2_5-3b-math-oft-ppo-adapter-critic-budget.sh
examples/high_precision/run-qwen2_5-3b-math-oft-ppo-adapter-critic-controlled.sh
examples/high_precision/run-qwen2_5-3b-math-oft-ppo-full-critic-budget.sh
examples/high_precision/run-qwen2_5-3b-math-oft-ppo-full-critic-controlled.sh
examples/high_precision/run-qwen2_5-3b-math-oft-ppo-lrprobe.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-full-b200.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-full-muon-kimi.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-full.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-lora-b200.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-lora.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-all.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-b32-b200.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-b32-kl-b200.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-b32-kl.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-b32.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-b64-b200.sh
examples/high_precision/run-qwen2_5-7b-bf16-openr1-oft-b64.sh
examples/high_precision/run-qwen3-1_7b-bf16-openreasoning-opd-full-vocab-lora-fkl.sh
examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-full-lr1e6.sh
examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-full-lr3e6.sh
examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-fullft-async.sh
examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-lora.sh
examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-oft-b32.sh
examples/high_precision/run-qwen3-30b-a3b-bf16-openr1-oft-b64.sh
examples/high_precision/run-qwen3-30b-a3b-instruct-2507-bf16-openr1-lora.sh
examples/high_precision/run-qwen3-30b-a3b-instruct-2507-bf16-openr1-oft.sh
examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh
examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-lora.sh
examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh
examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-fully-async.sh
examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh
```

**examples/judge** (1)

```
examples/judge/run-qwen2_5-0_5b-judge-smoke.sh
```

**examples/load_cuda13_2_orbit_env.sh** (1)

```
examples/load_cuda13_2_orbit_env.sh
```

**examples/low_precision** (17)

```
examples/low_precision/dsv4-common.sh
examples/low_precision/run-dsv4-mxfp4-math-oft-flash-debug.sh
examples/low_precision/run-dsv4-mxfp4-math-oft-pro-debug.sh
examples/low_precision/run-dsv4-mxfp4-openr1-oft-flash.sh
examples/low_precision/run-dsv4-mxfp4-openr1-oft-pro.sh
examples/low_precision/run-kimi-k25-int4-math-oft-debug6.sh
examples/low_precision/run-kimi-k25-int4-math-oft.sh
examples/low_precision/run-kimi-k25-int4-openr1-oft.sh
examples/low_precision/run-kimi-k25-nvfp4-math-oft.sh
examples/low_precision/run-kimi-k26-int4-openr1-oft.sh
examples/low_precision/run-qwen3-30b-a3b-fp8-math-oft.sh
examples/low_precision/run-qwen3-30b-a3b-instruct-2507-fp8-math-oft.sh
examples/low_precision/run-qwen3-30b-a3b-int4-math-oft.sh
examples/low_precision/run-qwen3-30b-a3b-nvfp4-math-oft.sh
examples/low_precision/run-qwen3-4b-fp8-math-oft.sh
examples/low_precision/run-qwen3-4b-int4-math-oft.sh
examples/low_precision/run-qwen3-4b-nvfp4-math-oft.sh
```

**examples/nemotron** (1)

```
examples/nemotron/run-nemotron-3-nano-4b-smoke.sh
```

**examples/on_policy_distillation** (17)

```
examples/on_policy_distillation/opd_teacher_cost_common.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-adapter-swap-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-blend-ppo-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-ema-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-free-teacher-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-full-vocab-gsm8k.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-full-vocab-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-mopd-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-sglang-smoke.sh
examples/on_policy_distillation/run-qwen2_5-0_5b-opd-teacher-pool-smoke.sh
examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-adapter.sh
examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-base.sh
examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-ema.sh
examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-load.sh
examples/on_policy_distillation/run-qwen2_5-3b-opd-cost-served.sh
examples/on_policy_distillation/run-qwen3-4B-opd-megatron.sh
examples/on_policy_distillation/run-qwen3-4B-opd-sglang.sh
```

**examples/optimizers** (2)

```
examples/optimizers/muon-kimi.env
examples/optimizers/run-muon-kimi-smoke.sh
```

**examples/peft_arena** (31)

```
examples/peft_arena/backend/LICENSE
examples/peft_arena/backend/README.md
examples/peft_arena/backend/eval/eval_math.sh
examples/peft_arena/backend/third_party/math_eval/data/aime24/test.jsonl
examples/peft_arena/backend/third_party/math_eval/data/amc23/test.jsonl
examples/peft_arena/backend/third_party/math_eval/data/math500/test.jsonl
examples/peft_arena/backend/third_party/math_eval/data_loader.py
examples/peft_arena/backend/third_party/math_eval/evaluate.py
examples/peft_arena/backend/third_party/math_eval/examples.py
examples/peft_arena/backend/third_party/math_eval/grader.py
examples/peft_arena/backend/third_party/math_eval/latex2sympy/__init__.py
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PS.interp
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PS.tokens
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PSLexer.interp
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PSLexer.py
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PSLexer.tokens
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PSListener.py
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/PSParser.py
examples/peft_arena/backend/third_party/math_eval/latex2sympy/gen/__init__.py
examples/peft_arena/backend/third_party/math_eval/latex2sympy/latex2sympy2.py
examples/peft_arena/backend/third_party/math_eval/math_eval.py
examples/peft_arena/backend/third_party/math_eval/model_utils.py
examples/peft_arena/backend/third_party/math_eval/parser.py
examples/peft_arena/backend/third_party/math_eval/python_executor.py
examples/peft_arena/backend/third_party/math_eval/sglang_adapter_utils.py
examples/peft_arena/backend/third_party/math_eval/trajectory.py
examples/peft_arena/backend/third_party/math_eval/utils.py
examples/peft_arena/backend/tools/merge_peft.py
examples/peft_arena/backend/tools/prepare_eval_checkpoint.py
examples/peft_arena/eval/README.md
examples/peft_arena/eval/eval-math-peft-arena.sh
```

**examples/sandbox** (1)

```
examples/sandbox/run-qwen2_5-0_5b-code-smoke.sh
```

**examples/search_r1** (13)

```
examples/search_r1/README.md
examples/search_r1/__init__.py
examples/search_r1/generate_with_search.py
examples/search_r1/local_search_server.py
examples/search_r1/qa_em_format.py
examples/search_r1/qwen2_5_3b_search_r1_ppo_common.sh
examples/search_r1/retrieval_server_jsonl.py
examples/search_r1/run-qwen2_5-0_5b-bf16-search-r1-ppo-full.sh
examples/search_r1/run-qwen2_5-0_5b-bf16-search-r1-ppo-lora.sh
examples/search_r1/run-qwen2_5-0_5b-bf16-search-r1-ppo-oft.sh
examples/search_r1/run-qwen2_5-3b-bf16-search-r1-ppo-full.sh
examples/search_r1/run-qwen2_5-3b-bf16-search-r1-ppo-lora.sh
examples/search_r1/run-qwen2_5-3b-bf16-search-r1-ppo-oft.sh
```

**examples/sft** (11)

```
examples/sft/README.md
examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh
examples/sft/run-llama3_1-8b-bf16-oft-sft-commonsenseqa.sh
examples/sft/run-llama3_1-8b-bf16-oft-sft-magicoder.sh
examples/sft/run-llama3_1-8b-bf16-oft-sft-numinamath.sh
examples/sft/run-llama3_1-8b-bf16-oft-sft-scienceqa-text.sh
examples/sft/run-qwen2_5-0_5b-bf16-sft-commonsenseqa.sh
examples/sft/run-qwen2_5-0_5b-bf16-sft-magicoder.sh
examples/sft/run-qwen2_5-0_5b-bf16-sft-numinamath.sh
examples/sft/run-qwen2_5-0_5b-bf16-sft-scienceqa-text.sh
examples/sft/run-qwen2_5-0_5b-bf16-sft-socialiqa.sh
```

**examples/swe** (1)

```
examples/swe/run-swe-patch-smoke.sh
```

**examples/swe_agent** (1)

```
examples/swe_agent/run-swe-agent-smoke.sh
```

**examples/tau_bench** (10)

```
examples/tau_bench/README.md
examples/tau_bench/__init__.py
examples/tau_bench/generate_with_tau.py
examples/tau_bench/openai_tool_adapter.py
examples/tau_bench/qwen3_4b_tau_bench_ppo_common.sh
examples/tau_bench/run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-full.sh
examples/tau_bench/run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-lora.sh
examples/tau_bench/run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-oft.sh
examples/tau_bench/sglang_tool_parser.py
examples/tau_bench/tau_tasks.py
```

**examples/true_on_policy** (2)

```
examples/true_on_policy/run-qwen3-0_6b-top-smoke.sh
examples/true_on_policy/run-qwen3-4b-top.sh
```

**orbit/audit** (2)

```
orbit/audit/__init__.py
orbit/audit/peft_wrap.py
```

**orbit/backends** (32)

```
orbit/backends/megatron_utils/bridge_peft_helpers.py
orbit/backends/megatron_utils/bridge_provider_overrides.py
orbit/backends/megatron_utils/critic_adapter.py
orbit/backends/megatron_utils/fp32_param_utils.py
orbit/backends/megatron_utils/low_precision_bootstrap.py
orbit/backends/megatron_utils/memory_attribution.py
orbit/backends/megatron_utils/model_state_manager.py
orbit/backends/megatron_utils/modelopt_state_shim.py
orbit/backends/megatron_utils/mtp_rl_patches.py
orbit/backends/megatron_utils/oft_utils.py
orbit/backends/megatron_utils/peft_offload.py
orbit/backends/megatron_utils/peft_transport/__init__.py
orbit/backends/megatron_utils/peft_transport/_gather.py
orbit/backends/megatron_utils/peft_transport/_payload.py
orbit/backends/megatron_utils/peft_transport/backends/__init__.py
orbit/backends/megatron_utils/peft_transport/backends/ipc.py
orbit/backends/megatron_utils/peft_transport/backends/nccl.py
orbit/backends/megatron_utils/peft_transport/backends/ray_object.py
orbit/backends/megatron_utils/peft_transport/interface.py
orbit/backends/megatron_utils/peft_transport/registry.py
orbit/backends/megatron_utils/peft_transport/runtime.py
orbit/backends/megatron_utils/peft_transport/slots.py
orbit/backends/megatron_utils/peft_utils.py
orbit/backends/megatron_utils/runtime_device.py
orbit/backends/megatron_utils/state_mode.py
orbit/backends/megatron_utils/tensor_semantics.py
orbit/backends/megatron_utils/update_weight/sync_metrics.py
orbit/backends/megatron_utils/update_weight/update_weight_from_distributed/bridge.py
orbit/backends/sglang_utils/compat_site/sitecustomize.py
orbit/backends/sglang_utils/native_ops.py
orbit/backends/training_utils/teacher_lm_head.py
orbit/backends/training_utils/vocab_parallel.py
```

**orbit/merge** (5)

```
orbit/merge/__init__.py
orbit/merge/bake_hf.py
orbit/merge/megatron_io.py
orbit/merge/oft_merge.py
orbit/merge/strategy.py
```

**orbit/rollout** (22)

```
orbit/rollout/fully_async_rollout.py
orbit/rollout/generate_utils/prefill_logprobs.py
orbit/rollout/genrm_judge.py
orbit/rollout/grader_errors.py
orbit/rollout/inference_rollout/eval_logging.py
orbit/rollout/llm_judge.py
orbit/rollout/opd_scoring.py
orbit/rollout/opd_sglang.py
orbit/rollout/reward_router.py
orbit/rollout/rm_hub/lean_rm.py
orbit/rollout/rm_hub/math_alignment.py
orbit/rollout/rm_hub/peft_arena_reward.py
orbit/rollout/rm_hub/ultra_agents.py
orbit/rollout/rm_hub/ultra_longtail.py
orbit/rollout/sandbox/__init__.py
orbit/rollout/sandbox/code_rm.py
orbit/rollout/sandbox/executor.py
orbit/rollout/sandbox/swe_rm.py
orbit/rollout/scoring_client.py
orbit/rollout/swe_agent/__init__.py
orbit/rollout/swe_agent/container_session.py
orbit/rollout/swe_agent/episode.py
```

**orbit/true_on_policy** (5)

```
orbit/true_on_policy/__init__.py
orbit/true_on_policy/config.py
orbit/true_on_policy/contracts.py
orbit/true_on_policy/model_profiles.py
orbit/true_on_policy/schema.py
```

**orbit/ultra** (2)

```
orbit/ultra/__init__.py
orbit/ultra/strict_json.py
```

**orbit/utils** (15)

```
orbit/utils/adapter_swap.py
orbit/utils/adapter_tensors.py
orbit/utils/chat_template_utils/deepseek_v4.py
orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja
orbit/utils/eval_nll.py
orbit/utils/llama3_chat_template.py
orbit/utils/logprob_compare.py
orbit/utils/opd_dump.py
orbit/utils/opd_teacher_pool.py
orbit/utils/opd_teacher_spec.py
orbit/utils/peft_param_match.py
orbit/utils/reward_normalization.py
orbit/utils/self_teacher.py
orbit/utils/self_teacher_checkpoint.py
orbit/utils/training_eta.py
```

**orbit_plugins/megatron_bridge** (13)

```
orbit_plugins/megatron_bridge/README.md
orbit_plugins/megatron_bridge/patches/__init__.py
orbit_plugins/megatron_bridge/patches/bridges/__init__.py
orbit_plugins/megatron_bridge/patches/bridges/nemotron_h.py
orbit_plugins/megatron_bridge/patches/bridges/qwen3_fp8_bridge.py
orbit_plugins/megatron_bridge/patches/conversion/__init__.py
orbit_plugins/megatron_bridge/patches/conversion/convert_checkpoints.py
orbit_plugins/megatron_bridge/patches/conversion/convert_fp8_checkpoint_direct.py
orbit_plugins/megatron_bridge/patches/conversion/convert_int4_checkpoint_direct.py
orbit_plugins/megatron_bridge/patches/conversion/convert_nvfp4_checkpoint_direct.py
orbit_plugins/megatron_bridge/patches/conversion/quantize_to_int4.py
orbit_plugins/megatron_bridge/patches/low_precision/__init__.py
orbit_plugins/megatron_bridge/patches/peft/__init__.py
```

**orbit_plugins/model_args** (56)

```
orbit_plugins/model_args/README.md
orbit_plugins/model_args/deepseek-v3-20layer.sh
orbit_plugins/model_args/deepseek-v3-5layer.sh
orbit_plugins/model_args/deepseek-v3.sh
orbit_plugins/model_args/deepseek-v4-flash-debug.sh
orbit_plugins/model_args/deepseek-v4-flash.sh
orbit_plugins/model_args/deepseek-v4-pro.sh
orbit_plugins/model_args/gemma-4-26b-a4b-it.sh
orbit_plugins/model_args/gemma-4-31b-it.sh
orbit_plugins/model_args/glm4-32B.sh
orbit_plugins/model_args/glm4-9B.sh
orbit_plugins/model_args/glm4.5-106B-A12B.sh
orbit_plugins/model_args/glm4.5-355B-A32B.sh
orbit_plugins/model_args/glm4.7-flash.sh
orbit_plugins/model_args/glm5-744B-A40B.sh
orbit_plugins/model_args/glm5-744B-A40B_20layer.sh
orbit_plugins/model_args/glm5-744B-A40B_4layer.sh
orbit_plugins/model_args/gpt-oss-20b.sh
orbit_plugins/model_args/kimi-k2-thinking.sh
orbit_plugins/model_args/kimi-k2.sh
orbit_plugins/model_args/kimi-k25-debug-6layer.sh
orbit_plugins/model_args/kimi-k25.sh
orbit_plugins/model_args/kimi-k26.sh
orbit_plugins/model_args/llama3-8B.sh
orbit_plugins/model_args/llama3.1-8B-Instruct.sh
orbit_plugins/model_args/llama3.2-3B-Instruct-amd.sh
orbit_plugins/model_args/llama3.2-3B-Instruct.sh
orbit_plugins/model_args/mimo-7B-rl.sh
orbit_plugins/model_args/moonlight.sh
orbit_plugins/model_args/nemotron-3-nano-30b-a3b.sh
orbit_plugins/model_args/nemotron-3-nano-4b.sh
orbit_plugins/model_args/qwen2.5-0.5B.sh
orbit_plugins/model_args/qwen2.5-1.5B.sh
orbit_plugins/model_args/qwen2.5-32B.sh
orbit_plugins/model_args/qwen2.5-3B.sh
orbit_plugins/model_args/qwen2.5-7B-4layer.sh
orbit_plugins/model_args/qwen2.5-7B.sh
orbit_plugins/model_args/qwen3-0.6B.sh
orbit_plugins/model_args/qwen3-1.7B.sh
orbit_plugins/model_args/qwen3-14B.sh
orbit_plugins/model_args/qwen3-235B-A22B.sh
orbit_plugins/model_args/qwen3-30B-A3B-4layer.sh
orbit_plugins/model_args/qwen3-30B-A3B-5layer.sh
orbit_plugins/model_args/qwen3-30B-A3B.sh
orbit_plugins/model_args/qwen3-32B.sh
orbit_plugins/model_args/qwen3-4B-Instruct-2507-w4a16.sh
orbit_plugins/model_args/qwen3-4B-Instruct-2507.sh
orbit_plugins/model_args/qwen3-4B.sh
orbit_plugins/model_args/qwen3-8B.sh
orbit_plugins/model_args/qwen3-next-80B-A3B.sh
orbit_plugins/model_args/qwen3.5-27B.sh
orbit_plugins/model_args/qwen3.5-35B-A3B.sh
orbit_plugins/model_args/qwen3.5-4B.sh
orbit_plugins/model_args/qwen3.5-9B.sh
orbit_plugins/model_args/qwen3.6-27B.sh
orbit_plugins/model_args/qwen3.6-35B-A3B.sh
```

**orbit_plugins/search_r1** (4)

```
orbit_plugins/search_r1/__init__.py
orbit_plugins/search_r1/generate_with_search.py
orbit_plugins/search_r1/local_search_server.py
orbit_plugins/search_r1/qa_em_format.py
```

**orbit_plugins/tau_bench** (4)

```
orbit_plugins/tau_bench/__init__.py
orbit_plugins/tau_bench/generate_with_tau.py
orbit_plugins/tau_bench/openai_tool_adapter.py
orbit_plugins/tau_bench/sglang_tool_parser.py
```

**results/backfill** (7)

```
results/backfill/e4_gsm8k_lr1.jsonl
results/backfill/e4_gsm8k_lr2.jsonl
results/backfill/e4_gsm8k_lr3.jsonl
results/backfill/e4_gsm8k_lr4.jsonl
results/backfill/e4_gsm8k_lr5.jsonl
results/backfill/e4_gsm8k_lr6.jsonl
results/backfill/e4_gsm8k_lr7.jsonl
```

**results/probe** (23)

```
results/probe/e1ot-full-na-na-lr6.28e-06-s0.jsonl
results/probe/e1ot-lora-r1-all-lr6.28e-05-s0.jsonl
results/probe/e1ot-oftscout-b1024-all-lr1e-05-s0.jsonl
results/probe/e1short-full-na-na-short-lr8.87e-06-s0.jsonl
results/probe/e1short-lora-r256-all-short-lr8.87e-05-s0.jsonl
results/probe/e1short-oftscout-b1024-all-short-lr1e-05-s0.jsonl
results/probe/e3-lora-r256-attn-lr6.28e-05-s0.jsonl
results/probe/e3-lora-r92-mlp-lr6.28e-05-s0.jsonl
results/probe/e3-oftscout-b1024-attn-lr1e-05-s0.jsonl
results/probe/e3-oftscout-b512-mlp-lr1e-05-s0.jsonl
results/probe/e4-full-na-na-lr3.16e-07-s0.jsonl
results/probe/e4-lora-r1-all-lr3.16e-06-s0.jsonl
results/probe/e4-oftscout-b1024-all-lr1e-06-s0.jsonl
results/probe/e4-oftscout-b1024-all-lr2.15e-05-s0.jsonl
results/probe/e4place-lora-r256-attn-lr3.16e-06-s0.jsonl
results/probe/e4place-lora-r92-mlp-lr3.16e-06-s0.jsonl
results/probe/e4place-oftscout-b1024-attn-lr1e-06-s0.jsonl
results/probe/e4place-oftscout-b1024-attn-lr2.15e-05-s0.jsonl
results/probe/e4place-oftscout-b512-mlp-lr1e-06-s0.jsonl
results/probe/e4place-oftscout-b512-mlp-lr2.15e-05-s0.jsonl
results/probe/e5rl-oft-b128-all-lr0.000316-s0.jsonl
results/probe/e5rl-oft-b32-all-lr0.000316-s0.jsonl
results/probe/e5rl-oft-b512-all-lr0.000316-s0.jsonl
```

**results/prompt_probe.jsonl** (1)

```
results/prompt_probe.jsonl
```

**results/prompt_probe2.jsonl** (1)

```
results/prompt_probe2.jsonl
```

**scripts/README.md** (1)

```
scripts/README.md
```

**scripts/check_oft_sharding_invariants.py** (1)

```
scripts/check_oft_sharding_invariants.py
```

**scripts/condor** (4)

```
scripts/condor/setup/README.md
scripts/condor/setup/install_env.sh
scripts/condor/setup/install_env.sub
scripts/condor/setup/verify_env.sh
```

**scripts/conversion** (8)

```
scripts/conversion/README.md
scripts/conversion/common.sh
scripts/conversion/configuration_deepseek_v4.py
scripts/conversion/convert_dsv4_hf_to_megatron.sh
scripts/conversion/convert_fp8_checkpoint_direct.sh
scripts/conversion/convert_int4_checkpoint_direct.sh
scripts/conversion/convert_nvfp4_checkpoint_direct.sh
scripts/conversion/deepseek_v4_chat_template.jinja
```

**scripts/inspect_oft_streamed_audit.py** (1)

```
scripts/inspect_oft_streamed_audit.py
```

**scripts/inspect_peft_wrap_audit.py** (1)

```
scripts/inspect_peft_wrap_audit.py
```

**scripts/lib** (9)

```
scripts/lib/common.sh
scripts/lib/driver.sh
scripts/lib/launcher.sh
scripts/lib/load_cuda13_2_orbit_env.sh
scripts/lib/paths.sh
scripts/lib/preflight.sh
scripts/lib/ray.sh
scripts/lib/tool_env.sh
scripts/lib/wandb.sh
```

**scripts/lora_regret** (107)

```
scripts/lora_regret/campaign.sh
scripts/lora_regret/coverage_probe.sh
scripts/lora_regret/coverage_probe_1gpu.sh
scripts/lora_regret/coverage_probe_4gpu.sh
scripts/lora_regret/coverage_probe_8gpu.sh
scripts/lora_regret/e4_protocol.sh
scripts/lora_regret/env2_rerun/README.md
scripts/lora_regret/env2_rerun/columns.sh
scripts/lora_regret/env2_rerun/condor/e4_gsm8k_ft_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_gsm8k_lora_r16_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_gsm8k_lora_r1_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_gsm8k_lora_r256_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_gsm8k_oft_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_math_ft_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_math_lora_r16_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_math_lora_r1_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_math_lora_r256_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/e4_math_oft_lr1_lr7.sub
scripts/lora_regret/env2_rerun/condor/job.sh
scripts/lora_regret/env2_rerun/condor/submit.sh
scripts/lora_regret/env2_rerun/env.sh
scripts/lora_regret/env2_rerun/run_column.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_ft_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lora_r16_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lora_r1_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lora_r256_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr1_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr2_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr3_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr4_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr5_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr6_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr1_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr2_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr3_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr4_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr5_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr6_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_gsm8k_oft_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_ft_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lora_r16_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lora_r1_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lora_r256_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr1_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr2_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr3_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr4_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr5_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr6_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr1_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr1_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr2_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr3_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr4_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr5_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr6_8gpu.sh
scripts/lora_regret/env2_rerun/run_e4_math_oft_lr7_8gpu.sh
scripts/lora_regret/env2_rerun/run_ft_column.sh
scripts/lora_regret/env2_rerun/run_lora_column.sh
scripts/lora_regret/env2_rerun/run_oft_column.sh
scripts/lora_regret/env2_rerun/sync_wandb.sh
scripts/lora_regret/env_v0516.sh
scripts/lora_regret/fetch_models.sh
scripts/lora_regret/run_e4_ft_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr0_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr1_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr2_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr3_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr4_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr5_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr6_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_lr7_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr0_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr1_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr2_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr3_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr4_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr5_8gpu.sh
scripts/lora_regret/run_e4_gsm8k_oft_lr6_8gpu.sh
scripts/lora_regret/run_e4_lora_8gpu.sh
scripts/lora_regret/run_e4_math_lora_verify_8gpu.sh
scripts/lora_regret/run_e4_math_lr0_8gpu.sh
scripts/lora_regret/run_e4_math_lr1_8gpu.sh
scripts/lora_regret/run_e4_math_lr2_8gpu.sh
scripts/lora_regret/run_e4_math_lr3_8gpu.sh
scripts/lora_regret/run_e4_math_lr4_8gpu.sh
scripts/lora_regret/run_e4_math_lr5_8gpu.sh
scripts/lora_regret/run_e4_math_lr6_8gpu.sh
scripts/lora_regret/run_e4_math_lr7_8gpu.sh
scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
scripts/lora_regret/run_e4_math_oft_b128_refine_a_8gpu.sh
scripts/lora_regret/run_e4_math_oft_b128_refine_b_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr0_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr1_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr2_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr4_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr5_8gpu.sh
scripts/lora_regret/run_e4_math_oft_lr6_8gpu.sh
scripts/lora_regret/run_e4_math_oft_verify_8gpu.sh
scripts/lora_regret/run_e4place_ft_8gpu.sh
scripts/lora_regret/run_e4place_lora_8gpu.sh
scripts/lora_regret/smoke_e4_8gpu.sh
scripts/lora_regret/sync_wandb.sh
```

**scripts/release** (1)

```
scripts/release/clean_room_gate.sh
```

**scripts/slurm** (14)

```
scripts/slurm/setup/cu128/BINARY_LAYER.md
scripts/slurm/setup/cu128/README.md
scripts/slurm/setup/cu128/extract_pins.py
scripts/slurm/setup/cu128/install_env.sh
scripts/slurm/setup/cu128/pins.env
scripts/slurm/setup/cu128/verify_env.py
scripts/slurm/setup/cu130/README.md
scripts/slurm/setup/cu130/extract_pins.py
scripts/slurm/setup/cu130/install_env.sh
scripts/slurm/setup/cu130/materialize_env.py
scripts/slurm/setup/cu130/miles-wheels-cu130-x86_64.sha256
scripts/slurm/setup/cu130/pins.env
scripts/slurm/setup/cu130/slurm_h200_runtime.sh
scripts/slurm/setup/cu130/verify_env.py
```

**tests/fast** (78)

```
tests/fast/dist_utils.py
tests/fast/fixtures/lora_regret/llama3_sample.jsonl
tests/fast/fixtures/lora_regret/no_robots_sample.jsonl
tests/fast/fixtures/lora_regret/smoke_lora_r256_eval_lines.log
tests/fast/rollout/test_sft_loss_mask_parity.py
tests/fast/rollout/test_sft_loss_mask_parity_llama3.py
tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py
tests/fast/scripts/slurm/setup/cu128/test_install_env.py
tests/fast/scripts/slurm/setup/cu128/test_verify_env.py
tests/fast/scripts/slurm/setup/cu130/test_materialize_env.py
tests/fast/test_actor_critic_sync.py
tests/fast/test_actor_ref_restore.py
tests/fast/test_aggregate_train_losses_extrema.py
tests/fast/test_async_off_policy_guard.py
tests/fast/test_async_offload_noop.py
tests/fast/test_bake_hf.py
tests/fast/test_distributed_update_weights_sync_metrics.py
tests/fast/test_get_responses_temperature.py
tests/fast/test_launcher_extra_train_args.py
tests/fast/test_launcher_topology_env.py
tests/fast/test_log_rollout_data_parity_gate.py
tests/fast/test_log_rollout_data_topk_keys.py
tests/fast/test_logprob_compare.py
tests/fast/test_loss_reduction_microbatch_invariance.py
tests/fast/test_megatron_bridge_orbit_namespace.py
tests/fast/test_megatron_merge.py
tests/fast/test_merge_cli.py
tests/fast/test_oft_merge.py
tests/fast/test_opd_cp_data.py
tests/fast/test_opd_dump.py
tests/fast/test_opd_teacher_equivalence.py
tests/fast/test_opd_topk_args.py
tests/fast/test_opd_topk_loss.py
tests/fast/test_opd_topk_ray_transport.py
tests/fast/test_opd_topk_transport.py
tests/fast/test_orthomerge_bridge.py
tests/fast/test_ppo_cp_advantages.py
tests/fast/test_ppo_gae_masks.py
tests/fast/test_prefill_cuda_graph_policy.py
tests/fast/test_rollout_timeline_binning.py
tests/fast/test_rollout_timeline_figure.py
tests/fast/test_rollout_timeline_probe.py
tests/fast/test_self_teacher_save_chain.py
tests/fast/test_sync_metrics.py
tests/fast/test_unbiased_kl_numerics.py
tests/fast/test_update_weights_sync_metrics.py
tests/fast/test_value_explained_var.py
tests/fast/test_vocab_parallel_topk.py
tests/fast/utils/test_eval_nll.py
tests/fast/utils/test_full_model_train_offload.py
tests/fast/utils/test_llama3_chat_template.py
tests/fast/utils/test_llama3_loss_mask.py
tests/fast/utils/test_lora_a_init_method_reaches_adapter.py
tests/fast/utils/test_lora_regret_analyze.py
tests/fast/utils/test_lora_regret_arms_coverage.py
tests/fast/utils/test_lora_regret_e5rl.py
tests/fast/utils/test_lora_regret_env2_rerun.py
tests/fast/utils/test_lora_regret_lr_columns.py
tests/fast/utils/test_lora_regret_models.py
tests/fast/utils/test_lora_regret_p3_check.py
tests/fast/utils/test_lora_regret_plot.py
tests/fast/utils/test_lora_regret_post_protocol.py
tests/fast/utils/test_lora_regret_preflight.py
tests/fast/utils/test_lora_regret_prepare_data.py
tests/fast/utils/test_lora_regret_probe.py
tests/fast/utils/test_lora_regret_prompt_rendering.py
tests/fast/utils/test_lora_regret_smoke.py
tests/fast/utils/test_lora_regret_sweep.py
tests/fast/utils/test_lora_regret_trace.py
tests/fast/utils/test_math_oft_b128_low_lr_sweep.py
tests/fast/utils/test_math_oft_b128_refinement_sweep.py
tests/fast/utils/test_memory_utils_allocator_counters.py
tests/fast/utils/test_modelopt_state_shim.py
tests/fast/utils/test_peft_arguments.py
tests/fast/utils/test_peft_param_match.py
tests/fast/utils/test_probe_steady_state.py
tests/fast/utils/test_train_actor_allocator_env.py
tests/fast/utils/test_wandb_run_naming.py
```

**tests/test_adapter_swap.py** (1)

```
tests/test_adapter_swap.py
```

**tests/test_bridge_provider_overrides.py** (1)

```
tests/test_bridge_provider_overrides.py
```

**tests/test_convert_hf_to_torch_dist.py** (1)

```
tests/test_convert_hf_to_torch_dist.py
```

**tests/test_critic_build_args.py** (1)

```
tests/test_critic_build_args.py
```

**tests/test_critic_checkpoint.py** (1)

```
tests/test_critic_checkpoint.py
```

**tests/test_critic_head_build.py** (1)

```
tests/test_critic_head_build.py
```

**tests/test_critic_low_precision.py** (1)

```
tests/test_critic_low_precision.py
```

**tests/test_critic_mode_args.py** (1)

```
tests/test_critic_mode_args.py
```

**tests/test_critic_peft_build.py** (1)

```
tests/test_critic_peft_build.py
```

**tests/test_critic_placement.py** (1)

```
tests/test_critic_placement.py
```

**tests/test_critic_train_phases.py** (1)

```
tests/test_critic_train_phases.py
```

**tests/test_critic_trunk_alias.py** (1)

```
tests/test_critic_trunk_alias.py
```

**tests/test_determinism_harness.py** (1)

```
tests/test_determinism_harness.py
```

**tests/test_distributed_utils.py** (1)

```
tests/test_distributed_utils.py
```

**tests/test_fp32_param_utils.py** (1)

```
tests/test_fp32_param_utils.py
```

**tests/test_full_vocab_parity_launcher.py** (1)

```
tests/test_full_vocab_parity_launcher.py
```

**tests/test_fullft_async_launcher.py** (1)

```
tests/test_fullft_async_launcher.py
```

**tests/test_gemma4_weight_sync_tolerances.py** (1)

```
tests/test_gemma4_weight_sync_tolerances.py
```

**tests/test_gemma_math_reward.py** (1)

```
tests/test_gemma_math_reward.py
```

**tests/test_generate_endpoint_peft_payload.py** (1)

```
tests/test_generate_endpoint_peft_payload.py
```

**tests/test_genrm_judge.py** (1)

```
tests/test_genrm_judge.py
```

**tests/test_lean_rm.py** (1)

```
tests/test_lean_rm.py
```

**tests/test_llm_judge.py** (1)

```
tests/test_llm_judge.py
```

**tests/test_lora_regret_reward_grading.py** (1)

```
tests/test_lora_regret_reward_grading.py
```

**tests/test_lora_regret_rl_launcher.py** (1)

```
tests/test_lora_regret_rl_launcher.py
```

**tests/test_model_provider.py** (1)

```
tests/test_model_provider.py
```

**tests/test_opd_advantage.py** (1)

```
tests/test_opd_advantage.py
```

**tests/test_opd_args.py** (1)

```
tests/test_opd_args.py
```

**tests/test_opd_critic_role.py** (1)

```
tests/test_opd_critic_role.py
```

**tests/test_opd_full_vocab.py** (1)

```
tests/test_opd_full_vocab.py
```

**tests/test_opd_jsd_loss.py** (1)

```
tests/test_opd_jsd_loss.py
```

**tests/test_opd_promotion.py** (1)

```
tests/test_opd_promotion.py
```

**tests/test_opd_rollout_data.py** (1)

```
tests/test_opd_rollout_data.py
```

**tests/test_opd_sample_merge.py** (1)

```
tests/test_opd_sample_merge.py
```

**tests/test_opd_scoring_stage.py** (1)

```
tests/test_opd_scoring_stage.py
```

**tests/test_opd_serve_teacher.py** (1)

```
tests/test_opd_serve_teacher.py
```

**tests/test_opd_sglang_postprocess.py** (1)

```
tests/test_opd_sglang_postprocess.py
```

**tests/test_opd_teacher_pool.py** (1)

```
tests/test_opd_teacher_pool.py
```

**tests/test_opd_teacher_spec.py** (1)

```
tests/test_opd_teacher_spec.py
```

**tests/test_opd_topk_scoring.py** (1)

```
tests/test_opd_topk_scoring.py
```

**tests/test_orbit_router_workers.py** (1)

```
tests/test_orbit_router_workers.py
```

**tests/test_peft_bridge_preload.py** (1)

```
tests/test_peft_bridge_preload.py
```

**tests/test_peft_broadcast_shm_refcount.py** (1)

```
tests/test_peft_broadcast_shm_refcount.py
```

**tests/test_peft_ipc_transport.py** (1)

```
tests/test_peft_ipc_transport.py
```

**tests/test_peft_ray_transport.py** (1)

```
tests/test_peft_ray_transport.py
```

**tests/test_peft_two_phase_resume.py** (1)

```
tests/test_peft_two_phase_resume.py
```

**tests/test_pion_optimizer.py** (1)

```
tests/test_pion_optimizer.py
```

**tests/test_ppo_critic_compare_launchers.py** (1)

```
tests/test_ppo_critic_compare_launchers.py
```

**tests/test_ppo_critic_distributed_checkpoint.py** (1)

```
tests/test_ppo_critic_distributed_checkpoint.py
```

**tests/test_ppo_launch_scripts.py** (1)

```
tests/test_ppo_launch_scripts.py
```

**tests/test_ppo_peft_distributed_preflight.py** (1)

```
tests/test_ppo_peft_distributed_preflight.py
```

**tests/test_ppo_peft_distributed_save_coordination.py** (1)

```
tests/test_ppo_peft_distributed_save_coordination.py
```

**tests/test_ppo_peft_distributed_save_preflight.py** (1)

```
tests/test_ppo_peft_distributed_save_preflight.py
```

**tests/test_ppo_ratio_numerics.py** (1)

```
tests/test_ppo_ratio_numerics.py
```

**tests/test_ppo_resume_orchestration.py** (1)

```
tests/test_ppo_resume_orchestration.py
```

**tests/test_prefill_logprobs.py** (1)

```
tests/test_prefill_logprobs.py
```

**tests/test_qwen2_true_on_policy_conversion.py** (1)

```
tests/test_qwen2_true_on_policy_conversion.py
```

**tests/test_response_only_loss_mask.py** (1)

```
tests/test_response_only_loss_mask.py
```

**tests/test_reward_router.py** (1)

```
tests/test_reward_router.py
```

**tests/test_rl_fullft_tensor_parallel.py** (1)

```
tests/test_rl_fullft_tensor_parallel.py
```

**tests/test_rollout_data_source_resume.py** (1)

```
tests/test_rollout_data_source_resume.py
```

**tests/test_sandbox_code_rm.py** (1)

```
tests/test_sandbox_code_rm.py
```

**tests/test_sandbox_executor.py** (1)

```
tests/test_sandbox_executor.py
```

**tests/test_scoring_client.py** (1)

```
tests/test_scoring_client.py
```

**tests/test_search_r1_example.py** (1)

```
tests/test_search_r1_example.py
```

**tests/test_search_r1_launch_scripts.py** (1)

```
tests/test_search_r1_launch_scripts.py
```

**tests/test_self_teacher.py** (1)

```
tests/test_self_teacher.py
```

**tests/test_sft_dataset_conversion.py** (1)

```
tests/test_sft_dataset_conversion.py
```

**tests/test_sft_jsonl_partition_split.py** (1)

```
tests/test_sft_jsonl_partition_split.py
```

**tests/test_sft_launch_scripts.py** (1)

```
tests/test_sft_launch_scripts.py
```

**tests/test_sft_mode.py** (1)

```
tests/test_sft_mode.py
```

**tests/test_sglang_native_ops.py** (1)

```
tests/test_sglang_native_ops.py
```

**tests/test_sglang_true_on_policy_deterministic_fallback.py** (1)

```
tests/test_sglang_true_on_policy_deterministic_fallback.py
```

**tests/test_swe_agent_episode.py** (1)

```
tests/test_swe_agent_episode.py
```

**tests/test_swe_rm.py** (1)

```
tests/test_swe_rm.py
```

**tests/test_tau_bench_example.py** (1)

```
tests/test_tau_bench_example.py
```

**tests/test_tau_bench_launch_scripts.py** (1)

```
tests/test_tau_bench_launch_scripts.py
```

**tests/test_training_checkpoint_resume.py** (1)

```
tests/test_training_checkpoint_resume.py
```

**tests/test_true_on_policy_config.py** (1)

```
tests/test_true_on_policy_config.py
```

**tests/test_true_on_policy_launch_scripts.py** (1)

```
tests/test_true_on_policy_launch_scripts.py
```

**tests/test_true_on_policy_logprobs.py** (1)

```
tests/test_true_on_policy_logprobs.py
```

**tests/test_ultra_agents.py** (1)

```
tests/test_ultra_agents.py
```

**tests/test_ultra_longtail.py** (1)

```
tests/test_ultra_longtail.py
```

**tests/test_update_weight_bridge_distributed.py** (1)

```
tests/test_update_weight_bridge_distributed.py
```

**tests/test_wandb_utils.py** (1)

```
tests/test_wandb_utils.py
```

**tools/README_merge_oft.md** (1)

```
tools/README_merge_oft.md
```

**tools/adapter_runtime_compare** (5)

```
tools/adapter_runtime_compare/__init__.py
tools/adapter_runtime_compare/analyze_a1.py
tools/adapter_runtime_compare/run_compare.py
tools/adapter_runtime_compare/test_analyze_a1.py
tools/adapter_runtime_compare/test_run_compare.py
```

**tools/bake_oft_to_hf.py** (1)

```
tools/bake_oft_to_hf.py
```

**tools/bench_kimi_int4_oft_gemm_latency.py** (1)

```
tools/bench_kimi_int4_oft_gemm_latency.py
```

**tools/check_checkpoint_parity.py** (1)

```
tools/check_checkpoint_parity.py
```

**tools/check_dsv4_checkpoint_parity.py** (1)

```
tools/check_dsv4_checkpoint_parity.py
```

**tools/check_dsv4_deepgemm_cross_repo_parity.py** (1)

```
tools/check_dsv4_deepgemm_cross_repo_parity.py
```

**tools/check_fp8_checkpoint_parity.py** (1)

```
tools/check_fp8_checkpoint_parity.py
```

**tools/check_fp8_runtime_parity.py** (1)

```
tools/check_fp8_runtime_parity.py
```

**tools/check_int4_checkpoint_parity.py** (1)

```
tools/check_int4_checkpoint_parity.py
```

**tools/check_int4_runtime_parity.py** (1)

```
tools/check_int4_runtime_parity.py
```

**tools/check_nvfp4_checkpoint_parity.py** (1)

```
tools/check_nvfp4_checkpoint_parity.py
```

**tools/check_nvfp4_runtime_parity.py** (1)

```
tools/check_nvfp4_runtime_parity.py
```

**tools/check_runtime_step0_parity.py** (1)

```
tools/check_runtime_step0_parity.py
```

**tools/checkpoint_parity_core.py** (1)

```
tools/checkpoint_parity_core.py
```

**tools/checkpoint_parity_utils.py** (1)

```
tools/checkpoint_parity_utils.py
```

**tools/compare_opd_teacher_logprobs.py** (1)

```
tools/compare_opd_teacher_logprobs.py
```

**tools/convert_checkpoints.py** (1)

```
tools/convert_checkpoints.py
```

**tools/convert_dsv4_hf_to_megatron.py** (1)

```
tools/convert_dsv4_hf_to_megatron.py
```

**tools/convert_fp8_checkpoint_direct.py** (1)

```
tools/convert_fp8_checkpoint_direct.py
```

**tools/convert_hf_to_int4_legacy.py** (1)

```
tools/convert_hf_to_int4_legacy.py
```

**tools/convert_int4_checkpoint_direct.py** (1)

```
tools/convert_int4_checkpoint_direct.py
```

**tools/convert_math_eval_to_orbit.py** (1)

```
tools/convert_math_eval_to_orbit.py
```

**tools/convert_nvfp4_checkpoint_direct.py** (1)

```
tools/convert_nvfp4_checkpoint_direct.py
```

**tools/convert_peftarena_data.py** (1)

```
tools/convert_peftarena_data.py
```

**tools/convert_sft_dataset_to_orbit.py** (1)

```
tools/convert_sft_dataset_to_orbit.py
```

**tools/convert_to_hf_legacy.py** (1)

```
tools/convert_to_hf_legacy.py
```

**tools/convert_torch_dist_to_hf_ray.py** (1)

```
tools/convert_torch_dist_to_hf_ray.py
```

**tools/eval_checkpoints_loop.sh** (1)

```
tools/eval_checkpoints_loop.sh
```

**tools/eval_checkpoints_once.sh** (1)

```
tools/eval_checkpoints_once.sh
```

**tools/gpu_process_memory_sampler.py** (1)

```
tools/gpu_process_memory_sampler.py
```

**tools/lean_rm_oracle.py** (1)

```
tools/lean_rm_oracle.py
```

**tools/lora_regret** (17)

```
tools/lora_regret/__init__.py
tools/lora_regret/analyze.py
tools/lora_regret/arms.py
tools/lora_regret/backfill.py
tools/lora_regret/g4_hf_nll.py
tools/lora_regret/models.py
tools/lora_regret/p3_check.py
tools/lora_regret/plot.py
tools/lora_regret/preflight.py
tools/lora_regret/prepare_data.py
tools/lora_regret/probe.py
tools/lora_regret/probe_log.py
tools/lora_regret/prompt_probe.py
tools/lora_regret/run_paths.py
tools/lora_regret/smoke.py
tools/lora_regret/sweep.py
tools/lora_regret/trace.py
```

**tools/merge_oft_adapters.py** (1)

```
tools/merge_oft_adapters.py
```

**tools/muon_kimi_equivalence.py** (1)

```
tools/muon_kimi_equivalence.py
```

**tools/orthomerge_bridge.py** (1)

```
tools/orthomerge_bridge.py
```

**tools/prepare_swe_subset.py** (1)

```
tools/prepare_swe_subset.py
```

**tools/quantize_to_int4.py** (1)

```
tools/quantize_to_int4.py
```

**tools/rollout_determinism_harness.py** (1)

```
tools/rollout_determinism_harness.py
```

**tools/rollout_timeline** (4)

```
tools/rollout_timeline/__init__.py
tools/rollout_timeline/binning.py
tools/rollout_timeline/figure.py
tools/rollout_timeline/probe.py
```

**tools/runtime_step0_parity_utils.py** (1)

```
tools/runtime_step0_parity_utils.py
```

**tools/split_sft_jsonl_partitions.py** (1)

```
tools/split_sft_jsonl_partitions.py
```

**tools/summarize_eval_results.py** (1)

```
tools/summarize_eval_results.py
```

**tools/swe_agent_oracle.py** (1)

```
tools/swe_agent_oracle.py
```

**tools/swe_rm_oracle.py** (1)

```
tools/swe_rm_oracle.py
```

</details>

### Layer 2 — modified miles files (the entangled layer)

Sorted by lines changed vs the fork base. The **conflicts** column marks the 54 files where
the transfer dry-run against miles@2026-08-27 produced a both-modified conflict — this column
is the actual work list for any transfer.

| Changed lines | File | Conflicts vs latest miles |
|--:|:--|:--:|
| 2,103 | `orbit/utils/arguments.py` |  |
| 1,039 | `orbit/backends/training_utils/loss.py` |  |
| 1,028 | `orbit/backends/megatron_utils/checkpoint.py` |  |
| 987 | `orbit/backends/megatron_utils/actor.py` |  |
| 619 | `orbit/backends/megatron_utils/update_weight/update_weight_from_tensor.py` |  |
| 614 | `orbit/backends/sglang_utils/sglang_engine.py` |  |
| 596 | `orbit/utils/ppo_utils.py` |  |
| 473 | `orbit/rollout/sglang_rollout.py` | yes |
| 467 | `orbit/backends/megatron_utils/lora_utils.py` |  |
| 397 | `examples/README.md` | yes |
| 392 | `orbit/backends/megatron_utils/model.py` | yes |
| 369 | `examples/on_policy_distillation/README.md` | yes |
| 331 | `orbit/ray/rollout.py` |  |
| 318 | `.github/workflows/pr-test.yml` | yes |
| 287 | `pyproject.toml` |  |
| 275 | `orbit/backends/training_utils/data.py` | yes |
| 273 | `train.py` | yes |
| 246 | `README.md` | yes |
| 242 | `orbit/rollout/data_source.py` |  |
| 219 | `orbit/backends/training_utils/log_utils.py` | yes |
| 199 | `tools/convert_hf_to_torch_dist.py` | yes |
| 199 | `orbit/utils/mask_utils.py` |  |
| 183 | `orbit/utils/types.py` | yes |
| 174 | `tests/fast/utils/test_lora_arguments.py` |  |
| 137 | `examples/low_precision/README.md` |  |
| 133 | `orbit/backends/sglang_utils/arguments.py` |  |
| 133 | `orbit/backends/megatron_utils/update_weight/common.py` | yes |
| 125 | `orbit/ray/actor_group.py` |  |
| 123 | `orbit/backends/megatron_utils/bridge_lora_helpers.py` |  |
| 121 | `orbit/rollout/generate_utils/sample_utils.py` |  |
| 119 | `orbit/router/router.py` | yes |
| 113 | `orbit/rollout/inference_rollout/inference_rollout_eval.py` |  |
| 100 | `orbit/utils/test_utils/mock_sglang_server.py` |  |
| 91 | `orbit/rollout/generate_utils/generate_endpoint_utils.py` |  |
| 90 | `orbit/utils/http_utils.py` | yes |
| 90 | `orbit/ray/placement_group.py` | yes |
| 76 | `orbit/backends/megatron_utils/replay_utils.py` |  |
| 75 | `orbit/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py` |  |
| 63 | `orbit/backends/megatron_utils/model_provider.py` | yes |
| 61 | `orbit_plugins/mbridge/qwen3_5.py` | yes |
| 58 | `orbit/utils/replay_base.py` | yes |
| 54 | `orbit/utils/external_utils/command_utils.py` | yes |
| 41 | `orbit/utils/distributed_utils.py` | yes |
| 39 | `orbit/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py` | yes |
| 38 | `requirements.txt` | yes |
| 38 | `.github/workflows/pr-test.yml.j2` |  |
| 37 | `orbit/backends/megatron_utils/arguments.py` |  |
| 35 | `orbit/utils/reloadable_process_group.py` |  |
| 35 | `orbit/utils/chat_template_utils/tito_tokenizer.py` | yes |
| 32 | `train_async.py` | yes |
| 32 | `orbit/utils/memory_utils.py` |  |
| 32 | `orbit/rollout/inference_rollout/inference_rollout_common.py` | yes |
| 32 | `.github/workflows/release-docs.yaml` |  |
| 28 | `orbit/ray/train_actor.py` | yes |
| 27 | `orbit/utils/data.py` | yes |
| 27 | `orbit/rollout/rm_hub/deepscaler.py` |  |
| 25 | `orbit/ray/utils.py` |  |
| 25 | `orbit/backends/megatron_utils/initialize.py` | yes |
| 24 | `.gitignore` | yes |
| 22 | `orbit/rollout/rm_hub/__init__.py` | yes |
| 22 | `orbit/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py` |  |
| 21 | `orbit/rollout/generate_utils/openai_endpoint_utils.py` | yes |
| 18 | `orbit/utils/wandb_utils.py` |  |
| 18 | `orbit/rollout/generate_hub/multi_turn.py` | yes |
| 18 | `orbit/backends/training_utils/cp_utils.py` |  |
| 18 | `orbit/backends/megatron_utils/megatron_to_hf/__init__.py` | yes |
| 16 | `orbit/utils/dumper_utils.py` |  |
| 16 | `orbit/rollout/inference_rollout/inference_rollout_train.py` | yes |
| 16 | `orbit/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py` | yes |
| 15 | `orbit/rollout/session/session_server.py` |  |
| 15 | `orbit/rollout/generate_hub/single_turn.py` | yes |
| 15 | `orbit/backends/megatron_utils/update_weight/hf_weight_iterator_base.py` | yes |
| 14 | `orbit/utils/tensor_backper.py` |  |
| 14 | `orbit/router/middleware_hub/radix_tree_middleware.py` |  |
| 14 | `orbit/rollout/base_types.py` | yes |
| 14 | `orbit/backends/megatron_utils/megatron_to_hf/qwen3moe.py` |  |
| 13 | `tools/fp8_cast_bf16.py` |  |
| 13 | `tools/convert_hf_to_fp8.py` |  |
| 12 | `orbit/utils/train_metric_utils.py` | yes |
| 12 | `orbit/rollout/generate_hub/agentic_tool_call.py` | yes |
| 12 | `orbit/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py` | yes |
| 11 | `orbit/utils/metric_utils.py` |  |
| 10 | `setup.py` | yes |
| 10 | `orbit_plugins/models/qwen3_5.py` | yes |
| 10 | `orbit/utils/processing_utils.py` | yes |
| 10 | `orbit/rollout/session/sessions.py` | yes |
| 10 | `orbit/rollout/session/linear_trajectory.py` | yes |
| 9 | `orbit/utils/misc.py` | yes |
| 8 | `orbit/utils/prometheus_utils.py` |  |
| 8 | `orbit/utils/chat_template_utils/__init__.py` | yes |
| 6 | `orbit_plugins/megatron_bridge/__init__.py` |  |
| 6 | `orbit/utils/typer_utils.py` | yes |
| 6 | `orbit/utils/tracking_utils.py` |  |
| 6 | `orbit/utils/test_utils/mock_trajectories.py` |  |
| 6 | `orbit/rollout/inference_rollout/compatibility.py` |  |
| 6 | `orbit/rollout/generate_hub/benchmarkers.py` |  |
| 5 | `orbit/backends/megatron_utils/__init__.py` |  |
| 4 | `tools/convert_torch_dist_to_hf.py` | yes |
| 4 | `orbit_plugins/models/glm5/glm5.py` | yes |
| 4 | `orbit/utils/test_utils/mock_tools.py` | yes |
| 4 | `orbit/utils/test_utils/chat_template_verify.py` | yes |
| 4 | `orbit/utils/profile_utils.py` |  |
| 4 | `orbit/utils/environ.py` | yes |
| 4 | `orbit/rollout/sft_rollout.py` |  |
| 4 | `orbit/rollout/generate_utils/tool_call_utils.py` |  |
| 4 | `orbit/rollout/filter_hub/dynamic_sampling_filters.py` |  |
| 4 | `orbit/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py` |  |
| 4 | `orbit/backends/megatron_utils/parallel.py` |  |
| 4 | `orbit/backends/megatron_utils/misc_utils.py` |  |
| 4 | `orbit/backends/megatron_utils/megatron_to_hf/qwen2.py` |  |
| 4 | `orbit/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py` |  |
| 4 | `orbit/backends/megatron_utils/megatron_to_hf/glm4.py` |  |
| 2 | `orbit_plugins/models/qwen3_next.py` |  |
| 2 | `orbit_plugins/models/hf_attention.py` |  |
| 2 | `orbit_plugins/models/cp_utils.py` |  |
| 2 | `orbit/utils/tensorboard_utils.py` |  |
| 2 | `orbit/utils/iter_utils.py` |  |
| 2 | `orbit/utils/eval_config.py` |  |
| 2 | `orbit/utils/env_report.py` |  |
| 2 | `orbit/utils/__init__.py` |  |
| 2 | `orbit/rollout/generate_hub/__init__.py` |  |
| 2 | `orbit/ray/ray_actor.py` |  |
| 2 | `orbit/backends/training_utils/ci_utils.py` |  |
| 2 | `orbit/backends/megatron_utils/megatron_to_hf/qwen3_next.py` |  |
| 2 | `orbit/backends/megatron_utils/megatron_to_hf/qwen3_5.py` |  |
| 2 | `orbit/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py` |  |
| 2 | `orbit/backends/megatron_utils/megatron_to_hf/llama.py` |  |
| 2 | `orbit/backends/megatron_utils/megatron_to_hf/glm4moe.py` |  |
| 2 | `orbit/backends/megatron_utils/ci_utils.py` |  |
| 1 | `tools/__init__.py` |  |
| 1 | `tests/__init__.py` |  |
| 1 | `orbit_plugins/__init__.py` |  |
| 1 | `examples/__init__.py` |  |

### Layer 3 — dropped miles files

| Dropped | Directory |
|--:|:--|
| 69 | `tests/fast/` |
| 43 | `tests/e2e/` |
| 39 | `scripts/models/` |
| 24 | `docs/en/` |
| 20 | `examples/eval/` |
| 14 | `orbit/backends/` |
| 11 | `examples/experimental/` |
| 9 | `examples/geo3k_vlm_multi_turn/` |
| 8 | `examples/retool/` |
| 8 | `examples/search-r1/` |
| 7 | `docs/_static/` |
| 7 | `examples/formal_math/` |
| 7 | `examples/tau-bench/` |
| 7 | `examples/true_on_policy/` |
| 6 | `examples/low_precision/` |
| 6 | `examples/multi_agent/` |
| 5 | `examples/train_infer_mismatch_helper/` |
| 5 | `tests/ci/` |
| 4 | `examples/eval_multi_task/` |
| 4 | `examples/geo3k_vlm/` |
| 4 | `examples/strands_sglang/` |
| 4 | `orbit/utils/` |
| 3 | `docker/glm5/` |
| 3 | `docker/patch/` |
| 3 | `examples/fully_async/` |
| 3 | `examples/lora/` |
| 3 | `examples/openai_format/` |
| 3 | `examples/retool_v2/` |
| 3 | `examples/true_on_policy_vlm/` |
| 2 | `docker/amd_patch/` |
| 2 | `examples/DrGRPO/` |
| 2 | `examples/on_policy_distillation/` |
| 2 | `examples/reproducibility/` |
| 1 | `.github/CODEOWNERS/` |
| 1 | `.github/workflows/` |
| 1 | `(root)/` |
| 1 | `docker/Dockerfile/` |
| 1 | `docker/Dockerfile.rocm_MI300/` |
| 1 | `docker/Dockerfile.rocm_MI350-5/` |
| 1 | `docker/Dockerfile_GB300/` |
| 1 | `docker/README.md/` |
| 1 | `docker/build.py/` |
| 1 | `docker/justfile/` |
| 1 | `docker/version.txt/` |
| 1 | `docs/README.md/` |
| 1 | `docs/build.sh/` |
| 1 | `docs/build_all.sh/` |
| 1 | `docs/conf.py/` |
| 1 | `docs/requirements.txt/` |
| 1 | `docs/serve.sh/` |
| 1 | `imgs/arch.png/` |
| 1 | `imgs/miles_logo.png/` |
| 1 | `imgs/miles_square.png/` |
| 1 | `scripts/amd/` |
| 1 | `scripts/run-deepseek-r1.sh/` |
| 1 | `scripts/run-glm4-9B-4xgpu-radixtree.sh/` |
| 1 | `scripts/run-glm4-9B.sh/` |
| 1 | `scripts/run-glm4.5-355B-A32B.sh/` |
| 1 | `scripts/run-glm4.7-flash.sh/` |
| 1 | `scripts/run-gpt-oss-20b-bf16.sh/` |
| 1 | `scripts/run-gptoss-20b-fsdp.sh/` |
| 1 | `scripts/run-kimi-k2-Instruct.sh/` |
| 1 | `scripts/run-kimi-k2-Thinking.sh/` |
| 1 | `scripts/run-mimo-7B-rl-eagle.sh/` |
| 1 | `scripts/run-moonlight-16B-A3B.sh/` |
| 1 | `scripts/run-qwen3-235B-A22B-sft.sh/` |
| 1 | `scripts/run-qwen3-235B-A22B.sh/` |
| 1 | `scripts/run-qwen3-32B.sh/` |
| 1 | `scripts/run-qwen3-4B-base-sft.sh/` |
| 1 | `scripts/run-qwen3-4B-fsdp.sh/` |
| 1 | `scripts/run-qwen3-4B.sh/` |
| 1 | `scripts/run-qwen3-4B_4xgpu-radixtree.sh/` |
| 1 | `scripts/run-qwen3-4B_4xgpu.sh/` |
| 1 | `scripts/run-qwen3-next-80B-A3B-8gpus.sh/` |
| 1 | `scripts/run-qwen3-next-80B-A3B-fsdp.sh/` |
| 1 | `scripts/run-qwen3-next-80B-A3B.sh/` |
| 1 | `scripts/run-qwen3.5-27B.sh/` |
| 1 | `scripts/run-qwen3.5-35B-A3B-mtp.sh/` |
| 1 | `scripts/run-qwen3.5-4B.sh/` |
| 1 | `scripts/run-qwen3.5-9B.sh/` |
| 1 | `scripts/run_deepseek.py/` |
| 1 | `scripts/run_glm45_355b_a32b.py/` |
| 1 | `scripts/run_glm47_flash.py/` |
| 1 | `scripts/run_glm5_744b_a40b.py/` |
| 1 | `scripts/run_mcore_fsdp.py/` |
| 1 | `scripts/run_qwen3_30b_a3b.py/` |
| 1 | `scripts/run_qwen3_4b.py/` |
| 1 | `scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py/` |
| 1 | `scripts/tools/` |
| 1 | `tests/test_chunked_gae.py/` |
| 1 | `tests/test_external_rollout.py/` |
| 1 | `tests/test_fsdp_import.py/` |
| 1 | `tests/test_fused_experts_backward.py/` |
| 1 | `tests/test_gspo.sh/` |
| 1 | `tests/utils/` |
| 1 | `tools/convert_hf_to_int4.py/` |
| 1 | `tools/convert_hf_to_int4_direct.py/` |
| 1 | `tools/convert_hf_to_nvfp4.py/` |
| 1 | `tools/convert_to_hf.py/` |

<details>
<summary>All 406 dropped files</summary>

```
.github/CODEOWNERS
.github/workflows/docker-build.yml
.gitmodules
docker/Dockerfile
docker/Dockerfile.rocm_MI300
docker/Dockerfile.rocm_MI350-5
docker/Dockerfile_GB300
docker/README.md
docker/amd_patch/latest/megatron.patch
docker/amd_patch/sglv0.5.10/megatron.patch
docker/build.py
docker/glm5/Dockerfile.dev-glm
docker/glm5/Dockerfile_glm5
docker/glm5/transformers.patch
docker/justfile
docker/patch/cu13/patch_fla_blackwell.py
docker/patch/latest/megatron.patch
docker/patch/latest/sglang.patch
docker/version.txt
docs/README.md
docs/_static/css/custom_log.css
docs/_static/css/readthedocs.css
docs/_static/image/blogs/release_v0.1.0/cuda_vmm.png
docs/_static/image/blogs/release_v0.1.0/overrall.png
docs/_static/image/logo.ico
docs/_static/image/logo.jpg
docs/_static/js/lang-toggle.js
docs/build.sh
docs/build_all.sh
docs/conf.py
docs/en/advanced/arch-support-beyond-megatron.md
docs/en/advanced/fault-tolerance.md
docs/en/advanced/miles-router.md
docs/en/advanced/miles_server_args.md
docs/en/advanced/p2p-weight-transfer.md
docs/en/advanced/pd-disaggregation.md
docs/en/advanced/speculative-decoding.md
docs/en/agentic/chat_template_verification.md
docs/en/developer_guide/debug.md
docs/en/developer_guide/migration.md
docs/en/examples/deepseek-r1.md
docs/en/examples/glm4-9B.md
docs/en/examples/glm4.5-355B-A32B.md
docs/en/examples/qwen3-30B-A3B.md
docs/en/examples/qwen3-4B.md
docs/en/examples/qwen3-4b-base-openhermes.md
docs/en/get_started/customization.md
docs/en/get_started/gen_endpoint.md
docs/en/get_started/oai_endpoint.md
docs/en/get_started/qa.md
docs/en/get_started/quick_start.md
docs/en/get_started/usage.md
docs/en/index.rst
docs/en/platform_support/amd_tutorial.md
docs/requirements.txt
docs/serve.sh
examples/DrGRPO/README.md
examples/DrGRPO/custom_reducer.py
examples/eval/__init__.py
examples/eval/eval_delegate.py
examples/eval/eval_delegate_rollout.py
examples/eval/nemo_skills/README.md
examples/eval/nemo_skills/__init__.py
examples/eval/nemo_skills/config/local_cluster.yaml
examples/eval/nemo_skills/skills_client.py
examples/eval/nemo_skills/skills_config.py
examples/eval/nemo_skills/skills_server.py
examples/eval/scripts/eval_tb_example.yaml
examples/eval/scripts/multi_tasks.yaml
examples/eval/scripts/run-eval-tb-qwen.sh
examples/eval/scripts/run-qwen3-32B.sh
examples/eval/scripts/run-qwen3-4B.sh
examples/eval/terminal_bench/README.md
examples/eval/terminal_bench/__init__.py
examples/eval/terminal_bench/requirements.txt
examples/eval/terminal_bench/tb_client.py
examples/eval/terminal_bench/tb_config.py
examples/eval/terminal_bench/tb_server.py
examples/eval_multi_task/README.md
examples/eval_multi_task/multi_task.sh
examples/eval_multi_task/multi_task.yaml
examples/eval_multi_task/requirements_ifbench.txt
examples/experimental/README.md
examples/experimental/swe-agent-v2/README.md
examples/experimental/swe-agent-v2/download_and_process_data.py
examples/experimental/swe-agent-v2/generate.py
examples/experimental/swe-agent-v2/prepare_harbor_tasks.py
examples/experimental/swe-agent-v2/run.py
examples/experimental/swe-agent-v2/swe_agent_function.py
examples/experimental/swe-agent/README.md
examples/experimental/swe-agent/download_and_process_data.py
examples/experimental/swe-agent/generate_with_swe_agent.py
examples/experimental/swe-agent/run-qwen3-4b-instruct.sh
examples/formal_math/single_round/README.md
examples/formal_math/single_round/kimina_wrapper.py
examples/formal_math/single_round/prepare_data.py
examples/formal_math/single_round/reward_fn.py
examples/formal_math/single_round/run.py
examples/formal_math/single_round/run_minimal.py
examples/formal_math/single_round/run_sft.py
examples/fully_async/README.md
examples/fully_async/fully_async_rollout.py
examples/fully_async/run-qwen3-4b-fully_async.sh
examples/geo3k_vlm/README.md
examples/geo3k_vlm/fsdp_vs_megatron.png
examples/geo3k_vlm/run_geo3k_vlm.sh
examples/geo3k_vlm/run_geo3k_vlm_sft.sh
examples/geo3k_vlm_multi_turn/README.md
examples/geo3k_vlm_multi_turn/__init__.py
examples/geo3k_vlm_multi_turn/base_env.py
examples/geo3k_vlm_multi_turn/env_geo3k.py
examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml
examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_reward.png
examples/geo3k_vlm_multi_turn/rollout.py
examples/geo3k_vlm_multi_turn/rollout_experiment_result_megatron.png
examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py
examples/lora/run-qwen2.5-0.5B-megatron-lora.sh
examples/lora/run-qwen3-4B-megatron-lora.sh
examples/lora/run-qwen3-4b-megatron-lora-result.sh
examples/low_precision/run-kimi-k2-Thinking-int4.sh
examples/low_precision/run-moonlight-16B-A3B-int4.sh
examples/low_precision/run-qwen3-235B-A22B-int4.sh
examples/low_precision/run-qwen3-30B-A3B-int4.sh
examples/low_precision/run-qwen3-30b-a3b-fp8-two-nodes.sh
examples/low_precision/run-qwen3-4b-fp8.sh
examples/multi_agent/README.md
examples/multi_agent/__init__.py
examples/multi_agent/agent_system.py
examples/multi_agent/prompts.py
examples/multi_agent/rollout_with_multi_agents.py
examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh
examples/on_policy_distillation/on_policy_distillation.py
examples/on_policy_distillation/run-qwen3-8B-opd.sh
examples/openai_format/__init__.py
examples/openai_format/dapo_math.py
examples/openai_format/run-qwen3-4B.sh
examples/reproducibility/README.md
examples/reproducibility/run-qwen2.5-0.5B-gsm8k.sh
examples/retool/README.md
examples/retool/generate_with_retool.py
examples/retool/requirements.txt
examples/retool/retool_qwen3_4b_rl.sh
examples/retool/retool_qwen3_4b_sft.sh
examples/retool/rl_data_preprocess.py
examples/retool/sft_data_processing.py
examples/retool/tool_sandbox.py
examples/retool_v2/README.md
examples/retool_v2/run_retool_multi_turn.py
examples/retool_v2/tool_sandbox.py
examples/search-r1/README.md
examples/search-r1/generate_with_search.py
examples/search-r1/google_search_server.py
examples/search-r1/local_dense_retriever/download.py
examples/search-r1/local_dense_retriever/retrieval_server.py
examples/search-r1/local_search_server.py
examples/search-r1/qa_em_format.py
examples/search-r1/run_qwen2.5_3B.sh
examples/strands_sglang/README.md
examples/strands_sglang/generate_with_strands.py
examples/strands_sglang/requirements.txt
examples/strands_sglang/strands_qwen3_8b.sh
examples/tau-bench/README.md
examples/tau-bench/generate_with_tau.py
examples/tau-bench/openai_tool_adapter.py
examples/tau-bench/run_qwen3_4B.sh
examples/tau-bench/sglang_tool_parser.py
examples/tau-bench/tau1_mock.py
examples/tau-bench/trainable_agents.py
examples/train_infer_mismatch_helper/README.md
examples/train_infer_mismatch_helper/mis.py
examples/train_infer_mismatch_helper/mis.yaml
examples/train_infer_mismatch_helper/run-qwen3-4b-fsdp-mis.sh
examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh
examples/true_on_policy/README.md
examples/true_on_policy/run_simple.py
examples/true_on_policy/src/aime.png
examples/true_on_policy/src/raw_reward.png
examples/true_on_policy/src/rollout_time.png
examples/true_on_policy/src/step_time.png
examples/true_on_policy/src/train_rollout_abs_diff.png
examples/true_on_policy_vlm/README.md
examples/true_on_policy_vlm/diff.png
examples/true_on_policy_vlm/run_simple.py
imgs/arch.png
imgs/miles_logo.png
imgs/miles_square.png
orbit/backends/experimental/__init__.py
orbit/backends/experimental/fsdp_utils/__init__.py
orbit/backends/experimental/fsdp_utils/actor.py
orbit/backends/experimental/fsdp_utils/arguments.py
orbit/backends/experimental/fsdp_utils/checkpoint.py
orbit/backends/experimental/fsdp_utils/kernels/__init__.py
orbit/backends/experimental/fsdp_utils/kernels/fused_experts.py
orbit/backends/experimental/fsdp_utils/kernels/fused_moe_triton_backward_kernels.py
orbit/backends/experimental/fsdp_utils/lr_scheduler.py
orbit/backends/experimental/fsdp_utils/models/__init__.py
orbit/backends/experimental/fsdp_utils/models/qwen3_moe.py
orbit/backends/experimental/fsdp_utils/models/qwen3_moe_hf.py
orbit/backends/experimental/fsdp_utils/parallel.py
orbit/backends/experimental/fsdp_utils/update_weight_utils.py
orbit/utils/debug_utils/__init__.py
orbit/utils/debug_utils/display_debug_rollout_data.py
orbit/utils/debug_utils/replay_reward_fn.py
orbit/utils/debug_utils/send_to_sglang.py
scripts/amd/run-qwen3-4B-amd.sh
scripts/models/deepseek-v3-20layer.sh
scripts/models/deepseek-v3-5layer.sh
scripts/models/deepseek-v3.sh
scripts/models/glm4-32B.sh
scripts/models/glm4-9B.sh
scripts/models/glm4.5-106B-A12B.sh
scripts/models/glm4.5-355B-A32B.sh
scripts/models/glm4.7-flash.sh
scripts/models/glm5-744B-A40B.sh
scripts/models/glm5-744B-A40B_20layer.sh
scripts/models/glm5-744B-A40B_4layer.sh
scripts/models/gpt-oss-20b.sh
scripts/models/kimi-k2-thinking.sh
scripts/models/kimi-k2.sh
scripts/models/llama3.1-8B-Instruct.sh
scripts/models/llama3.2-3B-Instruct-amd.sh
scripts/models/llama3.2-3B-Instruct.sh
scripts/models/mimo-7B-rl.sh
scripts/models/moonlight.sh
scripts/models/qwen2.5-0.5B.sh
scripts/models/qwen2.5-1.5B.sh
scripts/models/qwen2.5-32B.sh
scripts/models/qwen2.5-3B.sh
scripts/models/qwen2.5-7B.sh
scripts/models/qwen3-0.6B.sh
scripts/models/qwen3-1.7B.sh
scripts/models/qwen3-14B.sh
scripts/models/qwen3-235B-A22B.sh
scripts/models/qwen3-30B-A3B-5layer.sh
scripts/models/qwen3-30B-A3B.sh
scripts/models/qwen3-32B.sh
scripts/models/qwen3-4B-Instruct-2507.sh
scripts/models/qwen3-4B.sh
scripts/models/qwen3-8B.sh
scripts/models/qwen3-next-80B-A3B.sh
scripts/models/qwen3.5-27B.sh
scripts/models/qwen3.5-35B-A3B.sh
scripts/models/qwen3.5-4B.sh
scripts/models/qwen3.5-9B.sh
scripts/run-deepseek-r1.sh
scripts/run-glm4-9B-4xgpu-radixtree.sh
scripts/run-glm4-9B.sh
scripts/run-glm4.5-355B-A32B.sh
scripts/run-glm4.7-flash.sh
scripts/run-gpt-oss-20b-bf16.sh
scripts/run-gptoss-20b-fsdp.sh
scripts/run-kimi-k2-Instruct.sh
scripts/run-kimi-k2-Thinking.sh
scripts/run-mimo-7B-rl-eagle.sh
scripts/run-moonlight-16B-A3B.sh
scripts/run-qwen3-235B-A22B-sft.sh
scripts/run-qwen3-235B-A22B.sh
scripts/run-qwen3-32B.sh
scripts/run-qwen3-4B-base-sft.sh
scripts/run-qwen3-4B-fsdp.sh
scripts/run-qwen3-4B.sh
scripts/run-qwen3-4B_4xgpu-radixtree.sh
scripts/run-qwen3-4B_4xgpu.sh
scripts/run-qwen3-next-80B-A3B-8gpus.sh
scripts/run-qwen3-next-80B-A3B-fsdp.sh
scripts/run-qwen3-next-80B-A3B.sh
scripts/run-qwen3.5-27B.sh
scripts/run-qwen3.5-35B-A3B-mtp.sh
scripts/run-qwen3.5-4B.sh
scripts/run-qwen3.5-9B.sh
scripts/run_deepseek.py
scripts/run_glm45_355b_a32b.py
scripts/run_glm47_flash.py
scripts/run_glm5_744b_a40b.py
scripts/run_mcore_fsdp.py
scripts/run_qwen3_30b_a3b.py
scripts/run_qwen3_4b.py
scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py
scripts/tools/verify_chat_template.py
tests/ci/README.md
tests/ci/github_runner/.env.example
tests/ci/github_runner/.gitignore
tests/ci/github_runner/docker-compose.yml
tests/ci/gpu_lock_exec.py
tests/e2e/.gitkeep
tests/e2e/__init__.py
tests/e2e/ckpt/test_glm47_flash_ckpt.py
tests/e2e/ckpt/test_qwen3_4B_ckpt.py
tests/e2e/conftest_dumper.py
tests/e2e/fsdp/test_qwen3_0.6B_fsdp_distributed.py
tests/e2e/fsdp/test_qwen3_0.6B_megatron_fsdp_align.py
tests/e2e/fsdp/test_qwen3_4B_fsdp_true_on_policy.py
tests/e2e/fsdp/test_qwen3_vl_4B_fsdp.py
tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py
tests/e2e/long/test_qwen2.5_0.5B_gsm8k_async.py
tests/e2e/lora/test_lora_qwen2.5_0.5B.py
tests/e2e/megatron/test_glm47_flash_r3_mtp.py
tests/e2e/megatron/test_glm5_744b_a40b_4layer.py
tests/e2e/megatron/test_mimo_7B_mtp_only_grad.py
tests/e2e/megatron/test_moonlight_16B_A3B.py
tests/e2e/megatron/test_moonlight_16B_A3B_r3.py
tests/e2e/megatron/test_quick_start_glm4_9B.py
tests/e2e/megatron/test_qwen3_30B_A3B.py
tests/e2e/megatron/test_qwen3_30B_A3B_r3.py
tests/e2e/megatron/test_qwen3_4B_p2p.py
tests/e2e/megatron/test_qwen3_4B_ppo.py
tests/e2e/megatron/test_qwen3_5_35B_A3B_cp.py
tests/e2e/megatron/test_qwen3_5_mtp_bridge_mapping.py
tests/e2e/precision/test_hf_attention_cp_relayout.py
tests/e2e/precision/test_qwen3_0.6B_parallel_check.py
tests/e2e/precision/test_qwen3_5_cp_correctness.py
tests/e2e/sglang/__init__.py
tests/e2e/sglang/test_chat_input_ids_equivalence.py
tests/e2e/sglang/test_session_server_tool_call.py
tests/e2e/sglang/test_tito_logprob_equivalence.py
tests/e2e/sglang/utils/__init__.py
tests/e2e/sglang/utils/logprob_verify_generate.py
tests/e2e/sglang/utils/session_tool_agent.py
tests/e2e/sglang/utils/sglang_server.py
tests/e2e/sglang_config/test_sglang_config.py
tests/e2e/sglang_config/test_sglang_config_mixed_offload.py
tests/e2e/sglang_config/test_sglang_config_mixed_offload_ft.py
tests/e2e/short/__init__.py
tests/e2e/short/test_dumper.py
tests/e2e/short/test_qwen2.5_0.5B_gsm8k_async_short.py
tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py
tests/e2e/short/test_qwen3_0.6B_fsdp_colocated_2xGPU.py
tests/fast/backends/__init__.py
tests/fast/backends/megatron_utils/__init__.py
tests/fast/backends/megatron_utils/test_lora_checkpoint_helpers.py
tests/fast/backends/megatron_utils/test_lora_hf_weight_iterator.py
tests/fast/backends/megatron_utils/test_lora_model_branches.py
tests/fast/backends/megatron_utils/test_lora_update_weight.py
tests/fast/backends/megatron_utils/test_lora_utils.py
tests/fast/backends/megatron_utils/test_lora_weight_sync_validation.py
tests/fast/backends/training_utils/__init__.py
tests/fast/conftest.py
tests/fast/fixtures/__init__.py
tests/fast/fixtures/generation_fixtures.py
tests/fast/fixtures/rollout_fixtures.py
tests/fast/rollout/generate_hub/__init__.py
tests/fast/rollout/generate_hub/test_multi_turn.py
tests/fast/rollout/generate_hub/test_single_turn.py
tests/fast/rollout/generate_hub/test_tool_call_utils.py
tests/fast/rollout/generate_utils/__init__.py
tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py
tests/fast/rollout/generate_utils/test_sample_utils.py
tests/fast/rollout/inference_rollout/__init__.py
tests/fast/rollout/inference_rollout/conftest.py
tests/fast/rollout/inference_rollout/integration/__init__.py
tests/fast/rollout/inference_rollout/integration/test_agent_metadata.py
tests/fast/rollout/inference_rollout/integration/test_basic.py
tests/fast/rollout/inference_rollout/integration/test_deterministic.py
tests/fast/rollout/inference_rollout/integration/test_dynamic_filter.py
tests/fast/rollout/inference_rollout/integration/test_group_rm.py
tests/fast/rollout/inference_rollout/integration/test_multi_sample.py
tests/fast/rollout/inference_rollout/integration/test_multi_turn.py
tests/fast/rollout/inference_rollout/integration/test_over_sampling.py
tests/fast/rollout/inference_rollout/integration/test_sample_filter.py
tests/fast/rollout/inference_rollout/integration/test_semaphore.py
tests/fast/rollout/inference_rollout/integration/utils.py
tests/fast/rollout/inference_rollout/test_compatibility.py
tests/fast/rollout/rm_hub/__init__.py
tests/fast/rollout/rm_hub/test_deepscaler.py
tests/fast/rollout/rm_hub/test_f1.py
tests/fast/rollout/rm_hub/test_gpqa.py
tests/fast/rollout/rm_hub/test_math_dapo_utils.py
tests/fast/rollout/rm_hub/test_math_utils.py
tests/fast/rollout/rm_hub/test_rm_hub.py
tests/fast/router/__init__.py
tests/fast/router/test_linear_trajectory.py
tests/fast/router/test_router.py
tests/fast/router/test_session_pretokenized_e2e.py
tests/fast/router/test_session_race_conditions.py
tests/fast/router/test_sessions.py
tests/fast/test_megatron_cli_flags.py
tests/fast/utils/__init__.py
tests/fast/utils/chat_template_utils/__init__.py
tests/fast/utils/chat_template_utils/test_autofix.py
tests/fast/utils/chat_template_utils/test_pretokenized_chat.py
tests/fast/utils/chat_template_utils/test_template.py
tests/fast/utils/chat_template_utils/test_tito_tokenizer.py
tests/fast/utils/chat_template_utils/test_token_seq_comparator.py
tests/fast/utils/test_arguments.py
tests/fast/utils/test_async_utils.py
tests/fast/utils/test_dumper_utils.py
tests/fast/utils/test_env_report.py
tests/fast/utils/test_http_utils.py
tests/fast/utils/test_logging_utils.py
tests/fast/utils/test_mask_utils.py
tests/fast/utils/test_misc.py
tests/fast/utils/test_quantizer_ci.py
tests/fast/utils/test_types.py
tests/fast/utils/test_utils/__init__.py
tests/fast/utils/test_utils/test_mock_sglang_server.py
tests/fast/utils/test_utils/test_mock_tools.py
tests/test_chunked_gae.py
tests/test_external_rollout.py
tests/test_fsdp_import.py
tests/test_fused_experts_backward.py
tests/test_gspo.sh
tests/utils/test_sglang_config.py
tools/convert_hf_to_int4.py
tools/convert_hf_to_int4_direct.py
tools/convert_hf_to_nvfp4.py
tools/convert_to_hf.py
```

</details>

## Transfer dry-run: orbit's layer onto miles@2026-08-27

A real merge of `refs/miles/upstream` (dbbab156) into `miles-graft` was executed in a
throwaway worktree and aborted after the census:

| Outcome | Count | Meaning |
|:--|--:|:--|
| Auto-added | 1,094 | New upstream files, taken cleanly |
| Auto-merged | 59 | Upstream changes merged into lightly-touched files |
| **Both-modified conflicts** | **56 (45 code, 11 docs/CI)** | The real merge decisions |
| Delete/modify | 216 | Orbit dropped, miles touched — resolve: keep deleted |
| Add/delete adjudications | ~326 | Mostly mechanical (renamed-dir placement, upstream adds) |

The 45 code conflicts concentrate exactly where orbit rewrote the core: `initialize.py`,
`model.py`, `model_provider.py`, the `update_weight/` chain, rollout entry points. Everything
else the merge machinery handles because the graft gives git true ancestry.

**Recipe** (bounded at ~45 real decisions):

```
git checkout -b transfer-to-latest miles-graft
git -c merge.renameLimit=20000 merge refs/miles/upstream
# resolve the 45 code conflicts; keep-deleted for the 216 delete/modify
```

## Recommendation

Layer 1 needs no disentangling — it is already a clean overlay. The transfer cost is the 54
bolded files in Layer 2, and it is a bounded, git-assisted merge, not a rewrite. Upstream
uptake in the other direction stays selective: pick surgical fixes touching the 87 identical
+ 72 lightly-modified shared files, skip anything in the rewritten core (validated on
`e4152f61`, adopted; `cd464a1c4`, mostly duplicate of our own v0.5.18 work).

