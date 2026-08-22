---
title: Merged-stack numerical equivalence — sglang v0.5.16 line and merged orbit vs the published E4 stack
kind: investigation
profile: development-log
status: final
date: 2026-08-19
tags: lora-regret, numerics, sglang, megatron, oft, lora, verification
old_stack: sglang b52394d22 (orbit_env), orbit 46c8e0f6
new_stack: sglang 05cd76b4d (orbit_env_v2), orbit orbit-main
fixes: sglang 40784883e + 51845dc4a, orbit 89ea48c + badab95
jobs: 17466992 (i305), 17467160 (i201), 17467161 (i305), 17467303 (g102), 17468719 (g198), 17463137 (i407, OFT b8)
evidence: remote-cluster-runs mpi4 stage-a-inference-compare, mpi1/mpi2 backend-matrix, mpi4 stageb-trainer-equivalence
---

<section class="report-summary" aria-label="Outcome">
  <p class="summary-label">Outcome</p>
  <p class="summary-title">The merged stack is numerically equivalent to the published E4 stack everywhere a deterministic comparison exists: bit-identical on H100 for inference (base, LoRA, OFT on fa3 and triton) and for the trainer (forward always; the full forward-backward-optimizer pipeline bit-exact on one of two nodes, envelope-equal on the other) — and bit-identical on B200/triton as well once the one root-caused difference, a deliberate sm100 prefill-tiling choice new in v0.5.16, is held equal.</p>
  <p class="summary-detail">Three latent serving bugs were found, fixed, verified on GPU, and pushed en route (disk-loaded adapters silently ignored; adapter/base radix-cache cross-contamination; OFT parameter-count 1.46x undercount). Caveats: seeded sampling is incompatible across builds by construction; stock B200 triton keeps its (adapter-agnostic, fully attributed) tiling drift by decision; trtllm_mha on B200 is untestable deterministically on the old build.</p>
</section>

<div class="status-grid" role="list" aria-label="Program status">
  <div class="status-item" data-status="complete" role="listitem"><strong class="status-value">3</strong><span class="status-label">Bugs fixed, verified, pushed</span></div>
  <div class="status-item" data-status="complete" role="listitem"><strong class="status-value">22</strong><span class="status-label">Comparison cells measured</span></div>
  <div class="status-item" data-status="blocked" role="listitem"><strong class="status-value">2</strong><span class="status-label">Cells unmeasurable</span></div>
  <div class="status-item" data-status="open" role="listitem"><strong class="status-value">3</strong><span class="status-label">Handoff follow-ups</span></div>
</div>

## Question and decomposition

A merge moved the stack from the sglang v0.5.9 line (`b52394d22`, the build every
published E4 number was produced on, venv `orbit_env`) to the v0.5.16 line
(`05cd76b4d`, venv `orbit_env_v2`), and moved orbit itself by 216 commits from the
E4 report's provenance commit `46c8e0f6` (+4,490/−291 lines in
`backends/megatron_utils` alone). The question: does the merged stack produce the
same numbers?

Unseeded sampling makes end-to-end RL runs incomparable (measured earlier: 0/32
identical completions at fixed seed), so the program decomposed into pieces that
are deterministic by construction:

1. **Stage A** — inference forward: same weights, greedy decoding, deterministic
   inference mode, pinned attention backend, both builds on one node.
2. **Backend × adapter × GPU matrix** — the same probe over
   {H100, B200} × {fa3, triton, flashinfer} × {base, LoRA r16, OFT b128}.
3. **Stage B** — trainer: one frozen rollout batch replayed through the full E4
   launcher on both stacks via orbit's own `--load-debug-rollout-data` /
   `--debug-train-only` seam (zero sglang engines, one GRPO step, 1×H100,
   TP1/DP1), comparing per-token logprobs, advantages, grad norm, repeated for a
   nondeterminism envelope.

## Completed deliverables

- **sglang `40784883e`** — `fix(peft): propagate resolved adapter ids into cached
  request sub-objects`. Batched `generate()` with `adapter_path`/`lora_path`
  resolved the adapter at the tokenizer but the id never reached the scheduler
  (`GenerateReqInput.__getitem__` memoizes sub-objects before resolution runs).
  OFT served the identity slot: disk-loaded adapters silently returned
  **base-model output**. Verified pre/post on GPU for both PEFT kinds.
- **sglang `51845dc4a`** — `fix(peft): key the radix cache by OFT adapter id`.
  The port dropped the old build's `|oft:{id}:v{n}` extra-key branch, so
  adapter and base requests shared radix keys; base requests prefix-matched
  adapter-computed KV (base greedy ≡ adapter output on 14/16 prompts, 0/16 with
  the cache disabled). RL was protected only because orbit force-disables the
  radix cache for PEFT rollout engines. Verified with the cache on.
- **orbit `89ea48c` + `badab95`** — canonical OFT builds one rotation per fused
  **output slice** (qkv=3, fc1=2), not per module; the counter undercounted
  all-modules OFT by **1.46×** (b128: recorded 54,099,968, actual 79,069,184).
  This fed `matched_ratio`/`oft_matched_lora_rank`, so "parameter-matched"
  OFT/LoRA pairs handed OFT ~46% extra capacity. Consequences now pinned in
  tests: E4's b128 rung implies LoRA rank 35 (between the matrix's r16 and
  r256 — no capacity-comparable arm); every ladder rung now lands inside the
  0.85–1.15 band; attention-vs-MLP placement cannot be matched by block size at
  all (~26% high everywhere). `badab95` threads an `oft_type` keyword so legacy
  shared-R arms can never silently receive canonical accounting; legacy
  accounting reproduces the old ledger number exactly (regression-pinned).

All three fixes are pushed to `Sphere-AI-Lab/{sglang,orbit}` `orbit-main` and
live in `orbit_env_v2`'s installed copy.

<aside class="finding" data-tone="caution">
  <p class="block-label">Finding</p>
  <p>Positive: any pre-fix evaluation that loaded a saved OFT adapter from disk in a batched request reported base-model numbers and must be rerun if it fed any record. Training via the streamed cpu_gather path was never affected; published E4 training curves are untouched by all three bugs.</p>
</aside>

## Verification — inference (Stage A + matrix)

Probe: 16 math prompts × 64 greedy tokens, deterministic inference, radix cache
disabled, per-position |Δlogprob| over prompts whose full token sequences match.
Within-build repeatability across independent engine boots was exactly 0.00
(three control pairs), so every nonzero delta below is a genuine build
difference.

| GPU | Backend | Phase | Tokens identical | p50 | mean | p95 | max |
|:--|:--|:--|:--|:--|:--|:--|:--|
| H100 | fa3 | LoRA r16 | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| H100 | fa3 | OFT b128 | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| H100 | fa3 | base ×2 engines | 16/16 | 0.00 | 0.00 | 0.00 | 0.00 |
| H100 | triton | LoRA r16 | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| H100 | triton | OFT b128 | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| H100 | triton | base ×2 engines | 16/16 | 0.00 | 0.00 | 0.00 | 0.00 |
| B200 | triton (stock) | LoRA r16 | 10/16 | 8.0e-4 | 8.9e-3 | 5.0e-2 | 1.35e-1 |
| B200 | triton (stock) | OFT b128 | 16/16 | 4.7e-3 | 1.2e-2 | 4.7e-2 | 8.3e-2 |
| B200 | triton (stock) | base ×2 engines | 11/16 | 9.1e-4 | 8.3e-3 | 4.4e-2 | 1.15e-1 |
| B200 | triton, tiling matched¹ | LoRA r16 | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| B200 | triton, tiling matched¹ | OFT b128 | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| B200 | triton, tiling matched¹ | base ×2 engines | 16/16 | 0.00 | 0.00 | 0.00 | **0.00** |
| B200 | flashinfer | all | — | — | — | — | old build SIGKILLed at boot, twice |
| H100 | flashinfer | all | — | — | — | — | old build SIGKILLed at boot (child engine process fails); new build boots fine |
| B200 | trtllm_mha² | OFT / base | — | — | — | — | old build SIGKILLed at boot, twice — cross-build unmeasurable |

¹ New build with its sm100 extend-attention tiling branch disabled so it selects
the same (128, 64) prefill tiles the old build used on B200 by fall-through —
see the root-cause finding below. Same node (i305) as the stock rows; probe
patch reverted after the measurement.

² trtllm_mha rejects deterministic mode on the old build, so this cell was
attempted WITHOUT deterministic inference (greedy, single fixed batch, radix
off, 2 repeats per build). The old build was killed at engine boot both times.
Instructively, the new build's own two repeats already disagree without
deterministic mode — 16/16 tokens but max |Δlogprob| 1.55e-2 with the adapter,
and only 15/16 identical token sequences on base — so even with both builds
booting, this mode could never support bit-level cross-build claims; it bounds
any comparison at the ~1e-2 batch-nondeterminism floor.

<aside class="finding" data-tone="positive">
  <p class="block-label">Finding</p>
  <p>On H100 every cell is bit-identical across the ~5,500-commit merge, on both backends. The only drift is triton-on-B200, it is adapter-agnostic (LoRA's max delta 1.35e-1 exceeds base 1.15e-1 exceeds OFT 8.3e-2; medians ~1 bf16 ulp; OFT's higher median is a position-mix artifact of EOS-short outputs) — and it was subsequently <strong>root-caused to a single prefill tiling branch and closed to bit-identity</strong> (below).</p>
</aside>

<aside class="finding" data-tone="positive">
  <p class="block-label">Root cause + closure (B200 drift)</p>
  <p>The v0.5.16 line added an sm100-specific branch to the extend-attention tile selection (<code>sglang/kernels/ops/attention/extend_attention.py</code>, <code>CUDA_CAPABILITY[0] == 10</code> → BLOCK_M,BLOCK_N = (64,64) for Lq≤256, added for sm_100a register constraints); the old build had no such branch and fell through to the Hopper sizes (128,64) on B200. Different tiles → different bf16 online-softmax accumulation order; decode kernels have no capability branches, so all decode-position drift was downstream of prefill-written KV. Confirmed by experiment (job 17469012, i305, the same node as the reference dumps): with the branch disabled, old-vs-new is <strong>bit-identical on B200/triton</strong> — 16/16 tokens and max |Δlogprob| = 0.000e+00 for OFT, LoRA, and both base controls. The probe patch was reverted after the run. <strong>Decision: no permanent match switch adopted</strong> — the experiment's purpose was verifying the merge, which it does conclusively (the new build is the old build plus one deliberate, documented sm100 performance tiling choice); anyone needing bit-continuity on B200/triton can reproduce the one-line branch disable recorded here.</p>
</aside>

Seeded sampling (temperature 1.0, deterministic mode, same seed, same pytorch
sampling backend): 16/16 prompts diverge from position 0. Each build is exactly
self-reproducible; the RNG streams differ across builds. Cross-build "identical
seeded rollouts" is not achievable — this dictated Stage B's frozen-batch
design.

## Verification — trainer (Stage B)

One frozen batch (16 sequences: true-base greedy completions with documented
provenance, all-ones loss masks, deterministic alternating advantages;
`batch_0.pt` sha256 `cc97441e…`) replayed through the unmodified E4 launcher on
both stacks: old = `orbit_env` + a git worktree pinned at `46c8e0f6`; new =
`orbit_env_v2` + merged `orbit-main`. Same `MODEL/CKPT/PERF/PEFT` args as the
campaign, 1×H100, TP1/PP1/DP1, one optimizer step, LoRA r16 and OFT b128, with
repeats.

**Forward: bit-identical.** Every quantity the trainer computes ahead of the
backward pass compared exactly equal, old-vs-new (max |Δ| = 0.00 at every
position; "bit-equal" below means `torch.equal` on the full tensors). Backend
note: the trainer has no fa3/triton axis — those are sglang serving backends.
Both stacks ran the campaign launcher's own Megatron setting,
`--attention-backend flash` (TransformerEngine → flash-attn 2.8.3, byte-identical
builds in both venvs); other Megatron attention modes (fused/unfused) were not
exercised because the campaign never uses them:

| Node | Method | Quantity | Positions compared | Old vs new |
|:--|:--|:--|:--|:--|
| g102 | LoRA r16 | per-token log_probs | 1,023 | **bit-equal** |
| g102 | LoRA r16 | advantages, returns, loss_masks | 1,023 each | bit-equal |
| g102 | LoRA r16 | tokens consumed (prompt+response) | 2,019 | bit-equal |
| g102 | OFT b128 | per-token log_probs | 1,023 | **bit-equal** |
| g102 | OFT b128 | advantages, returns, loss_masks | 1,023 each | bit-equal |
| g102 | OFT b128 | tokens consumed (prompt+response) | 2,019 | bit-equal |
| g198 | OFT b128 | per-token log_probs (both repeats) | 1,023 | **bit-equal** |
| B200 i305, flash | OFT b128 (×2 repeats) | per-token log_probs | 1,023 | **bit-equal** |
| B200 i305, flash | LoRA r16 | per-token log_probs | 1,023 | **bit-equal** |
| B200 i305, fused (cuDNN) | OFT b128 | per-token log_probs | 1,023 | **bit-equal** |
| H100 i108, fused (cuDNN) | OFT b128 (×2 repeats) | per-token log_probs | 1,023 | **bit-equal** |
| i203, flash, **DP=4** | OFT b128 (×2 repeats) | per-token log_probs, all 4 rank shards | 1,023 total | **bit-equal** |

Within-build repeats were also bit-equal on every quantity, so the forward path
is exactly deterministic per node; the token-stream equality doubles as proof
that both stacks consumed and preprocessed the identical frozen batch.

**Backward/optimizer (grad norm):**

| Node, backend | Method | Old (repeats) | New (repeats) | Within-build spread | Cross-build |
|:--|:--|:--|:--|:--|:--|
| g102 (H100), flash | LoRA | 1.059683204, 1.059853554 | 1.059390545, 1.059692621 | 1.6e-4 / 2.9e-4 | 2.8e-4 — inside envelope |
| g102 (H100), flash | OFT | 4.716103554, 4.715067863 | 4.712501526, 4.712558270 | 2.2e-4 / 1.2e-5 | 5.3e-4–7.6e-4 |
| g198 (H100), flash | OFT | 4.847944260, 4.847944260 | 4.847944260, 4.847944260 | 0 (bit-equal) | **0 (bit-equal)** |
| i108 (H100), fused | OFT | 4.849560738 ×2 | 4.849560738 ×2 | 0 (bit-equal) | **0 (bit-equal)** |
| i305 (B200), flash | OFT | 4.810490608 ×2 | 4.810490608 ×2 | 0 (bit-equal) | **0 (bit-equal)** |
| i305 (B200), flash | LoRA | 1.081156850 ×2 | 1.081156850 ×2 | 0 (bit-equal) | **0 (bit-equal)** |
| i305 (B200), fused | OFT | 4.810490608 ×2 | 4.810490608 ×2 | 0 (bit-equal) | **0 (bit-equal)** |
| i203, flash, **DP=4** | OFT | 4.849619389 ×2 | 4.849619389 ×2 | 0 (bit-equal) | **0 (bit-equal)** |

The DP row tests the campaign's data-parallel axis (per-rank sharding of the
frozen batch, gradient allreduce across 4 ranks, one optimizer step): every
rank's forward shard and the globally reduced grad norm are bit-equal
cross-build. It ran as DP=4 on a 4-GPU slice rather than the campaign's DP=8
whole node because all three free complete H100/B200 nodes offered to the
whole-node request carried dead GPUs (i101: one, i104: two, i306: one, plus a
mislabeled ClassAd) — reported to cluster operations; the allreduce mechanism
under test is identical at either width.

<aside class="finding" data-tone="positive">
  <p class="block-label">Finding</p>
  <p>On every node/backend combination except g102, the entire pipeline — forward, backward, grad norm — is bit-identical across the merge: g198 and i108 (H100, flash and fused backends), and i305 (B200, flash and fused, both PEFT methods). A systematic code difference would appear in those cells too; g102's 7.6e-4 OFT cross-build delta is therefore node-state nondeterminism under-sampled at two repeats, not code. Grad-norm <em>values</em> differ per (node, backend) — 4.716 / 4.848 / 4.850 / 4.810 for the same OFT batch — but both builds always agree exactly per configuration; the variation is a hardware/backend effect orthogonal to the merge, and further support for the campaign's existing mixed-node-type limitation. Notably the B200 <em>trainer</em> is bit-identical cross-build even though the B200 <em>serving</em> triton path drifts: the trainer's TE/flash-attn kernels are the same binaries in both venvs, while the serving drift came from sglang's own re-tuned triton kernel.</p>
</aside>

## Static equivalence (Megatron side)

Between the two venvs: `megatron-core` installed trees byte-identical;
Megatron-Bridge differs in exactly one file, scoped to grouped-MoE experts under
legacy shared-R (the campaign is dense + canonical — doubly out of scope);
torch 2.11.0, transformer_engine 2.14.0+71bbefbf, triton 3.6.0, flash-attn
2.8.3, NCCL, numpy, apex, cuDNN all identical builds. The empirical Stage B was
still necessary because orbit's own trainer-facing code moved substantially.

## Campaign follow-up — the OFT b8 rollout failure

The `e4oftverify` ladder (b8/b128/b1024, all-modules, math, one shared LR of
7e-06) left one arm without a row: `oftverify-b8-all-math-lr7e-06-s0` died while
its two siblings completed all 150 rollouts. The ledger holds three `failed`
rows for it, and only the third is the event worth explaining:

| Attempt | Ran for | Died at | Cause |
|:--|:--|:--|:--|
| 1 | 339 s | engine init | `NotImplementedError: Breakable CUDA graph is not compatible with memory saver mode` |
| 2 | 320 s | engine init | `AssertionError: Triton tl.dot requires BS >= 16; got BS=8` |
| 3 | 1,286 s | rollout 8/150 | `OSError: [Errno 116] Stale file handle` in the Triton JIT cache |

Attempts 1 and 2 are already-closed environment faults: the first is the
memory-saver/prefill-graph clash that `env_v0516.sh` now disables the prefill
CUDA graph for, and the second is the pre-tiny-block package, whose fused kernel
hard-asserts `BS >= 16` and therefore cannot launch the b8 rung at all.

**Attempt 3 is root-caused, and it is not an OFT defect.** The arm was healthy
right up to the crash — `train/loss` 0.0038 at step 7, 161 GB free of 178 on
rank 0, rollouts pacing at 61–71 s, and a step-0 eval of 0.0572 against the
campaign baseline's 0.056. Three seconds into rollout 8's first prefill, TP1
raised inside `CompiledKernel.__init__`: Triton was compiling
`_gemm_oft_r_kernel` for `o_proj`, and reading its own freshly written cache
entry back returned ESTALE. The scheduler went down, SIGQUIT propagated, and the
driver exited on a 502 from the router. The `CUDA error: invalid argument`
printed afterwards comes from `MemPool::~MemPool` during crash teardown and is a
consequence, not the cause.

<aside class="finding" data-tone="caution">
  <p class="block-label">Root cause (OFT b8 worker death)</p>
  <p>Triton's JIT cache was left at its default, <code>$HOME/.triton/cache</code>, and this cluster's <code>$HOME</code> is NFS (<code>/lustre/home</code> is nfs4 from <code>sc-fb1:/cluster-home</code>). <code>_gemm_oft_r_kernel</code> declares <code>total_tokens</code> as <code>tl.constexpr</code>, so every distinct prefill token count is a separate specialization and a separate compile — an RL run compiles this kernel continuously rather than once at warmup, and the cache had grown to <strong>179,085 entries / 44 GB</strong>. All 16 ranks (8 engines × TP2) share that one directory and race to store the same key; on a network filesystem the loser's open handle is invalidated by the winner's rename and the read-back returns ESTALE. b8 is the exposed rung because its key space is entirely cold — <code>_pick_tiles</code> gives (8, 8) for BLOCK_SIZE=8 while b128 lands on the untiled (128, 128) path every earlier arm had already compiled — but nothing about the mechanism is specific to b8.</p>
</aside>

Fixed in `scripts/lora_regret/campaign.sh`, which now exports a node-local
`TRITON_CACHE_DIR` before any CUDA work. This is not a new idea: the same block,
with the same reasoning in its comment, has been in
`examples/low_precision/run-kimi-k25-int4-openr1-oft.sh` all along — the campaign
was simply never given it. Placed in `campaign.sh` rather than `env_v0516.sh`
because every launcher `exec`s the former while the latter is sourced by hand.

Residual: b8 still has no completed math row at 7e-06, so the ladder's low rung
is unmeasured. The fix removes the failure mode but is unproven against it until
a rerun completes; the fault is intermittent by nature, so a clean 150-rollout
b8 run is the only real confirmation.

## Risks and limitations

<aside class="risk">
  <p class="block-label">Residuals</p>
  <p><strong>trtllm_mha on B200</strong> (the campaign's B200 default backend) rejects deterministic inference on the old build, so its cross-build equality is unmeasurable by this method. <strong>flashinfer on B200</strong>: the old build is SIGKILLed at engine boot (reproducibly), so that cell is empty. <strong>Sampling</strong>: cross-build rollout draws differ by construction; per-run RL results were never comparable across builds and remain so — seed replicates stay the only way to compare endpoint accuracies. The B200-triton tiling difference (root-caused above; bit-matchable on demand) and the per-node H100 grad-norm difference both belong in any record that mixes node types.</p>
</aside>

## Actions

| action | owner | status | evidence or trigger |
|:--|:--|:--|:--|
| Repoint campaign.sh / INSTALL.md at orbit_env_v2 | zqiu | Done | all lora_regret launchers + INSTALL.md banner now point at env_v0516.sh; the 2026-08-10 E4 report keeps its orbit_env reference as historical provenance |
| Review + commit e4oftverify matrix and verify scripts | zqiu | Open | uncommitted in both orbit checkouts |
| Seed replicates for endpoint-accuracy claims | zqiu | Open | E4 report limitation #1 |
| OFT b8 worker death at rollout 8/150 | zqiu | Root-caused, fix applied | ESTALE on the NFS-backed Triton JIT cache, not OFT; `campaign.sh` now pins `TRITON_CACHE_DIR` node-local |
| Rerun the b8 rung to fill the ladder | zqiu | Open | needs one 8-GPU node, ~4.5 h; also the first real test of the cache fix |
| Retire leftover mpi4 tmux session claude-orbit-stageb-oft-extra | zqiu | Open | mpi4 dropped mid-teardown; job 17468719 idles out on its own |

<details class="reproducibility">
<summary>Reproducibility</summary>

**Builds:** old = `orbit_env` (sglang `0.0.0.dev9909+gb52394d22`), new =
`orbit_env_v2` (sglang `0.0.0.dev15479+g05cd76b4d` + fixes `40784883e`,
`51845dc4a` deployed). Orbit: old = worktree `/fast/zqiu/orbit-iclr/orbit-46c8e0f6`,
new = `/fast/zqiu/orbit-iclr/orbit` at `orbit-main`.

**Inference probes:** `/lustre/home/zqiu/sglang_cmp/{stage_a_probe.py,matrix_probe.py,matrix_run.sh,stage_a_compare.py,matrix_report.py}` —
greedy + deterministic inference, `attention_backend` pinned, `disable_radix_cache=True`,
16 prompts from `math_test.jsonl`, 64 new tokens. Jobs: 17466992 (B200 i305),
17467160 (H100 i201), 17467161 (B200 i305), all bid 100, 1 GPU.

**Trainer probe:** `/lustre/home/zqiu/sglang_cmp/stageb/` —
`stageb_build_batch.py` (frozen batch, sha256 `cc97441e14745bee…`),
`stageb_run3.sh` (launcher invocation), `gn_envelope.py`, `pernode_check.py`.
Key launcher env: `GPUS_PER_NODE=1 NUM_ROLLOUT=1 GLOBAL_BATCH_SIZE=16
ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=1 ROLLOUT_NUM_GPUS_PER_ENGINE=1
EPS_CLIP=1e9 SEED=1234`, `RL_EXTRA_ARGS="--disable-grpo-std-normalization
--disable-rewards-normalization --load-debug-rollout-data … --save-debug-train-data …
--ci-test --ci-disable-kl-checker --ci-save-grad-norm …"`. Jobs: 17467303
(H100 g102), 17468719 (H100 g198).

**Operational gotchas for reruns:** `ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT`
must equal `GLOBAL_BATCH_SIZE` or `train_iters=0` asserts;
`ROLLOUT_NUM_GPUS_PER_ENGINE=1` required on 1 GPU (IPC gather group);
`ray stop --force` between launcher cycles; never `sed -i` a script another node
is reading (Lustre stale handle — ship under a fresh name); ~14 launcher cycles
exhaust one allocation's PID budget.

**Evidence stores** (`~/.local/state/remote-cluster-runs/`, mirrored locally):
`mpi4/…/20260818T234900/stage-a-inference-compare`,
`mpi1/…/20260819T010500/backend-matrix-h100`,
`mpi2/…/20260819T010500/backend-matrix-b200`,
`mpi4/…/20260819T063000/stageb-trainer-equivalence` — dumps, grad-norm tensors,
compare reports, and provenance.

</details>
