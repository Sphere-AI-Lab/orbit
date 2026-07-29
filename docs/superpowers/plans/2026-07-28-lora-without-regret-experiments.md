# LoRA Without Regret — Experiment Plan

> **Anchor (corrected 2026-07-28):** the target is **the blog's own claims**
> (`https://thinkingmachines.ai/blog/lora/`), reproduced on **the blog's own setup**.
> michaelbzhu's GitHub repo is a third-party implementation kept only as a cross-check —
> its per-arm table (FullFT 1.8457, LoRA r256-all 1.8457, …) is **not** the reproduction
> target. The previous version of this file was built around that table; everything below
> replaces it.

> **For agentic workers:** this is a *campaign* plan, not a code plan. Most tasks are
> measurements whose acceptance criterion is a number. The prerequisites P0-P5 are real
> engineering and must be executed as code tasks under `superpowers:subagent-driven-development`.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure whether Orbit reproduces the five empirical claims of the LoRA-without-regret
post, using the models, datasets, and hyperparameter conventions the post itself used.

**Architecture:** Every arm is one Orbit launcher invocation with environment overrides,
driven by a sweep driver that appends one JSONL record per arm and skips completed arms.
Claims are read off measured argmins, learning-curve departure points, and orderings —
never off absolute loss values.

**Tech stack:** Orbit (Megatron backend) on Llama-3.1-8B base and Qwen3-30B-A3B-Base;
Tulu3 and OpenThoughts3 for SFT; MATH and GSM8K for RL.

Companion documents:
- Design and gate definitions: `docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md`
- Implementation plan (Tasks 1-14, Qwen3-era): `docs/superpowers/plans/2026-07-27-lora-without-regret-repro.md`
- Gate results so far: `docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md`

---

## The claims, and the experiment that decides each

| # | Claim (as the post states it) | Post's setup | Decided by |
|---|---|---|---|
| C1 | FullFT and high-rank LoRA share a learning curve (loss linear in log-steps); low-rank LoRA "falls off at a threshold of steps that correlates with rank" | Llama 3 / Qwen3, Tulu3 + OpenThoughts3, rank 1-512, single epoch | **E1** |
| C2 | Optimal LoRA LR is ~10x FullFT's (fitted multiplier 9.8); ~15x for short runs (~100 steps); optimal LR varies < 2x between rank 4 and rank 512 | 14 Llama/Qwen models, Tulu3 | **E1** (falls out of the same sweeps) |
| C3 | LoRA tolerates large batch sizes worse than FullFT — a persistent gap, caused by the product-of-matrices parametrization, **independent of rank** | 10,000-example OpenThoughts3 subset, batch 32 and larger | **E2** |
| C4 | Attention-only LoRA significantly underperforms MLP-only **at matched parameter count**, and adds nothing on top of MLP-only | Llama-3.1-8B (dense), Qwen3-30B-A3B-Base (MoE), ranks 128/256 | **E3** |
| C5 | LoRA matches FullFT under policy gradient **even at rank 1**, with a wider band of performant LRs | Llama-3.1-8B base, MATH + GSM8K, 32 samples/problem | **E4** |

Our own extension, not from the post: **C6 — matched-parameter OFT** behaves like LoRA on
C1/C2/C4. Kept as optional **E5**.

**What the anchor change buys us.** Every claim above is a ratio, an ordering, or a curve
shape — all internal comparisons among our own runs. The constant precision offset that
dominated the Qwen3-era gate work (Orbit-vs-HF 0.0032 nats, inside HF's own 0.0072-nat
bf16/fp32 spread) cancels in every one of them. Absolute-loss agreement with any third party
is no longer load-bearing, so gate G1 demotes from a blocker to an optional cross-check.

---

## Global Constraints

The post's hyperparameter conventions, quoted and then translated into Orbit knobs.

- **LoRA init:** "uniform distribution for A with scale 1/√d_in, zero initialization for B,
  the same LR for both, and α = 32."
  - Orbit setting: `LORA_A_INIT_METHOD=kaiming`, `LORA_ALPHA=32`, B pinned to `zero`.
  - These are **provably the same thing**, which is worth recording because it looks like a
    mismatch. PEFT's default is `kaiming_uniform_(a=√5)`, whose bound on a weight with fan-in
    `d_in` is `√(6 / ((1 + a²)·d_in))` = `√(6 / (6·d_in))` = `1/√d_in` — exactly the post's
    scale. Orbit's own default is `xavier`, which differs by ~2.4x in std and would shift
    every measured optimal LR, so this must be set explicitly on every arm.
- **Scaling:** α/r, so the effective scale falls as 1/r with fixed α=32. This is what makes
  the post's "learning curve is identical at the beginning of training regardless of rank"
  claim testable; do not switch to rank-stabilized scaling.
- **Schedule:** single epoch, constant LR, no warmup, no cooldown, `WEIGHT_DECAY=0.0`.
- **Precision:** bf16 throughout.
- **Metric:** training loss vs step for C1 (the post reads departure points off the training
  curve), and held-out token-weighted NLL for the LR argmins. Never `sample_mean` — the two
  differ by ~0.64 nats in our measurements.
- **Noise floor:** σ = 0.000992 nats, measured on Qwen3-4B/No Robots with three seeds
  (gate log). **It does not transfer to Llama-3.1-8B or to a different dataset** — E1-0
  re-measures it. Until then, treat any difference under ~0.002 as unresolved.
- **Seeds:** `SEED` is tied to `ROLLOUT_SEED` inside the repro launchers so a seed change
  varies data order as well as init. Sweep arms run at `SEED=0`.

---

## Prerequisites — none of E1-E4 is runnable today

Each of these is blocking, and each is a genuine gap rather than a formality.

- [ ] **P0: Multi-GPU. FullFT on 8B does not fit on one 80 GB card.**

Parameter arithmetic for Llama-3.1-8B (8.03B params) under Adam: bf16 weights 16.1 GB +
fp32 master 32.1 GB + Adam moments 64.2 GB + bf16 grads 16.1 GB ≈ **128 GB before
activations**. With Megatron's distributed optimizer sharding master+moments across DP ranks,
per-GPU cost is `32 GB + 96 GB/N`:

| GPUs | Per-GPU state | Verdict |
|---|---|---|
| 1 | 128 GB | impossible |
| 2 | 80 GB | over budget once activations land |
| 4 | 56 GB | workable |
| 8 | 44 GB | comfortable |

LoRA arms are unaffected — frozen 16 GB of base weights plus small adapters — and three
concurrent LoRA runs on one H100 is measured-safe (three Qwen3-4B runs finished in 26 min
wall clock). So the campaign tiers: **LoRA arms one GPU each, FullFT arms ≥4 GPUs.**
This box has exactly one H100, so FullFT needs an allocation that does not exist yet.

- [ ] **P1: Llama-3.1-8B *base* is not present.** `/lustre/fast/fast/zqiu/hf_models/` holds
  `Llama-3.1-8B-Instruct` and quantized variants only. The post uses the **base** model for
  both the layer study and RL. Download `meta-llama/Llama-3.1-8B` (gated — the HF license
  must be accepted on the account first) and convert to Megatron with
  `tools/convert_checkpoints.py`. Note `scripts/conversion/convert_checkpoint.sh` is unusable
  as shipped on this box: it hardcodes `CUDA_HOME=/is/software/nvidia/cuda-12.9` and calls
  bare `python`. The working recipe is recorded in the SDD ledger — interpreter
  `/lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python`, `CUDA_HOME=…/cuda-13.2`.
  `orbit_plugins/model_args/llama3.1-8B-Instruct.sh` already exists and the base model shares
  its architecture, so the model-args plugin should need only a checkpoint-path change —
  verify the config (rope_theta, vocab size, tied embeddings) rather than assume it.

- [x] **P2: There is no Llama-3 loss-mask generator, and it needs its own G3.**
  `--loss-mask-type` accepts exactly `{qwen, qwen3, distill_qwen}` (`orbit/utils/arguments.py`).
  Llama-3.1's chat template — `<|start_header_id|>`/`<|eot_id|>` — has no implementation.
  This must be written **and gated against HF token-for-token**, exactly as G3 was for Qwen3.
  That gate is not a formality: for Qwen3 it failed on first run and exposed a real bug
  (per-message `apply_chat_template` injecting 4 phantom `<think>` tokens before every
  non-final assistant turn, touching 13% of the held-out set). Assume the same class of bug
  exists for Llama until a parity test says otherwise.
  **Done:** `--loss-mask-type=llama3` now dispatches to `gen_multi_turn_loss_mask_llama3`,
  and gate G3-llama (`docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md`)
  is green on all 12 fixture rows against a char-offset HF oracle sharing no algorithm with
  the implementation. No bug was found this time (unlike Qwen3's G3), so the gate's value
  is prospective; `tools=`/`<|eom_id|>` and `step_loss_mask=0` remain untested by it — see
  the G3-llama entry for the full list of what it does and does not cover before this feeds E1.
  **Required at launch time:** the Llama-3.1-8B base checkpoint ships no `chat_template`, so
  any Llama-3 SFT run (this campaign's E1/E2/E3) must pass
  `--chat-template-path orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja` —
  without it, `load_tokenizer` raises `ValueError: Cannot use chat template functions because
  tokenizer.chat_template is not set` before training can start. That file is pinned
  byte-identical to the `LLAMA3_CHAT_TEMPLATE` constant this gate exercises (see
  `test_bundled_jinja_matches_the_pinned_python_constant` in
  `tests/fast/utils/test_llama3_chat_template.py`); do not hand-write a substitute .jinja, and
  do not pass `--chat-template-path` for the Instruct checkpoint, which already carries its own.

- [ ] **P3: Validate the held-out NLL reduction at DP > 1.** Orbit's eval returns
  `(sum_neg_logprob, n_tokens)` per actor and reduces over the DP group, because TP/PP
  replicas hold identical samples (double-counting) while DP shards hold different token
  counts (so a token-weighted mean is not the mean of per-rank means). That logic has only
  ever executed at `tp=pp=dp=1`. P0 forces DP>1 for every FullFT arm, so before trusting a
  single FullFT number: run one arm at DP=1 and the same arm at DP=4 and require **identical
  NLL to the printed precision and identical `tokens=`/`samples=` counts**.

- [ ] **P4: Dataset preparation.** `tools/lora_regret/prepare_data.py` handles No Robots and
  competition_math only. Needed: **Tulu3** (`allenai/tulu-3-sft-mixture`) for E1,
  **OpenThoughts3** 10,000-example subset for E2, **MATH** and **GSM8K** for E4. Each needs
  the same treatment No Robots got — conversion to Orbit's `{"prompt": [messages]}` JSONL, a
  held-out split, and a row-count assertion.
  **Pre-sweep requirement carried over from the llama3-loss-mask plan's Self-Review, not yet
  executed (only the 12-row fixture has been checked, not the real Tulu3 split — it is not in
  the local cache):** before Tulu3 feeds any sweep, this prep step must scan every row for
  assistant-message content containing the literal `<|start_header_id|>assistant<|end_header_id|>`
  (`gen_multi_turn_loss_mask_llama3` raises `ValueError` on this — it would crash a multi-hour
  sweep run partway through) or a literal `<|eot_id|>` (this silently truncates the scored span
  instead of raising — it would corrupt loss-mask spans without any error). Count both classes
  and filter the affected rows out before the first E1 run, not after a run fails or is
  discovered to be silently wrong.

- [ ] **P5: RL launcher.** Impl-plan Task 13 was never implemented. It must carry the tied
  `ROLLOUT_SEED` line itself (the shared library default is deliberately 42/untied), and note
  that `train_async.py` **refuses** `--eval-nll-data` by design — the async loop overlaps
  next-rollout generation with current-rollout training, so "weights at the moment of
  measurement" needs its own design. Either drive `train.py`, or measure validation accuracy
  instead of NLL.

---

### E1: Capacity, rank, and the 10x LR rule — decides C1 and C2

Llama-3.1-8B base, Tulu3, single epoch, constant LR, α=32, LoRA on all four projections.

**Arms:** FullFT plus LoRA at **rank ∈ {1, 4, 16, 64, 128, 256, 512}** — the post's stated
range, and wide enough that rank-1 must depart from the shared curve while rank-512 must not.

**LR grid:** five points at 0.3-decade spacing per arm, centered on the post's own prediction
so a confirmation is a hit rather than a fit — LoRA arms centered at 10x the FullFT center.
The grid must be re-centered, not extended, if any arm's argmin lands on an edge.

- [ ] **E1-0: Re-measure σ on this model and dataset (3 runs).** The Qwen3-4B σ = 0.000992
      does not transfer. Three seeds of LoRA r256-all at the center LR. Everything downstream
      is stated in units of the number this produces.
- [ ] **E1-1: Run the LR sweeps.** 8 arms × 5 LRs = **40 runs**. FullFT arms at ≥4 GPUs (P0),
      LoRA arms one GPU each, three concurrent.
- [ ] **E1-2: Read C1 off the training curves.** Plot loss against log-steps per rank at each
      rank's own argmin LR. Report, per rank, the **step at which it departs** from the
      FullFT/high-rank envelope, defined as the first step where it exceeds the pointwise
      minimum across arms by more than 2σ for 3 consecutive logging intervals. The claim
      predicts the departure step increases monotonically with rank, and that high ranks do
      not depart at all within the epoch.
- [ ] **E1-3: Read C2 off the argmins.** `argmin_LR(LoRA r256) / argmin_LR(FullFT)`; the post
      predicts 9.8, rising toward 15 for runs under ~100 steps. Also check the post's tighter
      claim that optimal LR moves **less than 2x between rank 4 and rank 512** — that one is
      falsifiable with the arms already in this sweep and costs nothing extra.

**Acceptance:** every run reports finite loss and identical held-out token/sample counts; any
arm whose argmin sits at a grid edge is re-run on a re-centered grid before its ratio is quoted.

---

### E2: Batch-size sensitivity — decides C3

The post's setup exactly: a **10,000-example subset of OpenThoughts3**, batch 32 and larger.

- [ ] **E2-1: Sweep batch × arm × LR.** Batch ∈ {32, 128, 512}; arms FullFT and LoRA r256;
      4 LRs per cell, re-centered per batch size because the optimum moves with batch.
      **24 runs.**
- [ ] **E2-2: Test the rank-independence half of the claim.** The post attributes the gap to
      the parametrization, *not* to capacity, so it must persist at a different rank. Add
      LoRA r16 at all three batch sizes, 4 LRs: **12 runs.** If the gap shrinks with rank, the
      post's mechanism is wrong and that is the finding.
- [ ] **E2-3: Report `best_LoRA(batch) − best_FullFT(batch)`** at each batch size, in units of
      σ. The claim is a *persistent* gap that grows with batch — a gap that vanishes at batch
      32 and appears at 512 is the signature; a constant offset at all batch sizes is not.

---

### E3: Layer placement at matched parameter count — decides C4

This is the claim our earlier plan got wrong, and the error is worth stating plainly: it
compared attention-only and MLP-only **at equal rank**, which in a transformer means unequal
parameter counts, confounding placement with capacity. The post deliberately matches
parameters and lets rank differ — attention-only rank-256 (0.25B) against MLP-only rank-128
(0.24B) on Llama-3.1-8B.

For Orbit's **fused** Megatron layout the matched pair must be recomputed, because
`linear_qkv` and `linear_fc1` bundle projections that HF keeps separate. Per layer, adapter
parameters are `r × (d_in + d_out)` summed over targeted modules:

```
attention   linear_qkv  r·(4096 + 6144) = 10240r     linear_proj r·(4096 + 4096) =  8192r
            attention total                                                        18432r

MLP         linear_fc1  r·(4096 + 28672) = 32768r    linear_fc2  r·(14336 + 4096) = 18432r
            MLP total                                                              51200r

ratio MLP/attention = 51200 / 18432 = 2.778   ⇒   attention r=256  ≡  MLP r=92
```

- [ ] **E3-1: Run the matched pair and the post's pair.** Arms: attention-only r256;
      MLP-only **r92** (our fused-layout match); MLP-only **r128** (the post's own pair, so a
      disagreement can be attributed to accounting rather than to physics); all-modules r256.
      5 LRs each: **20 runs.** Print realized adapter parameter counts next to every arm.
- [ ] **E3-2: Report `NLL(attn) − NLL(mlp)` at matched parameters**, in units of the σ from
      E1-0, and separately test the second half of the claim — that all-modules does not beat
      MLP-only by more than 2σ.
- [ ] **E3-3 (optional): the MoE arm.** `Qwen3-30B-A3B` is already local. The post applies
      LoRA per expert at rank = total rank / active experts (8). This needs no FullFT arm —
      it is an attention-vs-MLP comparison within LoRA — but 30B activations still exceed one
      card. Run only if P0 yields ≥4 GPUs to spare; skip otherwise and say so.

---

### E4: RL parity at low rank — decides C5

Llama-3.1-8B base, **MATH + GSM8K**, policy gradient with importance sampling and GRPO-like
centering, 32 samples per problem.

- [ ] **E4-1: Build the launcher (P5).**
- [ ] **E4-2: Run 4 arms × 4 LRs = 16 runs.** FullFT, LoRA r256, r16, **r1** — rank 1 is the
      claim's whole point, so it is not the arm to drop under budget pressure.
- [ ] **E4-3: Report validation-accuracy curves**, and the *width* of the performant LR band
      per arm — the post claims LoRA's is wider, which is a separate, checkable statement from
      peak parity. Note that σ for accuracy has never been measured; if the curves sit close,
      measuring it becomes a prerequisite exactly as E1-0 was for NLL.

---

### E5 (optional, ours): matched-parameter OFT

Carried over from the Qwen3-era plan and unchanged in spirit: does a rotation-parameterized
adapter at matched parameter count behave like LoRA on C1/C2/C4?
`orbit.utils.peft_param_match.matched_oft_block_size` already solves the block size, and its
LR scale is unknown a priori — it parameterizes a rotation, not an additive update — so it
needs a half-decade scout `{1e-5, 3e-5, 1e-4, 3e-4, 1e-3}` before any refinement grid.
**Run only after C1-C5 are settled.** This is the part of the campaign that is ours rather
than the post's, and it is worth nothing if the reproduction underneath it is unresolved.

---

## Cost

| Group | Runs | GPU tier |
|---|---|---|
| E1 (incl. E1-0 σ) | 43 | LoRA 1x, FullFT ≥4x |
| E2 | 36 | LoRA 1x, FullFT ≥4x |
| E3 | 20 (+MoE) | LoRA 1x, MoE ≥4x |
| E4 | 16 | LoRA 1x, FullFT ≥4x, plus rollout GPUs |
| **Total** | **~115** | |

I am not putting hours on this. Every timing number we have is from Qwen3-4B LoRA on a
6400-row epoch — 8.7 min per arm at three-way concurrency — and none of it survives the move
to an 8B model, a dataset two orders of magnitude larger, multi-GPU FullFT, or RL rollouts.
The first honest estimate comes from E1-0, which is also the first thing to run.

## What carries over from the Qwen3 work

Not wasted, despite the anchor change: the SFT launcher and its knobs, the held-out NLL eval
and its coverage guarantee (no floor-division truncation), the sweep driver and resumable
ledger, the OFT block-size solver, the `tool_env.sh` fix that lets any launcher start on this
box, and the discipline of measuring σ before quoting a difference. What does **not** carry
over: the Qwen3 loss mask (P2), the σ value itself (E1-0), the Megatron checkpoint (P1), and
the gate-log's G1/G2 framing, which existed to match a third party's absolute number.

## Known hazards, all previously observed

- **Shared `SAVE_DIR`.** The launcher default is one directory for every run; concurrent runs
  overwrite each other and one save took 293s instead of 97s under contention. Pass
  `SAVE_DIR` explicitly on every hand-run arm.
- **`codexlog` and env prefixes.** `codexlog <name> VAR=val cmd` fails with
  `command not found` — it execs its arguments directly. Use `codexlog <name> env VAR=val cmd`.
- **Determining that a run finished.** `phase=after_train` is emitted at *every* periodic
  eval; `pgrep` cannot see processes in another PID namespace; a bare `Traceback` may be
  benign wandb atexit noise. The only reliable end marker is
  `progress rollout=N/N … remaining=0` followed by `shutdown: dispose rollout done`.
- **`pytest tests/fast/` silently skips** `tests/fast/scripts/` and `tests/fast/tools/` —
  `norecursedirs` matches those basenames at any depth. Always use explicit paths.
