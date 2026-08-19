# Adapter-First RL: Experiment Program

## Objective

Produce the evidence that Orbit's three adapter-first re-designs — async RL with adapter-only sync, one-trunk PPO, and teacher-as-adapter-slot OPD/MOPD — are (a) algorithmically lossless, (b) systems-cheaper, and (c) enable regimes the full-model baseline cannot enter. Every experiment below is tagged with the claim type it serves:

- **Parity** — the adapter-first variant learns the same (reward vs samples, final benchmark accuracy, critic quality).
- **Cost collapse** — a resource that scaled with model size now scales with adapter size (sync latency, snapshot memory, critic trunk, teacher hosting).
- **Unlock** — something the baseline cannot run at all (trillion-scale single-node, mean-teacher RL, PPO where a second trunk does not fit).

Because every adapter-first advantage is O(adapter) vs O(model), experiments either put model size on the x-axis or go to the regime where the baseline is infeasible. Small-scale head-to-heads systematically understate the designs and must be framed as mechanism demonstrations, not as the headline.

## Standing constraints (design around these, do not re-run into them)

1. **The cheap push alone does not speed up sync RL.** Measured in `docs/orbit-adapter-async-db.html`: sync + adapter push has warm `update_weights` ≈ 0.105 s but step time ≈ 8.651 s — the serial loop is the bottleneck. The async claim is therefore about the composition (cheap push × overlap × double-buffered hot swap), never about the push in isolation.
2. **The adapter critic is ~23% slower per step at 3B** (62.3 s vs 48.1 s; `docs/reports/_src/2026-08-10-ppo-critic-comparison.md`) because its value phases serialize on the actor GPU while the full critic overlaps on its own GPU. The PPO claim is feasibility and GPU-hours (27.2 vs 29.3 at 3B), never step time. Do not compare `timing_s/actor_train` across critic modes.
3. **The fixed-budget panel premise failed at 3B math**: rollout was not the bottleneck, so the freed critic GPU bought nothing. A re-run is only meaningful on a workload whose profiled rollout fraction exceeds ~60% of step time (P2 pre-check).
4. **Quantized trunk + adapter critic is rejected** (`orbit/backends/megatron_utils/low_precision_bootstrap.py:151-156`): one-trunk aliasing shares `Parameter`s only; quantized trunk weights/scales live in checkpoint-created buffers. INT4/FP4 one-trunk PPO (P1-INT4, X2-PPO) is blocked until this is lifted; the BF16 feasibility frontier is runnable today.
5. **`self:*` OPD teachers are incompatible with `--adapter-double-buffer`** (`orbit/utils/arguments.py:1154-1166`), and sglang-local teacher scoring requires OFT (LoRA is single-active per batch). M2 runs trainer-side (`--opd-type megatron`) or single-slot.
6. **RESOLVED (I-0, 2026-08-17): `--offload-rollout` is structurally inert in async topologies — not a bug.** `needs_offload` is only set for engine groups whose GPUs overlap the Megatron slots, which happens only under `--colocate`; in every `train_async` run the startup offload releases nothing and engines stay resident, so no onload is needed. Documented in `train_async.py` / `orbit/ray/placement_group.py` and pinned by `tests/fast/test_async_offload_noop.py`. A3/A4 are unblocked.
7. **The double-buffer path currently pauses generation too** (found during I-2): `update_weights` dispatches the pause/flush/continue lifecycle unconditionally for all three sync paths (`update_weight_from_tensor.py`), so today's double-buffer runs report a real nonzero `perf/update_weights_pause_time`. The A1/A2 "no pause under double-buffer" asymmetry is therefore a *hypothesis about a possible optimization* (dropping the lifecycle when staging into an inactive slot), not current behavior — the instrument measures the actual window either way, and whether the lifecycle can be dropped is a candidate follow-up (I-7).

## Models and tasks

### Model ladder

Six rungs; each is chosen because its launcher family already exists, so scale points cost no new recipe engineering unless flagged under "Recipe gaps."

| Model | Role | Existing launchers |
|---|---|---|
| Qwen2.5-0.5B-Instruct | Phase-0 qualification and OPD smokes only; never reported as results (2026-08-06 rule) | OPD teacher-variant smokes, PPO/adapter-critic smokes, search-r1 0.5B |
| Qwen2.5-3B-Instruct, BF16 + canonical OFT | PPO workhorse (P2, P3, M1 measured table, M3); fully validated assets: torch_dist conversion, filtered OpenR1-49,990, aligned Math500/AIME/AMC evals | `ppo_critic_compare_common.sh` suite, GRPO/full-FT/head-critic variants, `search_r1/qwen2_5_3b_search_r1_ppo_common.sh` |
| Qwen3-4B-Instruct-2507, BF16 OFT | Async workhorse (A1–A4, M2); all published async numbers are at this scale, so new figures extend a measured baseline. Also X1 (FP8 twin) and the tau-bench P2 candidate | sync/async/fully-async triple in `examples/high_precision/`, `low_precision/run-qwen3-4b-fp8-math-oft.sh`, `tau_bench/qwen3_4b_tau_bench_ppo_common.sh` |
| Qwen2.5-7B, BF16 | Optional dense mid-point on the A1 curve; full-FT launchers exist, which A1's full-model-sync arm needs | `run-qwen2_5-7b-bf16-openr1-{full,lora,oft-*}` family |
| Qwen3-30B-A3B (+ Instruct-2507) | MoE scaling point (A1, P1 frontier, X1 at scale); the one model with BF16 and FP8/INT4 launchers side by side | `run-qwen3-30b-a3b-bf16-openr1-{full,lora,oft}`, `low_precision/run-qwen3-30b-a3b-{fp8,int4}-math-oft.sh` |
| Kimi-K2.6 INT4 / DSV4 MXFP4 (Flash, Pro) | Flagship X2 and the top of the A1 curve — the rung where no full-model baseline exists | `low_precision/run-kimi-k26-int4-openr1-oft.sh`, `low_precision/dsv4-*` pair |

### Task set

Exact-answer math carries every parity and systems claim (A1–A4, P1, P3, M1, M2, X1, X2): OpenR1-style 50k train JSONL, deterministic exact-match reward, Math500 primary and AIME 2024 / AMC 2023 secondary evals. Systems metrics are task-invariant, so a single task everywhere removes a confound, and math is the only task with validated data, a learned-reward-free grader, and launchers at every rung. Deviating from math requires a reason; there are exactly two:

1. **P2 needs a rollout-bound workload**, which single-turn math is not — that is why the 3B budget panel failed its premise. Candidates: tau-bench (multi-turn agentic tool use, Qwen3-4B, PPO launchers for full/LoRA/OFT) and Search-R1 (retrieval-augmented QA with EM reward, Qwen2.5-3B). The P2 pre-check profiles both and keeps whichever crosses the ~60% rollout fraction.
2. **M3 needs a domain the RL task does not cover**, to measure retention. The SFT suite ships NuminaMath, Magicoder, CommonsenseQA, and ScienceQA launchers: SFT a NuminaMath expert adapter, run blend-RL on OpenR1 math, evaluate Math500 plus a held-out Numina slice; Magicoder is the stretch variant (code expert preserved through math RL).

GSM8K appears only in the full-vocab OPD launcher and stays smoke-tier; the SWE / swe-agent examples are smoke-only and excluded.

### Assignment

| Experiment | Model(s) | Task |
|---|---|---|
| A1 | 0.5B → 3B → 4B → 7B → 30B-A3B → Kimi-K2.6 INT4 | math (task-invariant metric) |
| A2, A3, A4 | Qwen3-4B-Instruct-2507 | math |
| P1 | 7B → 30B-A3B (BF16) → R-1 if the wall is higher | math, few steps per point |
| P2 | Qwen3-4B (tau-bench) or Qwen2.5-3B (Search-R1) | pre-check winner |
| P3 | Qwen2.5-3B | math (validated suite) |
| M1 | qualify at 0.5B, measure at 3B (R-2) | math |
| M2 | Qwen3-4B | math |
| M3 | Qwen2.5-3B student + NuminaMath expert adapter (R-3) | math + Numina holdout |
| X1 | Qwen3-4B FP8 first; 30B-A3B FP8/INT4 confirm | math |
| X2 | Kimi-K2.6 INT4 (DSV4 MXFP4 alternate) | math (OpenR1) |

### Recipe gaps

- **R-1** *(Phase 4)* — If full-critic PPO on 8×B200 still fits at 30B-A3B, bracketing the P1 wall needs one new dense config around Qwen2.5-72B. Mechanical, but no launcher exists today.
- **R-2** *(Phase 1)* — M1's measured table requires porting the five teacher-variant flag blocks from the 0.5B smokes onto the 3B math recipe; the variants currently exist only as smokes.
- **R-3** *(Phase 3)* — M3 needs a Qwen2.5-3B SFT config by analogy with the existing 0.5B / Llama-8B SFT launchers; SFT and RL launchers do not currently share a model size above 0.5B.
- **R-4** *(Phase 3)* — M3's mixed-data baseline needs a joint task-reward + SFT-replay recipe, which no current flag provides (`--loss-type` is single-choice; `--use-opd` blends only distillation into advantages). Cheapest mechanical form: interleave `sft_loss` steps on Numina batches with RL steps at a fixed ratio; a `custom_loss` combination is the fallback.

## Prioritized matrix

| ID | Experiment | Claim | Tier | Hardware | Est. cost | Entry point | Blockers |
|---|---|---|---|---|---:|---|---|
| A1 | Sync-cost scaling curve | Cost collapse | 1 | 2–8 B200 per point | ~10 GPU-h/point | `tools/adapter_runtime_compare/run_compare.py` + full-FT arm | I-2, I-3 |
| A2 | Throughput timeline across an update | Cost collapse (mechanism) | 1 | 4–8 B200 | ~10 GPU-h | async 4B launchers | I-1 |
| M1 | Teacher-cost collapse table | Cost collapse | 1 | ≤4 B200 | ~20 GPU-h | `examples/on_policy_distillation/run-*.sh` | I-5 (correctness leg) |
| A3 | Async parity + speedup attribution | Parity + cost collapse | 1 | 8 B200 | ~450 GPU-h (3 arms × 3 seeds) | `run-qwen3-4b-...-oft-async.sh` vs sync twin + `...-fullft-async.sh` | I-0; full-FT LR from `lora_regret` e4 |
| X1 | Precision-gap 2×2 (one cell structurally empty) | Unlock | 2 | 8 B200 | ~200 GPU-h + PTQ/eval for cell (c, reuses A3 ckpts) | `examples/low_precision/run-qwen3-4b-fp8-math-oft.sh` + full-FT arm | named PTQ pipeline for arms (b)/(c) |
| P2 | Fixed-budget panel on a rollout-bound workload | Cost collapse | 2 | 4–8 B200 | ~150 GPU-h | `examples/tau_bench/` or `examples/search_r1/` PPO common | pre-check: ≥60% rollout + GPU-scaling probe |
| A4 | Staleness ablation (fully async) | Parity + mechanism | 2 | 4 B200 | ~100 GPU-h (4 settings) | `run-...-oft-fully-async.sh` | I-0 |
| P3 | Critic parity, 3 seeds + explained variance | Parity | 2 | 4 B200 | ~180 GPU-h (2 arms × 3 seeds) | `ppo_critic_compare_common.sh` wrappers | I-4 |
| M2 | Mean-teacher RL (EMA self-distillation blend) | Unlock (algorithmic) | 3 | 4–8 B200 | ~200 GPU-h (4 arms) | `run-...-opd-ema-smoke.sh` scaled up | constraint 5 |
| M3 | Expert-adapter distillation during RL | Unlock (compositional) | 3 | 8 B200 | ~330 GPU-h (4 arms + SFT) | `examples/sft/` + blend launcher | R-4 (mixed-data arm only) |
| P1 | PPO feasibility frontier | Unlock | 3 | 8 B200 | ~100 GPU-h | new wrappers over existing recipe | constraint 4 for INT4; BF16 runnable |
| X2 | Trillion-scale single-node flagship | Unlock | 3 | 8 B200 | ~1 node-week | Kimi/DSv4 recipes | constraint 4 for the PPO variant |

Costs are order-of-magnitude planning numbers anchored on the completed 3B benchmark (~30 GPU-h per arm-seed at 3B, 4-GPU layout); refresh them after the first qualification run of each experiment.

## Experiment specifications

### A1 — Sync-cost scaling curve

One figure, model size on x (0.5B → 3B → 4B → 7B optional → 30B-A3B → largest feasible), three arms: full-model broadcast (the `_send_base_params` path, exercised via a full-FT recipe), adapter single-slot, adapter double-buffer. Series: `update_weights` wall time, payload bytes, and engine pause time (all three paths currently dispatch the pause lifecycle — constraint 7 — so pause time is a measured series per arm, not an assumed zero for double-buffer). Derive one more column: **achieved fraction of link bandwidth** (payload bytes / wall time vs the nominal interconnect) — if the full-model broadcast runs near wire speed, the O(model)-bytes cost is physics and the arm cannot be dismissed as an unoptimized baseline; if it runs far below, say so and the honest comparison is bytes, not seconds. Expected shape: full-model grows linearly toward tens of seconds; adapter flat ≈ 0.1 s. Timing comes from existing metrics; payload bytes and pause time need I-2. The comparison harness runs paired async single-slot vs double-buffer today and needs a full-FT arm (I-3). Record the **PEFT transport per point**: this cluster's `env.sh` defaults `ORBIT_PEFT_ADAPTER_TRANSPORT=cpu_gather` (the B200 CUDA-IPC workaround), so colocated points measure CPU-gather rather than CUDA-IPC, while async points are NCCL regardless — without a transport column the colocated numbers do not compare across machines. Memory series can reuse the allocator-counter reporting merged in on 2026-08-17. At quantized bases the full-model arm is not even well-defined without requantization — state that in the figure caption rather than trying to measure it.

### A2 — Rollout-throughput timeline across a weight update

Rollout tokens/s in ~100 ms bins over a window containing 2–3 publications, one trace per arm (full-model, single-slot, double-buffer). This is the mechanism figure: the double-buffer trace is expected to be the shallowest because the broadcast lands in the inactive slot while the active slot serves — though the pause lifecycle currently still runs in all three paths (constraint 7), so the trace measures rather than assumes the asymmetry, quantifies the I-7 prize, and explains the measured +50.2% tok/GPU/s. Needs I-1 (done). Qwen3-4B, 4+4 layout, short run — cheap enough to iterate on until the figure is clean.

### A3 — Async parity, speedup, and attribution

Three arms, ≥3 seeds each: sync OFT (`run-qwen3-4b-instruct-2507-bf16-math-oft.sh`), async OFT + double-buffer (`run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh`, `ADAPTER_DOUBLE_BUFFER=1`), and **async full-FT** (`run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh`, the I-3 launcher). The third arm exists because constraint 1 makes the two-arm version attackable: sync→async overlap is a speedup any full-model async system also gets, so a sync-OFT vs async-OFT comparison cannot attribute anything to adapters. With three arms the wall-clock figure decomposes into sync→async-fullFT (overlap, not novel) and async-fullFT→async-OFT (the adapter contribution: pause window + payload), and the reward-vs-samples figure gains the program's only matched-pipeline full-FT quality anchor — everywhere else parity is adapter-vs-adapter, and outsourcing the adapter-vs-full-FT question entirely to the `lora_regret` e4 campaign (≤8B, MATH+GSM8K mix, different pipeline) is a cross-reference, not a control. **The full-FT arm must get its own learning rate** — the `lora_regret` sweeps put the full-FT and adapter optima about a decade apart, so reusing the OFT LR would manufacture a strawman; seed the full-FT LR from the e4 full-FT window and state it in the figure.

Two figures from the same runs: reward vs samples (adapter arms should coincide within the pre-registered margin; the full-FT arm anchors quality — the async off-policy guard enforces a correction; report which one is active) and reward vs wall-clock (adapter async expected ≈3.4× left of sync per the measured 8.651 → 2.531 s/step). State the staleness regime explicitly: `train_async`'s one-step overlap bounds staleness at one publication by construction (there is no `fully_async/staleness/*` metric in this mode because there is nothing to measure), so "async but off-policy by at most one version" is a structural statement in the figure caption, and the measured staleness distributions belong to A4. Report the noise floor across seeds; a parity claim without it is unfalsifiable. Never merge the two figures. Note: the held-out eval-NLL hook is deliberately unavailable here — `train_async.py` rejects `--eval-nll-data` because the overlap loop makes "weights at the moment of measurement" ill-defined — so A3's parity evidence stays reward curves plus benchmark evals.

### A4 — Staleness ablation in fully-async mode

`run-qwen3-4b-instruct-2507-bf16-math-oft-fully-async.sh` with `--max-weight-staleness` ∈ {1, 2, 4, unset}. Plot final reward, throughput, `fully_async/staleness/{mean,max}`, and `recycled_stale_groups`. The point: per-turn `adapter_version` stamping (enforced equal to `weight_version` at three layers) is what makes principled staleness control possible; the ablation shows the throughput/quality dial actually working. Secondary table: `--keep-old-actor` snapshot cost (time + bytes) under adapter state vs full-model backup, vs model size — one line of evidence per size point, harvestable from A1 runs.

### P1 — PPO feasibility frontier

Fixed hardware (8 B200), find the largest model where PPO-with-critic runs per mode. Full critic needs a second trunk + fp32 masters + Adam on its own GPUs; adapter critic adds ~0 trunk bytes (measured 44.6 GB vs 48.8 GB actor-alone at 3B). **The headline deliverable is the measured-bytes table** (critic trunk + fp32 masters + optimizer state per mode, from the allocator counters), with the "wall" bar chart as its illustration — a feasibility bar alone invites "you just didn't offload the critic," whereas O(model) measured bytes vs ~0 is independent of any offload policy. The wall itself is defined at matched parallelism with no CPU offload of trainable state, stated in the caption; every cell states its offload policy. **New sub-arm (merged 2026-08-17): adapter critic ± frozen-base offload** — `offload_megatron_frozen_base_to_cpu` (modes auto/flat/tms) is gated on PEFT being active, so only the adapter arm can offload its trunk during training phases; this pushes the adapter arm's wall further out and is itself an adapter-first unlock (full FT has no frozen parameters to offload). Every memory number must state the offload mode. Run BF16 now (bracket the wall with e.g. 14B/32B/72B dense); the INT4 trillion-scale version waits on constraint 4. Each point needs only a few steps to demonstrate fit + a stable loss, not a full training run. Memory series reuse the merged-in allocator counters.

### P2 — Fixed-budget panel, rollout-bound workload

Pre-check first, two conditions, both pre-registered: (i) profile rollout fraction of step time on the tau-bench and search-r1 PPO recipes and require it to exceed ~60%; (ii) run a 2-vs-3 rollout-GPU throughput probe and require rollout throughput to actually scale with the added GPU — a high rollout fraction whose bottleneck is env stepping (tau-bench tool calls, retrieval latency) would make the freed GPU worthless and the panel would fail its premise a second time. Proceed only where both hold. Then the budget layout from the 2026-08-06 design: full critic (N−1 rollout GPUs) vs adapter critic (N rollout GPUs) at equal total GPUs. Headline metric: GPU-hours to target reward. This directly repairs the failed 3B-math premise by choosing a workload where the freed GPU buys throughput.

### P3 — Critic parity with error bars

Extend the completed single-seed controlled panel (Math500 52.27±1.00 vs 52.33±1.26) to 3 seeds on the existing wrappers, and add critic explained variance (I-4) to show the aliased-trunk critic learns real values. Keep the head-critic collapse as the published negative control: value head alone fails, so the adapter is the minimal sufficient critic capacity. Optionally add one sparse/long-horizon task where PPO beats GRPO; if PPO never beats GRPO in the suite, say so and frame the critic work as infrastructure for when it does.

### M1 — Teacher-cost collapse table

Fixed student/task (0.5B–3B), one row per teacher realization: external server (`--opd-teacher-url`, +N GPUs), `load:` second Megatron model, `adapter:<path>` swap, `base` with KL on (aliases the ref forward — zero extra forwards), `self:ema`. Columns: extra GPUs, extra memory, extra forwards/step, step time. All arms have existing smoke launchers under `examples/on_policy_distillation/`. Correctness leg (I-5): identical `teacher_log_probs` on a fixed batch across the `alias_ref`, `adapter_off`, and external-URL plans, within numerical tolerance — this is what licenses the word "free." **Scope the claim explicitly: the collapse applies to same-trunk teachers** (self, EMA, expert adapters, the frozen base) — a larger cross-model teacher cannot be an adapter slot and still needs a server, so the table's claim is "same-trunk teacher hosting collapses," never "teacher hosting is free" unqualified.

### M2 — Mean-teacher RL

The EMA self-teacher is the variant whose cost collapses hardest: an EMA teacher of a full model is a second model copy — expensive, not impossible, and the framing must say so — while an EMA of an adapter is megabytes of FP32. Arms: RL-only; RL + `self:ema` blend at 2–3 `--opd-ema-decay` values; RL + `self:lag` (separates "averaging" from "delay"). Hypothesis: reduced entropy collapse, better pass@1 with pass@k retained. Runs trainer-side (`--opd-type megatron`) per constraint 5. Honest framing if neutral: the capability costs a flag, and M1 still stands.

### M3 — Expert-adapter distillation during RL

Train an SFT expert adapter (`examples/sft/`), then blend-mode RL with `--opd-teacher adapter:<expert>`: task reward + distillation toward the expert in one job, `--custom-rm-path` free for the real reward (impossible in the external-URL mode, which must hijack the reward hook). Baselines: sequential SFT→RL, RL-only, and **mixed-data RL** — task reward on math plus an SFT loss on replayed Numina data, no distillation — all at matched total compute. The mixed-data arm is the first alternative a reviewer proposes ("why distill from an adapter instead of replaying the data?"), so its absence would be read as dodging; if it ties the distillation arm, say so and the claim falls back to convenience (no data pipeline in the RL job), which is still real. Orbit has no joint policy+SFT loss today — `--loss-type` is single-choice and `--use-opd` blends only a distillation term into advantages — so this arm needs R-4 (interleaved `sft_loss` steps on replay batches, or a `custom_loss` combination). Metrics: final task accuracy + retention of the expert's domain (Numina holdout accuracy and NLL).

### X1 — Precision-gap experiment

A 2×2 design — {adapter, full-FT} × {train at deploy precision, train BF16 then quantize} — with one cell structurally empty: (a) adapter RL against the FP8/INT4 base, deployed as-is; (b) full-FT RL in BF16, then quantize; (c) adapter RL in BF16, then quantize — **reuses A3's sync-OFT checkpoints, so its marginal cost is PTQ + eval only**; (d) full-FT at deploy precision does not exist, which is the unlock and is stated as an empty cell, not omitted. Without (c) the two-arm version confounds adapter-vs-full-FT with precision; (c) separates "training at deploy precision" from "adapter vs full". **Name the PTQ recipe up front** — the same calibrated pipeline that produced the low-precision base checkpoints — and tune it in good faith; an untuned quantize step makes arm (b) a strawman and the whole figure dismissible. Measure train↔rollout logprob abs-diff during training (existing parity tooling) and final deployed accuracy. Arm (b) pays a requantization tax arm (a) structurally cannot pay; arm (a) at trillion scale has no baseline at all. Start at Qwen3-4B FP8 where all arms are cheap; the figure generalizes upward by the A1 argument.

### X2 — Trillion-scale single-node flagship

One end-to-end run: Kimi-K2.6 or DSV4 at INT4/FP4 on a single 8×B200 node, async + double-buffer, optionally a `base` free-teacher KL blend. Deliverables: reward curve, step-time breakdown, memory breakdown table where every baseline column reads "multi-node, high precision, N× hardware." GRPO-style estimator now; the PPO variant follows constraint 4. This is the existence proof that makes A1/P1's extrapolations land.

## Instrumentation and engineering pre-work

All of I-0 through I-5 landed on `orbit-main` on 2026-08-17 (five `instr/*` branches, merged after a green fast suite).

- **I-0 — DONE.** Resolved as not-a-bug (see constraint 6); comments + 9 pinning tests in `tests/fast/test_async_offload_noop.py`.
- **I-1 — DONE.** `tools/rollout_timeline/`: standalone `probe.py` polling SGLang's `sglang:realtime_tokens_total{mode="decode"}` counter from `/metrics` (engines need `--enable-metrics`; `/server_info` gauge as fallback), pure `binning.py` (counter resets and scrape gaps handled; a scrape gap during an update is itself signal), and trainer-side update markers gated on `ORBIT_TIMELINE_EVENTS_FILE`.
- **I-2 — DONE.** New per-update metrics through the existing perf flow: `perf/update_weights_payload_bytes`, `perf/update_weights_payload_num_tensors`, `perf/update_weights_num_chunks`, `perf/update_weights_pause_time` (pause dispatch → continue completion; see constraint 7 for the double-buffer finding). Implemented in `update_weight/sync_metrics.py` + transport send sites.
- **I-3 — DONE.** `tools/adapter_runtime_compare/` arms are now a registry; opt-in `async_fullft` arm (via `--modes`) plus the mechanical launcher `run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh`. Default arm selection unchanged (regression-tested).
- **I-4 — DONE.** `value_explained_var` computed exactly from five SUM-reduced token-level sufficient statistics (`value_ev/*`), finalized in `aggregate_train_losses`; NaN/degenerate-guarded; identical across critic modes.
- **I-5 — DONE.** `orbit/utils/logprob_compare.py` (stdlib-only comparison utility, shared with the future GPU/SGLang leg) + `tests/fast/test_opd_teacher_equivalence.py` pinning: `alias_ref` returns the ref list by identity with no forward run, `adapter_off` == adapter-free twin bitwise, `adapter_swap` == directly-built teacher module bitwise with exact restore — both through the real actor dispatch.
- **I-6** *(optional, unblocks P1-INT4 and X2-PPO)* — Extend one-trunk aliasing to quantized trunk buffers, or a buffer-sharing equivalent. Not started.
- **I-7** *(new, from constraint 7)* — Investigate dropping the pause/flush/continue lifecycle for the double-buffer path; the pause-time metric quantifies the prize first.

### Capabilities merged in from origin (2026-08-17, merge `a031b3c`) — reuse, do not rebuild

- **Frozen-base offload** (`offload_megatron_frozen_base_to_cpu`, PEFT-gated) — the P1 sub-arm above.
- **Held-out eval-NLL hook** (`--eval-nll-data`, `train.py` only; explicitly rejected in `train_async.py`) — cheap secondary learning-quality metric for the sync-driver experiments (P3, M2, M3, X1).
- **Allocator counters + per-arm W&B run naming** — the memory-series instrumentation A1/P1 planned to add; already present.
- **`ORBIT_PEFT_ADAPTER_TRANSPORT=cpu_gather` cluster default** in `env.sh` — colocated adapter sync routes over CPU-gather on B200; A1 records transport per point.
- **`tools/lora_regret/` campaign harness** — measures adapter-vs-full-FT learning quality with NLL probes; overlaps this program's parity tier at the algorithmic level. Cross-reference its results instead of re-measuring that question, and borrow its arms/sweep/analyze structure for the P2/P3 panels where it fits.

## Methodology standards

- ≥3 matched seeds for any learning-quality claim; single seeds only for qualification and systems timing. Always report the seed noise floor next to the effect size.
- **Every parity claim pre-registers an absolute equivalence margin before launch** (anchor: the completed 3B benchmark's observed ±1.00–1.26 eval CIs on Math500), and reports the effect with both seed-σ and eval CI against that margin. Parity means "effect inside the margin," equivalence-test style; "the curves coincide" with a post-hoc noise floor is not a decision rule — with 3 seeds, σ estimated after the fact cannot carry a claim.
- **Every adapter-sync run logs the train↔rollout logprob abs-diff** (existing parity tooling) as a standing guard metric. This repo has already shipped one silent adapter-sync corruption (the CanonicalOFT streamed-loader substring match that dropped every R update); the fast path is only evidence if each run carries proof it is also the correct path.
- Parity claims are per-task: math carries them, and the P2 workload is the program's only generality point. Say this in the paper rather than letting a reviewer discover it.
- Reward-vs-samples and reward-vs-wall-clock are separate figures answering separate questions; never conflate them (the standard async-RL reviewer objection).
- Controlled vs fixed-budget panels stay separate, as in the 2026-08-06 design.
- Headline efficiency metric is GPU-hours to target quality; step time is a diagnostic, not a claim.
- Pre-register the P2 bottleneck profile before choosing its workload.
- Reuse the validated 3B assets (model conversion, filtered OpenR1 data, aligned Math500/AIME/AMC evals) recorded in `2026-08-06-ppo-critic-comparison-design.md` wherever the model scale permits.
- Sync-driver experiments (P3, M2, M3, X1) additionally log held-out eval-NLL (`--eval-nll-data`) as a secondary learning-quality metric; async experiments cannot (by design — see A3 note) and claim parity on reward/benchmarks only.
- Systems figures state the active PEFT transport (NCCL / CUDA-IPC / cpu_gather) alongside every latency or pause measurement.

## Phasing

- **Phase 0** — I-0 through I-5 (code work DONE 2026-08-17); remaining: smoke-qualify every launcher named above at 0.5B scale (GPU runs).
- **Phase 1** — A1, A2, M1 (cheap, headline systems figures; no long training).
- **Phase 2** — A3, A4; P2 pre-check then P2; P3 seed extension.
- **Phase 3** — X1, M2, M3.
- **Phase 4** — P1 frontier (BF16 now, INT4 after I-6), X2 flagship.

Phases 1–2 are sufficient for a systems-paper submission; Phases 3–4 carry the unlock claims that differentiate the work.
