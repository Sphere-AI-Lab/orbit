---
title: E4 GSM8K + Math — FullFT vs LoRA across the complete RL learning-rate panel
kind: experiment
subtitle: 60 completed single-seed arms at 150 on-policy updates, with the two unstable Math LoRA 1e-3 arms explicitly abandoned
date: 2026-08-10
tags: lora-regret, e4, gsm8k, math, rl, learning-rate-sweep
matrix: e4 and e4lr0
completion: GSM8K 31/31; Math 29/31 plus 2 intentional abandonments
remote_branch: feat/lora-without-regret
remote_commit: 46c8e0f6d65a9630d3eff44d2db7c1e9dc38a18a
remote_state: tracked files clean; result ledgers and unrelated notes untracked
scheduler: HTCondor, whole-node interactive allocations managed by Codex
hardware: 8 GPUs per arm; mixed H100 80 GB and B200 178 GB nodes
jobs: final LR0 allocations 17448192.0 and 17448193.0; earlier job IDs not fully retained
datasets: gsm8k_train/gsm8k_test and math_train/math_test
seed: 0 (single seed, no variance estimate)
ledgers: results/e4_gsm8k_lr0..lr7.jsonl and results/e4_math_lr0..lr7.jsonl
logs: logs/lora_regret/
wandb_entity: zeju-qiu
wandb_projects: gsm8k-rl-rank-ft, gsm8k-rl-rank-lora, math-rl-rank-ft, math-rl-rank-lora
---

## Executive result

The complete short-horizon panel gives a mixed answer to the LoRA-without-regret
hypothesis. With learning rate tuned separately, LoRA r16 reaches **0.2758** on Math,
slightly above FullFT's **0.2660**. On GSM8K, FullFT reaches **0.7870** while the best
LoRA endpoint is **0.7551**. That LoRA endpoint is length-degenerate, however; the best
non-runaway LoRA endpoint is r256 at **0.7415**.

| dataset | base | best FullFT | best LoRA | LoRA − FullFT | LoRA/FullFT LR |
|:--|--:|:--|:--|--:|--:|
| GSM8K | 0.033 | 0.7870 @ 7e-07 | 0.7551 @ 3e-05, r16 | -0.0319 | 42.9× |
| Math | 0.056 | 0.2660 @ 7e-07 | 0.2758 @ 3e-05, r16 | +0.0098 | 42.9× |

<div class="callout warn">
<strong>This is not a full reproduction of rank-independent LoRA parity.</strong>
Math r1 peaks at 0.2536 and r256 at 0.2378, below both r16 and FullFT. GSM8K's three
LoRA ranks cluster more tightly at their tuned 3e-05 point, but even the best healthy
LoRA arm trails FullFT by 0.0455. Every number is one seed, so these are measured
endpoints rather than significance claims.
</div>

## Setup

All arms fine-tune `llama3.1-8b` with the protocol centralized in
`scripts/lora_regret/e4_protocol.sh`. GSM8K trains on 7,473 problems and evaluates on
1,319; Math trains on 7,498 and evaluates on 5,000. Dataset selection is the only
substantive configuration difference between paired GSM8K and Math scripts.

- `NUM_ROLLOUT=150`: 150 rollout batches and 150 optimizer updates; ledgers store the
  final zero-based step as `149`.
- `GLOBAL_BATCH_SIZE=1024`: one on-policy update per batch of 32 prompts × 32 samples.
- GRPO-style group-mean centering without standard-deviation normalization.
- PPO clipping disabled with `EPS_CLIP=EPS_CLIP_HIGH=1e9`.
- `EVAL_INTERVAL=25`: eval before training and after rollouts 24, 49, 74, 99, 124,
  and 149.
- No checkpoints (`SAVE_INTERVAL` empty); W&B writes offline and syncs from a login
  host to entity `zeju-qiu`.
- Every arm uses eight GPUs. FullFT is TP=4/DP=2. LoRA targets
  `linear_qkv,linear_proj,linear_fc1,linear_fc2`.

LoRA uses `lora_alpha=32` at every rank. Its adapter scaling α/r is therefore 32, 2,
and 0.125 for r1, r16, and r256. Rank and nominal learning rate are not independent
axes in this panel: a fixed nominal LR produces a 256× spread in adapter scaling.

### Learning-rate grids

LR0 is a LoRA-only point added below the original seven-column panel. Every other
column contains one FullFT arm and the three LoRA ranks.

| column | FullFT LR | LoRA LR | GSM8K | Math |
|:--|--:|--:|:--|:--|
| lr0 | — | 2e-06 | 3/3 complete | 3/3 complete |
| lr1 | 5e-08 | 5e-06 | 4/4 complete | 4/4 complete |
| lr2 | 1e-07 | 1e-05 | 4/4 complete | 4/4 complete |
| lr3 | 3e-07 | 3e-05 | 4/4 complete | 4/4 complete |
| lr4 | 7e-07 | 7e-05 | 4/4 complete | 4/4 complete |
| lr5 | 2e-06 | 2e-04 | 4/4 complete | 4/4 complete |
| lr6 | 4e-06 | 4e-04 | 4/4 complete | 4/4 complete |
| lr7 | 1e-05 | 1e-03 | 4/4 complete | FullFT + r1 complete; r16 stopped; r256 not launched |

### Reproduction commands

Each wrapper is resumable: an arm already recorded with `status: "ok"` is skipped.

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /fast/zqiu/orbit-iclr/orbit

bash scripts/lora_regret/run_e4_gsm8k_lr0_8gpu.sh  # repeat for lr1 … lr7
bash scripts/lora_regret/run_e4_math_lr0_8gpu.sh   # repeat for lr1 … lr7
```

## Results

### GSM8K endpoint accuracy

Held-out `gsm8k_test` accuracy after rollout 149. LR0 is blank for FullFT because the
extension deliberately added only LoRA arms.

| method | lr0 | lr1 | lr2 | lr3 | lr4 | lr5 | lr6 | lr7 | best |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| FullFT | — | 0.0637 | 0.2578 | 0.7028 | **0.7870** | 0.7854 | 0.5459 | 0.0000 | 0.7870 @ 7e-07 |
| LoRA r1 | 0.2858 | 0.6937 | 0.6892 | **0.7149** | 0.6808 | 0.0000 | 0.0000 | 0.0000 | 0.7149 @ 3e-05 |
| LoRA r16 | 0.1175 | 0.7051 | 0.7346 | **0.7551** | 0.6694 | 0.6217 | 0.0000 | 0.0000 | 0.7551 @ 3e-05 |
| LoRA r256 | 0.0538 | 0.3730 | 0.7187 | **0.7415** | 0.6513 | 0.6892 | 0.0000 | 0.0000 | 0.7415 @ 3e-05 |

FullFT's optimum is bracketed: 0.7870 at 7e-07 and 0.7854 at 2e-06 form a broad top,
then accuracy falls to 0.5459 at 4e-06 and zero at 1e-05. All three LoRA ranks peak
at 3e-05. LR0 confirms that 2e-06 is an under-training boundary, especially as rank
increases: r1/r16/r256 finish at 0.2858/0.1175/0.0538 and are still rising at the end.

The high-LR endpoint can hide a broken policy. LoRA r16 at 3e-05, the headline best,
reaches 0.7551 while its response length has already saturated near the 2,048-token
cap. At 2e-04, r16 and r256 still score 0.6217 and 0.6892 with 100% truncation. These
are gradeable answers followed by runaway text, not healthy policies. At 4e-04 and
1e-03 all LoRA ranks finish at zero.

### Math endpoint accuracy

Held-out `math_test` accuracy after rollout 149.

| method | lr0 | lr1 | lr2 | lr3 | lr4 | lr5 | lr6 | lr7 | best |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| FullFT | — | 0.0734 | 0.1228 | 0.2250 | **0.2660** | 0.1366 | 0.0300 | 0.0460 | 0.2660 @ 7e-07 |
| LoRA r1 | 0.1256 | 0.2240 | **0.2536** | 0.0000 | 0.2002 | 0.0000 | 0.0000 | 0.0000 | 0.2536 @ 1e-05 |
| LoRA r16 | 0.0776 | 0.1864 | 0.2690 | **0.2758** | 0.2722 | 0.1040 | 0.0000 | — | 0.2758 @ 3e-05 |
| LoRA r256 | 0.0682 | 0.1250 | 0.1736 | 0.2010 | **0.2378** | 0.0000 | 0.0000 | — | 0.2378 @ 7e-05 |

Math brackets the same FullFT optimum at 7e-07, but its LoRA optimum moves with rank:
r1 prefers 1e-05, r16 prefers 3e-05, and r256 prefers 7e-05. That ordered shift is
consistent with fixed α/r making the effective update smaller as rank increases.

The r1 1e-05 arm is not collapsed: its final complete segment rises
0.0590 → 0.0960 → 0.1680 → 0.1888 → 0.2202 → 0.2366 → 0.2536. The same file
contains a short earlier attempt ending at rollout 24; the successful ledger row and
last complete log segment identify the finished run. In contrast, r1 at 3e-05 reaches
0.2124 at rollout 74, then falls to 0.0006 at 99 and zero thereafter as truncation
rises to 97.7%.

### Representative evaluation trajectories

These checkpoints come from the last complete segment of each launcher log. They show
that endpoint zeros are genuine late collapse, not zero accuracy for the entire run.

| dataset / arm | 0 | 24 | 49 | 74 | 99 | 124 | 149 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| GSM8K FullFT 7e-07 | 0.035 | 0.309 | 0.672 | 0.758 | 0.752 | 0.761 | 0.787 |
| GSM8K LoRA r256 3e-05 | 0.036 | 0.168 | 0.590 | 0.593 | 0.675 | 0.676 | 0.741 |
| GSM8K LoRA r1 2e-04 | 0.033 | 0.582 | 0.484 | 0.668 | 0.475 | 0.000 | 0.000 |
| Math FullFT 7e-07 | 0.056 | 0.161 | 0.177 | 0.199 | 0.225 | 0.236 | 0.266 |
| Math LoRA r1 1e-05 | 0.059 | 0.096 | 0.168 | 0.189 | 0.220 | 0.237 | 0.254 |
| Math LoRA r16 3e-05 | 0.056 | 0.141 | 0.202 | 0.231 | 0.251 | 0.255 | 0.276 |
| Math LoRA r256 7e-05 | 0.057 | 0.168 | 0.158 | 0.180 | 0.208 | 0.204 | 0.238 |
| Math LoRA r1 3e-05 | 0.058 | 0.169 | 0.205 | 0.212 | 0.001 | 0.000 | 0.000 |

| collapse example | peak | final | final mean response | final truncated |
|:--|:--|--:|--:|--:|
| GSM8K LoRA r1, 2e-04 | 0.6679 @ 74 | 0.0000 | 2,047 | 99.9% |
| GSM8K FullFT, 1e-05 | 0.0326 @ 0 | 0.0000 | 2,048 | 100.0% |
| Math LoRA r1, 3e-05 | 0.2124 @ 74 | 0.0000 | 2,028 | 97.7% |
| Math LoRA r256, 2e-04 | 0.1986 @ 49 | 0.0000 | 1,628 | 78.8% |
| Math FullFT, 4e-06 | 0.0772 @ 24 | 0.0300 | 1,927 | 92.9% |

### Per-arm details

Wall time is the successful arm's elapsed time, not scheduler billing time. Summed over
successful rows it is 137.9 whole-node hours for GSM8K and 154.4 for Math. Hardware
varied between H100 and B200 nodes, so wall times are useful operational provenance but
not a controlled method comparison.

<details>
<summary>GSM8K — all 31 successful arms</summary>

| col | arm | method | LR | adapter params | accuracy | wall | rollouts |
|:--|:--|:--|--:|--:|--:|--:|--:|
| lr0 | lora-r1-all-gsm8k-lr2e-06-s0 | LoRA r1 | 2e-06 | 2.2 M | 0.2858 | 4.07 h | 150 |
| lr0 | lora-r16-all-gsm8k-lr2e-06-s0 | LoRA r16 | 2e-06 | 35.7 M | 0.1175 | 4.53 h | 150 |
| lr0 | lora-r256-all-gsm8k-lr2e-06-s0 | LoRA r256 | 2e-06 | 570.4 M | 0.0538 | 5.17 h | 150 |
| lr1 | full-na-na-gsm8k-lr5e-08-s0 | FullFT | 5e-08 | — | 0.0637 | 2.71 h | 150 |
| lr1 | lora-r1-all-gsm8k-lr5e-06-s0 | LoRA r1 | 5e-06 | 2.2 M | 0.6937 | 3.48 h | 150 |
| lr1 | lora-r16-all-gsm8k-lr5e-06-s0 | LoRA r16 | 5e-06 | 35.7 M | 0.7051 | 4.15 h | 150 |
| lr1 | lora-r256-all-gsm8k-lr5e-06-s0 | LoRA r256 | 5e-06 | 570.4 M | 0.3730 | 5.10 h | 150 |
| lr2 | full-na-na-gsm8k-lr1e-07-s0 | FullFT | 1e-07 | — | 0.2578 | 2.72 h | 150 |
| lr2 | lora-r1-all-gsm8k-lr1e-05-s0 | LoRA r1 | 1e-05 | 2.2 M | 0.6892 | 3.15 h | 150 |
| lr2 | lora-r16-all-gsm8k-lr1e-05-s0 | LoRA r16 | 1e-05 | 35.7 M | 0.7346 | 3.77 h | 150 |
| lr2 | lora-r256-all-gsm8k-lr1e-05-s0 | LoRA r256 | 1e-05 | 570.4 M | 0.7187 | 4.62 h | 150 |
| lr3 | full-na-na-gsm8k-lr3e-07-s0 | FullFT | 3e-07 | — | 0.7028 | 2.26 h | 150 |
| lr3 | lora-r1-all-gsm8k-lr3e-05-s0 | LoRA r1 | 3e-05 | 2.2 M | 0.7149 | 2.35 h | 150 |
| lr3 | lora-r16-all-gsm8k-lr3e-05-s0 | LoRA r16 | 3e-05 | 35.7 M | 0.7551 | 4.63 h | 150 |
| lr3 | lora-r256-all-gsm8k-lr3e-05-s0 | LoRA r256 | 3e-05 | 570.4 M | 0.7415 | 3.40 h | 150 |
| lr4 | full-na-na-gsm8k-lr7e-07-s0 | FullFT | 7e-07 | — | 0.7870 | 2.14 h | 150 |
| lr4 | lora-r1-all-gsm8k-lr7e-05-s0 | LoRA r1 | 7e-05 | 2.2 M | 0.6808 | 2.31 h | 150 |
| lr4 | lora-r16-all-gsm8k-lr7e-05-s0 | LoRA r16 | 7e-05 | 35.7 M | 0.6694 | 2.85 h | 150 |
| lr4 | lora-r256-all-gsm8k-lr7e-05-s0 | LoRA r256 | 7e-05 | 570.4 M | 0.6513 | 3.65 h | 150 |
| lr5 | full-na-na-gsm8k-lr2e-06-s0 | FullFT | 2e-06 | — | 0.7854 | 2.48 h | 150 |
| lr5 | lora-r1-all-gsm8k-lr0.0002-s0 | LoRA r1 | 2e-04 | 2.2 M | 0.0000 | 5.03 h | 150 |
| lr5 | lora-r16-all-gsm8k-lr0.0002-s0 | LoRA r16 | 2e-04 | 35.7 M | 0.6217 | 3.69 h | 150 |
| lr5 | lora-r256-all-gsm8k-lr0.0002-s0 | LoRA r256 | 2e-04 | 570.4 M | 0.6892 | 7.79 h | 150 |
| lr6 | full-na-na-gsm8k-lr4e-06-s0 | FullFT | 4e-06 | — | 0.5459 | 3.15 h | 150 |
| lr6 | lora-r1-all-gsm8k-lr0.0004-s0 | LoRA r1 | 4e-04 | 2.2 M | 0.0000 | 5.09 h | 150 |
| lr6 | lora-r16-all-gsm8k-lr0.0004-s0 | LoRA r16 | 4e-04 | 35.7 M | 0.0000 | 5.53 h | 150 |
| lr6 | lora-r256-all-gsm8k-lr0.0004-s0 | LoRA r256 | 4e-04 | 570.4 M | 0.0000 | 5.69 h | 150 |
| lr7 | full-na-na-gsm8k-lr1e-05-s0 | FullFT | 1e-05 | — | 0.0000 | 5.18 h | 150 |
| lr7 | lora-r1-all-gsm8k-lr0.001-s0 | LoRA r1 | 1e-03 | 2.2 M | 0.0000 | 8.79 h | 150 |
| lr7 | lora-r16-all-gsm8k-lr0.001-s0 | LoRA r16 | 1e-03 | 35.7 M | 0.0000 | 7.52 h | 150 |
| lr7 | lora-r256-all-gsm8k-lr0.001-s0 | LoRA r256 | 1e-03 | 570.4 M | 0.0000 | 10.90 h | 150 |

</details>

<details>
<summary>Math — all 29 successful arms</summary>

| col | arm | method | LR | adapter params | accuracy | wall | rollouts |
|:--|:--|:--|--:|--:|--:|--:|--:|
| lr0 | lora-r1-all-math-lr2e-06-s0 | LoRA r1 | 2e-06 | 2.2 M | 0.1256 | 4.50 h | 150 |
| lr0 | lora-r16-all-math-lr2e-06-s0 | LoRA r16 | 2e-06 | 35.7 M | 0.0776 | 5.00 h | 150 |
| lr0 | lora-r256-all-math-lr2e-06-s0 | LoRA r256 | 2e-06 | 570.4 M | 0.0682 | 5.74 h | 150 |
| lr1 | full-na-na-math-lr5e-08-s0 | FullFT | 5e-08 | — | 0.0734 | 3.01 h | 150 |
| lr1 | lora-r1-all-math-lr5e-06-s0 | LoRA r1 | 5e-06 | 2.2 M | 0.2240 | 4.03 h | 150 |
| lr1 | lora-r16-all-math-lr5e-06-s0 | LoRA r16 | 5e-06 | 35.7 M | 0.1864 | 4.81 h | 150 |
| lr1 | lora-r256-all-math-lr5e-06-s0 | LoRA r256 | 5e-06 | 570.4 M | 0.1250 | 5.69 h | 150 |
| lr2 | full-na-na-math-lr1e-07-s0 | FullFT | 1e-07 | — | 0.1228 | 3.02 h | 150 |
| lr2 | lora-r1-all-math-lr1e-05-s0 | LoRA r1 | 1e-05 | 2.2 M | 0.2536 | 3.66 h | 150 |
| lr2 | lora-r16-all-math-lr1e-05-s0 | LoRA r16 | 1e-05 | 35.7 M | 0.2690 | 4.36 h | 150 |
| lr2 | lora-r256-all-math-lr1e-05-s0 | LoRA r256 | 1e-05 | 570.4 M | 0.1736 | 5.45 h | 150 |
| lr3 | full-na-na-math-lr3e-07-s0 | FullFT | 3e-07 | — | 0.2250 | 3.39 h | 150 |
| lr3 | lora-r1-all-math-lr3e-05-s0 | LoRA r1 | 3e-05 | 2.2 M | 0.0000 | 4.68 h | 150 |
| lr3 | lora-r16-all-math-lr3e-05-s0 | LoRA r16 | 3e-05 | 35.7 M | 0.2758 | 4.22 h | 150 |
| lr3 | lora-r256-all-math-lr3e-05-s0 | LoRA r256 | 3e-05 | 570.4 M | 0.2010 | 4.54 h | 150 |
| lr4 | full-na-na-math-lr7e-07-s0 | FullFT | 7e-07 | — | 0.2660 | 2.68 h | 150 |
| lr4 | lora-r1-all-math-lr7e-05-s0 | LoRA r1 | 7e-05 | 2.2 M | 0.2002 | 4.72 h | 150 |
| lr4 | lora-r16-all-math-lr7e-05-s0 | LoRA r16 | 7e-05 | 35.7 M | 0.2722 | 5.54 h | 150 |
| lr4 | lora-r256-all-math-lr7e-05-s0 | LoRA r256 | 7e-05 | 570.4 M | 0.2378 | 4.37 h | 150 |
| lr5 | full-na-na-math-lr2e-06-s0 | FullFT | 2e-06 | — | 0.1366 | 3.10 h | 150 |
| lr5 | lora-r1-all-math-lr0.0002-s0 | LoRA r1 | 2e-04 | 2.2 M | 0.0000 | 7.56 h | 150 |
| lr5 | lora-r16-all-math-lr0.0002-s0 | LoRA r16 | 2e-04 | 35.7 M | 0.1040 | 8.78 h | 150 |
| lr5 | lora-r256-all-math-lr0.0002-s0 | LoRA r256 | 2e-04 | 570.4 M | 0.0000 | 5.38 h | 150 |
| lr6 | full-na-na-math-lr4e-06-s0 | FullFT | 4e-06 | — | 0.0300 | 3.58 h | 150 |
| lr6 | lora-r1-all-math-lr0.0004-s0 | LoRA r1 | 4e-04 | 2.2 M | 0.0000 | 8.85 h | 150 |
| lr6 | lora-r16-all-math-lr0.0004-s0 | LoRA r16 | 4e-04 | 35.7 M | 0.0000 | 9.35 h | 150 |
| lr6 | lora-r256-all-math-lr0.0004-s0 | LoRA r256 | 4e-04 | 570.4 M | 0.0000 | 10.02 h | 150 |
| lr7 | full-na-na-math-lr1e-05-s0 | FullFT | 1e-05 | — | 0.0460 | 5.15 h | 150 |
| lr7 | lora-r1-all-math-lr0.001-s0 | LoRA r1 | 1e-03 | 2.2 M | 0.0000 | 9.21 h | 150 |

</details>

## Negative results and run history

### Math LR7 was deliberately stopped after the result was already decisive

The Math LoRA r16 `1e-03` arm was stopped around rollout 50/150. Its evaluations at
rollouts 24 and 49 were both zero with mean response length 2,048 and 100% truncation.
The r256 `1e-03` arm was not launched. Their absence from the successful ledger is
intentional; they are not scheduler failures and should not be silently counted as
unfinished successes.

### Collapse means runaway length, not merely a low endpoint

Across both datasets, unstable arms drive generations to the 2,048-token cap. A
truncated response often loses its final `\boxed{}` answer and grades zero. Sometimes
the box appears before repeated text, so accuracy remains nonzero even at 100%
truncation. This is why accuracy alone cannot label a high-LR arm healthy.

### Ten stale GSM8K failures remain in the raw ledgers

The GSM8K ledgers contain ten `failed` rows in addition to the 31 unique successful
rows: three in lr1, two each in lr2 and lr3, and one each in lr5–lr7. Some use the
superseded FullFT grid; others are earlier attempts with no post-training evaluation.
They do not conflict with any successful endpoint. The report filters `status == "ok"`
and keeps failed history separate.

Several launcher logs also contain appended partial attempts before the final complete
segment. The endpoint matrices come from unique successful ledger rows; trajectory
tables use the last complete 0–149 segment rather than concatenating attempts.

### W&B upload is not completion evidence

Training ran with `WANDB_MODE=offline`, targeting the personal entity `zeju-qiu` during
login-node synchronization. This report does not treat W&B availability as evidence
that training finished: completion is established by the successful ledger rows and
the final rollout-149 evals. Final server-side synchronization was not re-audited here.

## Interpretation

1. **Tuning matters by roughly two orders of magnitude.** Both datasets choose
   7e-07 for FullFT and 3e-05 for the best r16 LoRA arm, a 42.9× LR ratio. Comparing
   the methods at one shared LR would be badly confounded.
2. **The stable LR window is finite and now bracketed.** Both FullFT curves peak at
   7e-07; both LoRA panels deteriorate above the 1e-05–7e-05 region. The added LR0
   point verifies the low side rather than improving the optimum.
3. **Rank independence is dataset-dependent at this budget.** GSM8K's tuned LoRA
   endpoints span 0.7149–0.7551. Math spans 0.2378–0.2758 and selects a different LR
   per rank. Fixed α/r and only 150 updates are plausible contributors, but the sweep
   does not isolate them.
4. **Endpoint accuracy overstates some arms.** GSM8K r16/r256 at 2e-04 and Math r16
   at 7e-05 retain high accuracy in runaway-length states. A parity claim should use
   both answer accuracy and response-health metrics.
5. **The best defensible result is narrower than the headline.** Math r16 matches
   FullFT under the measured protocol. GSM8K healthy LoRA comes within 0.0455 of
   FullFT. The broader statement that all three LoRA ranks match FullFT is not supported
   by this single-seed, 150-update panel.

## Limitations and next steps

- Run multiple seeds at the stable optima: FullFT 7e-07; GSM8K LoRA 3e-05; Math
  LoRA r1/r16/r256 at 1e-05/3e-05/7e-05.
- Add an eval stop condition or treat response length and truncation as first-class
  acceptance metrics so a gradeable box followed by 2,000 repeated tokens cannot look
  healthy.
- Extend selected low-LR arms to 234 rollouts (one dataset epoch) to distinguish slow
  learning from a genuinely poor LR, especially LR0 r16/r256.
- Test an α schedule that holds α/r constant across ranks, or retune each rank on a
  denser local grid, before interpreting rank as adapter capacity.
- Check final W&B synchronization separately if the online dashboard is needed; the
  durable result record is the ledger-backed report, not the dashboard state.
- No checkpoints were written, so these exact trained policies cannot be re-evaluated
  under a new grader or generation policy.

## Provenance and closure

The authoritative runtime checkout was `/fast/zqiu/orbit-iclr/orbit` on
`feat/lora-without-regret` at `46c8e0f6d65a9630d3eff44d2db7c1e9dc38a18a`.
Tracked files were clean; the 16 result ledgers and unrelated working notes were
untracked. A single bounded ledger snapshot was taken through `mpi2`; supporting
trajectories were extracted read-only from `logs/lora_regret/`.

The final LR0 allocations recovered from Condor history were `17448192.0` and
`17448193.0`; their execution nodes were `i101` and `i108`. Earlier interactive job
IDs were not fully retained, so this report does not invent them. All Codex-managed
tmux sessions were closed after completion and those allocations were released.

Successful-row closure:

```text
GSM8K  lr0=3, lr1..lr7=4 each  -> 31 unique successes
Math   lr0=3, lr1..lr6=4 each, lr7=2 -> 29 unique successes
Math   lr7 LoRA r16 stopped after persistent zero; r256 not launched
All 60 successful rows: final step=149, accuracy present, status=ok
```
