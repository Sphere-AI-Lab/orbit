---
title: E4 gsm8k learning-rate column 4 — FullFT vs LoRA r1/r16/r256
kind: experiment
subtitle: First completed column of the Figure 6 gsm8k panel; one point on each of C5's four curves
date: 2026-08-08
tags: lora-regret, e4, gsm8k, rl
matrix: e4
column: lr4 of 7
dataset: gsm8k
seed: 0 (single seed)
hardware: 8×B200 (178.35 GB)
ledger: results/e4_gsm8k_lr4.jsonl
logs: logs/lora_regret/
wandb: gsm8k-rl-rank-ft (FullFT), gsm8k-rl-rank-lora (LoRA)
---

## Setup

RL fine-tuning of `llama3.1-8b` on `gsm8k_train.jsonl`, scored once at the end on
`gsm8k_test.jsonl` alone. The protocol is baked into `scripts/lora_regret/e4_protocol.sh`
and shared by all fourteen column scripts: advantage centring **without** std
normalisation, clipping off, checkpoints off, 150 rollouts (steps 0–149), one final eval.
Every arm runs on 8 GPUs; FullFT is TP=4/DP=2, the only configuration it fits in
(runbook §22.2).

FullFT and LoRA sit on separate learning-rate grids an order of magnitude apart
(runbook §23.4). Column 4 pairs the 4th point of each: FullFT at 7e-07 — on the grid
re-placed a decade lower by the first gsm8k pass (§23.4.1, commit `56ea497`) — against
LoRA at 7e-05. LoRA applies to all four linear sites
(`linear_qkv, linear_proj, linear_fc1, linear_fc2`). One seed (`s0`); no variance
estimate exists for these numbers.

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
bash scripts/lora_regret/run_e4_gsm8k_lr4_8gpu.sh
```

The script is resumable: arms with `status: "ok"` in the ledger are skipped on re-run.

## Results

All four arms completed 150 rollouts with `status: "ok"` by 2026-08-07 09:11.
Final-eval accuracy on `gsm8k_test`:

| arm | method | lr | adapter params | steps | accuracy | wall time |
|:----|:-------|----:|--------------:|------:|---------:|----------:|
| full-na-na-gsm8k-lr7e-07-s0 | FullFT | 7e-07 | — | 149 | **0.7870** | 2.14 h |
| lora-r1-all-gsm8k-lr7e-05-s0 | LoRA r1 | 7e-05 | 2.2 M | 149 | 0.6808 | 2.31 h |
| lora-r16-all-gsm8k-lr7e-05-s0 | LoRA r16 | 7e-05 | 35.7 M | 149 | 0.6694 | 2.85 h |
| lora-r256-all-gsm8k-lr7e-05-s0 | LoRA r256 | 7e-05 | 570.4 M | 149 | 0.6513 | 3.65 h |

The untrained base model scores **0.0326** on `gsm8k_test` (its rollout-0 eval), so the
trained arms span a 0.03 → 0.79 dynamic range.

## Observations

- At this column the ordering is FullFT > r1 > r16 > r256, and accuracy falls
  monotonically as rank grows. This is one LR point per method, not a peak comparison:
  every C5 claim is about the shape **across** columns, so no argmin or regret statement
  can be read from this column alone (the column script says exactly this).
- All three LoRA ranks sit at the same lr 7e-05; if the per-rank optima differ, the
  ordering among ranks can change once the full curves exist.
- Wall time grows with rank — late-run rollouts take ~50 s (r1), ~60–100 s (r16),
  ~90–140 s (r256) — so r256 costs ~1.6× r1 per column.

## Next steps

- [ ] Columns lr5–lr7 (gsm8k): not yet run.
- [ ] Columns lr1–lr3 (gsm8k): rerun required — their FullFT rows (5e-07, 1e-06, 3e-06)
  are on the superseded grid.
- [ ] Decide on the math panel after the gsm8k panel completes (gsm8k has ~6× math's
  dynamic range; math's baseline is partly guessing).
