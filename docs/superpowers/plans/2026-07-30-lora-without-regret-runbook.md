# LoRA-Without-Regret — runbook

How to run the campaign on reserved nodes. Every command here is meant to be
pasted; every number it expects back is stated so a wrong one is visible
immediately.

Companions:
- Claims, arms, and acceptance criteria: `docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md`
- Port status and what is verified: `docs/superpowers/plans/2026-07-29-lora-without-regret-gap.md`
- Environment build: `INSTALL.md`

## What is ready, and what you still have to do

Ready and CPU-verified (779 tests, 0 failures, in the built env):

| Piece | Where |
|---|---|
| SFT launcher — LoRA, OFT and FullFT in one script | `examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh` |
| RL launcher (prerequisite P5) | `examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh` |
| Arm matrices `e1` / `e1ot` / `e1short` / `e2` / `e3` / `e4` / `e4place` / `e5scout` / `e5` / `sft82` | `tools/lora_regret/arms.py` |
| Base-model registry (checkpoints, dimensions, GPU floors) | `tools/lora_regret/models.py` |
| Sweep driver with resume ledger | `tools/lora_regret/sweep.py` |
| Data preparation for all five datasets | `tools/lora_regret/prepare_data.py` |
| Held-out token-weighted NLL eval | `orbit/utils/eval_nll.py`, wired in `train.py` |
| Preflight audit | `tools/lora_regret/preflight.py` |
| P3 DP-equality check | `tools/lora_regret/p3_check.py` |
| NLL trace extraction | `tools/lora_regret/trace.py` |
| σ, argmins, C1-C6 and C8 readings | `tools/lora_regret/analyze.py` |
| Figures from the analysis JSON | `tools/lora_regret/plot.py` (§19) |
| Coverage probe: one run per task per method | `scripts/lora_regret/coverage_probe.sh` (§20) |

The data is **materialized** (§2, done 2026-07-30) and the **smoke passes on it**
(§3, done 2026-07-30). Yours to do, in this order: **close P3** (§4, needs DP≥2 and
gates every FullFT number), then run experiments in the order §5 lays out.

Single-rank reachability is proven twice: on a 100-row fixture (2026-07-29) and on
the **real 1,000-row Tulu3 held-out split** (2026-07-30,
`logs/smoke_lora_r256_20260730_150952.log`) — two LoRA-r256 optimizer steps, NLL
1.209810 → 1.199709 → 1.194836, `tokens=308760 samples=1000` identical at all three
measurements, adapter written with `r=256 alpha=32`, exit 0. What has **never**
executed is the DP>1 reduction and anything multi-GPU. Treat §4 as mandatory.

## 1. Environment

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source env.sh                                  # LD_LIBRARY_PATH, z3 soname
source examples/load_cuda13_2_orbit_env.sh      # cudnn / flashinfer runtime
```

Activate **before** sourcing `env.sh`, so `$VIRTUAL_ENV` points it at the right
site-packages. `env.sh` is required even for CPU-only work: `megatron.core`
imports `deep_ep`, which asserts on an unset `CUDA_HOME`.

Confirm the env is real, not a skeleton of broken symlinks (that failure mode
imports *successfully* — see the gap plan's failure signature):

```bash
python -c "import torch, transformers, megatron.core as m; assert torch.__file__; print(torch.__version__, transformers.__version__, m.__version__)"
# expect: 2.11.0+cu130 4.57.1 0.18.0rc0
```

## 1.5 Preflight — run this first, every time

```bash
python -m tools.lora_regret.preflight --stage e1-lora
```

Checks the four imports have real files behind them (the dangling-symlink venv
imports *successfully*), the GPU count against the stage, both checkpoints, all
nine splits **at their row counts**, and that every matrix builds. Exits 1 with
the specific failure. Pass `--skip-gpu` to run it from a login node.

## 2. Materialize the datasets (CPU, no GPU, network required)

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
python -m tools.lora_regret.prepare_data --dataset campaign --out-dir "$DATA_DIR"
```

Expected row counts, all asserted by the tool — a mismatch raises and leaves the
previous file untouched rather than writing a short split:

**Materialized and verified on 2026-07-30** — these are measured counts, not
expectations:

| Output | Rows | Size | For |
|---|---|---|---|
| `tulu3_train.jsonl` / `tulu3_test.jsonl` | 938,343 / 1,000 | 2.95 GB | E1, E3 |
| `openthoughts3_train.jsonl` / `_test.jsonl` | 10,000 / 100 | 0.62 GB | E2 |
| `math_train.jsonl` / `math_test.jsonl` | 7,498 / 5,000 | 5 MB | E4 |
| `gsm8k_train.jsonl` / `gsm8k_test.jsonl` | 7,473 / 1,319 | 3 MB | E4 |
| `math_gsm8k_train.jsonl` | 14,971 | 8 MB | E4 (the launcher's `--prompt-data`) |

Every file was re-read afterwards and checked for row count, schema
(`{"prompt": [messages]}` for SFT, `{"prompt": str, "label": str}` for RL), absence
of the two Llama control-token literals, and absence of empty labels. All nine
passed.

**MATH is 7,498 rather than the official 7,500.** Two `number_theory` train rows
end `there are $\boxed{}$ primes` — a literally empty box where the answer is 0 —
and an empty label can never be earned honestly, while `grade_answer_verl(response,
"")` may match a model that also emits an empty box and reward it for saying
nothing. They are dropped and reported as `filtered=2`; the *source* count 12,500
is still asserted, so a changed dataset still fails loudly.

**Tulu3's control-token scan came back clean: `filtered=0`, across all 939,343
rows.** That closes the pre-sweep requirement carried over from the llama3
loss-mask plan, which had only ever been checked against the 12-row fixture. No row
carries a literal `<|start_header_id|>assistant<|end_header_id|>` (which would raise
mid-sweep) or `<|eot_id|>` (which would silently truncate a scored span), so the E1
denominator is the full 938,343.

The CLI prints `filtered=` / `assistant_header=` / `eot=` per dataset. Those count
rows whose assistant content contains a literal
`<|start_header_id|>assistant<|end_header_id|>` (which makes the llama3 mask
generator raise, killing a multi-hour run partway) or a literal `<|eot_id|>`
(which silently truncates the scored span — no error, corrupted spans). They are
removed before either split is written. **Record the counts**: they change the
denominator of every E1 number.

`--dataset campaign` appends a boxed-answer instruction to the MATH and GSM8K
prompts. Do not pass `--no-answer-instruction` unless you also change
`RM_TYPE`: the reward is `boxed_math`, which strips `\boxed{...}` from the
response before grading, and a Llama-3.1 *base* policy does not box unprompted —
every rollout would score 0 and every E4 arm would look identical.

**Read the count from the hub before a long stream, not after.** Tulu3's
assertion fires only once all 2.9 GB have been streamed, so a stale constant costs
the whole download:

```bash
python -c "
from datasets import load_dataset_builder
print({k: v.num_examples for k, v in load_dataset_builder('allenai/tulu-3-sft-mixture').info.splits.items()})"
# expect {'train': 939343} -- what TULU3_EXPECTED_ROWS is pinned to (verified 2026-07-30)
```

If it has moved, that is a question about the dataset before it is a number to
bump: the assertion exists to notice a changed mixture. OpenThoughts3 needs no such
check — it takes an exact 10,000/100 subset off the front of a 1.2M-row stream and
stops there, so nothing depends on the total.

## 3. Smoke one arm (1 GPU, ~10 minutes)

```bash
NUM_ROLLOUT=2 TRAIN_ROWS=64 \
LAUNCHER_NAME=smoke_lora_r256 \
SAVE_DIR=/lustre/fast/fast/zqiu/tmp/smoke_ckpt_20260730 \
EVAL_NLL_INTERVAL=1 \
bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh
```

Use `NUM_ROLLOUT`, not `SFT_EXTRA_ARGS="--num-rollout 2"`: the launcher already
puts `--num-rollout` in `ROLLOUT_ARGS`, and passing it twice leaves which one
wins up to argument-array ordering in `scripts/lib/launcher.sh`. `TRAIN_ROWS=64`
is only there to skip a `wc -l` over the 2 GB Tulu3 file, which the launcher
otherwise runs to derive its own `NUM_ROLLOUT` default.

Then confirm the eval line the sweep parses actually appears:

```bash
grep -c 'eval/test_nll' logs/smoke_lora_r256_*.log     # expect >= 2
```

**Result, 2026-07-30** (`logs/smoke_lora_r256_20260730_150952.log`, exit 0):

```
before_train  step 0   nll=1.209810  sample_mean=1.478078  tokens=308760  samples=1000
after_train   step 0   nll=1.199709  sample_mean=1.455645  tokens=308760  samples=1000
after_train   step 1   nll=1.194836  sample_mean=1.421378  tokens=308760  samples=1000
progress rollout=1/1 completed=2/2 remaining=0   elapsed=00:06:47
shutdown: dispose rollout done
```

`tokens=` and `samples=` must be **identical at every measurement** — a drifting
`samples=` means the held-out set is being floor-divided by the batch size (1,000
rows at global batch 32 would silently become 992), which makes the metric depend
on batch size and breaks E2 specifically. They were identical here, on the real
1,000-row split rather than a fixture.

Two further checks worth repeating after any change to the eval path or the
launcher's PEFT block, both of which unit tests cannot make:

```bash
# the sweep's parser reads the real log, not just synthesized lines
python -c "
from tools.lora_regret.sweep import parse_final_nll
print(parse_final_nll(open('logs/smoke_lora_r256_20260730_150952.log', errors='replace').read()))"
# -> (1.194836, 1): the highest-step after_train row

# the LoRA flags actually reached the adapter
python -c "
import json; print(json.load(open('<SAVE_DIR>/iter_0000001/adapter/adapter_config.json')))"
# -> peft_type=LORA r=256 lora_alpha=32 lora_dropout=0.0, all seven projections
# (the 07-30 run's SAVE_DIR was /lustre/fast/fast/zqiu/tmp/smoke_ckpt_20260730)
```

**Check the save is whole, not just present.** The 2026-07-29 fixture smoke left a
**truncated** adapter under `.../tmp/smoke_ckpt/` — 32 of 256 data records, 142 MB
where r256 all-modules is 1.14 GB — which `ls` cannot tell from a good one. The
cheap discriminator is a load, and it doubles as the parameter-count check E3 and
E5 depend on:

```bash
python -c "
import torch
sd = torch.load('<SAVE_DIR>/iter_0000001/adapter/adapter_megatron_tp0_pp0.pt', map_location='cpu')
print(len(sd), sum(v.numel() for v in sd.values()))"
# -> 256 570425344   for LoRA r256 all-modules on Llama-3.1-8B (32 layers)
```

**A quiet log is not a stalled run.** The launcher tees into `RUN_LOG` through a
buffer, so `grep` on it can lag the run by minutes — during this smoke the
`before_train` line existed at 15:16:12 but was not yet greppable at 15:19. Use
`tail -f`, or check `nvidia-smi` and the process, before concluding anything is
wedged.

## 4. Close P3 before trusting any FullFT number (needs DP ≥ 2, ideally 4)

The held-out NLL reduces `(sum_neg_logprob, n_tokens)` over the **DP group
only** — TP/PP replicas hold identical samples, DP shards hold different token
counts. That code has never run at DP>1, and P0 forces DP>1 for every FullFT
arm. So measure it:

```bash
# DP=1
GPUS_PER_NODE=1 LAUNCHER_NAME=p3_dp1 SAVE_DIR=/lustre/.../p3_dp1 \
  NUM_ROLLOUT=3 EVAL_NLL_INTERVAL=1 \
  bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh

# DP=4, same seed, same arm
GPUS_PER_NODE=4 LAUNCHER_NAME=p3_dp4 SAVE_DIR=/lustre/.../p3_dp4 \
  NUM_ROLLOUT=3 EVAL_NLL_INTERVAL=1 \
  bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh

python -m tools.lora_regret.p3_check logs/p3_dp1_*.log logs/p3_dp4_*.log
```

**Acceptance is the exit code.** It pairs measurements by `(phase, step)` and
asserts `nll` equal to six decimals with `tokens` and `samples` exactly equal.
A differing `tokens` means the reduction is double-counting or dropping a shard;
it says so, and the correct response is to stop — every FullFT number downstream
is wrong, and no amount of averaging fixes it.

## 4.5 Where the runs show up in wandb

**One project per task, one group per method.** The project name spells out
`<dataset>-<sft|rl>-<what is tested>`, so the sidebar is readable without the
plan in hand:

| `--matrix` | wandb project | groups inside it |
|---|---|---|
| `e1` | `tulu3-sft-rank` | full, lora |
| `e1long` | `tulu3-sft-curves` | full, lora |
| `e1short` | `tulu3-sft-lr-horizon` | full, lora |
| `e1ot` | `openthoughts3-sft-rank` | full, lora |
| `e2` | `openthoughts3-sft-batch` | full, lora |
| `e3` | `tulu3-sft-placement` | lora |
| `e4` | `math-gsm8k-rl-rank` | full, lora |
| `e4place` | `math-gsm8k-rl-placement` | lora |
| `e5scout` | `tulu3-sft-oft-scout` | oft |
| `e5` | `tulu3-sft-oft-match` | lora, oft |
| `sft82` | `tulu3-sft-bracket` | full, lora, oft |

The run name is the arm name, so `oft-b64-all-lr0.0001-s0` is readable without
opening it. The `<dataset>-<sft|rl>` head is asserted against each matrix's own
arms by `test_the_project_name_describes_the_arms_it_routes` — a name is a claim
about the runs inside it, and a project called `tulu3-sft-*` holding
OpenThoughts3 RL arms would be a worse lie than an opaque code.

The split is per *matrix* rather than per claim because the matrix is the unit
you schedule, resume and re-run: one project is one `--results` ledger is one
`analyze` invocation. Pooling them — which is what the launchers' single default
project does — would put E1's rank ladder, E3's placement pair and E5's OFT arms
in one flat namespace of 112 runs, where the run deciding C2 is indistinguishable
in the sidebar from the one deciding C6.

Every ledger row now carries `wandb_project` and `wandb_group`, so a number read
months later can be traced back to the dashboard it came off.

Hand-run arms (§3's smoke, §4's P3) do not go through the sweep and keep the
launcher's own `WANDB_PROJECT=lora-without-regret` default. Calling `run_arm`
directly without a matrix lands there too — where a hand-run arm lands, rather
than silently inside a task whose numbers you are quoting.

## 5. Execution order for E1-E5

Every stage below is gated by the ones above it. The gating is real, not
bureaucratic: each entry names what breaks if you skip it.

| # | Stage | Runs | GPUs | Gated by | Produces |
|---|---|---|---|---|---|
| §2 | ~~Materialize data~~ **done** | — | none | — | the nine splits every arm reads |
| §3 | ~~Smoke one arm~~ **done** | 1 | 1 | data | proof the eval line the parser needs is reached |
| §4 | P3: DP=1 vs DP=4 | 2 | 1 and 4 | smoke | permission to trust any FullFT number |
| §7 | **E1-0: σ** | 3 | 1 | data | the unit every later difference is quoted in |
| §8 | E1-1: LR sweeps | 45 | 1 / ≥4 | σ, P3 | argmins → **C2** |
| §8 | E1-2: long curves | 8 | 1 / ≥4 | E1-1's argmins (via --argmins-from) | departure steps → **C1** |
| §9 | E2: batch sweep | 48 | 1 / ≥4 | σ, P3 | best-per-batch gaps → **C3** |
| §10 | E3: placement | 35 | 1 | σ | matched-parameter deltas → **C4** |
| §16 | E1-OT: OpenThoughts3 rank ladder | 47 | 1 / ≥4 | σ(OT3) | curves + argmins on the second dataset → **C1/C2** |
| §17 | E1-short: 100-step multiplier | 21 | 1 / ≥4 | σ | short-vs-long LR ratio → **C8** |
| §18 | E4-place: RL layer placement | 20 | 8 | data, P3 | attention-vs-MLP under policy gradient → **C4** |
| §11 | E4: RL | 20 | 8 | data, P3 | accuracy curves + band width → **C5** |
| §12 | E5-1: OFT scout | 5 | 1 | σ | the OFT learning-rate decade |
| §12 | E5-2: OFT refine | 50 | 1 | the scout's argmin | OFT-vs-LoRA at matched params → **C6** |

**302 runs** (3 + 45 + 8 + 48 + 35 + 47 + 21 + 20 + 20 + 5 + 50), plus 3 preflight — one smoke
(done) and two for P3. One of E1-0's three seeds repeats an E1-1 arm; §7 says how to avoid
that if you care about the one run. E1-OT's 47 is 45 grid points plus the two
extra seeds that measure OpenThoughts3's own σ (§16) — Tulu3's does not transfer.

**60 of those 302 are the method-coverage arms** added so every task carries
FullFT, LoRA and OFT (§4.5). They decide no claim on their own: the FullFT arms
in E3 and E4-place are reference lines duplicating grids E1 and E4 already run,
and every OFT cell is an `oftscout` search until §12's scout produces a centre.
Drop them first under budget pressure — the claims survive; only the dashboards
get thinner.

Three orderings inside that are load-bearing rather than conventional:

**σ before everything it is quoted against.** E1-0 is three seeds, and it is the
first measurement rather than a footnote: every claim in the campaign is a
difference stated in units of σ, and the Qwen3-era σ = 0.000992 does not transfer
to a different model and dataset. Running the sweeps first and σ afterwards works
arithmetically and fails in practice — you will have read the results already.

**E1-1 before E1-2.** The long single-epoch runs that show C1's departure points
are only worth doing at each rank's *own* argmin LR, which E1-1 is what finds. Run
them at a shared LR and a rank that departs early is indistinguishable from a rank
whose LR was simply too high.

**The scout before the refinement.** `--matrix e5` refuses to start without
`--oft-lr-centre` for this reason; see §12.

E4 and E5 are independent of E1-E3 and of each other, so they can run on separate
allocations in parallel. E2 and E3 both depend only on σ. Within a stage, the
`--only` splits in each section are what parallelize it.

## 6. Cost arithmetic — read this before launching E1

One epoch of Tulu3 at the launcher's default batch is **29,323 optimizer
steps**:

```
(939,344 − 1,000 held out) / 32 = 29,323 rollouts, one optimizer step each
```

Forty of those arms is not a weekend. The plan's own structure splits cleanly,
and this is the recommended split:

- **E1-1 (the LR argmins, C2)** does not need a full epoch. An argmin is stable
  once the arms have separated; cap it. `NUM_ROLLOUT=2000` (≈6.8% of an epoch,
  64k samples) is the suggested starting point, and it is a *decision to record*,
  not a default to hide — quote it next to every ratio.
- **E1-2 (the learning curves, C1)** does need the long run, but only at each
  rank's own argmin LR: 8 arms, not 40. Departure points are what C1 is about,
  and a departure that happens at step 20,000 cannot be seen in 2,000.

Set the eval interval to roughly 1% of the run so the NLL trace has ~100 points
without dominating wall clock: `EVAL_NLL_INTERVAL=$((NUM_ROLLOUT / 100))`.

**Measured on this H100 by the §3 smoke, so the estimate is no longer a guess:**

| Quantity | Measured |
|---|---|
| training, steady state | **8.5 s/step** (104.7 TFLOPs, 2,327 tok/s) |
| training, first step | 47.4 s — kernel autotuning, a per-*arm* cost, not per-step |
| held-out eval, 1,000 Tulu3 rows | **67.7 s** |
| model build + Ray startup | ~85 s + ~200 s |

That makes a 2,000-rollout E1 arm ≈ **4.7 h of training + 1 h 53 min of eval** at
`EVAL_NLL_INTERVAL=20` — 2,000/20 = 100 measurements × 67.7 s = 6,770 s. The eval
is ~29% on top of the training, not a rounding error: **6.6 h per arm, ≈265
GPU-hours across E1-1's 40 arms.** The same arm at interval 1 would spend ~37 h
evaluating (2,000 × 67.7 s) — the smoke ran at `wait_time_ratio=0.89`, i.e. 89% of
its wall clock in the eval, which is right for a smoke and ruinous for an arm.
This is the single cheapest knob to get wrong.

Batch sizes per dataset, for reference: OpenThoughts3's 10,000 rows are 312
steps at batch 32 and 19 at batch 512 — E2 is cheap, E1/E3 are not.

## 7. E1-0 — measure σ, before anything is quoted against it

The σ = 0.000992 nats in the gate log was measured on **Qwen3-4B / No Robots**
and does not transfer to Llama-3.1-8B / Tulu3. Everything downstream is quoted
in units of σ, so measure it before quoting anything:

Drive it through the sweep rather than by hand, so the three NLLs land in a
ledger in the format §13 reads instead of being grepped out of three logs.

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
export NUM_ROLLOUT=2000 EVAL_NLL_INTERVAL=20 GPUS_PER_NODE=1

for seed in 0 1 2; do
  python -m tools.lora_regret.sweep --matrix e1 --seed $seed \
    --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
    --only 'lora-r256-all-lr0.00025' --results results/e1_0_sigma.jsonl
done
```

Each invocation selects exactly one arm (`--dry-run` first to see it). σ is the
standard deviation of the three `test_nll` values in `results/e1_0_sigma.jsonl`.

The seed-0 replicate is the same configuration as one of E1-1's 40 arms, and the
resume ledger is **per `--results` file** — so it does get run twice across the two
stages. If that matters, point these three at the same file as the E1-1 shard that
covers r256 (`results/e1_lora_c.jsonl` below) and the second invocation will skip
it; the shards are sequential with respect to E1-0, so there is no concurrent
append. Separate files are the default here because the provenance is easier to
read afterwards, and the cost is one run in 178.

Until σ exists, treat any difference under ~0.002 as unresolved. `SEED` is tied to
`ROLLOUT_SEED` inside the launcher, so each replicate varies data order as well
as initialization — which is what a seed replicate should vary.

## 8. E1 — capacity, rank, and the 10x LR rule (C1, C2)

40 arms: FullFT plus LoRA r ∈ {1, 4, 16, 64, 128, 256, 512}, five LRs each at
0.3-decade spacing, centred on 2.5e-5 (FullFT) and 2.5e-4 (LoRA — exactly 10x,
which is C2's prediction built into the grid rather than fitted out of it).

Inspect before spending anything:

```bash
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --num-layers 32 --dry-run | wc -l   # 40
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --num-layers 32 --dry-run | head -3
```

**LoRA arms — one GPU each.** Three-way concurrency on one card is measured
safe **for Qwen3-4B** (three runs, 26 min wall clock), and that measurement does
not transfer: Llama-3.1-8B holds 16 GB of frozen bf16 weights per process, so
three processes spend 48 GB of an 80 GB card before any activations. Run two
concurrently first and watch `nvidia-smi` before trusting three. Use disjoint
`--only` regexes and **separate results files**, one per shell, so two processes
never append to the same ledger:

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
export NUM_ROLLOUT=2000 EVAL_NLL_INTERVAL=20 GPUS_PER_NODE=1

python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --only '^lora-r(1|4)-'   --results results/e1_lora_a.jsonl &
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --only '^lora-r(16|64)-' --results results/e1_lora_b.jsonl &
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --only '^lora-r(128|256|512)-' --results results/e1_lora_c.jsonl &
wait
```

**FullFT arms — ≥4 GPUs**, on the allocation that has them:

```bash
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e1 \
  --hidden-size 4096 --ffn-size 14336 --num-layers 32 --only '^full-' --results results/e1_full.jsonl
```

The launcher **refuses** `PEFT_METHOD=none` below 4 GPUs and prints the
arithmetic (32 GB + 96 GB/N per GPU for optimizer state alone, so N=1 is 128 GB
and N=2 leaves nothing for activations). `ALLOW_SMALL_FULLFT=1` overrides it if
you want to watch it OOM.

The sweep is resumable: a completed arm appends `status: "ok"` to the ledger and
is skipped on the next invocation; a failed arm is retried. Killing and
restarting is safe.

**E1-3, reading C2 off the result:** `argmin_LR(LoRA r256) / argmin_LR(FullFT)`.
The post predicts 9.8, rising toward 15 for runs under ~100 steps. Also check
the tighter claim — optimal LR moves less than 2x between rank 4 and rank 512 —
which costs nothing extra since both arms are already in the sweep.

**Acceptance:** any arm whose argmin sits on a grid edge is **re-run on a
re-centred grid** before its ratio is quoted. Extend nothing; re-centre.

### E1-2 — the long curves that decide C1

Eight runs, one per arm, each at **that arm's own argmin LR** from E1-1 and each a
full Tulu3 epoch (29,323 steps — `NUM_ROLLOUT` unset so the launcher derives it).
This is the expensive stage, and it is eight runs rather than forty precisely
because E1-1 has already located the LRs.

```bash
python -m tools.lora_regret.sweep --matrix e1long \
  --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --argmins-from 'results/e1_*.jsonl' \
  --results results/e1_2.jsonl
```

E1-2 goes through the same driver as every other stage, so it gets the per-arm
`SAVE_DIR`, the resume ledger and uniform result records — which matter most
here, on 70-hour arms.

`--argmins-from` fails closed twice. Fewer than 8 arms recovered means a partial
E1-1 ledger, and running the 3 that happen to be there would look like a
completed stage. An argmin on a grid edge means the LR is a boundary value
rather than an optimum, and E1-2 is the most expensive place in the campaign to
act on an unchecked number — `--allow-edge-argmin` overrides it if you have
decided otherwise.

`NUM_ROLLOUT` is set to the **empty string** by these arms, not omitted: the
launcher's `${NUM_ROLLOUT:-...}` re-derives the full epoch on an empty value,
which also immunises the stage against a `NUM_ROLLOUT=2000` left exported in
your shell from E1-1. `EVAL_NLL_INTERVAL` is 293, ~1% of the epoch.

**E1-2, reading C1:** plot loss against log-steps per rank. Report, per rank, the
**step at which it departs** from the FullFT/high-rank envelope — the first step
where it exceeds the pointwise minimum across arms by more than 2σ for three
consecutive logging intervals. The claim predicts the departure step increases
monotonically with rank, and that high ranks do not depart at all within the
epoch. A rank that never departs and a rank whose run was too short look identical,
so state the step budget next to every departure point.

## 9. E2 — batch-size sensitivity (C3)

36 arms on the post's own setup: a 10,000-example OpenThoughts3 subset at batch
32, 128 and 512, for FullFT, LoRA r256 and LoRA r16.

```bash
python -m tools.lora_regret.sweep --matrix e2 --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --only '^lora' --results results/e2_lora.jsonl                       # 24 arms
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e2 \
  --hidden-size 4096 --ffn-size 14336 --num-layers 32 --only '^full' --results results/e2_full.jsonl   # 12 arms
```

E2 sets `GLOBAL_BATCH_SIZE` **and** `ROLLOUT_BATCH_SIZE` together and points
`TRAIN_JSONL`/`TEST_JSONL` at OpenThoughts3 automatically — nothing to export.
Each cell's four LRs are re-centred by √(batch/32); the edge-of-grid rule from §8
still applies, and it will fire more often here because the batch-size optimum is
less well predicted than the rank one.

E2 is the cheapest of the three SFT experiments: 10,000 rows is 312 optimizer
steps at batch 32 and 19 at batch 512, against E1's 29,323 for a Tulu3 epoch.

**E2-3, reading C3:** report `best_LoRA(batch) − best_FullFT(batch)` at each batch
size in units of σ. The claim is a gap that *grows* with batch — a gap that is
absent at 32 and present at 512 is the signature; a constant offset at all three
is not. **E2-2** is the rank-independence half: the post blames the
parametrization rather than capacity, so the gap must survive the change from
r256 to r16. If it shrinks with rank, the post's mechanism is wrong and that is
the finding.

## 10. E3 — layer placement at matched parameter count (C4)

20 arms on Tulu3, one GPU each.

```bash
python -m tools.lora_regret.sweep --matrix e3 --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --results results/e3.jsonl
```

E3's matched pair is **attention r256 against MLP r92**, solved for Orbit's fused
layout by `orbit.utils.peft_param_match.matched_mlp_rank` — per layer `18432·r`
for attention against `51200·r` for MLP, a ratio of 2.778. The post's own pair
(attention r256 / MLP r128) is also in the matrix, deliberately: if the two
disagree, the disagreement is parameter accounting rather than physics. Print
realized adapter parameter counts next to every arm before believing either.

**E3-2, reading C4:**

```bash
python -m tools.lora_regret.analyze c4 \
  --ledgers results/e3.jsonl --sigma-ledger results/e1_0_sigma.jsonl
```

Reports both halves: `NLL(attn) − NLL(mlp)` at matched parameters in units of σ,
then `NLL(all-modules) − NLL(mlp)`. Mind that the two are read in opposite
directions. The first upholds the claim when it is **positive** (attention-only
is worse); the second upholds it when it is near zero or positive, because the
claim is that all-modules *adds nothing* — a delta below −2σ is all-modules
winning by more than noise, and the tool labels that row `CONTRADICTS`.

**E3-3 (optional, the MoE arm)** is not wired: Qwen3-30B-A3B needs its own
model-args plugin and ≥4 GPUs for activations alone, and the post applies LoRA per
expert at rank = total/8. Skip it and say so, rather than substituting the dense
result.

## 11. E4 — RL parity at low rank (C5)

16 runs: FullFT, LoRA r256, r16, **r1** — rank 1 is the claim's whole point and
is not the arm to drop under budget pressure. The `e4` matrix drives them through
the same sweep driver, which knows to shell out to the RL launcher and to score
arms by **accuracy instead of NLL**:

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret

# Look first. The header line names the launcher and the metric.
python -m tools.lora_regret.sweep --matrix e4 --hidden-size 4096 --ffn-size 14336 --num-layers 32 --dry-run | wc -l   # 16

# LoRA arms
GPUS_PER_NODE=8 python -m tools.lora_regret.sweep --matrix e4 \
  --hidden-size 4096 --ffn-size 14336 --num-layers 32 --only '^lora' --results results/e4_lora.jsonl

# FullFT arms
GPUS_PER_NODE=8 python -m tools.lora_regret.sweep --matrix e4 \
  --hidden-size 4096 --ffn-size 14336 --num-layers 32 --only '^full' --results results/e4_full.jsonl
```

The grid is **half-decade** here, not E1's 0.3: the post gives a LR multiplier
for SFT and none for policy gradient, and C5's second half is about the *width*
of the performant band, which needs coverage more than resolution. LoRA is still
centred a decade above FullFT (1e-5 against 1e-6) as a prior carried over from
C2 — if the RL argmins disagree, that is a finding, not a grid error.

`accuracy` in the ledger is the mean of the per-dataset scores at the highest
rollout id, with `accuracy_per_dataset` alongside it. With `--rm-type boxed_math`
the reward is exactly 1 or 0, so a per-dataset score *is* accuracy on that split.
An eval where only one of the two datasets reported is **skipped, not averaged** —
a mean over a different set of splits than every other arm's is not comparable,
so it is not a number.

Defaults worth knowing: 32 samples per problem (the post's setting, and what
makes the GRPO baseline a per-problem mean rather than noise), constant LR, zero
weight decay, **KL and entropy coefficients zero**. Both are extra forces whose
strength interacts with the learning rate — the axis E4 sweeps — and a KL
penalty additionally pulls every arm toward the same reference policy, which is
exactly the between-arm difference C5 is about. Turn them on only if an arm
diverges, and say so when you report it.

**E4-3** reads validation-accuracy curves off `math_test` and `gsm8k_test`, plus
the *width* of the performant LR band per arm — the post claims LoRA's is wider,
which is a separate checkable statement from peak parity. σ for accuracy has
never been measured; if the curves sit close, measuring it becomes a
prerequisite exactly as E1-0 was for NLL.

## 12. E5 (optional, ours) — matched-parameter OFT

Only after C1-C5 are settled. Two matrices, and the second **cannot run until the
first has**, because OFT parameterizes a rotation rather than an additive update:
no LoRA learning rate transfers to it, not even the decade.

```bash
# Scout: 5 arms, half a decade apart, one block size (64).
python -m tools.lora_regret.sweep --matrix e5scout --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --results results/e5_scout.jsonl

# Refine: 50 arms, centred on the scout's argmin. Substitute the real number.
python -m tools.lora_regret.sweep --matrix e5 --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --oft-lr-centre 1e-4 --results results/e5.jsonl
```

Omitting `--oft-lr-centre` exits 2 and tells you to scout first; passing it to any
other matrix also exits 2, so it cannot silently look as though E1 were re-centred.

**How the matching works, and why not the obvious way.** A single global
`--oft-block-size` *cannot* match LoRA's parameter count across mixed shapes, and
this is arithmetic rather than tuning: OFT's count is `d_in·(b−1)/2` and ignores
`d_out` entirely, while LoRA's is `rank·(d_in + d_out)`. On Llama-3.1-8B at
`b = 64` the realized per-module ratios are 0.787 (`linear_qkv`), 0.984
(`linear_proj`), **0.246** (`linear_fc1`, whose fused gate+up makes
`d_out = 7·d_in`) and **1.531** (`linear_fc2`). Searching every divisor of 4096
and 14336 does not fix it — the best achievable all-modules ratio converges to
**0.764** for rank ≥ 4 — and Megatron takes one integer, so per-module block sizes
are not expressible either.

So E5 inverts the match: fix the block size, then solve for the **LoRA rank** with
the same parameter count. Rank is a much finer lattice than the divisors of `d_in`,
which brings every pair within a few percent:

| Axis | OFT arm | LoRA partner | realized ratio |
|---|---|---|---|
| capacity | all-modules b=32 | all-modules r6 | 0.988 |
| capacity | all-modules b=64 | all-modules r12 | 1.004 |
| capacity | all-modules b=256 | all-modules r49 | 0.995 |
| placement | attention-only b=64 | attention-only r14 | 1.000 |
| placement | MLP-only b=28 | MLP-only r5 | 1.004 |

The placement rows are a 2×2 of {OFT, LoRA} × {attention, MLP} **all at one
capacity**: the MLP block size is solved to match attention's realized count
(28, not 64) rather than reused, because OFT's count follows `d_in` and the MLP's
`d_in` sum is larger. Reusing 64 there would compare placement *and* capacity at
once — E3's mistake, one method over.

Every arm carries its `matched_ratio` into the ledger. Quote it, and mind the
direction: an OFT arm carrying slightly *fewer* parameters that still keeps up
strengthens the finding, while one carrying fewer and losing is confounded rather
than informative.

`OFT_BLOCK_SIZE` has **no default** in either launcher — a missing value fails at
launch instead of quietly comparing unmatched models.

Note that `sft82`'s own 40 OFT arms are **not** this design: they solve the block
size from the square attention shape (so all-modules lands at ratio 0.75) and put
35 of the 40 on LoRA's LR grid, which the campaign plan explicitly says is not
justified for OFT. Prefer `e5scout` + `e5`. `sft82` stays as-is because the gate
log records its dry run.

## 13. Reading results

The ledger is one JSON object per arm:

```json
{"arm": "lora-r256-all-lr0.00025-s0", "method": "lora", "rank": 256, "lr": 0.00025,
 "test_nll": 1.845700, "steps": 2000, "status": "ok"}
```

`test_nll` is the **last `phase=after_train` measurement**, chosen by highest
`step` rather than by file position, so an interleaved multi-rank log cannot make
a `before_train` row win. `status: "failed"` means either a non-zero exit or no
parseable metric — check the arm's log under `logs/lora_regret/<arm>.log`.

E4 arms carry `"metric": "accuracy"` instead, with `test_nll: null`:

```json
{"arm": "lora-r1-all-lr1e-05-s0", "method": "lora", "rank": 1, "lr": 1e-05,
 "metric": "accuracy", "accuracy": 0.44,
 "accuracy_per_dataset": {"math_test": 0.33, "gsm8k_test": 0.55},
 "steps": 100, "status": "ok"}
```

`steps` is the rollout id of that eval, not an optimizer step count — the RL eval
line does not carry a step, because `rollout.py` assigns `eval/step` *after* the
log call.

Argmins per arm:

```bash
python -m tools.lora_regret.analyze argmins --ledgers 'results/e1_*.jsonl'
python -m tools.lora_regret.analyze all \
  --ledgers 'results/e1_*.jsonl' --sigma-ledger results/e1_0_sigma.jsonl
```

The seed-0 filter, the edge-of-grid rule and the σ units are built in rather
than restated per reading. `argmins` marks edge-of-grid arms and still prints;
every *claim* subcommand exits 3 rather than quoting one, unless
`--allow-edge-argmin` is passed.

`--json` emits one JSON document on stdout and nothing else — the handoff to
plotting, and the only form worth piping. Exit codes do not change with it: a
refused edge-of-grid argmin still exits 3, with the reason inside the payload
under `edge_of_grid` and the refused claim simply absent, so a consumer cannot
plot a number the tool declined to quote.

The `seed != 0` filter is not cosmetic. E1-0's replicates are the same
configuration at seeds 1 and 2, and they are not grid points; measured on a
synthetic ledger, dropping the filter let a replicate at LR 9.95e-4 win r256's
argmin away from the real 2.5e-4 purely because that one run happened to score
better. Grid points are seed 0 only.

Then quote every difference in units of the σ from §7, and never off absolute
loss values — the constant Orbit-vs-HF precision offset (0.0032 nats, inside
HF's own 0.0072-nat bf16/fp32 spread) cancels in every ratio, ordering and
curve-shape claim this campaign makes, and cancels in nothing else.

## 14. Hazards, all previously observed

- **Shared `SAVE_DIR`.** The launcher default is one directory per recipe;
  concurrent runs overwrite each other, and one save took 293 s instead of 97 s
  under contention. The sweep sets a per-arm `SAVE_DIR` for you — pass it
  explicitly on every hand-run arm.
- **Knowing a run finished.** `phase=after_train` is emitted at *every* periodic
  eval, `pgrep` cannot see processes in another PID namespace, and a bare
  `Traceback` may be benign wandb atexit noise. The only reliable end marker is
  `progress rollout=N/N … remaining=0` followed by `shutdown: dispose rollout done`.
- **`codexlog` with env prefixes** fails with `command not found` — it execs its
  arguments directly. Use `codexlog <name> env VAR=val cmd`.
- **`pytest tests/fast/`** silently skips `tests/fast/scripts/` and
  `tests/fast/tools/`: `norecursedirs` matches those basenames at any depth. Use
  `pytest tests` or explicit paths.
- **Never `uv cache clean`.** uv installs in symlink mode here, so clearing the
  cache guts every env pointing into it — that is how the first build of
  `orbit_env` died. See the gap plan's failure signature.

## 15. GPU tiering, at a glance

| Arm class | GPUs | Why |
|---|---|---|
| LoRA / OFT SFT (any rank) | 1, three concurrent | frozen 16 GB base + small adapters |
| FullFT SFT | ≥4 | 32 GB + 96 GB/N optimizer state; launcher enforces |
| P3 DP check | 1 and 4 | the reduction is a no-op at DP=1 |
| RL, LoRA arms | 4-8 | policy + rollout engine share the node |
| RL, FullFT arms | 8 | optimizer state plus rollout engine |
| E3-3 MoE (Qwen3-30B-A3B) | ≥4 | 30B activations exceed one card; skip if unavailable and say so |

## 16. E1-OT — the rank ladder on OpenThoughts3 (C1, C2 on the second dataset)

**Measure this dataset's σ first.** The held-out split is **100 rows against
Tulu3's 1,000**, so its noise floor is a different number and E1-0's σ does not
transfer. Two extra runs:

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
for seed in 1 2; do
  python -m tools.lora_regret.sweep --matrix e1ot --seed $seed \
    --only 'lora-r256-all-lr0.00025' --results results/e1ot_0_sigma.jsonl
done
```

Seed 0 of that arm is already a grid point in the sweep below; point the σ
reading at both files. `analyze` **refuses** to quote a Tulu3 σ against these
arms — that guard is the reason to run this first rather than last.

```bash
python -m tools.lora_regret.sweep --matrix e1ot --only '^lora-r(1|4|16)-' --results results/e1ot_a.jsonl &
python -m tools.lora_regret.sweep --matrix e1ot --only '^lora-r(64|128)-'  --results results/e1ot_b.jsonl &
python -m tools.lora_regret.sweep --matrix e1ot --only '^lora-r(256|512)-' --results results/e1ot_c.jsonl &
wait
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e1ot --only '^full-' --results results/e1ot_full.jsonl
```

One epoch is **312 optimizer steps**, so these arms run to completion and give
both the argmins and the curves — there is no long-run counterpart to schedule.

```bash
python -m tools.lora_regret.analyze all --ledgers 'results/e1ot_*.jsonl' \
  --sigma-ledger results/e1ot_0_sigma.jsonl
```

## 17. E1-short — the ~100-step LR multiplier (C8)

14 arms, ~30 min each. The grid is **0.15-decade**, not the campaign's 0.3:
resolving 15x from 10x means resolving 0.176 decades, and on a 0.3-decade grid
adjacent points differ by 2x.

```bash
python -m tools.lora_regret.sweep --matrix e1short --only '^lora-' --results results/e1short_lora.jsonl
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e1short --only '^full-' --results results/e1short_full.jsonl

python -m tools.lora_regret.analyze c8 \
  --ledgers 'results/e1_*.jsonl' \
  --short-ledgers 'results/e1short_*.jsonl' \
  --sigma-ledger results/e1_0_sigma.jsonl
```

`--ledgers` is E1-1's long-horizon result and `--short-ledgers` is this stage's.
Passing one without the other exits 2: the claim is a comparison of two
horizons, and one horizon is not a comparison.

## 18. E4-place — layer placement under RL (C4 under policy gradient)

8 arms on 8 GPUs, on E4's own data and grid. The MLP arm is **r92** — E3's
solved match for attention r256 in Orbit's fused layout, not the post's r128.
There is no all-modules cell: E4 already ran it at these four learning rates,
so read it from `results/e4_lora.jsonl` and glob both files into `analyze`.

```bash
GPUS_PER_NODE=8 python -m tools.lora_regret.sweep --matrix e4place --results results/e4place.jsonl
python -m tools.lora_regret.analyze c4 --ledgers results/e4place.jsonl --metric accuracy --sigma ...
```

`--metric accuracy` is not optional here. `load_records` filters on the ledger's
own `metric` field, so omitting it reads zero records rather than reading them
in the wrong direction — but a `--metric` that disagrees with the ledger is
silence, not an error, so check the record count in the output.

σ for accuracy has never been measured. If the arms sit close, measuring it
becomes a prerequisite exactly as E1-0 was for NLL — say so rather than quoting
an unresolved difference.

## 19. Figures

`analyze --json` writes the document `plot.py` reads, so a figure can never show
a number the analysis declined to quote — an edge-of-grid argmin exits 3 before
the payload is written.

```bash
python -m tools.lora_regret.analyze all --ledgers 'results/e1_*.jsonl' \
  --sigma-ledger results/e1_0_sigma.jsonl --json > results/analysis.json
python -m tools.lora_regret.plot --analysis results/analysis.json --out results/figures
```

One PNG per panel the payload supports, and nothing for the panels it does not:
an empty axes reads as "measured, and flat". Each line printed names the post's
own figure to compare against, where one exists
(`third_party/lora-without-regret/figures/`). matplotlib is an extra —
`uv sync --extra plots` — and is imported lazily, so `plot.py` stays importable
without it.

## 20. Coverage probe — one run per task per method (before anything long)

```bash
bash scripts/lora_regret/coverage_probe.sh
```

24 runs, one per (task, method), three rollouts each, on a single 8×H100 node.

Rank, OFT block size, target modules and batch size are **not** probed
separately. They exercise the same code — the same wrap, the same launcher path,
the same optimizer — at different tensor shapes, so launching r512 after r256
re-runs a path that already passed. The axes carrying distinct code are the
method (which adapter, or none) and the task (which launcher, dataset, metric),
and those are what the 24 enumerate.

`PROBE_LEVEL=config` launches all 61 distinct configurations instead, collapsing
only the learning rate. Reach for it when hunting a *shape*-dependent failure —
an OOM at a batch size nothing has run at — rather than a code-path one.

Phases: 13 runs at 1 GPU (2 waves of 8), 5 at 4 GPUs (3 waves of 2), 6 at 8 GPUs
(sequential). Phase 3 dominates wall clock.

It answers exactly two questions:

1. **Does every method run?** Anything that cannot start, cannot wrap the model,
   or never reaches the eval line the parser needs fails here in minutes rather
   than on the 40th arm of a reserved node. OFT under policy gradient (§18) has
   never executed at all; this is where that is found out.
2. **How long is the real thing?** `train.py` logs `progress … last=` per
   rollout, so each probe yields a *measured* per-rollout time. The report
   multiplies it by that arm's own rollout count and by the number of arms of
   that method in that task.

   One caveat the report also prints: at method level a row stands for every
   rank, block size, placement **and batch size** in its task. Rank and placement
   barely move step time; batch size does, and E2 runs 32/128/512 — so E2's
   estimate is low by roughly the batch ratio for two thirds of its arms. Run
   `PROBE_LEVEL=config ONLY_PHASE=1` if you want that number tightened.

It deliberately cannot answer a third. Three-rollout rows carry
`probe_rollouts`, and `analyze` **exits 4** on any ledger containing one — a
90-second run produces a real-looking `test_nll`, and a globbed ledger would let
it win an argmin.

Every probe run goes to the **`lora-regret-smoke`** wandb project, never a
task's, with `group=<task>-<method>`. That routing is keyed off the rollout
count rather than a flag, so a probe cannot reach a real dashboard even
deliberately.

GPU sizing mirrors the real sweep, because a timing measured on the wrong number
of GPUs estimates nothing:

| phase | arms | GPUs each | concurrency |
|---|---|---|---|
| 1 | SFT LoRA / OFT | 1 | 8 at a time, one per device |
| 2 | SFT FullFT | 4 | 2 at a time |
| 3 | RL, all methods | 8 | sequential |

**Phase 1's numbers are upper bounds, not estimates.** Eight arms share NVLink,
host RAM and the filesystem; phases 2 and 3 are uncontended and their numbers are
estimates. Do not average the two. If you want clean 1-GPU numbers, re-run just
that phase alone: `ONLY_PHASE=1 PROBE_DIR=results/probe_solo bash …` with the
loop's barrier reached one run at a time.

Knobs: `PROBE_LEVEL` (`method` | `config`), `PROBE_ROLLOUTS` (3), `PROBE_DIR`
(`results/probe`), `ONLY_PHASE` (1, 4 or 8), `DRY_RUN=1`, `SKIP_PREFLIGHT=1`.

`e1long` and `sft82` are not probed and the plan says why: `e1long`'s arms come
from an E1-1 ledger that does not exist yet, and `sft82` is the frozen legacy
matrix whose methods are E1's and E5's. `e5`'s OFT cell has no scouted centre, so
the probe supplies 1e-4 — valid for plumbing and timing, and the real sweep still
refuses `--matrix e5` without the measured value.

Re-read the report at any time without re-running anything:

```bash
python -m tools.lora_regret.probe report --ledger 'results/probe/*.jsonl'
```
