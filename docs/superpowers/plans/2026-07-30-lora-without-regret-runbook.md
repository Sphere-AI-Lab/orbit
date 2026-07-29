# LoRA-Without-Regret — runbook

How to run the campaign on reserved nodes. Every command here is meant to be
pasted; every number it expects back is stated so a wrong one is visible
immediately.

Companions:
- Claims, arms, and acceptance criteria: `docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md`
- Port status and what is verified: `docs/superpowers/plans/2026-07-29-lora-without-regret-gap.md`
- Environment build: `INSTALL.md`

## What is ready, and what you still have to do

Ready and CPU-verified (439 tests, 0 failures, in the built env):

| Piece | Where |
|---|---|
| SFT launcher — LoRA, OFT and FullFT in one script | `examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh` |
| RL launcher (prerequisite P5) | `examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh` |
| Arm matrices `e1` / `e2` / `e3` / `sft82` | `tools/lora_regret/arms.py` |
| Sweep driver with resume ledger | `tools/lora_regret/sweep.py` |
| Data preparation for all five datasets | `tools/lora_regret/prepare_data.py` |
| Held-out token-weighted NLL eval | `orbit/utils/eval_nll.py`, wired in `train.py` |

Yours to do, in this order: **materialize the data** (§2, CPU only, no GPU),
**smoke one arm** (§3), **close P3** (§4, needs DP≥2 and gates every FullFT
number), then run experiments (§5-§9).

Single-rank reachability is already proven on an H100 — two LoRA-r256 optimizer
steps, all 100 held-out rows scored, the pinned NLL line emitted three times
(`logs/smoke_llama31_lora_20260729_214422.log`). What has **never** executed is
the DP>1 reduction and anything multi-GPU. Treat §4 as mandatory.

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

## 2. Materialize the datasets (CPU, no GPU, network required)

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
python -m tools.lora_regret.prepare_data --dataset campaign --out-dir "$DATA_DIR"
```

Expected row counts, all asserted by the tool — a mismatch raises and leaves the
previous file untouched rather than writing a short split:

| Output | Rows | For |
|---|---|---|
| `tulu3_train.jsonl` / `tulu3_test.jsonl` | 938,344 − filtered / 1,000 | E1, E3 |
| `openthoughts3_train.jsonl` / `_test.jsonl` | 10,000 / 100 | E2 |
| `math_train.jsonl` / `math_test.jsonl` | 7,500 / 5,000 | E4 |
| `gsm8k_train.jsonl` / `gsm8k_test.jsonl` | 7,473 / 1,319 | E4 |
| `math_gsm8k_train.jsonl` | 14,973 | E4 (the launcher's `--prompt-data`) |

The CLI prints `filtered=` / `assistant_header=` / `eot=` per dataset. Those are
Tulu3 rows whose assistant content contains a literal
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

If Tulu3's upstream row count has moved, the assertion fires with the actual
number. Re-run with `python -c "from tools.lora_regret.prepare_data import
prepare_tulu3; prepare_tulu3('$DATA_DIR', expected_source_rows=<actual>)"` only
after checking the dataset card — a changed count means a changed mixture.

## 3. Smoke one arm (1 GPU, ~10 minutes)

```bash
NUM_ROLLOUT=2 TRAIN_ROWS=64 \
LAUNCHER_NAME=smoke_lora_r256 \
SAVE_DIR=/lustre/fast/fast/zqiu/tmp/smoke_ckpt \
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

The line looks like
`eval/test_nll rollout_id=0 step=0 phase=before_train nll=1.968950 sample_mean=... tokens=17656 samples=100`.
`tokens=` and `samples=` must be **identical at every measurement** — a drifting
`samples=` means the held-out set is being floor-divided by the batch size, which
would make the metric depend on batch size and silently break E2.

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

grep 'eval/test_nll' logs/p3_dp1_*.log logs/p3_dp4_*.log
```

**Acceptance: identical `nll=` to the printed six decimals, and identical
`tokens=` and `samples=`.** A differing `tokens=` means the reduction is
double-counting or dropping a shard. If they differ, stop — every FullFT number
downstream is wrong, and no amount of averaging fixes it.

## 5. Cost arithmetic — read this before launching E1

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

Batch sizes per dataset, for reference: OpenThoughts3's 10,000 rows are 312
steps at batch 32 and 19 at batch 512 — E2 is cheap, E1/E3 are not.

## 6. E1 — capacity, rank, and the 10x LR rule (C1, C2)

40 arms: FullFT plus LoRA r ∈ {1, 4, 16, 64, 128, 256, 512}, five LRs each at
0.3-decade spacing, centred on 2.5e-5 (FullFT) and 2.5e-4 (LoRA — exactly 10x,
which is C2's prediction built into the grid rather than fitted out of it).

Inspect before spending anything:

```bash
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --dry-run | wc -l   # 40
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 --dry-run | head -3
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

python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 \
  --only '^lora-r(1|4)-'   --results results/e1_lora_a.jsonl &
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 \
  --only '^lora-r(16|64)-' --results results/e1_lora_b.jsonl &
python -m tools.lora_regret.sweep --matrix e1 --hidden-size 4096 --ffn-size 14336 \
  --only '^lora-r(128|256|512)-' --results results/e1_lora_c.jsonl &
wait
```

**FullFT arms — ≥4 GPUs**, on the allocation that has them:

```bash
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e1 \
  --hidden-size 4096 --ffn-size 14336 --only '^full-' --results results/e1_full.jsonl
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

## 7. E1-0 — re-measure σ first, actually

The σ = 0.000992 nats in the gate log was measured on **Qwen3-4B / No Robots**
and does not transfer to Llama-3.1-8B / Tulu3. Everything downstream is quoted
in units of σ, so measure it before quoting anything:

```bash
for seed in 0 1 2; do
  SEED=$seed LAUNCHER_NAME=e1_0_sigma_s$seed \
  SAVE_DIR=/lustre/.../e1_0_s$seed NUM_ROLLOUT=2000 EVAL_NLL_INTERVAL=20 \
  LORA_RANK=256 LR=2.5e-4 \
  bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh
done
```

σ is the standard deviation of the three final `nll=` values. Until it exists,
treat any difference under ~0.002 as unresolved. `SEED` is tied to
`ROLLOUT_SEED` inside the launcher, so each replicate varies data order as well
as initialization — which is what a seed replicate should vary.

## 8. E2 — batch-size sensitivity (C3), and E3 — layer placement (C4)

```bash
# E2: 36 arms, OpenThoughts3, batch {32,128,512}, FullFT + LoRA r256 + LoRA r16
python -m tools.lora_regret.sweep --matrix e2 --hidden-size 4096 --ffn-size 14336 \
  --only '^lora' --results results/e2_lora.jsonl
GPUS_PER_NODE=4 python -m tools.lora_regret.sweep --matrix e2 \
  --hidden-size 4096 --ffn-size 14336 --only '^full' --results results/e2_full.jsonl

# E3: 20 arms, Tulu3, matched-parameter attention vs MLP
python -m tools.lora_regret.sweep --matrix e3 --hidden-size 4096 --ffn-size 14336 \
  --results results/e3.jsonl
```

E2 sets `GLOBAL_BATCH_SIZE` **and** `ROLLOUT_BATCH_SIZE` together and points
`TRAIN_JSONL`/`TEST_JSONL` at OpenThoughts3 automatically — nothing to export.
Each cell's four LRs are re-centred by √(batch/32); the edge-of-grid rule from
§6 still applies, and it will fire more often here because the batch-size
optimum is less well predicted than the rank one.

E3's matched pair is **attention r256 against MLP r92**, solved for Orbit's
fused layout by `orbit.utils.peft_param_match.matched_mlp_rank` — per layer
`18432·r` for attention against `51200·r` for MLP, a ratio of 2.778. The post's
own pair (attention r256 / MLP r128) is also in the matrix, deliberately: if the
two disagree, the disagreement is parameter accounting rather than physics.
Print realized adapter parameter counts next to every arm before believing
either.

## 9. E4 — RL parity at low rank (C5)

16 runs: FullFT, LoRA r256, r16, **r1** — rank 1 is the claim's whole point and
is not the arm to drop under budget pressure. There is no sweep matrix for E4;
run the arms directly, because each is expensive enough to want individual
attention.

```bash
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
for rank in 1 16 256; do
  for lr in 3.3e-6 1e-5 3.3e-5 1e-4; do
    PEFT_METHOD=lora LORA_RANK=$rank LR=$lr \
    LAUNCHER_NAME=e4_lora_r${rank}_lr${lr} \
    SAVE_DIR=/lustre/.../e4_lora_r${rank}_lr${lr} \
    GPUS_PER_NODE=8 \
    bash examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh
  done
done

# FullFT arms, one decade down
for lr in 3.3e-7 1e-6 3.3e-6 1e-5; do
  PEFT_METHOD=none LR=$lr LAUNCHER_NAME=e4_full_lr${lr} \
  SAVE_DIR=/lustre/.../e4_full_lr${lr} GPUS_PER_NODE=8 \
  bash examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh
done
```

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

## 10. E5 (optional, ours) — matched-parameter OFT

Only after C1-C5 are settled. The `sft82` matrix is the only one carrying OFT
arms, and it includes a deliberate half-decade scout, because OFT parameterizes
a rotation rather than an additive update and its LR scale is unknown a priori:

```bash
python -m tools.lora_regret.sweep --matrix sft82 --hidden-size 4096 --ffn-size 14336 \
  --only '^oftscout' --results results/e5_scout.jsonl
```

The dry run prints the realized parameter ratio per rank. Check it: the block
size snaps to a divisor of `d_in`, which at large rank can move the ratio well
away from 1.0 — and an unmatched "matched" comparison is worse than none.
`OFT_BLOCK_SIZE` has **no default** in either launcher; it must come from
`matched_oft_block_size`, so a missing value fails at launch instead of quietly
comparing unmatched models.

## 11. Reading results

The ledger is one JSON object per arm:

```json
{"arm": "lora-r256-all-lr0.00025-s0", "method": "lora", "rank": 256, "lr": 0.00025,
 "test_nll": 1.845700, "steps": 2000, "status": "ok"}
```

`test_nll` is the **last `phase=after_train` measurement**, chosen by highest
`step` rather than by file position, so an interleaved multi-rank log cannot make
a `before_train` row win. `status: "failed"` means either a non-zero exit or no
parseable NLL line — check the arm's log under `logs/lora_regret/<arm>.log`.

Argmins per arm:

```bash
python - <<'PY'
import json, collections
best = {}
for path in ("results/e1_lora_a.jsonl","results/e1_lora_b.jsonl","results/e1_lora_c.jsonl","results/e1_full.jsonl"):
    for line in open(path):
        r = json.loads(line)
        if r["status"] != "ok": continue
        key = (r["method"], r["rank"])
        if key not in best or r["test_nll"] < best[key]["test_nll"]:
            best[key] = r
for key, r in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
    print(f"{key[0]:5} r={str(key[1]):4} argmin_lr={r['lr']:<9g} nll={r['test_nll']:.6f}")
PY
```

Then quote every difference in units of the σ from §7, and never off absolute
loss values — the constant Orbit-vs-HF precision offset (0.0032 nats, inside
HF's own 0.0072-nat bf16/fp32 spread) cancels in every ratio, ordering and
curve-shape claim this campaign makes, and cancels in nothing else.

## 12. Hazards, all previously observed

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

## 13. GPU tiering, at a glance

| Arm class | GPUs | Why |
|---|---|---|
| LoRA / OFT SFT (any rank) | 1, three concurrent | frozen 16 GB base + small adapters |
| FullFT SFT | ≥4 | 32 GB + 96 GB/N optimizer state; launcher enforces |
| P3 DP check | 1 and 4 | the reduction is a no-op at DP=1 |
| RL, LoRA arms | 4-8 | policy + rollout engine share the node |
| RL, FullFT arms | 8 | optimizer state plus rollout engine |
| E3-3 MoE (Qwen3-30B-A3B) | ≥4 | 30B activations exceed one card; skip if unavailable and say so |
