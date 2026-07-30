# LoRA-Without-Regret — campaign tooling design

What stands between "the sweep can run" and "reserve nodes, run the runbook, and
every number the campaign claims comes out of a committed tool." Six gaps, found
by auditing the tooling rather than the plans.

Companions:
- Operator's guide: `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md`
- Claims and acceptance criteria: `docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md`
- Port status: `docs/superpowers/plans/2026-07-29-lora-without-regret-gap.md`

## The gaps

Verified against the tree at `6376537`, not read off the plans. Every matrix
already produces its documented arm count (`e1`=40, `e2`=36, `e3`=20, `e4`=16,
`e5scout`=5, `e5`=50, `sft82`=82) and the CPU suite is **502 passed, 0 failed**,
so the launchers and matrices are sound. What is missing is entirely on the
operator side.

| # | Gap | Consequence |
|---|---|---|
| 1 | No preflight check | A bad path or a missing split is discovered inside a reservation, not before it |
| 2 | `--dry-run` prints an unsafe command line | It omits `SAVE_DIR`/`RUN_LOG`/`LAUNCHER_NAME`; pasting one by hand hits the launcher's shared default `SAVE_DIR` — hazard #1 of the runbook's §14, reintroduced by the preview tool |
| 3 | P3 is eyeballed | Acceptance is equality at the sixth decimal across three fields in two logs |
| 4 | No analysis module | σ, argmins, edge-of-grid, C1–C6 all live as heredocs inside a markdown file; the edge-of-grid **acceptance rule has no enforcement** |
| 5 | `adapter_params` is hardcoded `None` | E3 and E5 are entirely claims about matched parameter counts |
| 6 | No NLL-trace extraction | C1's departure step needs the whole curve; `parse_final_nll` returns one number |

Two things audited and found **not** to be problems, recorded so they are not
re-investigated: the Megatron checkpoint under the old repo path
(`/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints/Llama-3.1-8B_torch_dist`)
exists and is readable at 15 GB — a cross-repo dependency to pin in preflight,
not a break; and checkpoint writes total roughly 1 TB across the campaign
(E1-2's single FullFT arm is ~460 GB of it) against 242 T free on `/lustre/fast`,
so no retention knob is warranted.

## Architecture

Four new modules under `tools/lora_regret/`, plus two changes to existing files.
The organising constraint: **every piece is a pure function over text or JSON**,
so all of it is CPU-testable and none of it needs a GPU to verify — which is what
makes it safe to build before the nodes are held.

```
preflight.py   env / GPU / checkpoints / data / matrices   -> exit 0 | 1
trace.py       log text                                    -> list[NllPoint]
analyze.py     ledgers + traces                            -> claim readings
p3_check.py    two log paths                               -> exit 0 | 1
```

### Where traces come from

`sweep.py` already holds each arm's log text in memory when it parses the final
NLL, so it can write the whole trace into the ledger row for free. That cannot be
the *only* source: `p3_check` compares two hand-run logs, and any arm re-run by
hand outside the sweep produces a log and no ledger row. So `trace.py` parses
logs as its primary interface, and the sweep calls it to populate an `nll_trace`
field — one implementation, two entry points, rather than a ledger format that
silently cannot describe half the runs.

`trace.py` reuses `sweep.py`'s existing `_NLL_LINE` regex rather than re-spelling
it. That regex is built from `EVAL_NLL_METRIC_KEY` precisely so a rename cannot
desync the parser from the metric; a second copy would reintroduce the failure
mode the first copy was written to prevent. The shared constant moves to
`trace.py` and `sweep.py` imports it, so there is one definition and the
existing `TestLogFormatPins` tests keep pinning it.

## §1 — `preflight.py`

Fails fast on everything a reservation can discover expensively. One check per
line, each printing what it found, exiting non-zero on the first failure with the
specific cause rather than a summary.

- **Env**: `torch`, `transformers`, `megatron.core`, `orbit` import **and**
  `__file__ is not None`. The bare-import form is not sufficient — the
  broken-symlink failure mode recorded in the gap plan imports *successfully* and
  presents as a missing attribute.
- **GPUs**: visible device count and compute capability, checked against a
  `--stage` argument (`smoke`/`p3`/`e1-lora`/`e1-full`/`e4`) that names the
  requirement. Refuses `e1-full` below 4 devices for the same arithmetic the
  launcher enforces, so the failure happens before Ray starts rather than after.
- **Checkpoints**: `HF_CKPT` and `MEGATRON_LOAD` readable, and
  `latest_checkpointed_iteration.txt` present under the Megatron path.
- **Data**: all nine splits present under `DATA_DIR` at their recorded row
  counts (938,343 / 1,000 / 10,000 / 100 / 7,498 / 5,000 / 7,473 / 1,319 /
  14,971). Row counts, not just existence: a truncated split is the failure that
  otherwise surfaces as a wrong denominator in E1.
- **Matrices**: builds every matrix and asserts its documented arm count, so a
  matrix that raises does so now. `e5` is built with a dummy centre.

`--stage` selects which checks are required; all are reported either way.

## §2 — `trace.py`

```python
NllPoint = NamedTuple("NllPoint", rollout_id=int, step=int, phase=str,
                      nll=float, sample_mean=float, tokens=int, samples=int)

def parse_trace(log_text: str) -> list[NllPoint]:  ...
def trace_is_consistent(points) -> tuple[bool, str]: ...
```

`parse_trace` returns every measurement in `step` order, both phases retained —
`before_train` is gate G4's number and is meaningful, it just must never be
picked as an arm's *result*.

`trace_is_consistent` asserts `tokens` and `samples` are constant across the
trace. A drifting `samples` means the held-out set is being floor-divided by the
batch size, which makes the metric depend on batch size and breaks E2
specifically — the check §3 of the runbook currently does by eye.

## §3 — `analyze.py`

Subcommands, one per reading, plus `all`. Human-readable tables to stdout,
`--json` for machine consumption.

| Subcommand | Reads | Reports |
|---|---|---|
| `sigma` | E1-0 ledger | σ = stdev of the three `test_nll`; refuses on fewer than 3 |
| `argmins` | E1 ledgers | best LR per `(method, rank)`, **with edge-of-grid flags** |
| `c1` | E1-2 traces | departure step per rank |

`c1` reads each arm's trace from the `nll_trace` field of the E1-2 ledger, and
falls back to parsing `logs/lora_regret/<arm>.log` through `trace.py` when the
field is absent — which is the case for any arm run before this change lands, or
run by hand. It reports which source it used per arm, since a silently-empty
trace and a silently-truncated one both read as "no departure."
| `c2` | E1 ledgers | `argmin_LR(LoRA r256) / argmin_LR(FullFT)`; also the rank-4-to-512 spread |
| `c3` | E2 ledgers | `best_LoRA(batch) − best_FullFT(batch)` per batch, in σ; r256 vs r16 |
| `c4` | E3 ledger | `NLL(attn) − NLL(mlp)` at matched params, in σ; all-modules vs MLP-only |
| `c5` | E4 ledgers | peak accuracy per arm and performant-band width |
| `c6` | E5 ledgers | OFT vs LoRA at matched params, `matched_ratio` quoted alongside |

Three rules are load-bearing and are implemented once, here, rather than
restated per subcommand:

**Grid points are seed 0 only.** Every ledger read filters `status == "ok"` and
`seed == 0`. E1-0's replicates live in the same ledger directory and are not grid
points; the runbook records a synthetic-ledger measurement where dropping this
filter let a replicate at LR 9.95e-4 steal r256's argmin from the real 2.5e-4.

**Edge-of-grid is an error, not a note.** An argmin sitting on either end of its
arm's LR grid means the true optimum may lie outside the grid, so the ratio is
not quotable. `argmins` marks it, and every downstream subcommand refuses to
quote a claim that depends on a flagged arm unless `--allow-edge-argmin` is
passed. The runbook's instruction is to **re-centre, not extend** — the tool says
so in the failure message.

**σ gates significance, and where it does not exist the tool says so.**
Differences print in units of σ, read from the E1-0 ledger by default and
overridable with `--sigma`. For C5 there is no measured accuracy σ; `c5` prints
raw deltas, refuses to call any of them resolved, and names the measurement that
would settle it (an accuracy-σ replicate set, exactly as E1-0 is for NLL).

**C1's departure rule**, stated once: per rank, the first step at which its NLL
exceeds the pointwise minimum across all arms by more than 2σ for **three
consecutive** logging intervals. A rank that never departs and a rank whose run
was too short are indistinguishable, so `c1` prints the step budget beside every
departure point and reports `no departure within N steps` rather than a blank.

## §4 — `p3_check.py`

```
python -m tools.lora_regret.p3_check logs/p3_dp1_*.log logs/p3_dp4_*.log
```

Parses both traces, pairs measurements by `(phase, step)`, and asserts `nll`
equal to six decimals with `tokens` and `samples` exactly equal. Exit 0 or 1,
printing the first mismatching row. A differing `tokens` means the DP reduction
is double-counting or dropping a shard; the message says that, because the
correct response is to stop rather than to average.

## §5 — changes to existing files

**`arms.py` gains an `e1long` matrix.** Eight arms — FullFT plus LoRA r ∈ {1, 4,
16, 64, 128, 256, 512} — each at that rank's own argmin LR from the E1-1 ledgers.

Two new `Arm` fields carry what E1-2 needs and no other stage does:

- `full_epoch: bool = False` — when set, `arm_env` emits `NUM_ROLLOUT=""`. The
  launcher's `NUM_ROLLOUT=${NUM_ROLLOUT:-$((...))}` is the **colon** form, so an
  empty value re-derives the full epoch from the file's row count. Emitting the
  empty string rather than omitting the key is deliberate: it immunises the arm
  against a `NUM_ROLLOUT=2000` left exported in the shell from the E1-1 stage,
  which would otherwise silently turn a 29,323-step curve into a 2,000-step one
  and make every rank look like it never departs.
- `eval_nll_interval: int | None = None` — `e1long` sets 293, ~1% of the epoch,
  giving ~100 trace points for ~1.9 h of eval against ~70 h of training.

**`sweep.py` gains `--argmins-from`**, a glob over E1-1 ledgers. It builds the
`(method, rank) -> lr` map with the seed-0 filter, and **fails closed** in two
cases: fewer than 8 keys recovered (a partial ledger would otherwise silently
run three arms and look complete), or any recovered argmin flagged edge-of-grid
by the shared rule from §3 — overridable with `--allow-edge-argmin`, since that
is a judgment the operator may legitimately make.

`MATRICES` lambdas take a new keyword `argmins` alongside `oft_lr_centre`,
following the existing pattern. `--argmins-from` is rejected for any matrix but
`e1long`, and `e1long` requires it, mirroring the `e5`/`--oft-lr-centre` guard
that already exists.

The runbook's §8 heredoc generator is **deleted** and replaced by the
`--matrix e1long` invocation, so there is one driver rather than two.

**`sweep.py`'s dry-run prints the full environment** — `SAVE_DIR`, `RUN_LOG`,
`LAUNCHER_NAME`, `WANDB_GROUP` — not just `arm_env`. The current output is a
command line that is unsafe to paste, produced by the tool whose purpose is
previewing what will run.

**`adapter_params` is computed** from `peft_param_match`, which already has every
piece: `megatron_module_shapes(hidden, ffn, qkv_output_size)` filtered to the
arm's `target_modules`, then `lora_param_count_for_modules` or
`oft_param_count_for_modules`, times the layer count. Analytic rather than read
back from the written checkpoint, so it is available at dry-run time — before
compute is spent. `full` arms record `null`, which is meaningful rather than
missing.

The layer count is a new `--num-layers` CLI argument (32 for Llama-3.1-8B),
required alongside the existing `--hidden-size` / `--ffn-size` rather than
defaulted — the same reasoning that made those two explicit. `qkv_output_size`
comes from `arms.LLAMA31_8B_QKV_OUTPUT` (6144), which already exists and is
already threaded through `e3`/`e5`.

**This formula is verified, not assumed.** Measured 2026-07-30 against the
complete adapter at
`/lustre/fast/fast/zqiu/tmp/smoke_ckpt_20260730/iter_0000001/adapter/`: analytic
`lora_param_count_for_modules(256, shapes) × 32 = 570,425,344`, and the
checkpoint holds exactly 570,425,344 bf16 parameters across 256 tensors over 32
layers, with the per-module split matching shape-for-shape
(`linear_qkv` 83,886,080 / `linear_proj` 67,108,864 / `linear_fc1` 268,435,456 /
`linear_fc2` 150,994,944). E3's and E5's matched-parameter premise therefore rests
on a checked identity.

Note the oracle path: the *older* adapter at `.../tmp/smoke_ckpt/` is a
**truncated** save from the 2026-07-29 fixture smoke — 32 of 256 data records,
142 MB against 1.14 GB — and `torch.load` rejects it outright
(`failed locating file data/32 ... checkpoint file is corrupted`). It is not a
usable oracle and the gap plan's citation of it as written-adapter evidence
should be narrowed to the log line rather than the file.

**`sweep.py` writes `nll_trace`** into each ledger record via `trace.py`.

## §6 — testing

Every module above is a pure function over text or JSON, so the whole design is
CPU-verifiable. Tests go in `tests/fast/utils/`, alongside the existing
`test_lora_regret_sweep.py`.

Each detector gets a case it must **reject**, following the non-tautology
discipline the gap plan used throughout — a detector that only has passing cases
is untested:

- Edge-of-grid fires on an argmin at either boundary and stays silent one grid
  point in.
- The departure detector does **not** fire on a two-interval excursion when the
  rule says three, and does fire on the third.
- `argmins` drops a seed replicate that would otherwise win — the exact synthetic
  ledger the runbook records, so the regression is pinned rather than described.
- `trace_is_consistent` rejects a trace whose `samples` drops from 1000 to 992,
  which is the floor-division signature.
- `parse_trace` on the **real** smoke log
  (`logs/smoke_lora_r256_20260730_150952.log`) returns the three known points,
  and `parse_final_nll` over the same text still returns `(1.194836, 1)`.
- `adapter_params` for LoRA r256 all-modules equals **570,425,344**, the count
  measured in the real 07-30 adapter — a pinned constant, so a refactor of the
  shape arithmetic cannot drift away from a checkpoint that is no longer read.
- `e1long` with a `NUM_ROLLOUT=2000` polluted environment still emits
  `NUM_ROLLOUT=""`.
- `--argmins-from` over a 3-key ledger exits non-zero rather than running 3 arms.
- `p3_check` accepts two identical traces and rejects a pair differing only in
  `tokens`.

Acceptance for the whole change: the CPU suite stays at 502 passed plus the new
tests, 0 failures. No GPU run is required to land any of it.

## Out of scope

- Plotting. `analyze.py --json` emits the numbers; figures are a separate
  concern and the campaign's claims are all scalar comparisons.
- E3-3, the MoE arm. Unwired for the reasons the runbook gives (Qwen3-30B-A3B
  needs its own model-args plugin and ≥4 GPUs); skipping it and saying so is the
  documented decision.
- Any change to `prepare_data.py`, the launchers, or the eval-NLL wiring. All
  three are verified and the reservation depends on them not moving.
