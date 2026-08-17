"""One run per (task, method): does the method work, and how long is the real arm?

The probe runs every (matrix, method) pair once for `PROBE_ROLLOUTS` rollouts and
reports two things:

1. **Does it work.** A method that cannot start, cannot wrap the model, or never
   reaches the eval line fails here in minutes rather than on the 40th arm of a
   reserved node.
2. **How long is the real thing.** `train.py` already logs `progress ... last=`
   per rollout, so the probe reads a measured per-rollout time rather than
   inferring one from wall clock, and multiplies it by the rollout count that
   arm would really run.

A probe row is **not a measurement**. It carries a `probe_rollouts` field, and
`analyze` refuses any ledger containing one -- three rollouts produce a
real-looking `test_nll`, and a ledger that mixed the two would decide an argmin
from a learning rate that trained for ninety seconds.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from tools.lora_regret.arms import MATRICES, MATRICES_REQUIRING_OFT_CENTRE
from tools.lora_regret.models import DEFAULT_MODEL
from tools.lora_regret.models import get as get_model
from tools.lora_regret.probe_log import parse_rollout_seconds  # noqa: F401  (re-exported)
from tools.lora_regret.sweep import MATRIX_METRICS, wandb_project

# Three, not two: dropping rollout 1 (compile, weight load, the first allocator
# growth) leaves two steady rollouts, which is the fewest a median can be taken
# over. Two probe rollouts would leave one, and a single sample has no spread to
# show that the estimate is stable.
PROBE_ROLLOUTS = 3

# Matrices the probe does not run, and why. Recorded rather than omitted: a
# silently missing task looks the same as a task that passed.
EXCLUDED_MATRICES = {
    "e1long": (
        "its arms are read from an E1-1 ledger via --argmins-from, which does "
        "not exist before E1-1 runs. Its methods are E1's and are probed there."
    ),
    "sft82": (
        "the frozen legacy matrix. Its FullFT and LoRA arms are E1's and its OFT "
        "arms are E5's, so probing it would re-answer both questions at cost."
    ),
}

# Some matrices' OFT cells have no centre until a scout has run. Any value
# produces a valid *plumbing* probe -- the learning rate does not change how
# long a step takes -- so the probe supplies the midpoint of the scout span and
# says so. It must never leak into a real sweep, which is why those matrices
# still require the flag there. Which matrices they are is declared in `arms`,
# not tested for by name here: this line used to read `matrix == "e5"`, and that
# literal made the plan RAISE rather than skip when a second one was added.
PROBE_OFT_CENTRE = 1e-4

# Rollouts the REAL arm runs, per matrix. Three different sources, which is
# exactly why this is written down rather than derived:
#   * e1short  -- the arm carries num_rollout=100 itself.
#   * e1ot     -- full_epoch, so the launcher derives ceil(rows / batch).
#   * e1/e2/e3 -- the operator exports NUM_ROLLOUT=2000 (runbook section 8);
#                 nothing in the code says 2000, so nothing can derive it.
#   * e4/e4place -- the RL launcher's own default of 500.
#   * e5/e5scout -- Tulu3 SFT under the same runbook convention as e1.
OPENTHOUGHTS3_TRAIN_ROWS = 10_000
ROLLOUT_BATCH_SIZE = 32
SFT_SWEEP_ROLLOUTS = 2000
RL_LAUNCHER_ROLLOUTS = 500
FULL_RUN_ROLLOUTS = {
    "e1": SFT_SWEEP_ROLLOUTS,
    "e1ot": (OPENTHOUGHTS3_TRAIN_ROWS + ROLLOUT_BATCH_SIZE - 1) // ROLLOUT_BATCH_SIZE,
    "e1short": 100,
    "e2": SFT_SWEEP_ROLLOUTS,
    "e3": SFT_SWEEP_ROLLOUTS,
    "e4": RL_LAUNCHER_ROLLOUTS,
    "e4place": RL_LAUNCHER_ROLLOUTS,
    # The E4 LR column and the two Math BS128 OFT scouts all drive the same RL
    # launcher as e4, so they cost the same full run.
    "e4lr0": RL_LAUNCHER_ROLLOUTS,
    "e4oftb128low": RL_LAUNCHER_ROLLOUTS,
    "e4oftb128refine": RL_LAUNCHER_ROLLOUTS,
    "e5scout": SFT_SWEEP_ROLLOUTS,
    "e5": SFT_SWEEP_ROLLOUTS,
    "e5rl": RL_LAUNCHER_ROLLOUTS,
}


# What counts as one probe. Three levels, coarsest first.
#
# `path` is the default: one run per distinct **code path**, which is
# (launcher, dataset, method, target modules). 13 runs against `method`'s 24.
#
#   Carried on the axis, so it is probed:
#     launcher       -- SFT and RL are different scripts and different parsers
#     method         -- which adapter is wrapped, or none
#     target modules -- WHICH layers get wrapped. `linear_fc1` is Orbit's fused
#                       gate+up; wrapping it is not the same code as wrapping
#                       `linear_qkv`, so attn/mlp/all stay separate.
#     dataset        -- not a code difference but a shape one: OpenThoughts3
#                       rows are ~62 KB against Tulu3's ~3 KB, a 20x sequence
#                       length that moves both memory and step time.
#
#   Collapsed, because they are the same code at a different tensor shape:
#     rank, OFT block size, batch size. `e4/full` and `e4place/full` are the
#     same run twice; so are `e1/lora`, `e1short/lora` and `e5/lora`.
#
# The report still prints all 24 (task, method) rows -- each one reads the pace
# measured on ITS code path -- so nothing is lost from the estimate.
#
# `method` is the previous default: one run per (task, method), 24 runs. Use it
# if you want each task independently confirmed rather than inferred.
#
# `config` collapses only the learning rate, so every rank, block size,
# placement and batch size is launched once -- 61 runs. Worth it only when
# hunting a shape-dependent failure (an OOM at a batch size nothing has run at)
# rather than a code-path one.
PROBE_LEVELS = ("path", "method", "config")


@dataclass(frozen=True)
class ProbeRun:
    matrix: str
    method: str
    arm: str
    only: str
    gpus: int
    metric: str
    full_rollouts: int
    arms_of_method: int
    project: str
    label: str          # task/method/capacity/placement, for the report
    arms_in_config: int  # arms sharing this configuration -- the LR grid width


def _build(matrix: str):
    centre = PROBE_OFT_CENTRE if matrix in MATRICES_REQUIRING_OFT_CENTRE else None
    model = get_model(DEFAULT_MODEL)
    return MATRICES[matrix](
        model.hidden_size, model.ffn_size, model.qkv_output_size, 0, centre, None
    )


_MODULE_SHORT = {
    "linear_qkv,linear_proj,linear_fc1,linear_fc2": "all",
    "linear_qkv,linear_proj": "attn",
    "linear_fc1,linear_fc2": "mlp",
    "": "-",
}


def path_key(matrix: str, arm) -> tuple:
    """The distinct code path an arm exercises.

    Two arms with this key equal run the same script over the same data through
    the same wrapping code; only their tensor shapes differ. Probing both proves
    nothing the first did not.
    """
    launcher = "rl" if MATRIX_METRICS[matrix] == "accuracy" else "sft"
    return (launcher, arm.dataset or "tulu3", arm.method, arm.target_modules or "")


def config_key(arm) -> tuple:
    """Everything about an arm except its learning rate.

    This is the unit that has to be launched at least once: two arms with the
    same key differ only by a multiply, two arms with different keys can differ
    by 16x in rollout batch or 2x in adapter size.
    """
    return (arm.method, arm.rank, arm.oft_block_size, arm.target_modules,
            arm.global_batch_size)


def config_label(key: tuple) -> str:
    method, rank, block, modules, batch = key
    capacity = f"r{rank}" if method == "lora" else (f"b{block}" if method == "oft" else "-")
    label = f"{method}/{capacity}/{_MODULE_SHORT.get(modules, modules)}"
    return label + (f"/batch{batch}" if batch else "")


def _representative(arms: list):
    """The middle learning rate of a cell.

    Which LR is probed does not change the timing, so the choice is made for
    determinism and readability. The middle of the grid is the arm an operator
    recognises from the runbook.
    """
    ordered = sorted(arms, key=lambda a: (a.lr, a.name))
    return ordered[len(ordered) // 2]


def _gpus(method: str, metric: str) -> int:
    """What the real sweep gives this arm.

    The probe's timings are estimates of the real arms only if the real arms run
    on the same hardware, so this mirrors the runbook rather than the node.
    """
    if metric == "accuracy":
        return 8  # RL: policy plus the rollout engine share the node
    if method == "full":
        return get_model(DEFAULT_MODEL).min_gpus_fullft()
    return 1


def _probe_run(matrix: str, arm, label: str, multiplier: int) -> ProbeRun:
    metric = MATRIX_METRICS[matrix]
    return ProbeRun(
        matrix=matrix,
        method=arm.method,
        arm=arm.name,
        # Anchored at both ends: an unanchored name is a prefix of any longer
        # one, and `--only` takes a regex, so a bare name could select two arms
        # and bill the second to the first.
        only=f"^{re.escape(arm.name)}$",
        gpus=_gpus(arm.method, metric),
        metric=metric,
        full_rollouts=FULL_RUN_ROLLOUTS[matrix],
        arms_of_method=sum(
            1 for a in _build(matrix) if a.method == arm.method
        ),
        project=wandb_project(matrix),
        label=label,
        arms_in_config=multiplier,
    )


def probe_plan(level: str = "path") -> list[ProbeRun]:
    """One run per distinct code path by default; see PROBE_LEVELS."""
    if level not in PROBE_LEVELS:
        raise ValueError(f"unknown probe level {level!r}; known: {PROBE_LEVELS}")

    if level == "path":
        # Grouped ACROSS matrices, which is the whole point: `e4/full` and
        # `e4place/full` are the same script over the same data wrapping the
        # same (empty) module set, so running both proves nothing the first did
        # not. Grouping within a matrix would keep every duplicate.
        cells: dict[tuple, list[tuple[str, object]]] = {}
        for matrix in MATRICES:
            if matrix in EXCLUDED_MATRICES:
                continue
            for arm in _build(matrix):
                cells.setdefault(path_key(matrix, arm), []).append((matrix, arm))
        runs = []
        for key, members in cells.items():
            # Cheapest representative: the task with the fewest rollouts still
            # exercises the identical code, and there is no reason to probe on
            # the expensive one.
            matrix, arm = min(members, key=lambda m: (FULL_RUN_ROLLOUTS[m[0]], m[0]))
            launcher, dataset, method, modules = key
            label = f"{launcher}/{dataset}/{method}/{_MODULE_SHORT.get(modules, modules)}"
            runs.append(_probe_run(matrix, arm, label, len(members)))
        return sorted(runs, key=lambda r: (-r.gpus, r.label))

    runs = []
    for matrix in MATRICES:
        if matrix in EXCLUDED_MATRICES:
            continue
        cells = {}
        for arm in _build(matrix):
            key = config_key(arm) if level == "config" else (arm.method,)
            cells.setdefault(key, []).append(arm)
        for cell in cells.values():
            arm = _representative(cell)
            label = (
                config_label(config_key(arm)) if level == "config" else arm.method
            )
            runs.append(_probe_run(matrix, arm, label, len(cell)))
    return runs


def steady_seconds(rollout_seconds: list[float]) -> float | None:
    """Cheapest rollout time after dropping the first.

    Rollout 1 carries compilation, weight load and the first allocator growth,
    so it is excluded outright. Of what remains, the MINIMUM is taken rather
    than the median.

    The median was wrong, and measurably so. A probe runs three rollouts, so
    dropping the first leaves two -- and on two samples a median IS the mean.
    The last rollout of a probe also writes the run's checkpoint, so that cost
    landed squarely in the per-rollout figure. Measured on the FullFT RL arm on
    2026-08-01, `[308.0, 59.0, 677.0]`: the 677 is 59s of rollout plus a 616.5s
    write of 15 GB to Lustre, and `median(59, 677) = 368` put the campaign
    estimate at 931 h against a true ~453 h. Every OFT row was distorted the
    same way; LoRA all-modules escaped only because its adapter checkpoint is
    negligible.

    A rollout's duration is a fixed steady cost plus whatever one-off happened
    to land in it -- compile, eval, allocator growth, checkpoint. The minimum is
    the sample least contaminated by those, which is the same reasoning the
    kernel benchmark's `_time_ms` uses. It is not the fastest-possible rollout
    being passed off as typical: nothing here makes a rollout cheaper than
    steady state, only more expensive.

    Checkpoints are not thereby ignored -- they are priced explicitly, see
    `extra_saves`.
    """
    steady = rollout_seconds[1:]
    return min(steady) if steady else None


# The launcher's own cadence (`--save-interval "${SAVE_INTERVAL:-50}"`), pinned
# by test_the_launcher_save_interval_is_the_one_the_estimate_uses so a change
# there cannot leave this estimate silently wrong.
SAVE_INTERVAL = 50


def extra_saves(full_rollouts: int, probe_rollouts: int) -> int:
    """Checkpoints a real arm writes beyond the one the probe already paid for.

    A probe run writes its checkpoint once, at the end, and that cost is already
    inside its measured wall clock -- so it lands in `overhead`, which the
    estimate adds once. Only the ADDITIONAL writes a longer arm performs are
    charged on top; counting all of them would bill the first one twice.

    At SAVE_INTERVAL=50 a 500-rollout arm writes 10 and the probe wrote 1, so 9
    are added. For FullFT at ~616s each that is ~1.5h per arm -- small against
    the 8.2h of rollouts, but not nothing, and it is the entire reason the
    checkpoint is removed from `steady` rather than left to inflate it.
    """
    real = full_rollouts // SAVE_INTERVAL
    already_paid = 1 if probe_rollouts else 0
    return max(0, real - already_paid)


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}h{(total % 3600) // 60:02d}m"


def format_report(records: list[dict], level: str = "method") -> str:
    """The probe's answer to both questions, one row per planned run.

    Keyed on the arm name, not on (task, method): at `config` level a task has
    several rows per method, and a (task, method) key would collapse e2's batch
    32 onto its batch 512 -- which is precisely the pair whose difference the
    config level exists to measure.
    """
    if level == "path":
        # Rows are still the 24 (task, method) pairs -- that is the deliverable.
        # Each reads the pace measured on ITS code path, which is what makes 13
        # runs answer 24 questions.
        by_path = {}
        for r in records:
            matrix = r.get("matrix")
            if matrix not in MATRIX_METRICS:
                continue
            launcher = "rl" if MATRIX_METRICS[matrix] == "accuracy" else "sft"
            by_path[(launcher, r.get("dataset") or "tulu3", r.get("method"),
                     r.get("target_modules") or "")] = r
        by_key = {}
        for run in probe_plan("method"):
            arm = next(a for a in _build(run.matrix) if a.name == run.arm)
            record = by_path.get(path_key(run.matrix, arm))
            if record is not None:
                by_key[(run.matrix, run.arm)] = record
        level = "method"  # the rows below are per (task, method) from here on
    else:
        by_key = {(r.get("matrix"), r.get("arm")): r for r in records}
    lines = [
        f"{'task':9} {'configuration':22} {'gpu':>3} {'status':8} {'steady/roll':>12} "
        f"{'x rolls':>8} {'one arm':>8} {'arms':>5} {'all arms':>10}",
        "-" * 96,
    ]
    campaign = 0.0
    unknown = 0
    for run in probe_plan(level):
        record = by_key.get((run.matrix, run.arm))
        if record is None:
            lines.append(
                f"{run.matrix:9} {run.label:22} {run.gpus:>3} {'not run':8} "
                f"{'-':>12} {run.full_rollouts:>8} {'-':>8} {run.arms_in_config:>5} {'-':>10}"
            )
            unknown += 1
            continue
        status = "ok" if record.get("status") == "ok" else "FAILED"
        steady = steady_seconds(record.get("rollout_seconds") or [])
        if steady is None:
            lines.append(
                f"{run.matrix:9} {run.label:22} {run.gpus:>3} {status:8} "
                f"{'?':>12} {run.full_rollouts:>8} {'?':>8} {run.arms_in_config:>5} {'?':>10}"
            )
            unknown += 1
            continue
        # Startup is whatever the probe spent outside its rollouts, and every
        # real arm pays it once too -- so it is added once, not amortised away.
        # It also contains the probe's single checkpoint write, which is why
        # `extra_saves` charges only the additional ones.
        overhead = max(0.0, float(record.get("seconds") or 0.0) - steady * len(record["rollout_seconds"]))
        # Checkpoints, priced explicitly rather than smeared into `steady`.
        # A ledger written before save timings were recorded has no
        # `save_seconds`; those rows keep the old behaviour -- one save, already
        # inside `overhead` -- so the estimate is low rather than missing.
        saves = record.get("save_seconds") or []
        save_cost = (
            statistics.mean(saves)
            * extra_saves(run.full_rollouts, len(record.get("rollout_seconds") or []))
            if saves
            else 0.0
        )
        one_arm = overhead + steady * run.full_rollouts + save_cost
        # Arms sharing this configuration -- its LR grid width. At `config`
        # level that is what this row stands for; summing them reconstructs the
        # matrix exactly, with no arm counted twice and none omitted.
        multiplier = run.arms_in_config if level == "config" else run.arms_of_method
        all_arms = one_arm * multiplier
        campaign += all_arms
        lines.append(
            f"{run.matrix:9} {run.label:22} {run.gpus:>3} {status:8} "
            f"{steady:>11.1f}s {run.full_rollouts:>8} {_hms(one_arm):>8} "
            f"{multiplier:>5} {_hms(all_arms):>10}"
        )
    lines.append("-" * 96)
    lines.append(
        "campaign estimate, serial wall clock summed over every arm: "
        + (_hms(campaign) if campaign else "-- nothing measured yet")
    )
    if unknown:
        lines.append(
            f"{unknown} run(s) produced no per-rollout time, so the total is a "
            "LOWER BOUND -- it omits them rather than guessing."
        )
    lines.append(
        "Rows are 3-rollout probes. `one arm` = startup + steady x that arm's own "
        "rollout count; `all arms` multiplies by the arms this row stands for. "
        "Concurrency is not modelled: run 8 one-GPU arms at once and the wall "
        "clock divides, but each arm's own time does not."
        + (
            "\nAt method level one row stands for every rank, block size, "
            "placement and batch size in its task. Rank and placement barely move "
            "step time; BATCH SIZE does -- e2 runs 32/128/512, so its estimate is "
            "low by roughly the batch ratio for two thirds of its arms."
            if level == "method" else ""
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="one TSV line per planned run, for the shell driver")
    plan.add_argument("--gpus", type=int, default=None, help="only runs needing this many GPUs")
    plan.add_argument(
        "--level", choices=PROBE_LEVELS, default="method",
        help="method (default): one run per (task, method), 24 runs. Rank, block "
             "size, placement and batch size exercise the same code at different "
             "shapes, so they add no coverage. config: one run per distinct "
             "configuration, 61 runs -- for hunting a shape-dependent OOM.",
    )
    report = sub.add_parser("report", help="read probe ledgers and estimate the campaign")
    report.add_argument(
        "--level", choices=PROBE_LEVELS, default="method",
        help="must match the level the probe ran at",
    )
    report.add_argument(
        "--ledger", nargs="+", required=True,
        help="paths or globs. Concurrent probe runs each write their own file, "
             "because two processes appending to one ledger interleave lines.",
    )
    args = parser.parse_args(argv)

    if args.command == "plan":
        for run in probe_plan(args.level):
            if args.gpus is not None and run.gpus != args.gpus:
                continue
            print(
                "\t".join(
                    [run.matrix, run.method, run.arm, run.only, str(run.gpus),
                     run.metric, str(run.full_rollouts), run.label]
                )
            )
        return 0

    paths: list[Path] = []
    for entry in args.ledger:
        paths.extend(Path(p) for p in sorted(glob.glob(str(entry))))
    if not paths:
        # Still print the table: every row reads "not run", which is the honest
        # state of a probe that has not started and is more useful than an error.
        print(format_report([], args.level))
        return 1
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated final line from an interrupted write
    print(format_report(records, args.level))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
