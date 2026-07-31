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

from tools.lora_regret.arms import MATRICES
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

# E5's OFT cell has no centre until e5scout runs. Any value produces a valid
# *plumbing* probe -- the learning rate does not change how long a step takes --
# so the probe supplies the midpoint of the scout span and says so. It must
# never leak into a real sweep, which is why e5 still requires the flag there.
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
    "e5scout": SFT_SWEEP_ROLLOUTS,
    "e5": SFT_SWEEP_ROLLOUTS,
}


# What counts as one probe. `config` is the default and the honest unit: it
# collapses ONLY the learning rate, which is a scalar multiply and so changes
# neither step time nor memory. Every other axis -- rank, OFT block size, target
# modules, batch size -- changes both, and collapsing them is how a probe misses
# the arm that OOMs. `method` is the cheap version: one run per (task, method),
# which covers 26 of the 61 configurations and leaves e2's batch 512, e1's rank
# 512 and e5's block 256 unlaunched.
PROBE_LEVELS = ("config", "method")


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
    centre = PROBE_OFT_CENTRE if matrix == "e5" else None
    model = get_model(DEFAULT_MODEL)
    return MATRICES[matrix](model.hidden_size, model.ffn_size, 0, centre, None)


_MODULE_SHORT = {
    "linear_qkv,linear_proj,linear_fc1,linear_fc2": "all",
    "linear_qkv,linear_proj": "attn",
    "linear_fc1,linear_fc2": "mlp",
    "": "-",
}


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


def probe_plan(level: str = "config") -> list[ProbeRun]:
    """One run per distinct configuration (default) or per (task, method)."""
    if level not in PROBE_LEVELS:
        raise ValueError(f"unknown probe level {level!r}; known: {PROBE_LEVELS}")
    runs: list[ProbeRun] = []
    for matrix in MATRICES:
        if matrix in EXCLUDED_MATRICES:
            continue
        arms = _build(matrix)
        metric = MATRIX_METRICS[matrix]
        cells: dict[tuple, list] = {}
        for arm in arms:
            key = config_key(arm) if level == "config" else (arm.method,)
            cells.setdefault(key, []).append(arm)
        for cell in cells.values():
            arm = _representative(cell)
            method = arm.method
            label = (
                config_label(config_key(arm)) if level == "config" else f"{method}/*/*"
            )
            runs.append(
                ProbeRun(
                    matrix=matrix,
                    method=method,
                    arm=arm.name,
                    # Anchored at both ends: an unanchored name is a prefix of
                    # any longer one, and `--only` takes a regex, so a bare name
                    # could select two arms and bill the second to the first.
                    only=f"^{re.escape(arm.name)}$",
                    gpus=_gpus(method, metric),
                    metric=metric,
                    full_rollouts=FULL_RUN_ROLLOUTS[matrix],
                    # Arms of THIS method in this matrix, not the matrix total:
                    # the estimate is multiplied by it, and e1's 45 arms are 5
                    # FullFT plus 35 LoRA plus 5 OFT with very different costs.
                    arms_of_method=sum(1 for a in arms if a.method == method),
                    project=wandb_project(matrix),
                    label=label,
                    arms_in_config=len(cell),
                )
            )
    return runs


def steady_seconds(rollout_seconds: list[float]) -> float | None:
    """Median rollout time after dropping the first.

    Rollout 1 carries compilation, weight load and the first allocator growth.
    On a 2000-rollout arm, averaging it in moves the estimate by hours, and in
    the wrong direction -- it always overstates.
    """
    steady = rollout_seconds[1:]
    return statistics.median(steady) if steady else None


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}h{(total % 3600) // 60:02d}m"


def format_report(records: list[dict], level: str = "config") -> str:
    """The probe's answer to both questions, one row per planned run.

    Keyed on the arm name, not on (task, method): at `config` level a task has
    several rows per method, and a (task, method) key would collapse e2's batch
    32 onto its batch 512 -- which is precisely the pair whose difference the
    config level exists to measure.
    """
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
        overhead = max(0.0, float(record.get("seconds") or 0.0) - steady * len(record["rollout_seconds"]))
        one_arm = overhead + steady * run.full_rollouts
        # Arms sharing this configuration -- its LR grid width. At `config`
        # level that is what this row stands for; summing them reconstructs the
        # matrix exactly, with no arm counted twice and none omitted.
        multiplier = run.arms_in_config if level == "config" else run.arms_of_method
        all_arms = one_arm * multiplier
        campaign += all_arms
        lines.append(
            f"{run.matrix:10} {run.method:7} {run.gpus:>3} {status:8} "
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
        "rollout count; `all arms` multiplies by the arms sharing that "
        "configuration (its LR grid). Concurrency is not modelled: run 8 one-GPU "
        "arms at once and the wall clock divides, but each arm's own time does not."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="one TSV line per planned run, for the shell driver")
    plan.add_argument("--gpus", type=int, default=None, help="only runs needing this many GPUs")
    plan.add_argument(
        "--level", choices=PROBE_LEVELS, default="config",
        help="config (default): one run per distinct configuration, collapsing "
             "only the learning rate. method: one per (task, method) -- cheaper, "
             "and it never launches e2's batch 512, e1's rank 512 or e5's block "
             "256, which are the three most likely to OOM.",
    )
    report = sub.add_parser("report", help="read probe ledgers and estimate the campaign")
    report.add_argument(
        "--level", choices=PROBE_LEVELS, default="config",
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
