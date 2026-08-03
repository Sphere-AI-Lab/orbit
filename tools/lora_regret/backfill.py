"""Recover what the E4 ledgers lost, from the logs that were written anyway.

Seven gsm8k columns ran to completion and every arm landed in its ledger as
`accuracy: null, status: "failed"`. Neither training nor the node was at fault:

  1. train.py's generation-eval call omitted `num_rollout`, so
     `should_run_periodic_action`'s final-rollout branch was dead. At
     EVAL_INTERVAL=100000 -- chosen precisely to mean "once, at the end" -- the
     modulo never matched either, so the arms produced ZERO post-training evals.
     The only `eval` line in those logs is rollout 0's, from the separate
     eval-before-train branch: the untrained policy.
  2. `parse_final_accuracy` was handed a fixed ("math_test", "gsm8k_test") while
     `arm_env` had told the launcher to score gsm8k alone. It fails closed on a
     missing dataset, so even that rollout-0 eval parsed to None.

Both are fixed. Neither fix retrieves a number nobody measured, and the arms
saved no checkpoints (SAVE_INTERVAL is empty by protocol, deliberately), so
held-out accuracy for those arms is gone -- they have to be re-run.

What is NOT gone is the training-reward curve. `raw_reward` is the mean reward
over the batch the policy just generated, and with --rm-type math the reward is
exactly 1 or 0, so it is accuracy on the training batch: a real learning curve,
logged every rollout, for ~40 node-hours of finished work. It answers what the
campaign was actually asking -- which learning rates train and which blow up,
and how wide each method's stable band is -- just on train rather than on
held-out data. This writes it out.

Two rules this deliberately follows:

  Reward is never written to `accuracy`. `analyze` picks argmins off that field,
  and a figure built from training reward while labelled held-out accuracy is a
  worse outcome than a missing figure.

  A backfilled row is never promoted to `status: "ok"` on the strength of a
  reward curve. `campaign.sh` skips ok arms on resume, so promoting them would
  quietly retire exactly the arms that most need re-running. A row IS promoted
  when a genuine POST-TRAINING eval is recovered -- which is what the arms
  running under the fixed train.py, but recorded by an already-imported old
  sweep.py, will need.

Usage:

    python -m tools.lora_regret.backfill --ledgers 'results/e4_gsm8k_lr*.jsonl'

Writes results/backfill/<ledger>.jsonl by default and prints a summary table.
It does not touch the source ledgers unless asked (--in-place), because the
campaigns append to those files while they run.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from tools.lora_regret.probe_log import (
    last_run_segment,
    parse_reward_trace,
    parse_rollout_seconds,
)
from tools.lora_regret.sweep import parse_final_accuracy, rl_eval_datasets

# How many trailing rollouts make up the "final" reward. One rollout is 32
# problem groups and the batch-to-batch spread on a healthy arm is several
# points, so a single last value is noise; ten is ~1/15th of a 150-rollout run,
# short enough to still be the end of training.
FINAL_WINDOW = 10

# An arm counts as having learned something if its peak windowed reward clears
# this. Below it the run never left its starting reward (measured at 0.02-0.03
# on gsm8k) and "collapsed" would be the wrong word for it -- it never rose.
LEARNED_PEAK = 0.10

# Collapse: the end of the run is under this fraction of the peak. Deliberately
# blunt, because the collapses in this campaign are not subtle -- reward goes to
# exactly 0.000 and stays there, once response length reaches the 2,048-token
# cap and every answer grades 0 for having lost its \boxed{...}.
COLLAPSE_FRACTION = 0.05

# A ledger being appended to right now by a live campaign. --in-place rewrites
# the whole file, which would drop any row `append_result` adds between the read
# and the write.
LIVE_LEDGER_SECONDS = 1800


def _window_mean(trace: list[dict], start: int, stop: int) -> float | None:
    values = [point["reward"] for point in trace[start:stop]]
    return sum(values) / len(values) if values else None


def summarize(trace: list[dict]) -> dict:
    """Peak, final and a one-word verdict for a reward curve.

    The peak is over WINDOWED means, not raw rollouts: a single lucky batch on
    an otherwise-dead arm would otherwise set the peak and make every later
    rollout look like a collapse from it.
    """
    if not trace:
        return {"verdict": "no-trace", "reward_peak": None, "reward_final": None,
                "reward_peak_rollout": None, "collapse_rollout": None}

    windows = [
        (trace[start]["rollout"], _window_mean(trace, start, start + FINAL_WINDOW))
        for start in range(max(1, len(trace) - FINAL_WINDOW + 1))
    ]
    peak_rollout, peak = max(windows, key=lambda pair: pair[1])
    final = _window_mean(trace, len(trace) - FINAL_WINDOW, len(trace))

    collapse_rollout = None
    if peak >= LEARNED_PEAK and final < COLLAPSE_FRACTION * peak:
        verdict = "collapsed"
        # The first rollout after the peak from which nothing recovers. Reported
        # rather than just the fact of collapse because where an arm dies is the
        # measurement: a run that peaks at 0.67 by rollout 45 and dies at 90 is
        # a different statement about its learning rate than one that never rose.
        after_peak = [p for p in trace if p["rollout"] > peak_rollout]
        for index, point in enumerate(after_peak):
            rest = after_peak[index:]
            if all(p["reward"] < COLLAPSE_FRACTION * peak for p in rest) and len(rest) >= 5:
                collapse_rollout = point["rollout"]
                break
    elif peak < LEARNED_PEAK:
        verdict = "never-learned"
    else:
        verdict = "learned"

    return {
        "verdict": verdict,
        "reward_peak": peak,
        "reward_peak_rollout": peak_rollout,
        "reward_final": final,
        "collapse_rollout": collapse_rollout,
    }


def recover(log_text: str, datasets: tuple[str, ...]) -> dict:
    """Everything this log still has to say, from its most recent invocation."""
    segment = last_run_segment(log_text)
    trace = parse_reward_trace(segment)

    # Rollout 0's eval is the UNTRAINED policy -- the eval-before-train branch
    # fires on rollout 0 regardless of interval. Kept because it is a free
    # baseline check (0.032 on gsm8k, against the protocol's stated 0.02-0.03)
    # and thrown away as an arm's score, which is what it would silently have
    # become had `parse_final_accuracy` simply taken the highest eval it found.
    accuracy, eval_rollout, per_dataset = parse_final_accuracy(segment, datasets)
    before_train = accuracy if eval_rollout == 0 else None
    post_train = accuracy if (eval_rollout or 0) > 0 else None

    return {
        "reward_trace": trace,
        "rollouts_completed": len(trace),
        "runs_in_log": log_text.count("\nLogging to ") + log_text.startswith("Logging to "),
        "driver_exited": "Training driver exited" in segment,
        "rollout_seconds": parse_rollout_seconds(segment),
        "accuracy_before_train": before_train,
        "accuracy": post_train,
        "accuracy_per_dataset": per_dataset if post_train is not None else {},
        "eval_rollout": eval_rollout,
        **summarize(trace),
    }


def backfill_row(row: dict, logs_dir: Path) -> dict:
    """One ledger row, plus whatever its log still holds. Identity is preserved
    exactly; only measurement fields are added or replaced."""
    log_path = logs_dir / f"{row['arm']}.log"
    out = dict(row)
    if not log_path.exists():
        return {**out, "backfill": "no-log", "verdict": "no-log"}

    # The datasets THIS arm evaluated. Read off the row's own `dataset` rather
    # than assumed, for the same reason the bug existed: a gsm8k arm and a math
    # arm write different keys and neither writes both.
    dataset = row.get("dataset")
    datasets = rl_eval_datasets({"EVAL_DATASETS": dataset} if dataset else {})

    recovered = recover(log_path.read_text(encoding="utf-8", errors="replace"), datasets)
    out.update(recovered)
    # Promoted ONLY by a real post-training eval. A reward curve, however
    # complete, leaves the row `failed` so campaign.sh re-runs the arm.
    out["status"] = "ok" if recovered["accuracy"] is not None else row.get("status", "failed")
    out["backfill"] = "reward-trace" if recovered["accuracy"] is None else "accuracy+reward-trace"
    return out


def _round(row: dict, key: str) -> str:
    return "-" if row.get(key) is None else f"{row[key]:.3f}"


def _plain(row: dict, key: str) -> str:
    return "-" if row.get(key) is None else str(row[key])


def _live(path: Path) -> bool:
    return time.time() - path.stat().st_mtime < LIVE_LEDGER_SECONDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ledgers", nargs="+", default=["results/e4_gsm8k_lr*.jsonl"],
                        help="Glob(s) over ledger JSONL files.")
    parser.add_argument("--logs-dir", default="logs/lora_regret")
    parser.add_argument("--out-dir", default="results/backfill",
                        help="Where the enriched ledgers go. Ignored with --in-place.")
    parser.add_argument("--in-place", action="store_true",
                        help="Rewrite the source ledgers instead. Refuses on a ledger "
                             "touched in the last 30 minutes unless --force.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-trace", action="store_true",
                        help="Drop the per-rollout curve, keeping only the summary.")
    args = parser.parse_args(argv)

    paths = sorted({Path(p) for pattern in args.ledgers for p in glob.glob(pattern)})
    if not paths:
        print(f"no ledgers matched {args.ledgers}", file=sys.stderr)
        return 1

    logs_dir = Path(args.logs_dir)
    rows_by_path: dict[Path, list[dict]] = {}
    for path in paths:
        if args.in_place and _live(path) and not args.force:
            print(f"REFUSING {path}: modified {int(time.time() - path.stat().st_mtime)}s ago; a "
                  "campaign is probably appending to it. Drop --in-place, or pass --force.",
                  file=sys.stderr)
            return 2
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        rows_by_path[path] = [backfill_row(row, logs_dir) for row in rows]

    print(f"{'arm':<34}{'verdict':<15}{'peak':>7}{'@':>6}{'final':>8}"
          f"{'collapse':>10}{'rollouts':>10}{'runs':>6}")
    for rows in rows_by_path.values():
        for row in rows:
            print(f"{row['arm']:<34}{row.get('verdict', '-'):<15}"
                  f"{_round(row, 'reward_peak'):>7}{_plain(row, 'reward_peak_rollout'):>6}"
                  f"{_round(row, 'reward_final'):>8}{_plain(row, 'collapse_rollout'):>10}"
                  f"{_plain(row, 'rollouts_completed'):>10}{_plain(row, 'runs_in_log'):>6}")

    out_dir = Path(args.out_dir)
    for path, rows in rows_by_path.items():
        target = path if args.in_place else out_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                if args.no_trace:
                    row = {k: v for k, v in row.items() if k != "reward_trace"}
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {target}", file=sys.stderr)

    promoted = sum(1 for rows in rows_by_path.values() for r in rows if r["status"] == "ok")
    total = sum(len(rows) for rows in rows_by_path.values())
    print(f"\n{promoted}/{total} rows carry a post-training accuracy; the rest stay "
          "`failed` on purpose, so campaign.sh re-runs those arms.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
