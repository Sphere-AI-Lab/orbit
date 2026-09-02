#!/usr/bin/env python3
"""Summarize an adapter-runtime-compare campaign into the A3 parity/speedup tables.

A3 (docs/plans/2026-08-17-adapter-first-experiments-design.md) compares three
arms -- sync OFT, async double-buffer OFT, async full-FT -- over >= 3 seeds and
reports two figures from the same runs: reward vs samples (parity, with the
seed spread as the noise floor) and reward vs wall-clock (speedup), plus the
attribution decomposition sync -> async_fullft (overlap) -> async_db (adapter).

Per run this reads ``<campaign>/<run_id>/console.log`` (and ``run.log`` when
present) for the trainer's ``rollout N: {...}`` records (``rollout/raw_reward``),
the ``perf N: {...}`` records (step, update, pause, rollout, wait times) and the
``step N: {...}`` records (train/rollout log-prob gap). Wall-clock comes from the
``[YYYY-mm-dd HH:MM:SS]`` prefix the loggers stamp on every line, measured from
the first rollout record of the run. Samples per rollout come from the run
manifest (``ROLLOUT_BATCH_SIZE`` x ``N_SAMPLES_PER_PROMPT``).

Outputs markdown tables (per run, per arm over seeds, reward at sample
checkpoints, attribution) and, with ``--json``, the raw per-rollout series for
plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.adapter_runtime_compare.run_compare import parse_payload, strip_ansi

RUN_ID_RE = re.compile(
    r"r(?P<repeat>\d+)_(?P<branch>[^_]+)_(?P<model>.+)_(?P<precision>bf16|fp8|int4)_"
    r"(?P<peft>[^_]+)_(?P<mode>sync|async|async_db|async_fullft)_g"
)
RECORD_RE = re.compile(r"\b(?P<kind>rollout|perf|step) (?P<rollout>\d+): (?P<payload>\{.*\})")
STAMP_RE = re.compile(r"\[(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

REWARD_KEY = "rollout/raw_reward"
PERF_KEYS = (
    "perf/step_time",
    "perf/update_weights_time",
    "perf/update_weights_pause_time",
    "perf/rollout_time",
    "perf/train_wait_time",
    "perf/tokens_per_gpu_per_sec",
)
LOGPROB_GAP_KEY = "train/train_rollout_logprob_abs_diff"
ARM_ORDER = ("sync", "async_fullft", "async_db", "async")


@dataclass
class RunSeries:
    run_id: str
    mode: str
    seed: int | None
    samples_per_rollout: int
    reward: dict[int, float] = field(default_factory=dict)
    stamp: dict[int, datetime] = field(default_factory=dict)
    perf: dict[int, dict[str, float]] = field(default_factory=dict)
    logprob_gap: dict[int, float] = field(default_factory=dict)

    @property
    def rollouts(self) -> list[int]:
        return sorted(self.reward)

    def samples(self, rollout: int) -> int:
        return (rollout + 1) * self.samples_per_rollout

    def wall_s(self, rollout: int) -> float | None:
        """Seconds from the first rollout record to ``rollout``'s record."""
        if not self.stamp or rollout not in self.stamp:
            return None
        first = self.stamp[min(self.stamp)]
        return (self.stamp[rollout] - first).total_seconds()


def _parse_stamp(line: str) -> datetime | None:
    match = STAMP_RE.search(line)
    return datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S") if match else None


def _samples_per_rollout(run_dir: Path) -> int:
    manifest = run_dir / "run.json"
    if not manifest.exists():
        return 1
    env = json.loads(manifest.read_text()).get("env", {})
    return int(env.get("ROLLOUT_BATCH_SIZE", 1)) * int(env.get("N_SAMPLES_PER_PROMPT", 1))


def _seed(run_dir: Path, repeat: int) -> int | None:
    manifest = run_dir / "run.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text())
    return data.get("seed") if data.get("seed") is not None else None


def parse_run(run_dir: Path, id_match: re.Match[str]) -> RunSeries:
    series = RunSeries(
        run_id=run_dir.name,
        mode=id_match.group("mode"),
        seed=_seed(run_dir, int(id_match.group("repeat"))),
        samples_per_rollout=_samples_per_rollout(run_dir),
    )
    for log_name in ("console.log", "run.log"):
        log_path = run_dir / log_name
        if not log_path.exists():
            continue
        for raw in log_path.read_text(errors="replace").splitlines():
            line = strip_ansi(raw)
            match = RECORD_RE.search(line)
            if not match:
                continue
            payload = parse_payload(match.group("payload"))
            if not payload:
                continue
            rollout = int(match.group("rollout"))
            kind = match.group("kind")
            if kind == "rollout" and REWARD_KEY in payload:
                series.reward[rollout] = float(payload[REWARD_KEY])
                stamp = _parse_stamp(line)
                if stamp is not None:
                    series.stamp.setdefault(rollout, stamp)
            elif kind == "perf" and "perf/step_time" in payload:
                series.perf[rollout] = {k: float(payload[k]) for k in PERF_KEYS if k in payload}
            elif kind == "step" and LOGPROB_GAP_KEY in payload:
                series.logprob_gap.setdefault(rollout, float(payload[LOGPROB_GAP_KEY]))
    return series


def iter_runs(campaign_dir: Path) -> list[RunSeries]:
    runs = []
    for entry in sorted(Path(campaign_dir).iterdir()):
        id_match = RUN_ID_RE.match(entry.name)
        if entry.is_dir() and id_match:
            runs.append(parse_run(entry, id_match))
    return runs


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    return _mean(values), _std(values)


def run_summary(run: RunSeries, *, last_k: int, warm_from: int) -> dict[str, Any]:
    rollouts = run.rollouts
    tail = rollouts[-last_k:]
    warm = [r for r in run.perf if r >= warm_from]
    perf_mean = {
        key: _mean([run.perf[r][key] for r in warm if key in run.perf[r]]) for key in PERF_KEYS
    }
    last = rollouts[-1] if rollouts else None
    return {
        "run_id": run.run_id,
        "mode": run.mode,
        "seed": run.seed,
        "n_rollouts": len(rollouts),
        "samples": run.samples(last) if last is not None else 0,
        "wall_s": run.wall_s(last) if last is not None else None,
        "reward_last_k": _mean([run.reward[r] for r in tail]),
        "reward_mean": _mean([run.reward[r] for r in rollouts]),
        "step_s": perf_mean["perf/step_time"],
        "update_s": perf_mean["perf/update_weights_time"],
        "pause_s": perf_mean["perf/update_weights_pause_time"],
        "rollout_s": perf_mean["perf/rollout_time"],
        "train_wait_s": perf_mean["perf/train_wait_time"],
        "tok_per_gpu_s": perf_mean["perf/tokens_per_gpu_per_sec"],
        "logprob_gap": _mean(list(run.logprob_gap.values())),
    }


def reward_at_samples(run: RunSeries, target_samples: int, *, window: int) -> float | None:
    """Mean raw reward over the ``window`` rollouts ending at the last rollout
    whose cumulative sample count is <= ``target_samples`` (None if unreached)."""
    eligible = [r for r in run.rollouts if run.samples(r) <= target_samples]
    if not eligible:
        return None
    end = eligible[-1]
    tail = [r for r in run.rollouts if r <= end][-window:]
    return _mean([run.reward[r] for r in tail])


def arm_summary(
    runs: list[RunSeries], *, last_k: int, warm_from: int, checkpoints: list[int], window: int
) -> list[dict[str, Any]]:
    rows = []
    modes = [m for m in ARM_ORDER if any(r.mode == m for r in runs)]
    for mode in modes:
        arm_runs = [r for r in runs if r.mode == mode and r.rollouts]
        summaries = [run_summary(r, last_k=last_k, warm_from=warm_from) for r in arm_runs]
        row: dict[str, Any] = {"mode": mode, "n_seeds": len(arm_runs)}
        for key in ("reward_last_k", "wall_s", "step_s", "update_s", "pause_s", "train_wait_s", "tok_per_gpu_s"):
            row[key], row[key + "_std"] = _mean_std([s[key] for s in summaries if s[key] is not None])
        for target in checkpoints:
            values = [v for r in arm_runs if (v := reward_at_samples(r, target, window=window)) is not None]
            row[f"reward@{target}"] = _mean(values)
            row[f"reward@{target}_std"] = _std(values)
        rows.append(row)
    return rows


def attribution(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wall-clock decomposition: sync -> async_fullft (overlap, not adapter-specific)
    -> async_db (the adapter contribution: payload + swap). Missing arms are skipped."""
    by_mode = {row["mode"]: row for row in arm_rows if row.get("wall_s") is not None}
    steps = [("sync", "async_fullft", "overlap (any async system)"), ("async_fullft", "async_db", "adapter push + swap")]
    rows = []
    for src, dst, label in steps:
        if src in by_mode and dst in by_mode:
            a, b = by_mode[src]["wall_s"], by_mode[dst]["wall_s"]
            rows.append({"from": src, "to": dst, "what": label, "wall_s_from": a, "wall_s_to": b, "speedup": a / b if b else None})
    if "sync" in by_mode and "async_db" in by_mode:
        a, b = by_mode["sync"]["wall_s"], by_mode["async_db"]["wall_s"]
        rows.append({"from": "sync", "to": "async_db", "what": "total", "wall_s_from": a, "wall_s_to": b, "speedup": a / b if b else None})
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pm(mean: Any, std: Any, digits: int = 3) -> str:
    if mean is None:
        return "-"
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")


def render_markdown(
    runs: list[RunSeries], arm_rows: list[dict[str, Any]], attr_rows: list[dict[str, Any]], *,
    last_k: int, warm_from: int, checkpoints: list[int]
) -> str:
    out = []
    out.append("## Per run")
    out.append(
        f"| run | mode | seed | rollouts | samples | wall s | reward (last {last_k}) | step s | update s | "
        "pause s | wait s | tok/GPU/s | lp gap |"
    )
    out.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for run in runs:
        s = run_summary(run, last_k=last_k, warm_from=warm_from)
        out.append(
            f"| {s['run_id']} | {s['mode']} | {_fmt(s['seed'])} | {s['n_rollouts']} | {s['samples']} | {_fmt(s['wall_s'], 0)} | "
            f"{_fmt(s['reward_last_k'])} | {_fmt(s['step_s'], 2)} | {_fmt(s['update_s'])} | {_fmt(s['pause_s'])} | "
            f"{_fmt(s['train_wait_s'], 2)} | {_fmt(s['tok_per_gpu_s'], 0)} | {_fmt(s['logprob_gap'], 4)} |"
        )
    out.append("")
    out.append(f"## Per arm (mean ± std over seeds; timings over warm rollouts >= {warm_from})")
    out.append(f"| arm | seeds | reward (last {last_k}) | wall s | step s | update s | pause s | wait s | tok/GPU/s |")
    out.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for row in arm_rows:
        out.append(
            f"| {row['mode']} | {row['n_seeds']} | {_fmt_pm(row['reward_last_k'], row['reward_last_k_std'])} | "
            f"{_fmt_pm(row['wall_s'], row['wall_s_std'], 0)} | {_fmt_pm(row['step_s'], row['step_s_std'], 2)} | "
            f"{_fmt_pm(row['update_s'], row['update_s_std'])} | {_fmt_pm(row['pause_s'], row['pause_s_std'])} | "
            f"{_fmt_pm(row['train_wait_s'], row['train_wait_s_std'], 2)} | {_fmt_pm(row['tok_per_gpu_s'], row['tok_per_gpu_s_std'], 0)} |"
        )
    if checkpoints:
        out.append("")
        out.append("## Reward at matched sample counts (parity; mean ± std over seeds)")
        out.append("| arm | " + " | ".join(f"@{c}" for c in checkpoints) + " |")
        out.append("|:--|" + "--:|" * len(checkpoints))
        for row in arm_rows:
            cells = [_fmt_pm(row.get(f"reward@{c}"), row.get(f"reward@{c}_std")) for c in checkpoints]
            out.append(f"| {row['mode']} | " + " | ".join(cells) + " |")
    if attr_rows:
        out.append("")
        out.append("## Wall-clock attribution (same sample budget)")
        out.append("| from | to | what | wall s from | wall s to | speedup |")
        out.append("|:--|:--|:--|--:|--:|--:|")
        for row in attr_rows:
            out.append(
                f"| {row['from']} | {row['to']} | {row['what']} | {_fmt(row['wall_s_from'], 0)} | "
                f"{_fmt(row['wall_s_to'], 0)} | {_fmt(row['speedup'], 2)}× |"
            )
    return "\n".join(out) + "\n"


def series_json(runs: list[RunSeries]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.run_id,
            "mode": run.mode,
            "seed": run.seed,
            "samples_per_rollout": run.samples_per_rollout,
            "rollouts": [
                {
                    "rollout": r,
                    "samples": run.samples(r),
                    "wall_s": run.wall_s(r),
                    "reward": run.reward[r],
                    **{k.split("/", 1)[1]: v for k, v in run.perf.get(r, {}).items()},
                    "logprob_gap": run.logprob_gap.get(r),
                }
                for r in run.rollouts
            ],
        }
        for run in runs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaign_dir", help="<output-dir>/<campaign> holding the run directories")
    parser.add_argument("--last-k", type=int, default=20, help="rollouts in the final-reward window")
    parser.add_argument("--warm-from", type=int, default=1, help="first rollout counted in timing means")
    parser.add_argument("--checkpoints", default="", help="comma-separated sample counts for the parity table")
    parser.add_argument("--window", type=int, default=10, help="rollouts averaged at each checkpoint")
    parser.add_argument("--csv", help="write per-run rows here")
    parser.add_argument("--json", help="write per-rollout series here")
    args = parser.parse_args(argv)

    runs = iter_runs(Path(args.campaign_dir))
    runs = [r for r in runs if r.rollouts] or runs
    if not runs:
        print(f"no runs under {args.campaign_dir}", file=sys.stderr)
        return 1
    checkpoints = [int(c) for c in args.checkpoints.split(",") if c.strip()]
    arm_rows = arm_summary(runs, last_k=args.last_k, warm_from=args.warm_from, checkpoints=checkpoints, window=args.window)
    attr_rows = attribution(arm_rows)
    sys.stdout.write(render_markdown(runs, arm_rows, attr_rows, last_k=args.last_k, warm_from=args.warm_from, checkpoints=checkpoints))
    if args.csv:
        rows = [run_summary(r, last_k=args.last_k, warm_from=args.warm_from) for r in runs]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(series_json(runs), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
