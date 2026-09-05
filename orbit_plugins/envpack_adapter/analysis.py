"""Offline analysis helpers for envpack rollout dumps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GroupOutcome:
    step: int
    bucket: str
    group_id: str
    solved: int
    total: int

    @property
    def category(self) -> str:
        if self.solved <= 0:
            return "none_solved"
        if self.solved >= self.total:
            return "all_solved"
        return "mixed"


def summarize_dapo_groups(root: str | Path) -> list[dict[str, Any]]:
    """Summarize DAPO keep/drop categories from debug-dump ``record.json`` files.

    The input may be a ``train`` debug-dump directory, ``stepXXXX`` directory,
    or a single ``record.json``. Records are grouped by
    ``ids.step`` and ``ids.group_index`` because DAPO decides keep/drop at the
    prompt-group level, not per trajectory.
    """

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for path in _iter_record_paths(Path(root)):
        record = _load_record(path)
        step = int(((record.get("ids") or {}).get("step")) or 0)
        ids = record.get("ids") or {}
        group_index = ids.get("group_index")
        group_id = str(group_index if group_index is not None else path.parent.name)
        groups[(step, group_id)].append(record)

    outcomes = [_group_outcome(step, group_id, records) for (step, group_id), records in groups.items()]
    return _summarize_outcomes(outcomes)


def _iter_record_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.name != "record.json":
            raise ValueError(f"expected record.json file, got {root}")
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(root)
    yield from sorted(root.rglob("record.json"))


def _load_record(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _group_outcome(step: int, group_id: str, records: list[dict[str, Any]]) -> GroupOutcome:
    bucket_names = {_bucket_name(record) for record in records}
    bucket_names.discard("")
    bucket = sorted(bucket_names)[0] if bucket_names else "unknown_bucket"
    solved = sum(1 for record in records if _is_solved(record))
    return GroupOutcome(step=step, bucket=bucket, group_id=group_id, solved=solved, total=len(records))


def _bucket_name(record: dict[str, Any]) -> str:
    env = record.get("env") or {}
    bucket = env.get("bucket_name")
    if bucket:
        return str(bucket)
    solver_metrics = env.get("solver_metrics")
    if isinstance(solver_metrics, dict) and solver_metrics.get("bucket_name"):
        return str(solver_metrics["bucket_name"])
    return ""


def _is_solved(record: dict[str, Any]) -> bool:
    outcome = record.get("outcome") or {}
    return bool(outcome.get("traj_success"))


def _summarize_outcomes(outcomes: list[GroupOutcome]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[GroupOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.step, outcome.bucket)].append(outcome)
        grouped[(outcome.step, "_overall")].append(outcome)

    rows: list[dict[str, Any]] = []
    for (step, bucket), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        none_solved = sum(1 for value in values if value.category == "none_solved")
        mixed = sum(1 for value in values if value.category == "mixed")
        all_solved = sum(1 for value in values if value.category == "all_solved")
        total_groups = len(values)
        solved = sum(value.solved for value in values)
        total = sum(value.total for value in values)
        rows.append(
            {
                "step": step,
                "bucket": bucket,
                "groups": total_groups,
                "none_solved": none_solved,
                "mixed": mixed,
                "all_solved": all_solved,
                "dapo_keep_rate": mixed / total_groups if total_groups else 0.0,
                "solve_rate": solved / total if total else 0.0,
                "trajectories": total,
            }
        )
    return rows


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ("step", "bucket", "groups", "none_solved", "mixed", "all_solved", "dapo_keep_rate", "solve_rate")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["step"]),
                    str(row["bucket"]),
                    str(row["groups"]),
                    str(row["none_solved"]),
                    str(row["mixed"]),
                    str(row["all_solved"]),
                    f"{row['dapo_keep_rate']:.3f}",
                    f"{row['solve_rate']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _format_tsv(rows: list[dict[str, Any]]) -> str:
    headers = (
        "step",
        "bucket",
        "groups",
        "none_solved",
        "mixed",
        "all_solved",
        "dapo_keep_rate",
        "solve_rate",
        "trajectories",
    )
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["step"]),
                    str(row["bucket"]),
                    str(row["groups"]),
                    str(row["none_solved"]),
                    str(row["mixed"]),
                    str(row["all_solved"]),
                    f"{row['dapo_keep_rate']:.6f}",
                    f"{row['solve_rate']:.6f}",
                    str(row["trajectories"]),
                ]
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize envpack DAPO keep/drop categories from debug dumps.")
    parser.add_argument("root", help="Train debug-dump dir, step dir, or one record.json")
    parser.add_argument("--format", choices=("markdown", "json", "tsv"), default="markdown")
    args = parser.parse_args(argv)

    rows = summarize_dapo_groups(args.root)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "tsv":
        print(_format_tsv(rows))
    else:
        print(_format_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
