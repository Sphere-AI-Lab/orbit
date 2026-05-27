#!/usr/bin/env python3
"""Summarize PEFT-Arena math eval metrics into a curve-friendly CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PASS_K_COLUMNS = ("pass@1", "pass@2", "pass@4", "pass@8", "pass@16")
FIELDNAMES = (
    "iter",
    "iter_name",
    "dataset",
    "acc",
    "pass_acc",
    *PASS_K_COLUMNS,
    "metrics_file",
)


def _iter_sort_key(iter_name: str) -> int:
    match = re.search(r"(\d+)$", iter_name)
    return int(match.group(1)) if match else -1


def _find_iter_name(path: Path, run_dir: Path) -> str | None:
    try:
        rel_parts = path.relative_to(run_dir).parts
    except ValueError:
        return None
    for part in rel_parts:
        if part.startswith("iter_"):
            return part
    return None


def _metrics_row(metrics_path: Path, run_dir: Path) -> dict[str, str] | None:
    iter_name = _find_iter_name(metrics_path, run_dir)
    if iter_name is None:
        return None

    with metrics_path.open(encoding="utf-8") as f:
        metrics: dict[str, Any] = json.load(f)

    pass_at_k = metrics.get("pass@k") or {}
    dataset = metrics_path.parent.name
    row = {
        "iter": str(_iter_sort_key(iter_name)),
        "iter_name": iter_name,
        "dataset": dataset,
        "acc": _stringify_metric(metrics.get("acc")),
        "pass_acc": _stringify_metric(metrics.get("pass_acc")),
        "metrics_file": str(metrics_path),
    }
    for column in PASS_K_COLUMNS:
        key = column.removeprefix("pass@")
        row[column] = _stringify_metric(pass_at_k.get(key, pass_at_k.get(int(key))))
    return row


def _stringify_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def collect_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for metrics_path in sorted(run_dir.glob("iter_*/**/*metrics.json")):
        row = _metrics_row(metrics_path, run_dir)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row["iter"]), row["dataset"], row["metrics_file"]))


def write_summary(run_dir: Path, output_path: Path) -> None:
    rows = collect_rows(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PEFT-Arena math eval metrics.")
    parser.add_argument("--run-dir", required=True, help="Eval results run directory, e.g. eval_results/<run>")
    parser.add_argument("--output", default=None, help="CSV output path. Defaults to <run-dir>/summary.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else run_dir / "summary.csv"
    write_summary(run_dir, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
