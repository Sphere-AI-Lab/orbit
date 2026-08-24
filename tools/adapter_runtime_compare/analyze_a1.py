#!/usr/bin/env python3
"""Summarize adapter-runtime-compare logs into the A1 sync-cost table.

Reads every ``<output_dir>/<run_id>/*.log``, extracts ``perf N: {...}``
records carrying ``perf/update_weights_time``, and reports per (model, mode):
update wall time (mean/p50), payload MB, pause seconds, and the achieved
fraction of link bandwidth — the column that makes the full-model arm
strawman-proof (spec: A1). Transport is not in the logs; state it in the
figure caption (async arms: NCCL; colocated on this cluster: cpu_gather).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

from tools.adapter_runtime_compare.run_compare import METRIC_RE, parse_payload

RUN_ID_RE = re.compile(
    r"r\d+_(?P<branch>[^_]+)_(?P<model>.+)_(?P<precision>bf16|fp8|int4)_"
    r"(?P<peft>[^_]+)_(?P<mode>sync|async|async_db|async_fullft)_g"
)

TIME_KEY = "perf/update_weights_time"
BYTES_KEY = "perf/update_weights_payload_bytes"
PAUSE_KEY = "perf/update_weights_pause_time"


def iter_update_records(log_path: Path):
    for line in log_path.read_text(errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if not match or match.group("kind") != "perf":
            continue
        payload = parse_payload(match.group("payload"))
        if payload and TIME_KEY in payload:
            yield payload


def _iter_run_dirs(output_dir: Path):
    """Yield (run_dir, id_match) pairs, descending one level into any child
    directory whose name doesn't itself look like a run id -- run_compare.py
    (~line 339) nests runs under <output-dir>/<campaign>/<run_id>/ when
    --campaign is used, so the campaign directory needs one extra hop. Only
    one level of recursion is applied (no deeper)."""
    for entry in sorted(Path(output_dir).iterdir()):
        if not entry.is_dir():
            continue
        id_match = RUN_ID_RE.match(entry.name)
        if id_match:
            yield entry, id_match
            continue
        for child in sorted(entry.iterdir()):
            child_match = RUN_ID_RE.match(child.name)
            if child_match and child.is_dir():
                yield child, child_match


def summarize(output_dir: Path, link_gbps: float) -> list[dict]:
    rows = []
    for run_dir, id_match in _iter_run_dirs(output_dir):
        times, bytes_, pauses = [], [], []
        for log_path in sorted(run_dir.glob("*.log")):
            for rec in iter_update_records(log_path):
                times.append(float(rec[TIME_KEY]))
                bytes_.append(float(rec.get(BYTES_KEY, 0.0)))
                pauses.append(float(rec.get(PAUSE_KEY, 0.0)))
        if not times:
            print(f"warning: no {TIME_KEY} records in {run_dir.name}", file=sys.stderr)
            continue
        mean_t = statistics.mean(times)
        link_bytes_per_s = link_gbps / 8.0 * 1e9
        rows.append({
            "model": id_match.group("model"),
            "mode": id_match.group("mode"),
            "n_updates": len(times),
            "update_s_mean": mean_t,
            "update_s_p50": statistics.median(times),
            "payload_mb_mean": statistics.mean(bytes_) / 1e6,
            "pause_s_mean": statistics.mean(pauses),
            "bw_frac": (statistics.mean(bytes_) / mean_t) / link_bytes_per_s,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--link-gbps", type=float, required=True,
                        help="Nominal interconnect bandwidth for bw_frac (e.g. 400 for NDR IB)")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = parser.parse_args(argv)

    rows = summarize(args.output_dir, args.link_gbps)
    if not rows:
        print("no runs with update_weights records found", file=sys.stderr)
        return 1
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(
            f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
