#!/usr/bin/env python3
"""Summarize per-rank, per-step Stable-TP data from a PyTorch profiler trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any


PHASE_NAMES = {
    "operator_backward": "opd_dagger/stable_tp/operator_backward",
    "full_local_lse": "opd_dagger/stable_tp/full_local_lse",
    "full_tp_lse": "opd_dagger/stable_tp/full_tp_lse",
    "rest_local_lse": "opd_dagger/stable_tp/rest_local_lse",
    "rest_tp_lse": "opd_dagger/stable_tp/rest_tp_lse",
    "selected_tp_sum": "opd_dagger/stable_tp/selected_tp_sum",
}
FORWARD_PREFIX = "opd_dagger/stable_tp/operator_forward"
FORWARD_SHAPE = re.compile(r"\|rows=(\d+)\|vlocal=(\d+)\|k=(\d+)$")
RANK_PATTERN = re.compile(r"rank_(\d+)")
COLLECTIVE_PHASE_MULTIPLIERS = {
    "full_tp_lse": 2,
    "rest_tp_lse": 2,
    "selected_tp_sum": 1,
}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_ms(events: list[dict[str, Any]]) -> float:
    return sum(_number(event.get("dur")) or 0.0 for event in events) / 1000.0


def _complete_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("ph") == "X" and _number(event.get("ts")) is not None and _number(event.get("dur")) is not None
    ]


def _is_gpu_user_annotation(event: dict[str, Any]) -> bool:
    return "gpu_user_annotation" in str(event.get("cat", "")).lower()


def _is_cpu_user_annotation(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    return "user_annotation" in category and "gpu_user_annotation" not in category


def _interval(event: dict[str, Any]) -> tuple[float, float]:
    start = float(event["ts"])
    return start, start + float(event["dur"])


def _inside(event: dict[str, Any], intervals: list[tuple[float, float]]) -> bool:
    timestamp = _number(event.get("ts"))
    return timestamp is not None and any(start <= timestamp < end for start, end in intervals)


def _is_nccl_kernel(event: dict[str, Any]) -> bool:
    name = str(event.get("name", "")).lower()
    category = str(event.get("cat", "")).lower()
    return event.get("ph") == "X" and "nccl" in name and ("kernel" in name or "kernel" in category)


def _is_collective_api(event: dict[str, Any]) -> bool:
    """Match the one-per-call c10d all-reduce CPU event, not nested wrappers."""

    if event.get("ph") != "X" or _is_gpu_user_annotation(event) or _is_nccl_kernel(event):
        return False
    name = str(event.get("name", "")).lower()
    return name in {"c10d::allreduce", "c10d::allreduce_"}


def _memory_samples(events: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    samples = []
    for event in events:
        if event.get("name") != "[memory]":
            continue
        timestamp = _number(event.get("ts"))
        args = event.get("args") or {}
        allocated = _number(args.get("Total Allocated"))
        reserved = _number(args.get("Total Reserved"))
        if timestamp is None or allocated is None or reserved is None:
            continue
        samples.append((timestamp, allocated, reserved))
    return sorted(samples)


def _memory_stats(
    samples: list[tuple[float, float, float]],
    intervals: list[tuple[float, float]],
) -> tuple[float | str, float | str, float | str, float | str]:
    peak_allocated = None
    peak_reserved = None
    max_allocated_delta = None
    max_reserved_delta = None

    for start, end in intervals:
        prior = [sample for sample in samples if sample[0] <= start]
        baseline = prior[-1] if prior else None
        active = [sample for sample in samples if start <= sample[0] < end]
        values = ([baseline] if baseline is not None else []) + active
        if not values:
            continue

        interval_peak_allocated = max(sample[1] for sample in values)
        interval_peak_reserved = max(sample[2] for sample in values)
        peak_allocated = max(peak_allocated or interval_peak_allocated, interval_peak_allocated)
        peak_reserved = max(peak_reserved or interval_peak_reserved, interval_peak_reserved)
        if baseline is not None:
            allocated_delta = interval_peak_allocated - baseline[1]
            reserved_delta = interval_peak_reserved - baseline[2]
            max_allocated_delta = max(max_allocated_delta or allocated_delta, allocated_delta)
            max_reserved_delta = max(max_reserved_delta or reserved_delta, reserved_delta)

    def to_mib(value: float | None) -> float | str:
        return "" if value is None else round(value / (1024 * 1024), 6)

    return (
        to_mib(peak_allocated),
        to_mib(peak_reserved),
        to_mib(max_allocated_delta),
        to_mib(max_reserved_delta),
    )


def summarize_events(events: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    complete = _complete_events(events)
    cpu_annotations = [event for event in complete if _is_cpu_user_annotation(event)]
    gpu_annotations = [event for event in complete if _is_gpu_user_annotation(event)]
    steps = sorted(
        (event for event in cpu_annotations if str(event.get("name", "")).startswith("ProfilerStep#")),
        key=lambda event: float(event["ts"]),
    )
    if not steps:
        raise ValueError(f"{source}: no ProfilerStep events found")

    rank_match = RANK_PATTERN.search(source.name)
    rank = int(rank_match.group(1)) if rank_match else -1
    memory = _memory_samples(events)
    rows = []

    for step in steps:
        step_interval = [_interval(step)]
        step_events = [event for event in complete if _inside(event, step_interval)]
        step_cpu_annotations = [event for event in cpu_annotations if _inside(event, step_interval)]
        step_gpu_annotations = [event for event in gpu_annotations if _inside(event, step_interval)]
        forward_events = [
            event for event in step_cpu_annotations if str(event.get("name", "")).startswith(FORWARD_PREFIX)
        ]
        phase_events = {
            key: [event for event in step_cpu_annotations if event.get("name") == phase_name]
            for key, phase_name in PHASE_NAMES.items()
        }
        collective_phase_intervals = {
            key: [_interval(event) for event in phase_events[key]] for key in COLLECTIVE_PHASE_MULTIPLIERS
        }
        collective_intervals = [
            interval for intervals in collective_phase_intervals.values() for interval in intervals
        ]
        forward_intervals = [_interval(event) for event in forward_events]

        shapes = []
        for event in forward_events:
            match = FORWARD_SHAPE.search(str(event.get("name", "")))
            if match:
                shapes.append(tuple(int(value) for value in match.groups()))

        step_nccl = [event for event in step_events if _is_nccl_kernel(event)]
        opd_nccl = [event for event in step_nccl if _inside(event, collective_intervals)]
        collective_api_by_phase = {
            key: [
                event
                for event in step_events
                if _is_collective_api(event) and _inside(event, collective_phase_intervals[key])
            ]
            for key in COLLECTIVE_PHASE_MULTIPLIERS
        }
        opd_collective_api = [
            event for phase_events_list in collective_api_by_phase.values() for event in phase_events_list
        ]
        opd_gpu_annotations = [
            event for event in step_gpu_annotations if str(event.get("name", "")).startswith("opd_dagger/stable_tp/")
        ]
        step_memory = _memory_stats(memory, step_interval)
        operator_memory = _memory_stats(memory, forward_intervals)

        expected_collectives = sum(
            multiplier * len(phase_events[key]) for key, multiplier in COLLECTIVE_PHASE_MULTIPLIERS.items()
        )
        step_name = str(step.get("name", "ProfilerStep#unknown"))
        rows.append(
            {
                "trace_file": source.name,
                "rank": rank,
                "profiler_step": step_name.removeprefix("ProfilerStep#"),
                "step_ms": round(float(step["dur"]) / 1000.0, 6),
                "operator_forward_calls": len(forward_events),
                "operator_forward_ms": round(_duration_ms(forward_events), 6),
                "operator_backward_calls": len(phase_events["operator_backward"]),
                "operator_backward_ms": round(_duration_ms(phase_events["operator_backward"]), 6),
                "response_rows_total": sum(shape[0] for shape in shapes),
                "response_rows_max_call": max((shape[0] for shape in shapes), default=0),
                "vocab_local": ";".join(str(value) for value in sorted({shape[1] for shape in shapes})),
                "top_k": ";".join(str(value) for value in sorted({shape[2] for shape in shapes})),
                "full_local_lse_calls": len(phase_events["full_local_lse"]),
                "full_local_lse_ms": round(_duration_ms(phase_events["full_local_lse"]), 6),
                "full_tp_lse_calls": len(phase_events["full_tp_lse"]),
                "full_tp_lse_ms": round(_duration_ms(phase_events["full_tp_lse"]), 6),
                "rest_local_lse_calls": len(phase_events["rest_local_lse"]),
                "rest_local_lse_ms": round(_duration_ms(phase_events["rest_local_lse"]), 6),
                "rest_tp_lse_calls": len(phase_events["rest_tp_lse"]),
                "rest_tp_lse_ms": round(_duration_ms(phase_events["rest_tp_lse"]), 6),
                "selected_tp_sum_calls": len(phase_events["selected_tp_sum"]),
                "selected_tp_sum_ms": round(_duration_ms(phase_events["selected_tp_sum"]), 6),
                "expected_opd_tp_collectives": expected_collectives,
                "opd_collective_api_calls": len(opd_collective_api),
                "opd_collective_api_ms": round(_duration_ms(opd_collective_api), 6),
                "full_tp_lse_collective_api_calls": len(collective_api_by_phase["full_tp_lse"]),
                "rest_tp_lse_collective_api_calls": len(collective_api_by_phase["rest_tp_lse"]),
                "selected_tp_sum_collective_api_calls": len(collective_api_by_phase["selected_tp_sum"]),
                "opd_gpu_annotation_calls": len(opd_gpu_annotations),
                "opd_gpu_annotation_ms": round(_duration_ms(opd_gpu_annotations), 6),
                "opd_nccl_kernel_calls": len(opd_nccl),
                "opd_nccl_kernel_ms": round(_duration_ms(opd_nccl), 6),
                "step_nccl_kernel_calls": len(step_nccl),
                "step_nccl_kernel_ms": round(_duration_ms(step_nccl), 6),
                "step_peak_allocated_mib": step_memory[0],
                "step_peak_reserved_mib": step_memory[1],
                "operator_peak_allocated_mib": operator_memory[0],
                "operator_peak_reserved_mib": operator_memory[1],
                "operator_max_allocated_delta_mib": operator_memory[2],
                "operator_max_reserved_delta_mib": operator_memory[3],
            }
        )
    return rows


def _load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("traceEvents", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected traceEvents list")
    return payload


def _find_traces(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    traces = set(path.rglob("*.pt.trace.json")) | set(path.rglob("*.pt.trace.json.gz"))
    return sorted(traces)


def _validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        identity = f"rank={row['rank']} step={row['profiler_step']}"
        forward_calls = int(row["operator_forward_calls"])
        if forward_calls == 0:
            errors.append(f"{identity}: no Stable-TP forward range")
            continue
        for key in (
            "operator_backward_calls",
            "full_local_lse_calls",
            "full_tp_lse_calls",
            "rest_local_lse_calls",
            "rest_tp_lse_calls",
            "selected_tp_sum_calls",
        ):
            if int(row[key]) != forward_calls:
                errors.append(f"{identity}: {key}={row[key]}, expected {forward_calls}")
        if int(row["expected_opd_tp_collectives"]) != 5 * forward_calls:
            errors.append(
                f"{identity}: expected_opd_tp_collectives={row['expected_opd_tp_collectives']}, "
                f"expected {5 * forward_calls}"
            )
        expected_observed_by_phase = {
            "full_tp_lse_collective_api_calls": 2 * forward_calls,
            "rest_tp_lse_collective_api_calls": 2 * forward_calls,
            "selected_tp_sum_collective_api_calls": forward_calls,
        }
        for key, expected in expected_observed_by_phase.items():
            if int(row[key]) != expected:
                errors.append(f"{identity}: {key}={row[key]}, expected observed c10d count {expected}")
        if int(row["opd_collective_api_calls"]) != 5 * forward_calls:
            errors.append(
                f"{identity}: opd_collective_api_calls={row['opd_collective_api_calls']}, "
                f"expected observed c10d count {5 * forward_calls}"
            )
        if row["operator_peak_allocated_mib"] == "":
            errors.append(f"{identity}: no CUDA memory samples inside Stable-TP forward ranges")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path, help="Profiler trace file or directory")
    parser.add_argument("--output", type=Path, default=None, help="Per-rank/per-step CSV output path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when CPU phase, observed c10d collective, or memory invariants are missing",
    )
    args = parser.parse_args()

    traces = _find_traces(args.trace_path)
    if not traces:
        parser.error(f"no *.pt.trace.json[.gz] files found under {args.trace_path}")

    rows = []
    for trace in traces:
        rows.extend(summarize_events(_load_trace(trace), trace))
    rows.sort(key=lambda row: (int(row["rank"]), str(row["profiler_step"]), str(row["trace_file"])))

    output_dir = args.trace_path if args.trace_path.is_dir() else args.trace_path.parent
    output = args.output or output_dir / "03p-step-profile.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    errors = _validate_rows(rows)
    print(f"wrote {len(rows)} rank-step rows from {len(traces)} traces to {output}")
    if errors:
        print("profile validation findings:")
        for error in errors:
            print(f"  - {error}")
        return 2 if args.strict else 0
    print("profile CPU-phase/observed-c10d/memory invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
