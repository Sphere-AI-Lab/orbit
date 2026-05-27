#!/usr/bin/env python3
"""Sample per-process GPU memory with NVML.

This is intentionally external to the training process. It lets us identify
which PID/process owns the peak without adding hooks to Megatron or SGLang.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def classify_process(command: str) -> str:
    lowered = command.lower()
    if "megatrontrainrayactor" in lowered or "train_actor" in lowered:
        return "megatron_train"
    if "sglang" in lowered or "launch_server" in lowered or "scheduler" in lowered:
        return "sglang"
    if "raylet" in lowered or "gcs_server" in lowered or "ray::" in lowered:
        return "ray"
    return "other"


def update_peak_records(peaks: dict[tuple[int, int], dict[str, Any]], record: dict[str, Any]) -> None:
    key = (int(record["gpu"]), int(record["pid"]))
    previous = peaks.get(key)
    if previous is None or int(record["used_bytes"]) > int(previous["used_bytes"]):
        peaks[key] = dict(record)


def _read_cmdline(pid: int) -> str:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        raw = proc_cmdline.read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _process_name(pid: int) -> str:
    proc_comm = Path(f"/proc/{pid}/comm")
    try:
        return proc_comm.read_text().strip()
    except OSError:
        return ""


def _nvml_processes(nvml, handle):
    processes = []
    for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
        getter = getattr(nvml, getter_name, None)
        if getter is None:
            continue
        try:
            processes.extend(getter(handle))
        except Exception:
            continue
    return processes


def sample_once(nvml) -> list[dict[str, Any]]:
    timestamp = time.time()
    records: list[dict[str, Any]] = []
    device_count = nvml.nvmlDeviceGetCount()
    for gpu in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
        for proc in _nvml_processes(nvml, handle):
            pid = int(proc.pid)
            used_bytes = int(getattr(proc, "usedGpuMemory", 0) or 0)
            command = _read_cmdline(pid)
            name = _process_name(pid)
            records.append(
                {
                    "time": timestamp,
                    "gpu": gpu,
                    "pid": pid,
                    "used_bytes": used_bytes,
                    "used_gib": used_bytes / 1024**3,
                    "name": name,
                    "component": classify_process(f"{name} {command}"),
                    "cmdline": command,
                }
            )
    return records


def _open_output(path: str | None):
    if path is None or path == "-":
        return sys.stdout
    return open(path, "a", buffering=1)


def _write_summary(path: str | None, peaks: dict[tuple[int, int], dict[str, Any]]) -> None:
    if not path:
        return
    ordered = sorted(peaks.values(), key=lambda item: (item["gpu"], -item["used_bytes"], item["pid"]))
    Path(path).write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds; 0 means run until killed.")
    parser.add_argument("--output", default="-", help="JSONL output path; '-' means stdout.")
    parser.add_argument("--summary", default=None, help="Optional JSON file with peak memory per GPU/PID.")
    args = parser.parse_args(argv)

    import pynvml

    pynvml.nvmlInit()
    peaks: dict[tuple[int, int], dict[str, Any]] = {}
    start = time.monotonic()
    output = _open_output(args.output)
    try:
        while True:
            for record in sample_once(pynvml):
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                update_peak_records(peaks, record)
            _write_summary(args.summary, peaks)
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            time.sleep(args.interval)
    finally:
        if output is not sys.stdout:
            output.close()
        pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
