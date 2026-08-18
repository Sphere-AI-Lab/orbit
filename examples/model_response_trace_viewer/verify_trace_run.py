#!/usr/bin/env python3
"""Check that a finished trace-viewer smoke run actually produced traces.

    python3 examples/model_response_trace_viewer/verify_trace_run.py <run-dir>

Stdlib only, so it runs with the login node's bare python3 -- no conda env, no
PIL. Reports every check rather than stopping at the first failure, because
when a run is expensive you want the whole picture in one pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Emitted by Miles' own rollout metrics. Its presence proves the hook returned
# False and layered, rather than suppressing the built-in logging.
DEFAULT_METRICS_RE = re.compile(r"\bperf \d+:")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")
        if not ok:
            self.failures.append(label)
        return ok


def _expected_from_args(run_dir: Path) -> tuple[int | None, int | None]:
    """Read the run's own args.json so expectations track the recipe, not a guess.

    launch_miles.sbatch writes a flat {"flag-name": value} mapping (no leading
    dashes). An argv list is also accepted so this keeps working if that changes.
    """
    args_path = run_dir / "args.json"
    if not args_path.is_file():
        return None, None
    try:
        parsed = json.loads(args_path.read_text())
    except json.JSONDecodeError:
        return None, None

    def value(flag: str) -> int | None:
        raw = None
        if isinstance(parsed, dict):
            raw = parsed.get(flag, parsed.get(f"--{flag}"))
        elif isinstance(parsed, list):
            for i, tok in enumerate(parsed[:-1]):
                if tok in (f"--{flag}", flag):
                    raw = parsed[i + 1]
                    break
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    steps = value("num-rollout")
    batch, samples = value("rollout-batch-size"), value("n-samples-per-prompt")
    return steps, (batch * samples if batch and samples else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expect-steps", type=int, default=None)
    parser.add_argument("--expect-per-step", type=int, default=None)
    opts = parser.parse_args()

    run_dir: Path = opts.run_dir.expanduser().resolve()
    report = Report()
    print(f"run dir: {run_dir}")

    if not report.check(run_dir.is_dir(), "run directory exists"):
        return 1

    steps_from_args, per_step_from_args = _expected_from_args(run_dir)
    expect_steps = opts.expect_steps if opts.expect_steps is not None else steps_from_args
    expect_per_step = opts.expect_per_step if opts.expect_per_step is not None else per_step_from_args
    print(
        f"expecting: steps={expect_steps if expect_steps is not None else '?'} "
        f"records/step={expect_per_step if expect_per_step is not None else '?'}\n"
    )

    train_dir = run_dir / "traces" / "train"
    if not report.check(train_dir.is_dir(), "trace tree written", str(train_dir)):
        print(
            "\nNo traces at all — the hook never ran. Check run.log for the "
            "--custom-rollout-log-function-path value and any import error."
        )
        return 1

    steps = sorted(p for p in train_dir.iterdir() if p.is_dir() and p.name.startswith("step"))
    report.check(bool(steps), "at least one trace step", f"{len(steps)} found")
    if expect_steps is not None:
        report.check(len(steps) == expect_steps, "step count matches --num-rollout", f"{len(steps)} vs {expect_steps}")

    total_records = 0
    bad_json: list[str] = []
    bad_png: list[str] = []
    without_messages: list[str] = []
    with_images = 0
    multi_turn = 0

    for step in steps:
        records = sorted(p for p in step.iterdir() if p.is_dir())
        total_records += len(records)
        if expect_per_step is not None and len(records) != expect_per_step:
            report.check(False, f"{step.name} record count", f"{len(records)} vs {expect_per_step}")
        for record in records:
            payload = record / "record.json"
            if not payload.is_file():
                bad_json.append(str(record))
                continue
            try:
                data = json.loads(payload.read_text())
            except json.JSONDecodeError:
                bad_json.append(str(payload))
                continue
            if not data.get("conversation", {}).get("messages"):
                without_messages.append(str(payload))
            if data.get("counts", {}).get("n_images", 0) > 0:
                with_images += 1
            if data.get("outcome", {}).get("num_turns", 0) > 1:
                multi_turn += 1
            for image in record.glob("*.png"):
                if image.read_bytes()[:8] != PNG_MAGIC:
                    bad_png.append(str(image))

    report.check(total_records > 0, "trace records written", f"{total_records} total")
    report.check(not bad_json, "every record.json parses", f"{len(bad_json)} bad")
    report.check(not bad_png, "every image is a real PNG", f"{len(bad_png)} bad")
    report.check(not without_messages, "every record carries a conversation", f"{len(without_messages)} without")
    # Images prove the prompt diagram survived into the trace, not just text.
    report.check(with_images > 0, "records carry images", f"{with_images}/{total_records}")
    # >1 turn proves multi-turn capture reached the trace via metadata, rather
    # than only the single final response Miles would record on its own.
    report.check(multi_turn > 0, "at least one multi-turn record", f"{multi_turn} found")

    log = run_dir / "run.log"
    if log.is_file():
        text = log.read_text(errors="replace")
        report.check(
            bool(DEFAULT_METRICS_RE.search(text)),
            "default rollout metrics still logged (hook layered, not suppressed)",
        )
        # Third-party libraries (wandb) print tracebacks from atexit handlers at
        # interpreter shutdown. Those are teardown noise, not run failures.
        lines = text.splitlines()
        real = [
            i
            for i, line in enumerate(lines)
            if "Traceback (most recent call last)" in line
            and "Exception ignored in atexit callback" not in (lines[i - 1] if i else "")
        ]
        total = sum("Traceback (most recent call last)" in line for line in lines)
        report.check(
            not real,
            "no tracebacks in run.log (atexit teardown noise excluded)",
            f"{len(real)} real, {total - len(real)} atexit",
        )
    else:
        print("  [skip] run.log not found — cannot check default-metrics layering")

    print()
    if report.failures:
        print(f"FAILED ({len(report.failures)}): " + "; ".join(report.failures))
        return 1
    print("All checks passed — the trace viewer works through the customization hook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
