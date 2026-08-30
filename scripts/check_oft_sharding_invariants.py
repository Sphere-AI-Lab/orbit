#!/usr/bin/env python3
"""Cross-rank sharding invariant checker for split OFT audits.

Consumes:
  - PEFT wrap audit JSONL files written by ``miles/orbit/audit/peft_wrap.py``:
    ``<wrap-base>.megatron.tp{T}_pp{P}.jsonl`` (one file per TP/PP rank).
    Each file contains module records (no per-record tp_rank field) plus
    one summary record with ``summary: true`` and ``tp_rank``/``tp_size``/
    ``pp_rank``/``ep_rank``/``ep_size``. The checker reads tp_rank from
    the filename and pairs each file's modules with that file's summary.
  - SGLang streamed audit JSONL files written by ``_streamed_audit.py``:
    ``<streamed-base>.sglang_streamed.tp{N}.jsonl``. Each record carries
    ``tp_rank`` directly.

Asserts:
  A. ColumnParallel split wrappers (OFTLinearSplitQKV, OFTLinearSplitFC1UpGate,
     OFTLinearGroupedSplitFC1UpGate) have IDENTICAL adapter R_shape across
     all TP ranks at the same (module, pp_rank).
  B. RowParallel adapters (modules ending linear_proj, linear_fc2, o_proj,
     down_proj) have the same R_shape[0] on every TP rank, AND records appear
     for all TP ranks in the run.
  C. For grouped expert wraps: len(adapter_gate_R_shapes) ==
     len(adapter_up_R_shapes) per record, and num_local_experts * ep_size
     equals --num-moe-experts when supplied.
  D. Streamed audit: no expert_id appears under more than one tp_rank for
     a given (layer_id, proj).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


_WRAP_FILE_RE = re.compile(r"\.megatron\.tp(?P<tp>\d+)_pp(?P<pp>\d+)\.jsonl$")
_STREAMED_FILE_RE = re.compile(r"\.sglang_streamed\.tp(?P<tp>\d+)\.jsonl$")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_wrap_files(wrap_arg: Path) -> list[Path]:
    """Accept either a base path (will glob ``.megatron.tp*_pp*.jsonl``
    siblings), a directory (recursively glob), or a single file."""
    if wrap_arg.is_file():
        return [wrap_arg]
    if wrap_arg.is_dir():
        return sorted(wrap_arg.rglob("*.megatron.tp*_pp*.jsonl"))
    parent = wrap_arg.parent if wrap_arg.parent.exists() else Path(".")
    pattern = f"{wrap_arg.name}.megatron.tp*_pp*.jsonl"
    return sorted(parent.glob(pattern))


def _resolve_streamed_files(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if base.is_dir():
        return sorted(base.rglob("*.sglang_streamed.tp*.jsonl"))
    parent = base.parent if base.parent.exists() else Path(".")
    pattern = f"{base.name}.sglang_streamed.tp*.jsonl"
    return sorted(parent.glob(pattern))


def _decorate_wrap_events(files: list[Path]) -> list[dict]:
    """Load each per-rank file and attach tp_rank/pp_rank to every record
    (extracted from the filename -- module records don't carry it themselves).
    """
    events: list[dict] = []
    for path in files:
        match = _WRAP_FILE_RE.search(path.name)
        tp_rank = int(match.group("tp")) if match else 0
        pp_rank = int(match.group("pp")) if match else 0
        for event in _load_jsonl(path):
            event.setdefault("tp_rank", tp_rank)
            event.setdefault("pp_rank", pp_rank)
            events.append(event)
    return events


def _decorate_streamed_events(files: list[Path]) -> list[dict]:
    events: list[dict] = []
    for path in files:
        match = _STREAMED_FILE_RE.search(path.name)
        tp_rank = int(match.group("tp")) if match else 0
        for event in _load_jsonl(path):
            event.setdefault("tp_rank", tp_rank)
            events.append(event)
    return events


def _summary(events: list[dict]) -> dict[int, dict]:
    """Index summary records by tp_rank."""
    out: dict[int, dict] = {}
    for event in events:
        if event.get("summary"):
            out[int(event["tp_rank"])] = event
    return out


def _adapter_R_shape_for(event: dict) -> tuple | None:
    """Pick the first per-adapter R shape that uniquely identifies the wrapper."""
    for key in ("adapter_R_shape", "adapter_q_R_shape", "adapter_gate_R_shape"):
        shape = event.get(key)
        if shape is not None:
            return tuple(shape)
    return None


def _check_column_parallel_replicated(wrap_events: list[dict]) -> list[str]:
    failures: list[str] = []
    column_wrappers = {
        "OFTLinearSplitQKV",
        "OFTLinearSplitFC1UpGate",
        "OFTLinearGroupedSplitFC1UpGate",
    }
    grouped: dict[tuple, set[tuple]] = defaultdict(set)
    for event in wrap_events:
        if event.get("wrapper") not in column_wrappers:
            continue
        shape = _adapter_R_shape_for(event)
        if shape is None:
            continue
        key = (event.get("module"), int(event.get("pp_rank", 0)))
        grouped[key].add(shape)
    for key, shapes in grouped.items():
        if len(shapes) > 1:
            failures.append(
                f"column-parallel wrapper at {key[0]} (pp_rank={key[1]}) "
                f"has inconsistent R shapes across TP ranks: {sorted(shapes)}"
            )
    return failures


def _check_row_parallel_sliced(
    wrap_events: list[dict], summaries: dict[int, dict]
) -> list[str]:
    failures: list[str] = []
    row_suffixes = ("linear_proj", "linear_fc2", "o_proj", "down_proj")
    grouped: dict[str, dict[int, int]] = defaultdict(dict)
    for event in wrap_events:
        if event.get("summary"):
            continue
        module = event.get("module") or ""
        if not any(module.endswith(suffix) for suffix in row_suffixes):
            continue
        shape = _adapter_R_shape_for(event)
        if shape is None or not shape:
            continue
        grouped[module][int(event["tp_rank"])] = int(shape[0])
    tp_size_seen = max((s["tp_size"] for s in summaries.values()), default=1)
    for module, by_rank in grouped.items():
        unique = set(by_rank.values())
        if len(unique) > 1:
            failures.append(
                f"row-parallel module {module} has uneven per-rank R block counts: "
                f"{by_rank}"
            )
        if tp_size_seen > 1 and len(by_rank) != tp_size_seen:
            failures.append(
                f"row-parallel module {module} missing TP ranks: "
                f"saw {sorted(by_rank)} of {tp_size_seen}"
            )
    return failures


def _check_grouped_expert_local_count(
    wrap_events: list[dict],
    summaries: dict[int, dict],
    num_moe_experts: int | None,
) -> list[str]:
    failures: list[str] = []
    per_ep_local: dict[int, int] = {}
    ep_size_seen = 1
    for event in wrap_events:
        if event.get("wrapper") != "OFTLinearGroupedSplitFC1UpGate":
            continue
        gate_shapes = event.get("adapter_gate_R_shapes") or []
        up_shapes = event.get("adapter_up_R_shapes") or []
        if len(gate_shapes) != len(up_shapes):
            failures.append(
                f"grouped expert wrap {event.get('module')} has gate/up bank "
                f"size mismatch: gate={len(gate_shapes)}, up={len(up_shapes)}"
            )
        tp_rank = int(event.get("tp_rank", 0))
        summary = summaries.get(tp_rank, {})
        ep_rank = int(summary.get("ep_rank", 0))
        per_ep_local[ep_rank] = len(gate_shapes)
        ep_size_seen = max(ep_size_seen, int(summary.get("ep_size", 1)))
    if not per_ep_local:
        return failures
    counts = set(per_ep_local.values())
    if len(counts) > 1:
        failures.append(
            f"grouped expert wraps have uneven num_local_experts across EP ranks: "
            f"{per_ep_local}"
        )
    if num_moe_experts is not None:
        n_local = next(iter(counts))
        if n_local * ep_size_seen != num_moe_experts:
            failures.append(
                f"num_local_experts ({n_local}) * ep_size ({ep_size_seen}) "
                f"!= num_moe_experts ({num_moe_experts})"
            )
    return failures


def _check_streamed_expert_coverage(streamed_events: list[dict]) -> list[str]:
    failures: list[str] = []
    per_proj: dict[tuple[int, str], dict[int, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for event in streamed_events:
        if event.get("event") != "expert_partition":
            continue
        layer = int(event.get("layer_id", -1))
        proj = str(event.get("proj"))
        tp_rank = int(event.get("tp_rank", 0))
        expert_id = int(event.get("expert_id", -1))
        per_proj[(layer, proj)][tp_rank].add(expert_id)
    for key, by_rank in per_proj.items():
        flat: dict[int, int] = {}
        for tp_rank, expert_ids in by_rank.items():
            for expert_id in expert_ids:
                if expert_id in flat and flat[expert_id] != tp_rank:
                    failures.append(
                        f"layer {key[0]} proj {key[1]} expert {expert_id} "
                        f"observed on tp_rank {flat[expert_id]} and {tp_rank}"
                    )
                else:
                    flat[expert_id] = tp_rank
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrap-base",
        type=Path,
        required=True,
        help="Base path passed to PEFT_WRAP_AUDIT_DUMP, OR a directory of "
             "wrap-audit files, OR a single .megatron.tp*_pp*.jsonl file.",
    )
    parser.add_argument(
        "--streamed-base",
        type=Path,
        required=False,
        help="Base path passed to SGLANG_OFT_STREAMED_AUDIT; sibling "
             ".sglang_streamed.tp*.jsonl files are loaded.",
    )
    parser.add_argument("--num-moe-experts", type=int, default=None)
    args = parser.parse_args()

    wrap_files = _resolve_wrap_files(args.wrap_base)
    if not wrap_files:
        print(f"FAIL: no wrap audit files found for {args.wrap_base}")
        return 1
    wrap_events = _decorate_wrap_events(wrap_files)
    summaries = _summary(wrap_events)

    streamed_files = _resolve_streamed_files(args.streamed_base) if args.streamed_base else []
    streamed_events = _decorate_streamed_events(streamed_files)

    failures: list[str] = []
    failures.extend(_check_column_parallel_replicated(wrap_events))
    failures.extend(_check_row_parallel_sliced(wrap_events, summaries))
    failures.extend(_check_grouped_expert_local_count(wrap_events, summaries, args.num_moe_experts))
    failures.extend(_check_streamed_expert_coverage(streamed_events))

    print(f"wrap_files={len(wrap_files)} wrap_records={len(wrap_events)}")
    print(f"streamed_files={len(streamed_files)} streamed_records={len(streamed_events)}")
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
