#!/usr/bin/env python3
"""Inspect PEFT_WRAP_AUDIT_DUMP JSONL files and assert coverage.

Usage:
    inspect_peft_wrap_audit.py <base_path> \
        --num-hidden-layers N \
        [--require-megatron-wrapper OFTLinearSplitQKV ...] \
        [--require-target qkv_proj --require-target gate_up_proj ...] \
        [--forbid-target gate_up_proj ...]

Aggregates across all rank-suffixed files:

    <base_path>.megatron.tp{T}_pp{P}.jsonl   (one per training (TP,PP) shard)
    <base_path>.sglang.tp{T}.jsonl           (one per SGLang TP rank)

For the Megatron side, unique module names across all rank files are
counted (TP slices share module names; PP shards hold disjoint module
sets so the union is the full model). For the SGLang side, layer indices
are deduped per target (TP slices replicate column-parallel targets and
slice row-parallel targets, but the layer count is identical).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_megatron(base_path: Path) -> list[dict]:
    pattern = str(base_path) + ".megatron.tp*_pp*.jsonl"
    records: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        records.extend(_load(Path(path)))
    return records


def _load_sglang(base_path: Path) -> list[dict]:
    pattern = str(base_path) + ".sglang.tp*.jsonl"
    records: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        records.extend(_load(Path(path)))
    return records


def _summarise_megatron(records: list[dict]) -> dict:
    # Dedup by unique (chunk, module) across all rank files. TP slices replicate
    # module names; PP shards add disjoint module names; the union is the model.
    by_module_id: dict[tuple[int, str], dict] = {}
    for record in records:
        if record.get("summary"):
            continue
        key = (record.get("chunk", 0), record.get("module", "?"))
        by_module_id.setdefault(key, record)

    wrapper_counts: collections.Counter[str] = collections.Counter(
        rec.get("wrapper", "?") for rec in by_module_id.values()
    )
    per_wrapper_modules: dict[str, list[str]] = collections.defaultdict(list)
    for (chunk, module_name), rec in by_module_id.items():
        per_wrapper_modules[rec.get("wrapper", "?")].append(module_name)

    return {
        "wrapper_counts": dict(wrapper_counts),
        "per_wrapper_modules": {k: sorted(v) for k, v in per_wrapper_modules.items()},
        "rank_files_seen": sum(1 for r in records if r.get("summary")),
    }


def _summarise_sglang(records: list[dict]) -> dict:
    # Dedup R-buffer rows by (target, layer) across rank files. Per-target
    # shape is collected as a SET across ranks — for column-parallel targets
    # the shape is identical on every rank; for row-parallel targets the
    # input dim is sliced, so different shapes are expected and not a bug.
    layers_by_target: dict[str, set[int]] = collections.defaultdict(set)
    shapes_by_target: dict[str, set[tuple]] = collections.defaultdict(set)
    rank_files_seen = 0
    for record in records:
        if record.get("summary"):
            rank_files_seen += 1
            continue
        target = record.get("target")
        if target is None:
            continue
        layer = record.get("layer")
        if layer is not None:
            layers_by_target[target].add(layer)
        shape = record.get("R_buffer_shape")
        if shape:
            shapes_by_target[target].add(tuple(shape))

    target_layer_counts = {target: len(layers) for target, layers in layers_by_target.items()}
    return {
        "target_layer_counts": target_layer_counts,
        "per_target_shapes": {k: sorted(v) for k, v in shapes_by_target.items()},
        "rank_files_seen": rank_files_seen,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path, help="Same value as PEFT_WRAP_AUDIT_DUMP")
    parser.add_argument("--num-hidden-layers", type=int, required=True)
    parser.add_argument(
        "--require-megatron-wrapper", action="append", default=[],
        help="Wrapper class name that MUST appear at least num_hidden_layers times across the model",
    )
    parser.add_argument(
        "--require-target", action="append", default=[],
        help="SGLang fused target name that MUST have exactly num_hidden_layers layer entries",
    )
    parser.add_argument(
        "--forbid-target", action="append", default=[],
        help="SGLang fused target name that must NOT appear",
    )
    parser.add_argument(
        "--allow-row-parallel-shape-mismatch", action="store_true",
        help="By default the inspector flags multiple distinct shapes per target as suspicious. "
             "Row-parallel targets (o_proj, down_proj) are sliced under TP so they legitimately "
             "have the same shape on every rank — this flag exists only to silence the warning "
             "if a future audit format ever surfaces per-rank shapes.",
    )
    parser.add_argument(
        "--forbid-grouped-expert-fc1-oftlinear",
        action="store_true",
        help="Fail if decoder.layers.*.mlp.experts.linear_fc1 is wrapped by plain OFTLinear.",
    )
    parser.add_argument(
        "--require-replicated-r",
        action="append",
        default=[],
        help="Module suffix whose adapter R_shape must be identical across "
             "every TP rank for the same module. Repeatable.",
    )
    parser.add_argument(
        "--require-row-parallel-r",
        action="append",
        default=[],
        help="Module suffix where adapter R_shape[0] must be equal across "
             "all TP ranks AND a record must exist on every TP rank. Repeatable.",
    )
    args = parser.parse_args()

    megatron_events = _load_megatron(args.base_path)
    megatron = _summarise_megatron(megatron_events)
    sglang = _summarise_sglang(_load_sglang(args.base_path))

    print(f"=== Megatron (aggregated across {megatron['rank_files_seen']} rank file(s)) ===")
    if not megatron["wrapper_counts"]:
        print("  (no records — did the launcher start? Was PEFT_WRAP_AUDIT_DUMP set in the right process?)")
    for wrapper, count in sorted(megatron["wrapper_counts"].items()):
        print(f"  {wrapper}: {count} unique modules")
    print(f"=== SGLang (aggregated across {sglang['rank_files_seen']} rank file(s)) ===")
    if not sglang["target_layer_counts"]:
        print("  (no records)")
    for target, count in sorted(sglang["target_layer_counts"].items()):
        shapes = sglang["per_target_shapes"].get(target, [])
        print(f"  {target}: {count} layer(s), shapes={shapes}")

    failures: list[str] = []

    if not megatron["wrapper_counts"]:
        failures.append(f"megatron audit empty for base {args.base_path}")
    if not sglang["target_layer_counts"]:
        failures.append(f"sglang audit empty for base {args.base_path}")

    for wrapper in args.require_megatron_wrapper:
        actual = megatron["wrapper_counts"].get(wrapper, 0)
        if actual < args.num_hidden_layers:
            failures.append(
                f"megatron: required wrapper {wrapper!r} appeared {actual} times, "
                f"expected >= num_hidden_layers={args.num_hidden_layers}"
            )

    if args.forbid_grouped_expert_fc1_oftlinear:
        for event in megatron_events:
            module = event.get("module", "")
            wrapper = event.get("wrapper", "")
            if module.endswith(".mlp.experts.linear_fc1") and wrapper == "OFTLinear":
                failures.append(
                    f"grouped expert FC1 {module} used plain OFTLinear; expected OFTLinearGroupedSplitFC1UpGate"
                )

    for target in args.require_target:
        actual = sglang["target_layer_counts"].get(target, 0)
        if actual != args.num_hidden_layers:
            failures.append(
                f"sglang: target {target!r} has {actual} layer entries, "
                f"expected {args.num_hidden_layers}"
            )

    for target in args.forbid_target:
        actual = sglang["target_layer_counts"].get(target, 0)
        if actual > 0:
            failures.append(
                f"sglang: target {target!r} appeared {actual} times but was forbidden"
            )

    if not args.allow_row_parallel_shape_mismatch:
        for target, shapes in sglang["per_target_shapes"].items():
            if len(shapes) > 1:
                failures.append(
                    f"sglang: target {target!r} has multiple R_buffer shapes across layers/ranks "
                    f"({shapes}) — every layer should have the same shape under TP replication"
                )

    # --- Sharding invariants via filename-decorated per-rank records ---
    if args.require_replicated_r or args.require_row_parallel_r:
        import re
        wrap_re = re.compile(r"\.megatron\.tp(?P<tp>\d+)_pp(?P<pp>\d+)\.jsonl$")
        decorated: list[dict] = []
        tp_size_seen = 1
        for path_str in sorted(glob.glob(str(args.base_path) + ".megatron.tp*_pp*.jsonl")):
            match = wrap_re.search(path_str)
            tp_rank = int(match.group("tp")) if match else 0
            for event in _load(Path(path_str)):
                event = dict(event)
                event["tp_rank"] = event.get("tp_rank", tp_rank)
                if event.get("summary") and "tp_size" in event:
                    tp_size_seen = max(tp_size_seen, int(event["tp_size"]))
                decorated.append(event)

        def _R_shape_for(event: dict):
            for key in ("adapter_R_shape", "adapter_q_R_shape", "adapter_gate_R_shape"):
                shape = event.get(key)
                if shape is not None:
                    return tuple(shape)
            return None

        for suffix in args.require_replicated_r:
            grouped: dict[str, set[tuple]] = collections.defaultdict(set)
            for event in decorated:
                module = event.get("module") or ""
                if not module.endswith(suffix):
                    continue
                shape = _R_shape_for(event)
                if shape is not None:
                    grouped[module].add(shape)
            for module, shapes in grouped.items():
                if len(shapes) > 1:
                    failures.append(
                        f"replicated-r suffix {suffix!r}: module {module} "
                        f"has inconsistent R shapes across TP ranks: {sorted(shapes)}"
                    )

        for suffix in args.require_row_parallel_r:
            per_module: dict[str, dict[int, int]] = collections.defaultdict(dict)
            for event in decorated:
                if event.get("summary"):
                    continue
                module = event.get("module") or ""
                if not module.endswith(suffix):
                    continue
                shape = _R_shape_for(event)
                if shape is None or not shape:
                    continue
                per_module[module][int(event["tp_rank"])] = int(shape[0])
            for module, by_rank in per_module.items():
                if len(set(by_rank.values())) > 1:
                    failures.append(
                        f"row-parallel suffix {suffix!r}: module {module} has uneven "
                        f"per-rank R block counts: {by_rank}"
                    )
                if tp_size_seen > 1 and len(by_rank) != tp_size_seen:
                    failures.append(
                        f"row-parallel suffix {suffix!r}: module {module} missing TP "
                        f"ranks: saw {sorted(by_rank)} of {tp_size_seen}"
                    )

    if failures:
        print("\n=== FAIL ===")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\n=== OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
