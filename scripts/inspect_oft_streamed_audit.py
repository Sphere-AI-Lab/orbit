#!/usr/bin/env python3
"""Parse SGLANG_OFT_STREAMED_AUDIT rank-suffixed JSONLs and assert expected writes.

The argument is the *base path* passed to ``SGLANG_OFT_STREAMED_AUDIT``. The
inspector globs ``<base>.sglang_streamed.tp*.jsonl`` and aggregates events
across all ranks.

Examples:
    inspect_oft_streamed_audit.py logs/run_streamed_audit \\
        --require-dense qkv_proj --require-dense gate_up_proj \\
        --require-dense o_proj --require-dense down_proj \\
        --forbid-expert-proj gate_proj --forbid-expert-proj down_proj

    inspect_oft_streamed_audit.py logs/run_streamed_audit \\
        --require-dense qkv_proj --require-dense o_proj \\
        --require-expert-proj gate_proj --require-expert-proj down_proj \\
        --forbid-dense gate_up_proj
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_base(base: Path) -> tuple[list[dict], list[Path]]:
    matches = sorted(base.parent.glob(f"{base.name}.sglang_streamed.tp*.jsonl"))
    events: list[dict] = []
    for path in matches:
        for line in path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events, matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audit_base",
        type=Path,
        help="Base path passed to SGLANG_OFT_STREAMED_AUDIT (no .jsonl suffix).",
    )
    parser.add_argument("--require-dense", action="append", default=[])
    parser.add_argument("--forbid-dense", action="append", default=[])
    parser.add_argument("--require-expert-proj", action="append", default=[])
    parser.add_argument("--forbid-expert-proj", action="append", default=[])
    parser.add_argument("--min-nonzero", type=int, default=1)
    args = parser.parse_args()

    events, files = _load_base(args.audit_base)
    if not files:
        print(f"FAIL: no rank files matched {args.audit_base}.sglang_streamed.tp*.jsonl")
        return 1
    dense = [event for event in events if event.get("event") == "dense_write"]
    expert = [event for event in events if event.get("event") == "expert_partition"]
    dense_targets = Counter(event.get("fused_target") for event in dense)
    expert_projs = Counter(event.get("proj") for event in expert)
    ranks = sorted({int(event.get("tp_rank", 0)) for event in events})

    failures: list[str] = []
    for target in args.require_dense:
        if dense_targets[target] == 0:
            failures.append(f"missing dense target {target}")
    for target in args.forbid_dense:
        if dense_targets[target] > 0:
            failures.append(
                f"forbidden dense target {target} appeared {dense_targets[target]} times"
            )
    for proj in args.require_expert_proj:
        if expert_projs[proj] == 0:
            failures.append(f"missing expert proj {proj}")
    for proj in args.forbid_expert_proj:
        if expert_projs[proj] > 0:
            failures.append(
                f"forbidden expert proj {proj} appeared {expert_projs[proj]} times"
            )

    # min_nonzero applies once the first sync has happened (R drifts away from
    # identity). The audit captures every sync; the very first sync may have
    # near-zero deltas, so check the last 25% of events for each target.
    def _tail_for(events_list, key, value):
        filtered = [e for e in events_list if e.get(key) == value]
        if not filtered:
            return []
        cutoff = max(1, len(filtered) // 4)
        return filtered[-cutoff:]

    for target in args.require_dense:
        tail = _tail_for(dense, "fused_target", target)
        max_nonzero = (
            max(int(e["tensor"].get("nonzero", 0)) for e in tail) if tail else 0
        )
        if tail and max_nonzero < args.min_nonzero:
            failures.append(
                f"dense target {target} stayed at <={args.min_nonzero} nonzeros "
                "and never received a real update"
            )

    print(f"rank_files={[str(p) for p in files]}")
    print(f"tp_ranks_seen={ranks}")
    print(f"dense_targets={dict(dense_targets)}")
    print(f"expert_projs={dict(expert_projs)}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
