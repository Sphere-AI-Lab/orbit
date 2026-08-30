#!/usr/bin/env python3
"""Compare two OPD teacher-logprob dumps (see orbit/opd/opd_dump.py).

Records are keyed by ``(rollout, sample_index)``. A matched pair whose
``tokens`` (the real ``Sample.tokens`` field -- full prompt+response ids)
differ is treated as "not the same underlying sample" and is a hard error
(exit 2), not a silent skip -- this is what makes the numeric comparison
below it trustworthy.
"""

from __future__ import annotations

import argparse
import json
import sys

from orbit.utils.logprob_compare import compare_logprobs, summarize_reports


def load(path: str) -> dict:
    records = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            records[(rec["rollout"], rec["sample_index"])] = rec
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args(argv)

    ref, cand = load(args.reference), load(args.candidate)
    common = sorted(set(ref) & set(cand))
    if not common:
        print("no common (rollout, sample_index) keys", file=sys.stderr)
        return 2
    reports = []
    for key in common:
        if ref[key]["tokens"] != cand[key]["tokens"]:
            print(f"token ids differ at {key}: not the same batch", file=sys.stderr)
            return 2
        reports.append(
            compare_logprobs(ref[key]["teacher_log_probs"], cand[key]["teacher_log_probs"])
        )
    summary = summarize_reports(reports)
    print(f"samples={len(common)} {summary}")
    ok = summary.within(args.atol)
    print("PASS" if ok else f"FAIL (atol={args.atol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
