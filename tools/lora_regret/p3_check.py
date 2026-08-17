"""P3: assert the DP>1 held-out NLL reduction matches the DP=1 answer.

    python -m tools.lora_regret.p3_check logs/p3_dp1_*.log logs/p3_dp4_*.log

The eval reduces `(sum_neg_logprob, n_tokens)` over the **DP group only** --
TP/PP replicas hold identical samples, DP shards hold different token counts.
That code has never executed at DP>1, and P0 forces DP>1 for every FullFT arm,
so every FullFT number in the campaign is downstream of this check.

Exits 1 on any mismatch. A differing `tokens` in particular means the reduction
is double-counting or dropping a shard; the correct response is to stop, not to
average.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.lora_regret.trace import NllPoint, parse_trace_file


def compare_traces(
    dp1: list[NllPoint],
    dpn: list[NllPoint],
    decimals: int = 6,
) -> list[str]:
    """Problems found, empty if the two traces agree. Never raises.

    Measurements are paired by `(phase, step)` rather than by position: the two
    runs may log at different wall-clock moments and interleave differently, but
    a measurement at the same phase and step is the same measurement.

    `nll` is compared to `decimals` places because train.py prints `%.6f` --
    comparing the parsed floats exactly would compare digits the log never
    carried.
    """
    if not dp1 or not dpn:
        return [
            f"empty trace: dp1 has {len(dp1)} measurements, dpN has {len(dpn)}; "
            "two runs that logged nothing are not two runs that agreed"
        ]
    left = {(p.phase, p.step): p for p in dp1}
    right = {(p.phase, p.step): p for p in dpn}
    problems: list[str] = []
    for key in sorted(set(left) - set(right)):
        problems.append(f"{key[0]} step={key[1]}: only in the dp1 log")
    for key in sorted(set(right) - set(left)):
        problems.append(f"{key[0]} step={key[1]}: only in the dpN log")
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        where = f"{key[0]} step={key[1]}"
        if round(a.nll, decimals) != round(b.nll, decimals):
            problems.append(f"{where}: nll {a.nll:.{decimals}f} != {b.nll:.{decimals}f}")
        if a.tokens != b.tokens:
            problems.append(
                f"{where}: tokens {a.tokens} != {b.tokens} -- the DP reduction is "
                "double-counting or dropping a shard"
            )
        if a.samples != b.samples:
            problems.append(
                f"{where}: samples {a.samples} != {b.samples} -- the held-out set "
                "differs between the two runs, so the comparison is not a DP test"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dp1_log", type=Path, help="log from the GPUS_PER_NODE=1 run")
    parser.add_argument("dpn_log", type=Path, help="log from the GPUS_PER_NODE=N run")
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()

    dp1 = parse_trace_file(args.dp1_log)
    dpn = parse_trace_file(args.dpn_log)
    print(f"dp1: {len(dp1)} measurements from {args.dp1_log}")
    print(f"dpN: {len(dpn)} measurements from {args.dpn_log}")

    problems = compare_traces(dp1, dpn, args.decimals)
    if problems:
        print("\nP3 FAILED -- do not trust any FullFT number:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"\nP3 PASSED: {len(dp1)} measurements identical to {args.decimals} decimals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
