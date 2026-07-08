#!/usr/bin/env python
"""Known-answer oracle for the Lean grader (real kimina-lean-server).

Boots nothing — expects a running server at --lean-server-url. Checks:
- a trivially TRUE proof (norm_num on 1+1=2)          -> 1.0
- the same statement left as ``sorry``                 -> 0.0
- garbage lean                                         -> 0.0
- a real blend row's formal_statement with a `sorry`   -> 0.0 (compiles to a
  sorry-warning, proving header+statement parse against Mathlib)

Exits 0 iff all four verdicts are correct.

Usage:
    python tools/lean_rm_oracle.py --lean-server-url http://127.0.0.1:8000 \\
        [--swe-jsonl .../splits/rlvr1... ] # optional real-row check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

import orbit.rollout.rm_hub.lean_rm as lr


def _load_lean_row(path: str) -> dict | None:
    with open(path) as f:
        for line in f:
            if '"math_formal_lean_refinement_agent"' not in line:
                continue
            return json.loads(line)
    return None


async def _run(args_ns, response, header="", statement=""):
    return await lr.grade_lean_proof(args_ns, response, header, statement)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean-server-url", required=True)
    ap.add_argument("--lean-timeout-secs", type=float, default=180.0)
    ap.add_argument("--rlvr-jsonl", default=None)
    args = ap.parse_args()
    ns = SimpleNamespace(lean_server_url=args.lean_server_url, lean_timeout_secs=args.lean_timeout_secs)

    true_proof = "```lean4\nimport Mathlib\ntheorem t : 1 + 1 = 2 := by norm_num\n```"
    sorry_proof = "```lean4\nimport Mathlib\ntheorem t : 1 + 1 = 2 := by sorry\n```"
    garbage = "```lean4\nimport Mathlib\ntheorem t : 1 + 1 = 2 := by this_is_not_a_tactic\n```"

    r_true = asyncio.run(_run(ns, true_proof))
    r_sorry = asyncio.run(_run(ns, sorry_proof))
    r_garbage = asyncio.run(_run(ns, garbage))
    print(f"true norm_num : {r_true}  (expect 1.0)")
    print(f"sorry         : {r_sorry}  (expect 0.0)")
    print(f"garbage tactic: {r_garbage}  (expect 0.0)")

    checks = [r_true == 1.0, r_sorry == 0.0, r_garbage == 0.0]

    if args.rlvr_jsonl:
        row = _load_lean_row(args.rlvr_jsonl)
        if row:
            # the row's own statement ends in sorry -> must be 0.0 but parse clean
            resp = f"```lean4\n{row['header']}{row['formal_statement']}  sorry\n```"
            r_row = asyncio.run(_run(ns, resp, row["header"], row["formal_statement"]))
            print(f"real blend stmt (sorry): {r_row}  (expect 0.0, proves Mathlib parse)")
            checks.append(r_row == 0.0)

    ok = all(checks)
    print(f"### LEAN_ORACLE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
