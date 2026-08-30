#!/usr/bin/env python
"""Golden-patch oracle for the SWE patch reward (rung 2a acceptance gate).

Every SWE-rebench row ships its golden fix, giving swe_rm a perfect
end-to-end test against the real container:

- the GOLDEN patch must earn reward 1.0 (FAIL_TO_PASS turn green,
  PASS_TO_PASS stay green);
- a GARBAGE patch must earn 0.0;
- an empty response must earn 0.0 without touching the container.

Usage:
    python tools/swe_rm_oracle.py \\
        --swe-jsonl /path/to/swe.train.jsonl \\
        --sif-cache /path/to/sif_cache \\
        [--instance-id python-markdown__markdown-1529] [--timeout-secs 600]

Exits 0 iff all three verdicts are correct; prints "### SWE_RM_ORACLE PASS|FAIL".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

from miles.orbit.rewards.sandbox import swe_rm
from miles.utils.types import Sample


def _load_instance(path: str, instance_id: str | None) -> dict:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            md = row["responses_create_params"]["metadata"]
            if instance_id is None or md.get("instance_id") == instance_id:
                return json.loads(md["instance_dict"])
    raise SystemExit(f"instance {instance_id!r} not found in {path}")


def _sample(inst: dict, response: str) -> Sample:
    return Sample(
        prompt=[{"role": "user", "content": inst["problem_statement"][:2000]}],
        response=response,
        metadata={
            "swe": {
                "image_name": inst["image_name"],
                "test_patch": inst.get("test_patch") or "",
                "fail_to_pass": inst.get("FAIL_TO_PASS") or [],
                "pass_to_pass": inst.get("PASS_TO_PASS") or [],
            }
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe-jsonl", required=True)
    ap.add_argument("--sif-cache", required=True)
    ap.add_argument("--instance-id", default=None)
    ap.add_argument("--timeout-secs", type=float, default=600.0)
    args = ap.parse_args()

    inst = _load_instance(args.swe_jsonl, args.instance_id)
    print(f"instance: {inst['instance_id']}  image: {inst['image_name']}")
    print(f"tests: {len(inst.get('FAIL_TO_PASS') or [])} FAIL_TO_PASS + {len(inst.get('PASS_TO_PASS') or [])} PASS_TO_PASS")

    rm_args = SimpleNamespace(swe_rm_sif_cache=args.sif_cache, swe_rm_timeout_secs=args.timeout_secs)

    golden = f"```diff\n{inst['patch']}\n```"
    garbage = (
        "```diff\ndiff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-x\n+definitely not a fix\n```"
    )

    verdicts = {}
    verdicts["golden"] = asyncio.run(swe_rm.reward_func(rm_args, _sample(inst, golden)))
    verdicts["garbage"] = asyncio.run(swe_rm.reward_func(rm_args, _sample(inst, garbage)))
    verdicts["empty"] = asyncio.run(swe_rm.reward_func(rm_args, _sample(inst, "I cannot fix this.")))

    print(f"golden  -> {verdicts['golden']}   (expect 1.0)")
    print(f"garbage -> {verdicts['garbage']}   (expect 0.0)")
    print(f"empty   -> {verdicts['empty']}   (expect 0.0)")

    ok = verdicts["golden"] == 1.0 and verdicts["garbage"] == 0.0 and verdicts["empty"] == 0.0
    print(f"### SWE_RM_ORACLE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
