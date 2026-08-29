#!/usr/bin/env python
"""Build an orbit-trainable subset of the Nemotron-RL-Ultra swe blend.

Filters SWE-rebench rows to an easy, pytest-friendly slice, converts them to
orbit rows (prompt/label/metadata with the swe verification contract), and
pre-pulls the per-instance Apptainer images into the SIF cache swe_rm reads.

Usage:
    python tools/prepare_swe_subset.py \\
        --swe-jsonl .../splits/swe.train.jsonl \\
        --out .../orbit/swe_easy.train.jsonl \\
        --sif-cache .../sif_cache \\
        --num-instances 40 [--no-pull] [--pull-concurrency 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from orbit.peft.rewards.sandbox.swe_rm import _sif_path

_PROMPT_TEMPLATE = """You are an expert software engineer. Fix the following GitHub issue.

Repository: {repo} (checked out at commit {base_commit})

Issue:
{problem_statement}

Write a fix as a single unified diff against the repository root. Reply with
ONLY the patch, in one fenced block:

```diff
diff --git a/<path> b/<path>
...
```"""


def _select(path: str, n: int) -> list[dict]:
    picked = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            md = row["responses_create_params"]["metadata"]
            if md.get("dataset_name") != "nebius/SWE-rebench-V2":
                continue
            inst = json.loads(md["instance_dict"])
            meta = inst.get("meta") or {}
            llm_md = meta.get("llm_metadata") or {}
            if inst.get("language") != "python":
                continue
            if llm_md.get("difficulty") not in ("easy",):
                continue
            f2p = inst.get("FAIL_TO_PASS") or []
            p2p = inst.get("PASS_TO_PASS") or []
            if not (1 <= len(f2p) <= 5) or len(p2p) > 200:
                continue
            if len(inst.get("problem_statement") or "") > 6000:
                continue
            picked.append(inst)
            if len(picked) >= n:
                break
    return picked


def _to_orbit_row(inst: dict) -> dict:
    prompt = _PROMPT_TEMPLATE.format(
        repo=inst["repo"],
        base_commit=inst["base_commit"][:12],
        problem_statement=inst["problem_statement"].strip(),
    )
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "label": None,
        "metadata": {
            "agent": "swe_agents_train",
            "swe": {
                "image_name": inst["image_name"],
                "test_patch": inst.get("test_patch") or "",
                "fail_to_pass": inst.get("FAIL_TO_PASS") or [],
                "pass_to_pass": inst.get("PASS_TO_PASS") or [],
            },
        },
    }


async def _pull_all(instances: list[dict], cache_dir: str, concurrency: int) -> tuple[int, int]:
    os.makedirs(cache_dir, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    ok = skipped = 0

    async def pull(inst):
        nonlocal ok, skipped
        sif = _sif_path(cache_dir, inst["image_name"])
        if os.path.exists(sif):
            skipped += 1
            return
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                "apptainer",
                "pull",
                "--force",
                sif,
                f"docker://{inst['image_name'].removeprefix('docker.io/')}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                # registry connections through the proxy can wedge silently;
                # a stuck pull must not hang the whole batch
                _, err = await asyncio.wait_for(proc.communicate(), timeout=900)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                print(f"PULL TIMEOUT {inst['image_name']}", flush=True)
                return
            if proc.returncode == 0:
                ok += 1
                print(f"pulled {os.path.basename(sif)}", flush=True)
            else:
                print(f"PULL FAILED {inst['image_name']}: {err.decode()[-200:]}", flush=True)

    await asyncio.gather(*(pull(i) for i in instances))
    return ok, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sif-cache", required=True)
    ap.add_argument("--num-instances", type=int, default=40)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--pull-concurrency", type=int, default=4)
    args = ap.parse_args()

    instances = _select(args.swe_jsonl, args.num_instances)
    print(f"selected {len(instances)} easy python instances")

    with open(args.out, "w") as f:
        for inst in instances:
            f.write(json.dumps(_to_orbit_row(inst), ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")

    if not args.no_pull:
        ok, skipped = asyncio.run(_pull_all(instances, args.sif_cache, args.pull_concurrency))
        print(f"SIF cache: {ok} pulled, {skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
