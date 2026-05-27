"""Convert math_eval test JSONLs into Orbit eval JSONL.

Supports two output modes:

- ``orbit515``: preserve the existing orbit-515 behavior
- ``math_alignment``: canonical COT prompt + metadata for aligned judging
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_DATASETS = ("math500", "aime24", "amc23")
BOXED_INSTRUCTION = " Please reason step by step, and put your final answer within \\boxed{}."


def _project_row_orbit515(row: dict, dataset_name: str, append_instruction: bool) -> dict:
    problem = row["problem"]
    answer = row["answer"]
    content = problem.rstrip()
    if append_instruction:
        content = content + BOXED_INSTRUCTION
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": str(answer),
        "metadata": {
            "data_source": "openr1",
            "dataset_name": dataset_name,
            "answer": str(answer),
        },
    }


def _project_row_math_alignment(row: dict, dataset_name: str) -> dict:
    prompt = f"Question: {row['problem'].rstrip()}\nAnswer:"
    metadata = {
        "rm_type": "math_alignment",
        "dataset_name": dataset_name,
        "answer": str(row.get("answer", "")),
    }
    if "solution" in row:
        metadata["solution"] = row["solution"]
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "label": str(row.get("answer", "")),
        "metadata": metadata,
    }


def _project_row(row: dict, dataset_name: str, mode: str, append_instruction: bool) -> dict:
    if mode == "orbit515":
        return _project_row_orbit515(row, dataset_name, append_instruction)
    if mode == "math_alignment":
        return _project_row_math_alignment(row, dataset_name)
    raise ValueError(f"Unknown mode {mode}")


def _convert(src_path: str, dst_path: str, dataset_name: str, mode: str, append_instruction: bool) -> int:
    count = 0
    with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            projected = _project_row(row, dataset_name=dataset_name, mode=mode, append_instruction=append_instruction)
            fout.write(json.dumps(projected, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    orbit_root = os.path.dirname(here)
    default_src = os.path.join(orbit_root, "examples/peft_arena/backend/third_party/math_eval/data")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_dir", default=default_src, help="Root of math_eval/data with <name>/test.jsonl subdirs.")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Dataset subdirs to convert.")
    ap.add_argument("--output_dir", required=True, help="Where to write <name>.jsonl files.")
    ap.add_argument(
        "--mode",
        choices=("orbit515", "math_alignment"),
        default="orbit515",
        help="Output schema/semantics to write.",
    )
    ap.add_argument(
        "--no-append-instruction",
        dest="append_instruction",
        action="store_false",
        help="For orbit515 mode only: don't append the boxed-answer suffix.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing JSONL outputs.")
    ap.set_defaults(append_instruction=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for name in args.datasets:
        src = os.path.join(args.source_dir, name, "test.jsonl")
        dst = os.path.join(args.output_dir, f"{name}.jsonl")
        if not os.path.isfile(src):
            sys.exit(f"missing source: {src}")
        if not args.force and os.path.exists(dst):
            sys.exit(f"refusing to overwrite {dst} (use --force).")
        count = _convert(src, dst, dataset_name=name, mode=args.mode, append_instruction=args.append_instruction)
        print(f"wrote {dst}: {count} rows ({args.mode})")
    if args.mode == "orbit515" and args.append_instruction:
        print("(appended '\\boxed{}' instruction; pass --no-append-instruction to disable)")


if __name__ == "__main__":
    main()
