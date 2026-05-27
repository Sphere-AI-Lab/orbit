"""Convert PEFT-Arena training parquets into Orbit's RL JSONL format.

Source dataset: huggingface.co/datasets/SphereLab/PEFTArena-data
  - data/openr1-50k/{train,test}.parquet  (math, OpenR1 50k filtered)
  - data/medthink-23k/{train,test}.parquet (medical, optional)

Output: train.jsonl / test.jsonl in <output_dir>, each row shaped as
  {"prompt": [{"role": "user", "content": ...}], "label": <ground_truth>, "metadata": {...}}
matching what Orbit's launchers expect at TRAIN_JSONL / TEST_JSONL.

By default the converter appends Orbit's standard boxed-answer instruction
to the user content so the model learns to emit \\boxed{...} answers that
PEFT-Arena's grader can extract. Disable with --no-append-instruction.

Example:
  python tools/convert_peftarena_data.py \\
      --output_dir /path/to/peft_arena_openr1_50k

  TRAIN_JSONL=/path/to/peft_arena_openr1_50k/train.jsonl \\
  TEST_JSONL=/path/to/peft_arena_openr1_50k/test.jsonl \\
      bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh
"""

import argparse
import json
import os
import sys

import pandas as pd
from huggingface_hub import snapshot_download

BOXED_INSTRUCTION = " Please reason step by step, and put your final answer within \\boxed{}."


def _data_source_for_subset(subset: str) -> str:
    if subset == "openr1-50k":
        return "openr1"
    if subset == "medthink-23k":
        return "med_23k_think"
    return subset


def _project_row(row, append_instruction: bool, subset: str = "openr1-50k") -> dict:
    prompt = [dict(msg) for msg in row["prompt"]]
    if append_instruction and prompt and prompt[-1].get("role") == "user":
        prompt[-1]["content"] = prompt[-1]["content"].rstrip() + BOXED_INSTRUCTION
    label = row["reward_model"]["ground_truth"]
    return {
        "prompt": prompt,
        "label": label,
        "metadata": {
            "data_source": _data_source_for_subset(subset),
            "peft_arena_subset": subset,
        },
    }


def _convert_split(parquet_path: str, output_path: str, append_instruction: bool, subset: str) -> int:
    df = pd.read_parquet(parquet_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(_project_row(row, append_instruction, subset), ensure_ascii=False) + "\n")
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default="SphereLab/PEFTArena-data")
    ap.add_argument("--subset", default="openr1-50k", help="Subdir under data/ (e.g. openr1-50k, medthink-23k).")
    ap.add_argument("--output_dir", required=True, help="Where to write {train,test}.jsonl.")
    ap.add_argument("--cache_dir", default=None, help="HF dataset cache dir (default: ~/.cache/huggingface).")
    ap.add_argument(
        "--no-append-instruction",
        dest="append_instruction",
        action="store_false",
        help="Don't append Orbit's '\\boxed{}' suffix to user prompts.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing JSONL outputs.")
    ap.set_defaults(append_instruction=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_out = os.path.join(args.output_dir, "train.jsonl")
    test_out = os.path.join(args.output_dir, "test.jsonl")

    if not args.force and (os.path.exists(train_out) or os.path.exists(test_out)):
        existing = [p for p in (train_out, test_out) if os.path.exists(p)]
        sys.exit(f"refusing to overwrite existing JSONL: {existing}. Re-run with --force.")

    snapshot_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=f"data/{args.subset}/*.parquet",
        cache_dir=args.cache_dir,
    )
    parquet_dir = os.path.join(snapshot_dir, "data", args.subset)
    if not os.path.isdir(parquet_dir):
        sys.exit(f"subset not found in snapshot: {parquet_dir}")

    train_in = os.path.join(parquet_dir, "train.parquet")
    test_in = os.path.join(parquet_dir, "test.parquet")
    for p in (train_in, test_in):
        if not os.path.isfile(p):
            sys.exit(f"missing parquet split: {p}")

    n_train = _convert_split(train_in, train_out, args.append_instruction, args.subset)
    n_test = _convert_split(test_in, test_out, args.append_instruction, args.subset)
    print(f"wrote {train_out}: {n_train} rows")
    print(f"wrote {test_out}: {n_test} rows")
    if args.append_instruction:
        print("(appended '\\boxed{}' instruction to each user prompt; pass --no-append-instruction to disable)")


if __name__ == "__main__":
    main()
