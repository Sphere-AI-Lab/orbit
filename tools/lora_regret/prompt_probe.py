"""Measure what reward the *untrained* base policy earns, per prompt rendering.

This is the plan's Phase 0 gate, and it is the only way to know that a rendering
works: every argument for one frame over another is an argument about a
distribution nobody has sampled. It runs the real reward function
(`miles.rollout.rm_hub.async_rm`) against real problems, so a number out of here
is the same number the campaign would earn on rollout 0.

    python -m tools.lora_regret.prompt_probe --style completion --style chat

What it reports, and why each line is load-bearing:

    reward            mean binary reward over all samples -- the y-intercept of
                      Figure 6. The post's base model sits near 0.06 on GSM8K
                      and 0.035 on MATH.
    solvable_groups   fraction of problems with at least one correct sample out
                      of `--n-samples`, and NOT all of them correct. This, not
                      `reward`, is what decides whether RL can learn: advantage
                      is reward minus the group mean, so a group whose samples
                      all agree contributes exactly zero gradient however right
                      or wrong it is. A rendering with a lower `reward` but more
                      `solvable_groups` is the better rendering.
    boxed             fraction of responses containing a \\boxed{...} at all.
                      `--rm-type math` grades the box; an unboxed correct answer
                      scores 0, so this is the ceiling on `reward`.
    truncated         fraction that hit the token cap. A truncated response has
                      lost its box, so truncation converts to reward 0 directly.

Single GPU: an 8B policy in bf16 is ~16 GB of weights, and this only generates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from types import SimpleNamespace

import orbit  # noqa: F401  -- arms orbit's async_rm patch (orbit's rm_types)
from miles.rollout.rm_hub import async_rm
from miles.utils.types import Sample

from tools.lora_regret.prepare_data import ANSWER_INSTRUCTION, COMPLETION_STOP, render_prompt

DEFAULT_MODEL = "/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B"
DEFAULT_DATA_DIR = Path("/lustre/fast/fast/groups/ei-slm/data/lora_regret")

# The rendering the campaign used until 2026-08-02: Llama-3.1 *Instruct*'s turn
# structure wrapped around a base checkpoint. Kept as a probe candidate so the
# change is a measured comparison rather than an assertion about it.
CHAT_PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
)
CHAT_SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def build_prompt(problem: str, style: str) -> str:
    """Render one problem. `problem` is the raw statement, with no instruction."""
    if style == "chat":
        return CHAT_PREFIX + problem + ANSWER_INSTRUCTION + CHAT_SUFFIX
    return render_prompt(problem, answer_instruction=ANSWER_INSTRUCTION, style=style)


def stop_words(style: str) -> list[str]:
    return [] if style == "chat" else [COMPLETION_STOP]


def load_problems(path: Path, n: int, seed: int) -> list[dict]:
    """Read prepared rows and recover the raw problem statement.

    The prepared jsonl already carries whatever rendering was current when it
    was written, so the frame is stripped back off here and re-applied per
    candidate style. Otherwise the probe would measure the file, not the style.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        problem = row["prompt"]
        if problem.startswith("Problem:\n"):
            problem = problem[len("Problem:\n") :]
        problem = problem.split("\n\nSolution:")[0]
        if problem.endswith(ANSWER_INSTRUCTION):
            problem = problem[: -len(ANSWER_INSTRUCTION)]
        row["problem"] = problem
    random.Random(seed).shuffle(rows)
    return rows[:n]


def grade(response: str, label: str) -> int:
    args = SimpleNamespace(custom_rm_path=None, rm_type="math", rm_url=None)
    return asyncio.run(async_rm(args, Sample(prompt="", response=response, label=label)))


def summarise(style: str, dataset: str, records: list[dict], n_samples: int) -> dict:
    rewards = [r["reward"] for r in records]
    groups = [rewards[i : i + n_samples] for i in range(0, len(rewards), n_samples)]
    # Sampling-health check, reported rather than assumed. If this is ~1 the
    # draws within a group are not independent and `solvable_groups` is
    # measuring the harness, not the policy -- see the batching note above.
    response_groups = [
        records[i : i + n_samples] for i in range(0, len(records), n_samples)
    ]
    distinct = [len({r["response"] for r in g}) for g in response_groups]
    # A group teaches nothing unless its samples disagree: advantage is reward
    # minus the group mean, so an all-0 or all-1 group has zero advantage.
    informative = [g for g in groups if 0 < sum(g) < len(g)]
    return {
        "style": style,
        "dataset": dataset,
        "problems": len(groups),
        "samples": len(records),
        "reward": sum(rewards) / max(len(rewards), 1),
        "solvable_groups": len(informative) / max(len(groups), 1),
        "any_correct_groups": sum(1 for g in groups if sum(g) > 0) / max(len(groups), 1),
        "distinct_per_group": sum(distinct) / max(len(distinct), 1),
        "boxed": sum(1 for r in records if "\\boxed" in r["response"]) / max(len(records), 1),
        "truncated": sum(1 for r in records if r["truncated"]) / max(len(records), 1),
        "mean_response_chars": sum(len(r["response"]) for r in records) / max(len(records), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dataset", action="append", choices=["gsm8k", "math"], default=None, help="repeatable; default both"
    )
    parser.add_argument(
        "--style", action="append", choices=["completion", "raw", "chat"], default=None, help="repeatable"
    )
    parser.add_argument("--n-problems", type=int, default=64)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="completions per problem. The campaign uses 32; 8 is enough to see whether groups disagree.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dp",
        type=int,
        default=None,
        help=(
            "Data-parallel replicas; defaults to the visible GPU count. An 8B "
            "policy fits on one card, so replicas beat sharding here -- this is "
            "pure generation with no optimizer state to split."
        ),
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel size. 1 is right for 8B in bf16.")
    parser.add_argument("--out", type=Path, default=Path("results/prompt_probe.jsonl"))
    parser.add_argument("--samples-out", type=Path, default=None, help="write every graded completion here")
    args = parser.parse_args()

    datasets = args.dataset or ["gsm8k", "math"]
    styles = args.style or ["completion", "chat"]

    import torch

    import sglang as sgl

    dp_size = args.dp or max(torch.cuda.device_count(), 1)
    print(f"engine: dp={dp_size} tp={args.tp} model={args.model}", flush=True)
    engine = sgl.Engine(
        model_path=args.model,
        random_seed=args.seed,
        mem_fraction_static=0.85,
        dp_size=dp_size,
        tp_size=args.tp,
    )
    summaries = []
    all_records = []
    try:
        for dataset in datasets:
            path = args.data_dir / f"{dataset}_test.jsonl"
            problems = load_problems(path, args.n_problems, args.seed)
            for style in styles:
                # One round per sample, each round submitting the DISTINCT
                # prompts once. Not `[p]*n_samples` flattened into a single
                # call: on 2026-08-02 that returned 8 byte-identical
                # completions for all 128 GSM8K groups while MATH, same params
                # same engine, returned 7.95 distinct out of 8. Whatever the
                # mechanism, duplicate requests inside one batch are not 8
                # independent draws, and the failure is invisible in the
                # aggregate -- it reads as a policy with no sampling variance,
                # which is exactly what `solvable_groups` exists to detect.
                prompts = [build_prompt(row["problem"], style) for row in problems]
                labels = [row["label"] for row in problems]
                records = []
                for _ in range(args.n_samples):
                    outputs = engine.generate(
                        prompts,
                        {
                            "temperature": args.temperature,
                            "top_p": 1.0,
                            "max_new_tokens": args.max_new_tokens,
                            "stop": stop_words(style),
                        },
                    )
                    for prompt, label, out in zip(prompts, labels, outputs, strict=True):
                        text = out["text"]
                        meta = out.get("meta_info", {})
                        records.append(
                            {
                                "dataset": dataset,
                                "style": style,
                                "prompt": prompt,
                                "response": text,
                                "label": label,
                                "reward": grade(text, label),
                                "truncated": meta.get("finish_reason", {}).get("type") == "length",
                            }
                        )
                # Records arrive sample-major; `summarise` slices problem-major.
                order = {prompt: i for i, prompt in enumerate(prompts)}
                records.sort(key=lambda r: order[r["prompt"]])
                summary = summarise(style, dataset, records, args.n_samples)
                summaries.append(summary)
                all_records.extend(records)
                print(
                    f"{dataset:6s} {style:11s} reward={summary['reward']:.4f} "
                    f"solvable_groups={summary['solvable_groups']:.3f} "
                    f"any_correct={summary['any_correct_groups']:.3f} "
                    f"boxed={summary['boxed']:.3f} truncated={summary['truncated']:.3f} "
                    f"distinct/group={summary['distinct_per_group']:.2f}/{args.n_samples}",
                    flush=True,
                )
                if summary["distinct_per_group"] < 1.5 and args.n_samples > 1:
                    print(
                        "    WARNING: draws within a group are near-identical; "
                        "solvable_groups is not measuring the policy.",
                        flush=True,
                    )
                first = records[0]
                print(f"    sample response: {first['response'][:220]!r}\n", flush=True)
    finally:
        engine.shutdown()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary) + "\n")
    print(f"wrote {args.out}")
    if args.samples_out:
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_out.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record) + "\n")
        print(f"wrote {args.samples_out}")


if __name__ == "__main__":
    main()
