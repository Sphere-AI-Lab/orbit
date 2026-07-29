"""Convert the LoRA-without-regret datasets into Orbit's JSONL format.

Two different schemas, because two different consumers:

* No Robots feeds the SFT path. `sft_rollout.generate_rollout` reads
  `sample.prompt` and hands it straight to `MultiTurnLossMaskGenerator.get_loss_mask`,
  which expects a list of `{"role", "content"}` dicts. So `prompt` stays a list and
  the launcher must NOT pass `--apply-chat-template`.
* competition_math feeds the RL path, which uses the standard
  `--input-key prompt --label-key label` contract with a string prompt.

Also builds `tests/fast/fixtures/lora_regret/llama3_sample.jsonl`: a small,
real, multi-turn-heavy Tulu3 sample (same list-of-messages `prompt` schema as
No Robots) that the llama-3 loss-mask parity gate diffs against an HF oracle.
See `select_llama3_conversations` for why "multi-turn-heavy" is load-bearing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

NO_ROBOTS_REPO = "HuggingFaceH4/no_robots"
COMPETITION_MATH_REPO = "qwedsacf/competition_math"
TULU3_REPO = "allenai/tulu-3-sft-mixture"


def _load_split(name: str, split: str) -> list[dict[str, Any]]:
    """Load one split as a list of dicts. Split out so tests can monkeypatch it."""
    from datasets import load_dataset

    return list(load_dataset(name, split=split))


def _load_streamed_prefix(name: str, split: str, limit: int) -> Iterable[dict[str, Any]]:
    """Yield up to `limit` rows from a streamed split, in dataset order.

    Split out (same convention as `_load_split`) so tests can monkeypatch it
    with a small in-memory iterable instead of hitting the network.

    Streaming, rather than `_load_split`'s full-materialize-as-list, matters
    here specifically: Tulu3 (`allenai/tulu-3-sft-mixture`) is ~940k rows, and
    the caller only ever needs a dozen of them. `load_dataset(..., streaming=True)`
    never downloads more than the caller actually consumes, so a consumer that
    stops early (see `select_llama3_conversations`) bounds the real cost to
    however far into the mixture the last required example happens to sit --
    not to `limit`, which is only an outer safety cap.
    """
    from datasets import load_dataset

    ds = load_dataset(name, split=split, streaming=True)
    for i, row in enumerate(ds):
        if i >= limit:
            return
        yield row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def extract_boxed(solution: str) -> str | None:
    """Return the contents of the last \\boxed{...} in a solution, or None.

    Uses arbitrary-depth brace counting, not a fixed-depth regex: competition_math
    solutions routinely nest braces two or more levels deep (e.g.
    ``\\boxed{\\frac{2\\sqrt{35}}{35}}``), which a single-level regex silently
    mis-drops as "no boxed answer".

    This mirrors ``last_boxed_only_string`` / ``remove_boxed`` in
    `orbit/rollout/rm_hub/math_utils.py` (the same extraction the RL reward path
    uses to grade rollouts, so ground-truth extraction here stays consistent with
    how answers are later graded) rather than importing them: importing that
    module pulls in the whole `orbit.rollout.rm_hub` package, which transitively
    imports torch and ray (~2400 extra modules, ~9s just to import) — heavy
    runtime deps this CPU-only, dependency-light dataset-prep script has no
    other reason to need.
    """
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    depth = 0
    right_brace_idx = None
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed = solution[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if boxed[: len(left)] != left or boxed[-1] != "}":
        return None
    return boxed[len(left) : -1].strip()


def prepare_no_robots(out_dir: Path, n_train: int = 6400, n_test: int = 100) -> tuple[Path, Path]:
    """Write No Robots train/test JSONL in Orbit chat format.

    Returns (train_path, test_path).
    """
    out_dir = Path(out_dir)
    train_raw = _load_split(NO_ROBOTS_REPO, "train")[:n_train]
    test_raw = _load_split(NO_ROBOTS_REPO, "test")[:n_test]

    def _convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"prompt": row["messages"]} for row in rows]

    train_path = _write_jsonl(out_dir / "no_robots_train.jsonl", _convert(train_raw))
    test_path = _write_jsonl(out_dir / "no_robots_test.jsonl", _convert(test_raw))
    return train_path, test_path


def prepare_competition_math(
    out_dir: Path,
    n_train: int = 7500,
    val_start: int = 7500,
    val_end: int = 8500,
) -> tuple[Path, Path]:
    """Write competition_math train/val JSONL in Orbit prompt/label format.

    Rows whose solution has no \\boxed{...} answer are dropped, since the math
    reward function cannot grade them.

    Returns (train_path, val_path).
    """
    out_dir = Path(out_dir)
    raw = _load_split(COMPETITION_MATH_REPO, "train")

    def _convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            answer = extract_boxed(row["solution"])
            if answer is None:
                continue
            out.append({"prompt": row["problem"], "label": answer})
        return out

    train_path = _write_jsonl(out_dir / "competition_math_train.jsonl", _convert(raw[:n_train]))
    val_path = _write_jsonl(out_dir / "competition_math_val.jsonl", _convert(raw[val_start:val_end]))
    return train_path, val_path


def select_llama3_conversations(
    rows: Iterable[dict[str, Any]],
    n: int = 12,
    min_multi_turn: int = 6,
    min_system: int = 1,
    min_long: int = 1,
    long_threshold: int = 8,
) -> list[list[dict[str, Any]]]:
    """Deterministically pick n conversations' `messages` lists from an ordered stream.

    Built for the llama-3 loss-mask parity fixture, whose whole point is multi-turn
    coverage: the equivalent Qwen3 gate passed on every single-turn conversation and
    failed on multi-turn ones (a chat-template quirk for non-final assistant turns),
    and that bug reached 13% of the held-out set. A hand-built or single-turn-only
    fixture would have certified the same kind of broken implementation, so selection
    here is driven by real rows and real predicates, not curated by hand.

    Single pass over `rows`, in the order given -- never sampled, so re-running this
    against the same dataset revision reproduces the same fixture byte for byte.
    Rows are classified as they arrive:
      - "multi-turn": >=2 assistant messages (the exact shape that broke the Qwen3
        gate: a non-final assistant turn rendered differently from a final one).
      - "system": contains a system message.
      - "long": >=`long_threshold` messages total (a deep, multi-round exchange).

    Selection is priority order, not raw first-N-overall: the multi-turn quota fills
    first, then the first system-message row and the first long row are added (each
    may already be one of the multi-turn picks -- overlap is fine and only means
    fewer filler rows are needed). Remaining slots up to `n` are padded with the
    earliest rows encountered that were not already selected, so every row in the
    fixture keeps its original dataset order and the fixture stays deterministic.

    The scan continues past the point the predicates are all satisfied until at
    least `n` rows total have been seen, so there are always enough candidates
    left over to pad with -- otherwise a dataset where the predicates happen to
    be satisfied within the first few rows could stop scanning before collecting
    enough filler, even though plenty more rows were one step away.

    Raises ValueError if the stream is exhausted before the predicate minimums, or
    before `n` total rows, can be satisfied.
    """
    multi_turn: list[tuple[int, list[dict[str, Any]]]] = []
    system_row: tuple[int, list[dict[str, Any]]] | None = None
    long_row: tuple[int, list[dict[str, Any]]] | None = None
    order: list[tuple[int, list[dict[str, Any]]]] = []

    for i, row in enumerate(rows):
        messages = row["messages"]
        order.append((i, messages))

        if len(multi_turn) < min_multi_turn:
            if sum(m["role"] == "assistant" for m in messages) >= 2:
                multi_turn.append((i, messages))
        if system_row is None and any(m["role"] == "system" for m in messages):
            system_row = (i, messages)
        if long_row is None and len(messages) >= long_threshold:
            long_row = (i, messages)

        if (
            len(multi_turn) >= min_multi_turn
            and system_row is not None
            and long_row is not None
            and len(order) >= n
        ):
            break

    if len(multi_turn) < min_multi_turn:
        raise ValueError(
            f"stream exhausted with only {len(multi_turn)}/{min_multi_turn} multi-turn rows"
        )
    if min_system and system_row is None:
        raise ValueError("stream exhausted with no system-message row found")
    if min_long and long_row is None:
        raise ValueError(f"stream exhausted with no row of >={long_threshold} messages found")

    selected: dict[int, list[dict[str, Any]]] = dict(multi_turn)
    if system_row is not None:
        selected[system_row[0]] = system_row[1]
    if long_row is not None:
        selected[long_row[0]] = long_row[1]

    for idx, messages in order:
        if len(selected) >= n:
            break
        selected.setdefault(idx, messages)

    if len(selected) < n:
        raise ValueError(f"only found {len(selected)} candidate rows, need {n}")

    ordered_ids = sorted(selected)[:n]
    return [selected[idx] for idx in ordered_ids]


def prepare_llama3_sample(
    out_dir: Path,
    n: int = 12,
    scan_limit: int = 200_000,
) -> Path:
    """Write the llama-3 loss-mask parity fixture: n real, multi-turn-heavy Tulu3
    conversations, in Orbit's `{"prompt": [messages]}` JSONL format.

    Streams `allenai/tulu-3-sft-mixture` (see `_load_streamed_prefix`) rather than
    loading the ~940k-row split into memory -- `scan_limit` is only an outer safety
    cap; `select_llama3_conversations` stops consuming the stream as soon as its
    predicates are satisfied, which in practice (dataset revision at authoring time)
    is well under `scan_limit`.

    Returns the fixture path.
    """
    out_dir = Path(out_dir)
    raw = _load_streamed_prefix(TULU3_REPO, "train", scan_limit)
    conversations = select_llama3_conversations(raw, n=n)
    rows = [{"prompt": messages} for messages in conversations]
    return _write_jsonl(out_dir / "llama3_sample.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/lustre/fast/fast/groups/ei-slm/data/lora_regret"),
        help="Directory to write the JSONL files into.",
    )
    parser.add_argument(
        "--dataset",
        choices=["no_robots", "competition_math", "both", "llama3_sample"],
        default="both",
    )
    args = parser.parse_args()

    if args.dataset in ("no_robots", "both"):
        train, test = prepare_no_robots(args.out_dir)
        print(f"no_robots: {train} {test}")
    if args.dataset in ("competition_math", "both"):
        train, val = prepare_competition_math(args.out_dir)
        print(f"competition_math: {train} {val}")
    if args.dataset == "llama3_sample":
        # Not part of "both": this regenerates the tiny (12-row) parity fixture,
        # e.g. `--out-dir tests/fast/fixtures/lora_regret`, not a training split.
        fixture = prepare_llama3_sample(args.out_dir)
        print(f"llama3_sample: {fixture}")


if __name__ == "__main__":
    main()
