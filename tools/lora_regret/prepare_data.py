"""Convert the LoRA-without-regret datasets into Orbit's JSONL format.

There are two output schemas because there are two consumers:

* Tulu3, OpenThoughts3, and No Robots feed SFT. `sft_rollout.generate_rollout`
  hands `sample.prompt` to `MultiTurnLossMaskGenerator`, so `prompt` stays a
  list of `{"role", "content"}` messages and launchers must not pass
  `--apply-chat-template`.
* MATH, GSM8K, and competition_math feed RL. They use Orbit's standard
  `--input-key prompt --label-key label` contract with a string prompt.

Also builds `tests/fast/fixtures/lora_regret/llama3_sample.jsonl`: a small,
real, multi-turn-heavy Tulu3 sample (same list-of-messages `prompt` schema as
No Robots) that the llama-3 loss-mask parity gate diffs against an HF oracle.
See `select_llama3_conversations` for why "multi-turn-heavy" is load-bearing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NO_ROBOTS_REPO = "HuggingFaceH4/no_robots"
COMPETITION_MATH_REPO = "qwedsacf/competition_math"
TULU3_REPO = "allenai/tulu-3-sft-mixture"
OPENTHOUGHTS3_REPO = "open-thoughts/OpenThoughts3-1.2M"
MATH_REPO = "EleutherAI/hendrycks_math"
GSM8K_REPO = "openai/gsm8k"

# Verified against the hub's own split metadata on 2026-07-30 via
# `load_dataset_builder(TULU3_REPO).info.splits` -- 939,343 rows / 2.91 GB. The
# previous value here was 939,344, off by one, which would have failed the
# assertion only after streaming the whole 2.9 GB mixture. Check the metadata
# before changing this again: the point of the assertion is to notice a changed
# mixture, so a mismatch is a question about the dataset, not a number to bump.
TULU3_EXPECTED_ROWS = 939_343
OPENTHOUGHTS3_TRAIN_ROWS = 10_000
OPENTHOUGHTS3_TEST_ROWS = 100
MATH_EXPECTED_TRAIN_ROWS = 7_500
MATH_EXPECTED_TEST_ROWS = 5_000
GSM8K_EXPECTED_TRAIN_ROWS = 7_473
GSM8K_EXPECTED_TEST_ROWS = 1_319

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

ASSISTANT_HEADER_LITERAL = "<|start_header_id|>assistant<|end_header_id|>"
EOT_LITERAL = "<|eot_id|>"

# `qwedsacf/competition_math`, split positionally: rows [0:7500] train,
# [7500:8500] validation.
#
# **This is not the blog post's protocol**, and it was labelled as such here from
# 2026-08-02 until the post itself was read. It is the split used by
# michaelbzhu/lora-without-regret, a *community reproduction* that was vendored
# under `third_party/` and mistaken for the post's own code; the prompt template
# that went with it has been removed along with that directory.
#
# The post uses **MATH + GSM8K on Llama-3.1-8B base** -- `prepare_math` and
# `prepare_gsm8k` below, which is what the campaign's `e4` already reads. This
# split is kept because it is a real, usable dataset path with its assertions
# now in place, not because anything in the campaign wants it.
COMPETITION_MATH_EXPECTED_ROWS = 12_500
COMPETITION_MATH_TRAIN_ROWS = 7_500
COMPETITION_MATH_VAL_START = 7_500
COMPETITION_MATH_VAL_END = 8_500

# Appended to RL prompts so `--rm-type math` has something to extract.
# grade_answer_verl pulls the final \boxed{...} out of the response and grades
# that; a Llama-3.1 *base* policy does not box unprompted, so without this every
# rollout scores 0 and every E4 arm looks identical. Off by default in the
# library (do not mutate source text silently), on by default in the CLI (which
# builds runnable data).
ANSWER_INSTRUCTION = "\n\nPut your final answer in \\boxed{}."

# --- prompt rendering -------------------------------------------------------
#
# The policy is `Llama-3.1-8B`, the BASE checkpoint. It has no instruction
# tuning and no turn structure: the Instruct chat template's control tokens are
# in its vocabulary, but it was never trained to condition on them as
# delimiters, and the 2026-07-31 probe recorded what it emits after an assistant
# header -- web-scrape noise and private-use codepoints, reward 0 on all 1,024
# rollouts of every step.
#
# So the prompt is rendered as ordinary text that a pretraining corpus is full
# of: a `Problem:` block and a `Solution:` cue the model continues. The frame is
# part of the DATA, not of the launcher, so the exact bytes the policy sees are
# in the jsonl and are identical for FullFT and every LoRA rank. Rendering two
# arms differently would confound the axis E4 sweeps.
#
# COMPLETION_STOP is passed to the engine as a stop word: a base model continues
# the pattern past its own answer and starts writing the next problem. Without
# it, rollouts run to the token cap (10.2% truncated at 2,048 in the probe) and
# a truncated response has lost its \boxed{...}, so it grades 0 whatever it
# argued.
COMPLETION_STOP = "\n\nProblem:"
PROMPT_STYLES = ("completion", "raw")


def render_prompt(problem: str, *, answer_instruction: str = "", style: str = "completion") -> str:
    """Render one problem into the exact string the policy is conditioned on.

    `raw` is the pre-2026-08-02 behaviour -- the bare problem text, which only
    makes sense downstream of `--apply-chat-template`. It is kept because the
    chat-template path is still a legitimate configuration for an *Instruct*
    checkpoint, not because anything in this campaign uses it.
    """
    if style == "raw":
        return problem + answer_instruction
    if style == "completion":
        return f"Problem:\n{problem}{answer_instruction}\n\nSolution:"
    raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")


@dataclass(frozen=True)
class PreparedDataset:
    """Paths and counts emitted by one preparation job."""

    name: str
    train_path: Path
    source_rows: int
    train_rows: int
    test_rows: int
    # None for the RL mix, which has no single held-out file: E4 evaluates the
    # MATH and GSM8K test splits separately, so per-dataset accuracy stays
    # visible instead of being averaged away.
    test_path: Path | None = None
    filtered_rows: int = 0
    assistant_header_rows: int = 0
    eot_rows: int = 0


def _load_split(name: str, split: str) -> list[dict[str, Any]]:
    """Load one split as a list of dicts. Split out so tests can monkeypatch it."""
    from datasets import load_dataset

    return list(load_dataset(name, split=split))


def _load_config_split(name: str, config: str, split: str) -> list[dict[str, Any]]:
    """Load one configured split. Split out so tests avoid network access."""
    from datasets import load_dataset

    return list(load_dataset(name, config, split=split))


def _load_stream(name: str, split: str) -> Iterable[dict[str, Any]]:
    """Stream a dataset split without materializing it in memory."""
    from datasets import load_dataset

    return load_dataset(name, split=split, streaming=True)


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


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]], expected_rows: int) -> Path:
    """Atomically write exactly `expected_rows`, or leave the prior file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    count = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        if count != expected_rows:
            raise ValueError(f"{path.name}: expected {expected_rows} rows, got {count}")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def _normalize_messages(
    messages: Any,
    *,
    role_key: str = "role",
    content_key: str = "content",
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversation must be a non-empty list")

    role_map = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "system": "system",
    }
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("every conversation message must be an object")
        raw_role = message.get(role_key)
        content = message.get(content_key)
        if raw_role not in role_map:
            raise ValueError(f"unsupported conversation role {raw_role!r}")
        if not isinstance(content, str):
            raise ValueError(f"message content must be a string, got {type(content).__name__}")
        normalized.append({"role": role_map[raw_role], "content": content})
    return normalized


def _llama_control_token_hazards(messages: list[dict[str, str]]) -> tuple[bool, bool]:
    assistant_contents = [
        message["content"] for message in messages if message["role"] == "assistant"
    ]
    has_assistant_header = any(ASSISTANT_HEADER_LITERAL in content for content in assistant_contents)
    has_eot = any(EOT_LITERAL in content for content in assistant_contents)
    return has_assistant_header, has_eot


def _prepare_streamed_chat_dataset(
    *,
    name: str,
    rows: Iterable[dict[str, Any]],
    out_dir: Path,
    train_filename: str,
    test_filename: str,
    convert: Callable[[dict[str, Any]], list[dict[str, str]]],
    n_test: int,
    n_train: int | None,
    expected_source_rows: int | None,
) -> PreparedDataset:
    """Partition a chat stream with exact counts and atomic final outputs.

    The first valid rows form the held-out split. This is deterministic for a
    fixed upstream dataset order and requires only one streaming pass.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / train_filename
    test_path = out_dir / test_filename
    train_tmp = train_path.with_name(train_path.name + ".tmp")
    test_tmp = test_path.with_name(test_path.name + ".tmp")

    source_count = 0
    train_count = 0
    test_count = 0
    filtered_count = 0
    assistant_header_count = 0
    eot_count = 0

    try:
        with train_tmp.open("w", encoding="utf-8") as train_fh, test_tmp.open(
            "w", encoding="utf-8"
        ) as test_fh:
            for row in rows:
                source_count += 1
                messages = convert(row)
                has_assistant_header, has_eot = _llama_control_token_hazards(messages)
                assistant_header_count += int(has_assistant_header)
                eot_count += int(has_eot)
                if has_assistant_header or has_eot:
                    filtered_count += 1
                    continue

                record = {"prompt": messages}
                if test_count < n_test:
                    test_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    test_count += 1
                elif n_train is None or train_count < n_train:
                    train_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    train_count += 1

                if n_train is not None and train_count == n_train and test_count == n_test:
                    break

        if expected_source_rows is not None and source_count != expected_source_rows:
            raise ValueError(
                f"{name}: expected {expected_source_rows} source rows, got {source_count}"
            )
        if test_count != n_test:
            raise ValueError(f"{name}: expected {n_test} held-out rows, got {test_count}")
        expected_train = n_train
        if expected_train is None:
            expected_train = source_count - filtered_count - test_count
        if train_count != expected_train:
            raise ValueError(f"{name}: expected {expected_train} train rows, got {train_count}")

        os.replace(train_tmp, train_path)
        os.replace(test_tmp, test_path)
    except BaseException:
        train_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)
        raise

    return PreparedDataset(
        name=name,
        train_path=train_path,
        test_path=test_path,
        source_rows=source_count,
        train_rows=train_count,
        test_rows=test_count,
        filtered_rows=filtered_count,
        assistant_header_rows=assistant_header_count,
        eot_rows=eot_count,
    )


def extract_boxed(solution: str) -> str | None:
    """Return the contents of the last \\boxed{...} in a solution, or None.

    Uses arbitrary-depth brace counting, not a fixed-depth regex: competition_math
    solutions routinely nest braces two or more levels deep (e.g.
    ``\\boxed{\\frac{2\\sqrt{35}}{35}}``), which a single-level regex silently
    mis-drops as "no boxed answer".

    This follows ``last_boxed_only_string`` / ``remove_boxed`` in
    `miles/rollout/rm_hub/math_utils.py` (the same extraction the RL reward path
    uses to grade rollouts, so ground-truth extraction here stays consistent with
    how answers are later graded) rather than importing them: importing that
    module pulls in the whole `miles.rollout.rm_hub` package, which transitively
    imports torch and ray (~2400 extra modules, ~9s just to import) — heavy
    runtime deps this CPU-only, dependency-light dataset-prep script has no
    other reason to need.

    **One deliberate divergence:** this handles TeX's brace-less ``\\boxed 9``
    and the grader does not (``remove_boxed`` requires a literal ``\\boxed{``, and
    ``last_boxed_only_string`` returns None with no brace to count). Diverging is
    right for a *label*: it recovers two correct ground truths that would
    otherwise be discarded. It does mean a model that answers ``\\boxed 9``
    scores 0 in E4 even when correct — a real quirk of the reward path, recorded
    here because the prompt asks for ``\\boxed{}`` and so it should stay rare.
    """
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return None

    # TeX's brace-less form: `\boxed 9` is legal and takes a single token. Two of
    # MATH's 12,500 rows use it (algebra/train #888 `$\boxed 2$` and #1011
    # `$\boxed 9$`), and without this they read as "no boxed answer" -- which
    # would mean either raising on real data or asserting 7,498 rows instead of
    # the official 7,500.
    #
    # Read to the closing `$` rather than taking literally one token, following
    # the reference implementations: TeX would box only the "1" of `\boxed 12`,
    # and a label of "1" where the answer is 12 is silently wrong, which is worse
    # than either alternative.
    after_command = solution[idx + len("\\boxed") :].lstrip()
    if not after_command.startswith("{"):
        return after_command.split("$")[0].strip() or None

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
    # `or None` for the empty box: two number_theory rows end "there are
    # $\boxed{}$ primes", an empty ground truth for an answer of 0. An empty label
    # can never be earned honestly, and grade_answer_verl(response, "") may match a
    # model that also emits an empty box -- rewarding it for saying nothing. Treat
    # it as no answer, the same as the brace-less branch above.
    return boxed[len(left) : -1].strip() or None


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
    n_train: int = COMPETITION_MATH_TRAIN_ROWS,
    val_start: int = COMPETITION_MATH_VAL_START,
    val_end: int = COMPETITION_MATH_VAL_END,
    *,
    prompt_template: str | None = None,
    expected_source_rows: int | None = COMPETITION_MATH_EXPECTED_ROWS,
) -> PreparedDataset:
    """Write competition_math train/val JSONL.

    Rows whose solution has no \\boxed{...} answer are dropped and reported as
    `filtered_rows`, since the math reward function cannot grade them. The
    *source* count stays asserted, so a changed dataset is still caught -- the
    same contract `prepare_math` uses, and the reason a few unusable rows do not
    block the rest.

    `prompt_template` wraps each problem; None keeps the bare problem text, which
    is the default. It is a parameter rather than a constant applied
    unconditionally because the library must not mutate source text silently -- a
    hidden instruction would be invisible in the resulting JSONL's provenance.

    Returns a `PreparedDataset`; `test_path` is the validation split.
    """
    out_dir = Path(out_dir)
    raw = _load_split(COMPETITION_MATH_REPO, "train")
    if expected_source_rows is not None and len(raw) != expected_source_rows:
        raise ValueError(
            f"competition_math: expected {expected_source_rows} source rows, got {len(raw)}. "
            "The post's split is positional (rows 0-7500 train, 7500-8500 val), so a "
            "changed row count silently changes which problems are trained on."
        )
    if not 0 <= n_train <= val_start <= val_end <= len(raw):
        raise ValueError(
            f"competition_math: split bounds 0 <= {n_train} <= {val_start} <= {val_end} "
            f"<= {len(raw)} do not hold; a val range overlapping train leaks the "
            "training set into the reported accuracy."
        )

    dropped: list[dict[str, Any]] = []

    def _convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            answer = extract_boxed(row["solution"])
            if answer is None:
                dropped.append(row)
                continue
            problem = row["problem"]
            prompt = prompt_template.replace("{question}", problem) if prompt_template else problem
            out.append(
                {
                    "prompt": prompt,
                    "label": answer,
                    "metadata": {"dataset": "competition_math"},
                }
            )
        return out

    train_rows = _convert(raw[:n_train])
    val_rows = _convert(raw[val_start:val_end])
    train_path = _write_jsonl_atomic(
        out_dir / "competition_math_train.jsonl", train_rows, len(train_rows)
    )
    val_path = _write_jsonl_atomic(
        out_dir / "competition_math_val.jsonl", val_rows, len(val_rows)
    )
    return PreparedDataset(
        name="competition_math",
        train_path=train_path,
        test_path=val_path,
        source_rows=len(raw),
        train_rows=len(train_rows),
        test_rows=len(val_rows),
        filtered_rows=len(dropped),
    )


def prepare_tulu3(
    out_dir: Path,
    *,
    n_test: int = 1_000,
    expected_source_rows: int = TULU3_EXPECTED_ROWS,
) -> PreparedDataset:
    """Stream Tulu3 into full-train and held-out chat JSONL files.

    Rows containing literal Llama control tokens in assistant content are
    counted and removed. Such rows either raise in the loss-mask generator or
    silently terminate an assistant span early.
    """
    return _prepare_streamed_chat_dataset(
        name="tulu3",
        rows=_load_stream(TULU3_REPO, "train"),
        out_dir=out_dir,
        train_filename="tulu3_train.jsonl",
        test_filename="tulu3_test.jsonl",
        convert=lambda row: _normalize_messages(row["messages"]),
        n_test=n_test,
        n_train=None,
        expected_source_rows=expected_source_rows,
    )


def prepare_openthoughts3(
    out_dir: Path,
    *,
    n_train: int = OPENTHOUGHTS3_TRAIN_ROWS,
    n_test: int = OPENTHOUGHTS3_TEST_ROWS,
) -> PreparedDataset:
    """Stream an exact OpenThoughts3 subset into Orbit chat JSONL files."""
    return _prepare_streamed_chat_dataset(
        name="openthoughts3",
        rows=_load_stream(OPENTHOUGHTS3_REPO, "train"),
        out_dir=out_dir,
        train_filename="openthoughts3_train.jsonl",
        test_filename="openthoughts3_test.jsonl",
        convert=lambda row: _normalize_messages(
            row["conversations"], role_key="from", content_key="value"
        ),
        n_test=n_test,
        n_train=n_train,
        expected_source_rows=None,
    )


def _math_rows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset: str,
    category: str | None = None,
    answer_instruction: str = "",
    prompt_style: str = "completion",
    dropped: list[dict[str, Any]] | None = None,
):
    """Convert MATH-shaped rows, setting aside any without a usable answer.

    Skips rather than raises, and the caller counts what was skipped: two of the
    12,500 real rows carry a literally empty `\\boxed{}` (number_theory/train),
    and two bad source rows should not block the other 12,498. The *source* counts
    stay asserted by the caller, so upstream drift is still caught -- what changes
    is only that an unusable row becomes a reported drop instead of a crash.
    """
    for row in rows:
        answer = extract_boxed(row["solution"])
        if answer is None:
            if dropped is None:
                raise ValueError(f"{dataset}: solution has no complete \\\\boxed{{...}} answer")
            dropped.append(row)
            continue
        metadata = {"dataset": dataset}
        if category is not None:
            metadata["category"] = category
        yield {
            "prompt": render_prompt(row["problem"], answer_instruction=answer_instruction, style=prompt_style),
            "label": answer,
            "metadata": metadata,
        }


def prepare_math(
    out_dir: Path,
    *,
    expected_train_rows: int = MATH_EXPECTED_TRAIN_ROWS,
    expected_test_rows: int = MATH_EXPECTED_TEST_ROWS,
    answer_instruction: str = "",
    prompt_style: str = "completion",
) -> PreparedDataset:
    """Convert every official MATH category and preserve its train/test split.

    `expected_train_rows`/`expected_test_rows` are asserted against the **source**
    split sizes, not the output: rows with no usable answer are dropped and
    reported as `filtered_rows`, so the assertion keeps catching a changed dataset
    while two unusable rows do not block the other 12,498.
    """
    train_rows = []
    test_rows = []
    dropped: list[dict[str, Any]] = []
    source_train = source_test = 0
    for config in MATH_CONFIGS:
        raw_train = _load_config_split(MATH_REPO, config, "train")
        raw_test = _load_config_split(MATH_REPO, config, "test")
        source_train += len(raw_train)
        source_test += len(raw_test)
        train_rows.extend(
            _math_rows(
                raw_train,
                dataset="math",
                category=config,
                answer_instruction=answer_instruction,
                prompt_style=prompt_style,
                dropped=dropped,
            )
        )
        test_rows.extend(
            _math_rows(
                raw_test,
                dataset="math",
                category=config,
                answer_instruction=answer_instruction,
                prompt_style=prompt_style,
                dropped=dropped,
            )
        )

    if source_train != expected_train_rows:
        raise ValueError(
            f"math train split: expected {expected_train_rows} source rows, got {source_train}"
        )
    if source_test != expected_test_rows:
        raise ValueError(
            f"math test split: expected {expected_test_rows} source rows, got {source_test}"
        )
    out_dir = Path(out_dir)
    train_path = _write_jsonl_atomic(out_dir / "math_train.jsonl", train_rows, len(train_rows))
    test_path = _write_jsonl_atomic(out_dir / "math_test.jsonl", test_rows, len(test_rows))
    return PreparedDataset(
        name="math",
        train_path=train_path,
        test_path=test_path,
        source_rows=source_train + source_test,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        filtered_rows=len(dropped),
    )


def extract_gsm8k_answer(answer: str) -> str:
    """Extract the final answer after GSM8K's `####` delimiter."""
    _, separator, final_answer = answer.rpartition("####")
    if not separator or not final_answer.strip():
        raise ValueError("gsm8k: answer has no non-empty `####` final answer")
    return final_answer.strip()


def _gsm8k_rows(rows: Iterable[dict[str, Any]], *, answer_instruction: str = "", prompt_style: str = "completion"):
    for row in rows:
        yield {
            "prompt": render_prompt(row["question"], answer_instruction=answer_instruction, style=prompt_style),
            "label": extract_gsm8k_answer(row["answer"]),
            "metadata": {"dataset": "gsm8k"},
        }


def prepare_gsm8k(
    out_dir: Path,
    *,
    expected_train_rows: int = GSM8K_EXPECTED_TRAIN_ROWS,
    expected_test_rows: int = GSM8K_EXPECTED_TEST_ROWS,
    answer_instruction: str = "",
    prompt_style: str = "completion",
) -> PreparedDataset:
    """Convert GSM8K's official main train/test splits."""
    train_rows = list(
        _gsm8k_rows(
            _load_config_split(GSM8K_REPO, "main", "train"),
            answer_instruction=answer_instruction,
            prompt_style=prompt_style,
        )
    )
    test_rows = list(
        _gsm8k_rows(
            _load_config_split(GSM8K_REPO, "main", "test"),
            answer_instruction=answer_instruction,
            prompt_style=prompt_style,
        )
    )
    if len(train_rows) != expected_train_rows:
        raise ValueError(
            f"gsm8k_train.jsonl: expected {expected_train_rows} rows, got {len(train_rows)}"
        )
    if len(test_rows) != expected_test_rows:
        raise ValueError(
            f"gsm8k_test.jsonl: expected {expected_test_rows} rows, got {len(test_rows)}"
        )
    out_dir = Path(out_dir)
    train_path = _write_jsonl_atomic(
        out_dir / "gsm8k_train.jsonl", train_rows, expected_train_rows
    )
    test_path = _write_jsonl_atomic(
        out_dir / "gsm8k_test.jsonl", test_rows, expected_test_rows
    )
    return PreparedDataset(
        name="gsm8k",
        train_path=train_path,
        test_path=test_path,
        source_rows=len(train_rows) + len(test_rows),
        train_rows=len(train_rows),
        test_rows=len(test_rows),
    )


def prepare_rl_mix(
    out_dir: Path,
    *,
    sources: tuple[str, ...] = ("math_train.jsonl", "gsm8k_train.jsonl"),
    train_filename: str = "math_gsm8k_train.jsonl",
) -> PreparedDataset:
    """Concatenate the RL training splits into the one file the launcher takes.

    C5 is claimed over MATH *and* GSM8K, but `--prompt-data` accepts a single
    path. Reads back what `prepare_math`/`prepare_gsm8k` already wrote rather
    than re-deriving it, so the concatenation cannot disagree with the per-
    dataset files an eval run scores against.

    Missing sources raise instead of being skipped: a silently half-sized mix
    would train on one dataset and still be reported as MATH+GSM8K.
    """
    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for source in sources:
        path = out_dir / source
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run --dataset math and --dataset gsm8k first")
        # Iterate the file rather than `read_text().splitlines()`. splitlines()
        # also breaks on U+2028/U+2029/VT/FF/NEL, and `ensure_ascii=False` writes
        # those raw inside JSON strings -- gsm8k_train.jsonl really does carry two
        # U+2028 (measured 2026-07-30: 7,475 splitlines() fragments for 7,473
        # lines), so splitlines() tears two records in half. The file is valid
        # JSONL regardless: JSON allows an unescaped U+2028 in a string, and
        # pyarrow -- what Orbit's loader goes through -- splits on "\n" only.
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())

    train_path = _write_jsonl_atomic(out_dir / train_filename, rows, len(rows))
    return PreparedDataset(
        name="math_gsm8k",
        train_path=train_path,
        source_rows=len(rows),
        train_rows=len(rows),
        test_rows=0,
    )


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
        choices=[
            "no_robots",
            "competition_math",
            "both",
            "llama3_sample",
            "tulu3",
            "openthoughts3",
            "math",
            "gsm8k",
            "rl_mix",
            "campaign",
        ],
        default="both",
    )
    parser.add_argument(
        "--no-answer-instruction",
        action="store_true",
        help=(
            "Do not append the boxed-answer instruction to MATH/GSM8K prompts. "
            "Only use this if the reward is not --rm-type math: without the "
            "instruction a base policy never boxes, so every rollout scores 0."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="completion",
        help=(
            "How MATH/GSM8K prompts are framed. `completion` writes the exact "
            "text the base policy is conditioned on (Problem:/Solution:), which "
            "is what the RL launcher feeds through unmodified. `raw` writes the "
            "bare problem and only makes sense with --apply-chat-template, i.e. "
            "against an Instruct checkpoint."
        ),
    )
    args = parser.parse_args()
    answer_instruction = "" if args.no_answer_instruction else ANSWER_INSTRUCTION

    if args.dataset in ("no_robots", "both"):
        train, test = prepare_no_robots(args.out_dir)
        print(f"no_robots: {train} {test}")
    summaries = []
    if args.dataset in ("competition_math", "both"):
        summaries.append(prepare_competition_math(args.out_dir))
    if args.dataset == "llama3_sample":
        # Not part of "both": this regenerates the tiny (12-row) parity fixture,
        # e.g. `--out-dir tests/fast/fixtures/lora_regret`, not a training split.
        fixture = prepare_llama3_sample(args.out_dir)
        print(f"llama3_sample: {fixture}")
    if args.dataset in ("tulu3", "campaign"):
        summaries.append(prepare_tulu3(args.out_dir))
    if args.dataset in ("openthoughts3", "campaign"):
        summaries.append(prepare_openthoughts3(args.out_dir))
    if args.dataset in ("math", "campaign"):
        summaries.append(prepare_math(args.out_dir, answer_instruction=answer_instruction, prompt_style=args.prompt_style))
    if args.dataset in ("gsm8k", "campaign"):
        summaries.append(prepare_gsm8k(args.out_dir, answer_instruction=answer_instruction, prompt_style=args.prompt_style))
    if args.dataset in ("rl_mix", "campaign"):
        # Last, so `--dataset campaign` writes the mix from the files it just
        # produced rather than from a stale pair.
        summaries.append(prepare_rl_mix(args.out_dir))
    for summary in summaries:
        print(
            f"{summary.name}: train={summary.train_path} ({summary.train_rows}) "
            f"test={summary.test_path} ({summary.test_rows}) "
            f"source={summary.source_rows} filtered={summary.filtered_rows} "
            f"assistant_header={summary.assistant_header_rows} eot={summary.eot_rows}"
        )


if __name__ == "__main__":
    main()
