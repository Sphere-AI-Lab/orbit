"""Convert supported Hugging Face SFT datasets into Orbit chat JSONL.

The output schema is the format consumed by Orbit's SFT rollout:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Supported dataset keys:

    numinamath      -> AI-MO/NuminaMath-CoT
    magicoder       -> ise-uiuc/Magicoder-OSS-Instruct-75K
    commonsenseqa   -> tau/commonsense_qa
    socialiqa       -> allenai/social_i_qa
    scienceqa-text  -> derek-thomas/ScienceQA, skipping image rows by default

Example:

    python tools/convert_sft_dataset_to_orbit.py \\
        --dataset commonsenseqa \\
        --output-dir data/sft/commonsenseqa \\
        --splits train validation \\
        --force
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import urllib.request
import zipfile


CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SOCIALIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
SOCIALIQA_ARCHIVE_DIR = "socialiqa-train-dev"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    default_splits: tuple[str, ...]
    trust_remote_code: bool = False


@dataclass
class ConversionStats:
    seen: int = 0
    written: int = 0
    skipped: int = 0


DATASET_SPECS: dict[str, DatasetSpec] = {
    "numinamath": DatasetSpec(
        key="numinamath",
        repo_id="AI-MO/NuminaMath-CoT",
        default_splits=("train", "test"),
    ),
    "magicoder": DatasetSpec(
        key="magicoder",
        repo_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        default_splits=("train",),
    ),
    "commonsenseqa": DatasetSpec(
        key="commonsenseqa",
        repo_id="tau/commonsense_qa",
        default_splits=("train", "validation"),
    ),
    "socialiqa": DatasetSpec(
        key="socialiqa",
        repo_id="allenai/social_i_qa",
        default_splits=("train", "validation"),
        trust_remote_code=True,
    ),
    "scienceqa-text": DatasetSpec(
        key="scienceqa-text",
        repo_id="derek-thomas/ScienceQA",
        default_splits=("train", "validation", "test"),
    ),
}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _valid_messages(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(message, dict) and "role" in message and "content" in message for message in value
    )


def _copy_messages(messages: Sequence[dict]) -> list[dict]:
    return [dict(message) for message in messages]


def _with_system_prompt(messages: list[dict], system_prompt: str | None) -> list[dict]:
    if not system_prompt:
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def _chat_row(
    dataset_key: str,
    user_content: str,
    assistant_content: str,
    metadata: dict | None = None,
    system_prompt: str | None = None,
) -> dict | None:
    user_content = _clean_text(user_content)
    assistant_content = _clean_text(assistant_content)
    if not user_content or not assistant_content:
        return None
    messages = _with_system_prompt(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        system_prompt,
    )
    return {
        "messages": messages,
        "metadata": {
            "dataset": dataset_key,
            **(metadata or {}),
        },
    }


def _format_choice_prompt(question: str, choices: Sequence[tuple[str, str]]) -> str:
    choice_lines = "\n".join(f"{label}. {text}" for label, text in choices)
    return f"Question: {question}\n\nChoices:\n{choice_lines}\n\nChoose the best answer."


def _choice_lookup(choices: Sequence[tuple[str, str]], answer_label: str) -> str | None:
    normalized = answer_label.strip().upper()
    for label, text in choices:
        if label.upper() == normalized:
            return f"{label}. {text}"
    return None


def _project_numinamath(row: dict, system_prompt: str | None) -> dict | None:
    if _valid_messages(row.get("messages")):
        messages = _with_system_prompt(_copy_messages(row["messages"]), system_prompt)
        return {
            "messages": messages,
            "metadata": {
                "dataset": "numinamath",
                "source_dataset": DATASET_SPECS["numinamath"].repo_id,
                "source": row.get("source"),
            },
        }
    return _chat_row(
        "numinamath",
        _clean_text(row.get("problem")),
        _clean_text(row.get("solution")),
        metadata={
            "source_dataset": DATASET_SPECS["numinamath"].repo_id,
            "source": row.get("source"),
        },
        system_prompt=system_prompt,
    )


def _project_magicoder(row: dict, system_prompt: str | None) -> dict | None:
    return _chat_row(
        "magicoder",
        _clean_text(row.get("problem")),
        _clean_text(row.get("solution")),
        metadata={
            "source_dataset": DATASET_SPECS["magicoder"].repo_id,
            "lang": row.get("lang"),
            "raw_index": row.get("raw_index"),
            "index": row.get("index"),
        },
        system_prompt=system_prompt,
    )


def _project_commonsenseqa(row: dict, system_prompt: str | None) -> dict | None:
    answer_key = _clean_text(row.get("answerKey")).upper()
    if not answer_key:
        return None
    choices_obj = row.get("choices") or {}
    labels = choices_obj.get("label") or []
    texts = choices_obj.get("text") or []
    choices = [(str(label), _clean_text(text)) for label, text in zip(labels, texts, strict=False)]
    answer = _choice_lookup(choices, answer_key)
    if answer is None:
        return None
    return _chat_row(
        "commonsenseqa",
        _format_choice_prompt(_clean_text(row.get("question")), choices),
        answer,
        metadata={
            "source_dataset": DATASET_SPECS["commonsenseqa"].repo_id,
            "id": row.get("id"),
            "question_concept": row.get("question_concept"),
            "answer_key": answer_key,
        },
        system_prompt=system_prompt,
    )


def _project_socialiqa(row: dict, system_prompt: str | None) -> dict | None:
    label = _clean_text(row.get("label"))
    if label not in {"1", "2", "3"}:
        return None
    choices = [
        ("A", _clean_text(row.get("answerA"))),
        ("B", _clean_text(row.get("answerB"))),
        ("C", _clean_text(row.get("answerC"))),
    ]
    answer = choices[int(label) - 1]
    prompt = "\n\n".join(
        [
            f"Context: {_clean_text(row.get('context'))}",
            _format_choice_prompt(_clean_text(row.get("question")), choices),
        ]
    )
    return _chat_row(
        "socialiqa",
        prompt,
        f"{answer[0]}. {answer[1]}",
        metadata={
            "source_dataset": DATASET_SPECS["socialiqa"].repo_id,
            "label": label,
        },
        system_prompt=system_prompt,
    )


def _scienceqa_has_image(row: dict) -> bool:
    image = row.get("image")
    return image not in (None, "")


def _project_scienceqa_text(
    row: dict,
    system_prompt: str | None,
    include_image_rows: bool,
) -> dict | None:
    if _scienceqa_has_image(row) and not include_image_rows:
        return None
    choices = [(CHOICE_LETTERS[i], _clean_text(text)) for i, text in enumerate(row.get("choices") or [])]
    answer_index = row.get("answer")
    if not isinstance(answer_index, int) or answer_index < 0 or answer_index >= len(choices):
        return None

    prompt_parts = [f"Question: {_clean_text(row.get('question'))}"]
    if _clean_text(row.get("hint")):
        prompt_parts.append(f"Hint: {_clean_text(row.get('hint'))}")
    if _clean_text(row.get("lecture")):
        prompt_parts.append(f"Lecture: {_clean_text(row.get('lecture'))}")
    prompt_parts.append("Choices:\n" + "\n".join(f"{label}. {text}" for label, text in choices))
    prompt_parts.append("Choose the best answer.")

    answer_label, answer_text = choices[answer_index]
    assistant = f"{answer_label}. {answer_text}"
    if _clean_text(row.get("solution")):
        assistant = f"{assistant}\n\nExplanation: {_clean_text(row.get('solution'))}"

    return _chat_row(
        "scienceqa-text",
        "\n\n".join(prompt_parts),
        assistant,
        metadata={
            "source_dataset": DATASET_SPECS["scienceqa-text"].repo_id,
            "grade": row.get("grade"),
            "subject": row.get("subject"),
            "topic": row.get("topic"),
            "category": row.get("category"),
            "skill": row.get("skill"),
            "answer_index": answer_index,
            "text_only": not _scienceqa_has_image(row),
        },
        system_prompt=system_prompt,
    )


def project_row(
    dataset_key: str,
    row: dict,
    *,
    system_prompt: str | None = None,
    scienceqa_include_image_rows: bool = False,
) -> dict | None:
    """Project one source row into Orbit SFT chat JSONL, or None when unlabeled."""

    if dataset_key == "numinamath":
        return _project_numinamath(row, system_prompt)
    if dataset_key == "magicoder":
        return _project_magicoder(row, system_prompt)
    if dataset_key == "commonsenseqa":
        return _project_commonsenseqa(row, system_prompt)
    if dataset_key == "socialiqa":
        return _project_socialiqa(row, system_prompt)
    if dataset_key == "scienceqa-text":
        return _project_scienceqa_text(row, system_prompt, scienceqa_include_image_rows)
    raise ValueError(f"unknown dataset key: {dataset_key}")


def _iter_projected(
    dataset_key: str,
    rows: Iterable[dict],
    stats: ConversionStats,
    *,
    max_rows: int | None = None,
    system_prompt: str | None = None,
    scienceqa_include_image_rows: bool = False,
) -> Iterator[dict]:
    for row in rows:
        stats.seen += 1
        projected = project_row(
            dataset_key,
            row,
            system_prompt=system_prompt,
            scienceqa_include_image_rows=scienceqa_include_image_rows,
        )
        if projected is None:
            stats.skipped += 1
            continue
        stats.written += 1
        yield projected
        if max_rows is not None and stats.written >= max_rows:
            break


def convert_rows(
    dataset_key: str,
    rows: Iterable[dict],
    *,
    max_rows: int | None = None,
    system_prompt: str | None = None,
    scienceqa_include_image_rows: bool = False,
) -> tuple[list[dict], ConversionStats]:
    records: list[dict] = []
    stats = ConversionStats()
    for projected in _iter_projected(
        dataset_key,
        rows,
        stats,
        max_rows=max_rows,
        system_prompt=system_prompt,
        scienceqa_include_image_rows=scienceqa_include_image_rows,
    ):
        records.append(projected)
    return records, stats


def write_jsonl(output_path: str | Path, records: Iterable[dict], *, force: bool = False) -> None:
    path = Path(output_path)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_converted_jsonl(
    output_path: str | Path,
    dataset_key: str,
    rows: Iterable[dict],
    *,
    split: str,
    force: bool = False,
    max_rows: int | None = None,
    system_prompt: str | None = None,
    scienceqa_include_image_rows: bool = False,
) -> ConversionStats:
    path = Path(output_path)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats()
    with path.open("w", encoding="utf-8") as fout:
        for projected in _iter_projected(
            dataset_key,
            rows,
            stats,
            max_rows=max_rows,
            system_prompt=system_prompt,
            scienceqa_include_image_rows=scienceqa_include_image_rows,
        ):
            projected.setdefault("metadata", {})["split"] = split
            fout.write(json.dumps(projected, ensure_ascii=False) + "\n")
    return stats


def _iter_socialiqa_archive(archive_path: str | Path, split: str) -> Iterator[dict]:
    split_files = {
        "train": ("train.jsonl", "train-labels.lst"),
        "validation": ("dev.jsonl", "dev-labels.lst"),
    }
    if split not in split_files:
        raise ValueError("SocialIQA raw fallback supports only train and validation splits.")

    jsonl_name, labels_name = split_files[split]
    jsonl_path = f"{SOCIALIQA_ARCHIVE_DIR}/{jsonl_name}"
    labels_path = f"{SOCIALIQA_ARCHIVE_DIR}/{labels_name}"

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(labels_path) as labels_file:
            labels = [line.decode("utf-8").strip() for line in labels_file]
        with archive.open(jsonl_path) as rows_file:
            for index, line in enumerate(rows_file):
                row = json.loads(line.decode("utf-8"))
                row["label"] = labels[index]
                yield row


def _download_socialiqa_archive(cache_dir: str | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "orbit" / "sft_datasets"
    archive_path = root / "socialiqa" / "socialiqa-train-dev.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urllib.request.urlretrieve(SOCIALIQA_URL, archive_path)
    return archive_path


def load_hf_split(
    spec: DatasetSpec,
    split: str,
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    revision: str | None = None,
) -> Iterable[dict]:
    if spec.key == "socialiqa":
        if revision is not None:
            raise ValueError("--revision is not supported for the SocialIQA raw-data fallback.")
        archive_path = _download_socialiqa_archive(cache_dir)
        return _iter_socialiqa_archive(archive_path, split)

    from datasets import load_dataset

    kwargs = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": spec.trust_remote_code,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(spec.repo_id, **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for <split>.jsonl outputs.")
    parser.add_argument("--splits", nargs="+", default=None, help="HF splits to convert. Defaults by dataset.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    parser.add_argument("--revision", default=None, help="Optional dataset revision.")
    parser.add_argument("--streaming", action="store_true", help="Use datasets streaming mode.")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum written rows per split.")
    parser.add_argument("--system-prompt", default=None, help="Optional system message prepended to each chat.")
    parser.add_argument(
        "--scienceqa-include-image-rows",
        action="store_true",
        help="Keep ScienceQA rows that contain images, while still emitting text-only prompts.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSONL outputs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = DATASET_SPECS[args.dataset]
    splits = tuple(args.splits or spec.default_splits)

    for split in splits:
        rows = load_hf_split(
            spec,
            split,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
            revision=args.revision,
        )
        output_path = args.output_dir / f"{split}.jsonl"
        stats = write_converted_jsonl(
            output_path,
            args.dataset,
            rows,
            split=split,
            force=args.force,
            max_rows=args.max_rows,
            system_prompt=args.system_prompt,
            scienceqa_include_image_rows=args.scienceqa_include_image_rows,
        )
        print(f"wrote {output_path}: {stats.written} rows ({stats.skipped} skipped, {stats.seen} seen)")


if __name__ == "__main__":
    main()
