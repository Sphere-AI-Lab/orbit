import json
import zipfile

import pytest

from tools.convert_sft_dataset_to_orbit import (
    _iter_socialiqa_archive,
    convert_rows,
    project_row,
    write_jsonl,
)


def test_numinamath_preserves_source_messages_when_available():
    source_messages = [
        {"role": "user", "content": "Compute 2 + 2."},
        {"role": "assistant", "content": "2 + 2 = 4."},
    ]

    projected = project_row(
        "numinamath",
        {"messages": source_messages, "source": "synthetic_math", "problem": "ignored", "solution": "ignored"},
    )

    assert projected["messages"] == source_messages
    assert projected["metadata"]["dataset"] == "numinamath"
    assert projected["metadata"]["source"] == "synthetic_math"


def test_magicoder_projects_problem_and_solution_to_chat_messages():
    projected = project_row(
        "magicoder",
        {
            "lang": "python",
            "raw_index": 7,
            "problem": "Write a function that returns the square of n.",
            "solution": "def square(n):\n    return n * n",
        },
    )

    assert projected["messages"] == [
        {"role": "user", "content": "Write a function that returns the square of n."},
        {"role": "assistant", "content": "def square(n):\n    return n * n"},
    ]
    assert projected["metadata"]["lang"] == "python"
    assert projected["metadata"]["raw_index"] == 7


def test_commonsenseqa_projects_labeled_choice_row():
    projected = project_row(
        "commonsenseqa",
        {
            "id": "example",
            "question": "Where would a person store soup?",
            "question_concept": "soup",
            "choices": {"label": ["A", "B"], "text": ["bowl", "shoe"]},
            "answerKey": "A",
        },
    )

    assert "Where would a person store soup?" in projected["messages"][0]["content"]
    assert "A. bowl" in projected["messages"][0]["content"]
    assert "B. shoe" in projected["messages"][0]["content"]
    assert projected["messages"][1]["content"] == "A. bowl"
    assert projected["metadata"]["answer_key"] == "A"


def test_socialiqa_maps_numeric_label_to_choice_letter_and_text():
    projected = project_row(
        "socialiqa",
        {
            "context": "Sydney helped Robin carry boxes.",
            "question": "How would Robin feel afterward?",
            "answerA": "grateful",
            "answerB": "confused",
            "answerC": "unrelated",
            "label": "1",
        },
    )

    assert "Context: Sydney helped Robin carry boxes." in projected["messages"][0]["content"]
    assert "A. grateful" in projected["messages"][0]["content"]
    assert projected["messages"][1]["content"] == "A. grateful"
    assert projected["metadata"]["label"] == "1"


def test_socialiqa_archive_loader_pairs_rows_with_labels(tmp_path):
    archive_path = tmp_path / "socialiqa-train-dev.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "socialiqa-train-dev/dev.jsonl",
            json.dumps(
                {
                    "context": "Alex apologized to Taylor.",
                    "question": "How would Taylor feel?",
                    "answerA": "forgiven",
                    "answerB": "ignored",
                    "answerC": "hungry",
                }
            )
            + "\n",
        )
        archive.writestr("socialiqa-train-dev/dev-labels.lst", "1\n")

    rows = list(_iter_socialiqa_archive(archive_path, "validation"))

    assert rows == [
        {
            "context": "Alex apologized to Taylor.",
            "question": "How would Taylor feel?",
            "answerA": "forgiven",
            "answerB": "ignored",
            "answerC": "hungry",
            "label": "1",
        }
    ]


def test_scienceqa_text_skips_image_rows_by_default_and_uses_solution_when_text_only():
    image_row = {
        "image": object(),
        "question": "Which state is farthest north?",
        "choices": ["West Virginia", "Louisiana"],
        "answer": 0,
    }
    assert project_row("scienceqa-text", image_row) is None

    projected = project_row(
        "scienceqa-text",
        {
            "image": None,
            "question": "Which material is magnetic?",
            "choices": ["wood", "iron"],
            "answer": 1,
            "hint": "Think about metals.",
            "lecture": "Magnets attract some metals.",
            "solution": "Iron is attracted to magnets.",
            "grade": "grade3",
        },
    )

    assert "Hint: Think about metals." in projected["messages"][0]["content"]
    assert "Lecture: Magnets attract some metals." in projected["messages"][0]["content"]
    assert projected["messages"][1]["content"] == "B. iron\n\nExplanation: Iron is attracted to magnets."
    assert projected["metadata"]["text_only"] is True


def test_convert_rows_counts_skipped_unlabeled_examples():
    records, stats = convert_rows(
        "commonsenseqa",
        [
            {
                "question": "Where would a person store soup?",
                "choices": {"label": ["A", "B"], "text": ["bowl", "shoe"]},
                "answerKey": "A",
            },
            {
                "question": "Unlabeled test row",
                "choices": {"label": ["A", "B"], "text": ["yes", "no"]},
                "answerKey": "",
            },
        ],
    )

    assert len(records) == 1
    assert stats.written == 1
    assert stats.skipped == 1


def test_convert_rows_counts_all_skipped_examples():
    records, stats = convert_rows(
        "commonsenseqa",
        [
            {
                "question": "Unlabeled test row",
                "choices": {"label": ["A", "B"], "text": ["yes", "no"]},
                "answerKey": "",
            }
        ],
    )

    assert records == []
    assert stats.seen == 1
    assert stats.written == 0
    assert stats.skipped == 1


def test_write_jsonl_refuses_to_overwrite_without_force(tmp_path):
    output_path = tmp_path / "train.jsonl"
    output_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_jsonl(output_path, [{"messages": []}], force=False)

    write_jsonl(output_path, [{"messages": [{"role": "user", "content": "x"}]}], force=True)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "messages": [{"role": "user", "content": "x"}]
    }
