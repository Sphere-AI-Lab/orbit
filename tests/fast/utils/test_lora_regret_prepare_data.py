"""Schema tests for the LoRA-without-regret data preparation.

These run without network access by monkeypatching the dataset loader, so the
JSONL contract is pinned independently of HuggingFace availability.
"""

import json
from pathlib import Path

import pytest

from tools.lora_regret.prepare_data import (
    extract_boxed,
    prepare_competition_math,
    prepare_no_robots,
    _write_jsonl,
)


def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [{"prompt": [{"role": "user", "content": "hi"}]}, {"prompt": "x", "label": "1"}]
    out = tmp_path / "out.jsonl"
    _write_jsonl(out, rows)
    read_back = [json.loads(line) for line in out.read_text().splitlines()]
    assert read_back == rows


def test_no_robots_emits_messages_list(tmp_path: Path, monkeypatch):
    fake = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
    monkeypatch.setattr(
        "tools.lora_regret.prepare_data._load_split",
        lambda name, split: fake,
    )
    train, test = prepare_no_robots(tmp_path, n_train=6, n_test=2)

    train_rows = [json.loads(l) for l in train.read_text().splitlines()]
    test_rows = [json.loads(l) for l in test.read_text().splitlines()]

    assert len(train_rows) == 6
    assert len(test_rows) == 2
    # The contract sft_rollout depends on: prompt is a list of message dicts.
    assert isinstance(train_rows[0]["prompt"], list)
    assert train_rows[0]["prompt"][0]["role"] == "user"
    assert set(train_rows[0].keys()) == {"prompt"}


def test_no_robots_train_test_are_disjoint_prefixes(tmp_path: Path, monkeypatch):
    fake = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
    calls = []

    def _fake_load(name, split):
        calls.append(split)
        return fake

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_split", _fake_load)
    prepare_no_robots(tmp_path, n_train=6, n_test=2)
    # train comes from the train split, test from the test split — never a slice of one.
    assert calls == ["train", "test"]


def test_competition_math_emits_prompt_label(tmp_path: Path, monkeypatch):
    fake = [{"problem": f"p{i}", "solution": rf"x \boxed{{{i}}} y"} for i in range(20)]
    monkeypatch.setattr(
        "tools.lora_regret.prepare_data._load_split",
        lambda name, split: fake,
    )
    train, val = prepare_competition_math(tmp_path, n_train=5, val_start=5, val_end=8)

    train_rows = [json.loads(l) for l in train.read_text().splitlines()]
    val_rows = [json.loads(l) for l in val.read_text().splitlines()]

    assert len(train_rows) == 5
    assert len(val_rows) == 3
    assert train_rows[0] == {"prompt": "p0", "label": "0"}
    assert val_rows[0] == {"prompt": "p5", "label": "5"}


def test_competition_math_skips_rows_without_boxed_answer(tmp_path: Path, monkeypatch):
    fake = [
        {"problem": "good", "solution": r"\boxed{42}"},
        {"problem": "bad", "solution": "no answer here"},
        {"problem": "good2", "solution": r"\boxed{7}"},
        # Nested braces (frac containing sqrt) must NOT be treated as "no boxed
        # answer" — regression coverage for the single-level-nesting regex bug.
        {"problem": "good3", "solution": r"\boxed{\frac{2\sqrt{35}}{35}}"},
    ]
    monkeypatch.setattr("tools.lora_regret.prepare_data._load_split", lambda name, split: fake)
    train, _ = prepare_competition_math(tmp_path, n_train=4, val_start=4, val_end=4)
    rows = [json.loads(l) for l in train.read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["good", "good2", "good3"]
    assert rows[2]["label"] == r"\frac{2\sqrt{35}}{35}"


def test_extract_boxed_handles_nested_braces():
    # Two levels of nesting (sqrt inside frac inside boxed) is exactly the case
    # the original single-level regex silently mis-dropped.
    assert extract_boxed(r"\boxed{\frac{1}{\sqrt{2}}}") == r"\frac{1}{\sqrt{2}}"
    assert extract_boxed(r"x \boxed{\frac{2\sqrt{35}}{35}} y") == r"\frac{2\sqrt{35}}{35}"


def test_extract_boxed_returns_none_when_absent():
    assert extract_boxed("no answer here") is None
