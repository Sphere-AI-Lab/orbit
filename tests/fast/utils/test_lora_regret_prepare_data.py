"""Schema tests for the LoRA-without-regret data preparation.

These run without network access by monkeypatching the dataset loader, so the
JSONL contract is pinned independently of HuggingFace availability.
"""

import json
from pathlib import Path

import pytest

from tools.lora_regret.prepare_data import (
    ANSWER_INSTRUCTION,
    MATH_CONFIGS,
    extract_boxed,
    extract_gsm8k_answer,
    prepare_competition_math,
    prepare_gsm8k,
    prepare_math,
    prepare_no_robots,
    prepare_openthoughts3,
    prepare_rl_mix,
    prepare_tulu3,
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

    train_rows = [json.loads(line) for line in train.read_text().splitlines()]
    test_rows = [json.loads(line) for line in test.read_text().splitlines()]

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

    train_rows = [json.loads(line) for line in train.read_text().splitlines()]
    val_rows = [json.loads(line) for line in val.read_text().splitlines()]

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
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["good", "good2", "good3"]
    assert rows[2]["label"] == r"\frac{2\sqrt{35}}{35}"


def test_extract_boxed_handles_nested_braces():
    # Two levels of nesting (sqrt inside frac inside boxed) is exactly the case
    # the original single-level regex silently mis-dropped.
    assert extract_boxed(r"\boxed{\frac{1}{\sqrt{2}}}") == r"\frac{1}{\sqrt{2}}"
    assert extract_boxed(r"x \boxed{\frac{2\sqrt{35}}{35}} y") == r"\frac{2\sqrt{35}}{35}"


def test_extract_boxed_returns_none_when_absent():
    assert extract_boxed("no answer here") is None


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tulu3_filters_llama_control_token_hazards_and_asserts_counts(tmp_path: Path, monkeypatch):
    rows = [
        {"messages": [{"role": "user", "content": "heldout"}, {"role": "assistant", "content": "ok"}]},
        {
            "messages": [
                {"role": "user", "content": "bad header"},
                {
                    "role": "assistant",
                    "content": "literal <|start_header_id|>assistant<|end_header_id|>",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "bad eot"},
                {"role": "assistant", "content": "literal <|eot_id|>"},
            ]
        },
        {"messages": [{"role": "user", "content": "train"}, {"role": "assistant", "content": "ok"}]},
    ]
    monkeypatch.setattr("tools.lora_regret.prepare_data._load_stream", lambda name, split: rows)

    result = prepare_tulu3(tmp_path, n_test=1, expected_source_rows=4)

    assert result.source_rows == 4
    assert result.train_rows == 1
    assert result.test_rows == 1
    assert result.filtered_rows == 2
    assert result.assistant_header_rows == 1
    assert result.eot_rows == 1
    assert _read_jsonl(result.test_path)[0]["prompt"][0]["content"] == "heldout"
    assert _read_jsonl(result.train_path)[0]["prompt"][0]["content"] == "train"


def test_tulu3_row_count_drift_leaves_no_partial_outputs(tmp_path: Path, monkeypatch):
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    monkeypatch.setattr("tools.lora_regret.prepare_data._load_stream", lambda name, split: rows)

    with pytest.raises(ValueError, match="expected 2 source rows, got 1"):
        prepare_tulu3(tmp_path, n_test=1, expected_source_rows=2)

    assert not (tmp_path / "tulu3_train.jsonl").exists()
    assert not (tmp_path / "tulu3_test.jsonl").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_openthoughts3_normalizes_roles_and_writes_exact_subset(tmp_path: Path, monkeypatch):
    rows = [
        {
            "conversations": [
                {"from": "human", "value": f"q{i}"},
                {"from": "gpt", "value": f"a{i}"},
            ]
        }
        for i in range(5)
    ]
    monkeypatch.setattr("tools.lora_regret.prepare_data._load_stream", lambda name, split: rows)

    result = prepare_openthoughts3(tmp_path, n_train=3, n_test=1)

    assert (result.source_rows, result.train_rows, result.test_rows) == (4, 3, 1)
    test_messages = _read_jsonl(result.test_path)[0]["prompt"]
    train_messages = _read_jsonl(result.train_path)[0]["prompt"]
    assert test_messages == [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": "a0"},
    ]
    assert train_messages[0]["content"] == "q1"


def test_openthoughts3_rejects_unknown_roles(tmp_path: Path, monkeypatch):
    rows = [{"conversations": [{"from": "tool", "value": "x"}]}]
    monkeypatch.setattr("tools.lora_regret.prepare_data._load_stream", lambda name, split: rows)

    with pytest.raises(ValueError, match="unsupported conversation role"):
        prepare_openthoughts3(tmp_path, n_train=0, n_test=1)


def test_math_combines_categories_and_preserves_official_splits(tmp_path: Path, monkeypatch):
    def _fake_load(name, config, split):
        suffix = "tr" if split == "train" else "te"
        return [
            {
                "problem": f"{config}-{suffix}",
                "solution": rf"work \boxed{{{len(config)}}}",
            }
        ]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    result = prepare_math(
        tmp_path,
        expected_train_rows=len(MATH_CONFIGS),
        expected_test_rows=len(MATH_CONFIGS),
    )

    train_rows = _read_jsonl(result.train_path)
    test_rows = _read_jsonl(result.test_path)
    assert len(train_rows) == len(MATH_CONFIGS)
    assert len(test_rows) == len(MATH_CONFIGS)
    assert train_rows[0]["metadata"] == {"dataset": "math", "category": MATH_CONFIGS[0]}
    assert train_rows[0]["prompt"].endswith("-tr")
    assert test_rows[0]["prompt"].endswith("-te")


def test_math_reports_rather_than_raises_on_a_missing_boxed_answer(tmp_path: Path, monkeypatch):
    """This assertion was inverted on 2026-07-30. `prepare_math` used to raise
    here, which meant two unusable rows in the real 12,500 (number_theory/train's
    empty `\\boxed{}`) blocked the entire dataset. It now drops them and reports
    the count, and fail-closed moved to the *source* count assertion — see
    test_math_still_fails_closed_on_a_wrong_source_count. `_math_rows` still raises
    when no caller is collecting drops, so the strict path is not lost."""
    monkeypatch.setattr(
        "tools.lora_regret.prepare_data._load_config_split",
        lambda name, config, split: [{"problem": "p", "solution": "no boxed answer"}],
    )
    result = prepare_math(
        tmp_path, expected_train_rows=len(MATH_CONFIGS), expected_test_rows=len(MATH_CONFIGS)
    )
    assert (result.train_rows, result.test_rows) == (0, 0)
    assert result.filtered_rows == 2 * len(MATH_CONFIGS)

    from tools.lora_regret.prepare_data import _math_rows

    with pytest.raises(ValueError, match="no complete"):
        list(_math_rows([{"problem": "p", "solution": "nope"}], dataset="math"))


def test_extract_gsm8k_answer():
    assert extract_gsm8k_answer("reasoning\n#### 1,234") == "1,234"
    with pytest.raises(ValueError, match="no non-empty"):
        extract_gsm8k_answer("reasoning only")


def test_gsm8k_preserves_official_splits_and_extracts_labels(tmp_path: Path, monkeypatch):
    def _fake_load(name, config, split):
        marker = "train" if split == "train" else "test"
        return [{"question": marker, "answer": "work\n#### 72"}]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    result = prepare_gsm8k(tmp_path, expected_train_rows=1, expected_test_rows=1)

    assert _read_jsonl(result.train_path) == [{"prompt": "train", "label": "72", "metadata": {"dataset": "gsm8k"}}]
    assert _read_jsonl(result.test_path)[0]["prompt"] == "test"


# ---------------------------------------------------------------------------
# E4's RL inputs. Two requirements that come from the reward function rather
# than from the datasets: the prompt must ask for a boxed answer, and the two
# training splits must arrive as one file.
# ---------------------------------------------------------------------------


def test_math_appends_the_answer_instruction_when_asked(tmp_path: Path, monkeypatch):
    """`--rm-type boxed_math` strips \\boxed{...} from the response before
    grading. A Llama-3.1 *base* policy does not box unprompted, so without this
    every rollout scores 0 and every E4 arm looks identical."""

    def _fake_load(name, config, split):
        return [{"problem": "2+2?", "solution": r"\boxed{4}"}]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    result = prepare_math(
        tmp_path,
        expected_train_rows=len(MATH_CONFIGS),
        expected_test_rows=len(MATH_CONFIGS),
        answer_instruction=ANSWER_INSTRUCTION,
    )

    prompt = _read_jsonl(result.train_path)[0]["prompt"]
    assert prompt.startswith("2+2?")
    assert ANSWER_INSTRUCTION in prompt
    # The label is the bare answer either way -- the instruction changes the
    # prompt, never the grading target.
    assert _read_jsonl(result.train_path)[0]["label"] == "4"


def test_gsm8k_appends_the_answer_instruction_when_asked(tmp_path: Path, monkeypatch):
    def _fake_load(name, config, split):
        return [{"question": "how many?", "answer": "work\n#### 72"}]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    result = prepare_gsm8k(
        tmp_path,
        expected_train_rows=1,
        expected_test_rows=1,
        answer_instruction=ANSWER_INSTRUCTION,
    )

    row = _read_jsonl(result.train_path)[0]
    assert row["prompt"].startswith("how many?")
    assert ANSWER_INSTRUCTION in row["prompt"]
    assert row["label"] == "72"


def test_answer_instruction_is_off_by_default(tmp_path: Path, monkeypatch):
    """The library default does not mutate the source text; the CLI turns the
    instruction on, because that is where a runnable dataset is being built."""

    def _fake_load(name, config, split):
        return [{"question": "how many?", "answer": "work\n#### 72"}]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    result = prepare_gsm8k(tmp_path, expected_train_rows=1, expected_test_rows=1)
    assert _read_jsonl(result.train_path)[0]["prompt"] == "how many?"


def test_rl_mix_concatenates_math_and_gsm8k(tmp_path: Path):
    """The RL launcher takes one --prompt-data path, and C5 is claimed over
    MATH + GSM8K together."""
    _write_jsonl(tmp_path / "math_train.jsonl", [{"prompt": "m", "label": "1"}])
    _write_jsonl(
        tmp_path / "gsm8k_train.jsonl",
        [{"prompt": "g1", "label": "2"}, {"prompt": "g2", "label": "3"}],
    )

    result = prepare_rl_mix(tmp_path)

    rows = _read_jsonl(result.train_path)
    assert result.train_path.name == "math_gsm8k_train.jsonl"
    assert [row["prompt"] for row in rows] == ["m", "g1", "g2"]
    assert result.train_rows == 3


def test_rl_mix_refuses_when_a_source_split_is_missing(tmp_path: Path):
    """Failing here beats an RL run that silently trains on half the campaign."""
    _write_jsonl(tmp_path / "math_train.jsonl", [{"prompt": "m", "label": "1"}])

    with pytest.raises(FileNotFoundError, match="gsm8k_train.jsonl"):
        prepare_rl_mix(tmp_path)


# ---------------------------------------------------------------------------
# TeX's brace-less \boxed form. Measured on the real MATH train split
# (2026-07-30): 2 of 12,500 rows use it -- algebra/train #888 (`$\boxed 2$`) and
# #1011 (`$\boxed 9$`). Both contain the literal \boxed, so they are a syntax
# variant rather than unboxed solutions, and dropping them would mean asserting
# 7,498/5,000 instead of the official split sizes.
# ---------------------------------------------------------------------------


def test_extract_boxed_handles_the_brace_less_single_token_form():
    assert extract_boxed(r"It follows that $x^2 + y^2 = \boxed 9$.") == "9"
    assert extract_boxed(r"our answer is $\boxed 2$.") == "2"


def test_extract_boxed_takes_the_whole_brace_less_argument_not_one_character():
    """TeX itself would box only the first token, but every reference
    implementation reads to the closing `$` -- and a silent "1" where the answer
    is "12" is a wrong label, which is worse than either."""
    assert extract_boxed(r"$x = \boxed 12$") == "12"
    assert extract_boxed(r"$x = \boxed -3$") == "-3"


def test_extract_boxed_prefers_the_braced_form_when_both_appear():
    assert extract_boxed(r"first $\boxed 1$ then $\boxed{42}$") == "42"


def test_extract_boxed_brace_less_form_without_a_terminator_still_extracts():
    assert extract_boxed(r"the answer is \boxed 7") == "7"


def test_extract_boxed_returns_none_when_boxed_has_no_argument():
    assert extract_boxed(r"a bare \boxed") is None
    assert extract_boxed(r"a bare \boxed$") is None


def test_rl_mix_survives_unicode_line_separators_in_values(tmp_path: Path):
    """`str.splitlines()` splits on U+2028/U+2029/VT/FF/NEL as well as \\n, and
    `ensure_ascii=False` writes those raw inside JSON strings -- so reading back
    with splitlines() tears a record in half and raises JSONDecodeError.

    Measured on the real data (2026-07-30): gsm8k_train.jsonl carries 2 raw
    U+2028, giving 7,475 splitlines() fragments for 7,473 actual lines. The file
    is valid JSONL either way -- JSON permits unescaped U+2028 inside a string,
    and pyarrow (which Orbit's loader uses) splits on \\n only -- so the writer is
    right and the reader has to iterate lines, not splitlines() a blob.
    """
    _write_jsonl(
        tmp_path / "math_train.jsonl",
        [{"prompt": "what is 2+2? show your work", "label": "4"}],
    )
    _write_jsonl(tmp_path / "gsm8k_train.jsonl", [{"prompt": "g", "label": "7"}])

    result = prepare_rl_mix(tmp_path)

    rows = _read_jsonl_strict(result.train_path)
    assert result.train_rows == 2
    assert rows[0]["prompt"] == "what is 2+2? show your work"
    assert rows[0]["label"] == "4"


def _read_jsonl_strict(path: Path):
    """Iterate lines rather than splitlines(), for the reason above."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_extract_boxed_treats_an_empty_box_as_no_answer():
    r"""hendrycks_math number_theory/train has two rows whose solution ends
    `there are $\boxed{}$ primes` -- a literally empty box where the intended
    answer is 0. Returning "" there is worse than returning None: an empty label
    can never be earned honestly, and `grade_answer_verl(response, "")` may match a
    model that also emits an empty box, which rewards saying nothing."""
    assert extract_boxed(r"there are $\boxed{}$ primes") is None
    assert extract_boxed(r"$\boxed{ }$") is None
    assert extract_boxed(r"$\boxed{0}$") == "0"


def test_math_drops_unusable_rows_and_counts_them(tmp_path: Path, monkeypatch):
    """Dropping beats raising: two bad source rows should not block all of MATH.
    The *source* counts stay asserted, so upstream drift is still caught, and the
    drop shows up as filtered_rows rather than as a silently smaller file."""

    def _fake_load(name, config, split):
        good = {"problem": f"{config}-{split}-good", "solution": r"\boxed{7}"}
        empty = {"problem": f"{config}-{split}-empty", "solution": r"answer is $\boxed{}$"}
        return [good, empty]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    result = prepare_math(
        tmp_path,
        expected_train_rows=2 * len(MATH_CONFIGS),
        expected_test_rows=2 * len(MATH_CONFIGS),
    )

    assert result.source_rows == 4 * len(MATH_CONFIGS)
    assert result.filtered_rows == 2 * len(MATH_CONFIGS)
    assert result.train_rows == len(MATH_CONFIGS)
    assert result.test_rows == len(MATH_CONFIGS)
    assert all(row["label"] == "7" for row in _read_jsonl(result.train_path))


def test_math_still_fails_closed_on_a_wrong_source_count(tmp_path: Path, monkeypatch):
    def _fake_load(name, config, split):
        return [{"problem": "p", "solution": r"\boxed{7}"}]

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_config_split", _fake_load)
    with pytest.raises(ValueError, match="expected"):
        prepare_math(tmp_path, expected_train_rows=99, expected_test_rows=99)
    assert not list(tmp_path.glob("*.jsonl"))
