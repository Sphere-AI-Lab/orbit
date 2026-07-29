"""Unit tests for the Ultra single-turn graders (rm_hub/ultra_agents.py).

Rule-based graders for the NeMo Gym agents whose rows are single-turn and
self-contained: tool-argument comparison, MCQA, structured outputs, and
instruction following (open-instruct IFEvalG registry — NOT allenai/IFBench,
whose instruction ids are disjoint from the blend's).
"""

from __future__ import annotations

import json

import orbit.rollout.rm_hub.ultra_agents as ua

TOOL_CALL_RESPONSE = (
    "I'll book that for you.\n<tool_call>\n"
    '{"name": "book_service", "arguments": {"user_id": "u1", "pet_ids": ["p1", "p2"], "count": 2}}'
    "\n</tool_call>"
)

EXPECTED_CALL = {
    "type": "function_call",
    "name": "book_service",
    "arguments": json.dumps({"pet_ids": ["p1", "p2"], "user_id": "u1", "count": 2}),
}


# ---------------------------------------------------------------------------
# Tool-argument comparison
# ---------------------------------------------------------------------------


def test_tool_call_exact_match_scores_one():
    assert ua.grade_tool_call(TOOL_CALL_RESPONSE, EXPECTED_CALL) == 1.0


def test_tool_call_key_order_and_numeric_type_are_insensitive():
    resp = '<tool_call>{"name": "f", "arguments": {"a": 1.0, "b": "x"}}</tool_call>'
    expected = {"type": "function_call", "name": "f", "arguments": '{"b": "x", "a": 1}'}
    assert ua.grade_tool_call(resp, expected) == 1.0


def test_tool_call_wrong_name_or_args_scores_zero():
    wrong_name = TOOL_CALL_RESPONSE.replace("book_service", "cancel_service")
    assert ua.grade_tool_call(wrong_name, EXPECTED_CALL) == 0.0
    wrong_args = TOOL_CALL_RESPONSE.replace('"count": 2', '"count": 3')
    assert ua.grade_tool_call(wrong_args, EXPECTED_CALL) == 0.0


def test_tool_call_missing_call_scores_zero():
    assert ua.grade_tool_call("I think we should book a service.", EXPECTED_CALL) == 0.0


def test_last_tool_call_wins():
    resp = (
        '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'
        "hmm, actually:"
        '<tool_call>{"name": "g", "arguments": {"b": 2}}</tool_call>'
    )
    expected = {"type": "function_call", "name": "g", "arguments": '{"b": 2}'}
    assert ua.grade_tool_call(resp, expected) == 1.0


def test_message_expected_rewards_not_calling():
    expected = {"type": "message", "content": "reference text"}
    assert ua.grade_tool_call("Sure — could you confirm the dates?", expected) == 1.0
    assert ua.grade_tool_call(TOOL_CALL_RESPONSE, expected) == 0.0
    assert ua.grade_tool_call("", expected) == 0.0  # empty response is not an answer


def test_malformed_tool_call_json_scores_zero():
    resp = "<tool_call>{not json}</tool_call>"
    assert ua.grade_tool_call(resp, EXPECTED_CALL) == 0.0


# ---------------------------------------------------------------------------
# MCQA
# ---------------------------------------------------------------------------


def test_mcqa_regex_extraction_and_match():
    regex = r"<final_answer>\s*([A-Za-z])\s*</final_answer>"
    assert ua.grade_mcqa("reasoning... <final_answer>I</final_answer>", "I", regex) == 1.0
    assert ua.grade_mcqa("<final_answer> i </final_answer>", "I", regex) == 1.0
    assert ua.grade_mcqa("<final_answer>F</final_answer>", "I", regex) == 0.0


def test_mcqa_last_match_wins_and_missing_scores_zero():
    regex = r"<final_answer>\s*([A-Za-z])\s*</final_answer>"
    two = "<final_answer>F</final_answer> wait no <final_answer>I</final_answer>"
    assert ua.grade_mcqa(two, "I", regex) == 1.0
    assert ua.grade_mcqa("the answer is I", "I", regex) == 0.0


def test_mcqa_default_regex_when_row_has_none():
    assert ua.grade_mcqa("... <final_answer>B</final_answer>", "B", None) == 1.0


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

SCHEMA = json.dumps(
    {
        "type": "object",
        "required": ["name", "count"],
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    }
)


def test_structured_valid_json_in_fence_scores_one():
    resp = 'Here you go:\n```json\n{"name": "x", "count": 3}\n```'
    assert ua.grade_structured_output(resp, SCHEMA, "json") == 1.0


def test_structured_bare_json_scores_one():
    assert ua.grade_structured_output('{"name": "x", "count": 3}', SCHEMA, "json") == 1.0


def test_structured_schema_violation_scores_zero():
    assert ua.grade_structured_output('{"name": "x"}', SCHEMA, "json") == 0.0
    assert ua.grade_structured_output('{"name": "x", "count": "three"}', SCHEMA, "json") == 0.0


def test_structured_unparseable_or_missing_scores_zero():
    assert ua.grade_structured_output("no json here", SCHEMA, "json") == 0.0
    assert ua.grade_structured_output("{broken", SCHEMA, "json") == 0.0


def test_structured_non_json_schema_type_scores_zero():
    assert ua.grade_structured_output('{"a": 1}', SCHEMA, "xml") == 0.0


# ---------------------------------------------------------------------------
# Instruction following (IFEvalG registry; skips if repo unavailable)
# ---------------------------------------------------------------------------

import pytest


def _ifeval_available():
    try:
        ua._ifeval_registry()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ifeval_available(), reason="open-instruct IFEvalG unavailable")
def test_instruction_following_strict_verdicts():
    ids = ["first_word:first_word_answer", "last_word:last_word_sent"]
    kwargs = [{"first_word": "development"}, {"last_word": "limit"}]
    good = "development pushes the limit. Every sentence honors the limit"
    assert ua.grade_instruction_following(good, ids, kwargs) == 1.0
    assert ua.grade_instruction_following("wrong start but ends with limit", ids, kwargs) == 0.0
    assert ua.grade_instruction_following("development start, wrong ending", ids, kwargs) == 0.0


@pytest.mark.skipif(not _ifeval_available(), reason="open-instruct IFEvalG unavailable")
def test_instruction_following_ignores_terminal_qwen_control_token():
    ids = ["startend:quotation", "last_word:last_word_answer"]
    kwargs = [None, {"last_word": "ask"}]

    assert ua.grade_instruction_following('"answer ask"<|im_end|>', ids, kwargs) == 1.0
    assert ua.grade_instruction_following('"answer ask"<|im_end|> trailing', ids, kwargs) == 0.0


@pytest.mark.skipif(not _ifeval_available(), reason="open-instruct IFEvalG unavailable")
def test_instruction_following_unknown_id_and_empty():
    assert ua.grade_instruction_following("text", ["bogus:not_a_rule"], [{}]) == 0.0
    assert ua.grade_instruction_following("", ["first_word:first_word_answer"], [{"first_word": "x"}]) == 0.0
