"""Unit tests for the long-tail Ultra graders (rm_hub/ultra_longtail.py)."""

from __future__ import annotations

import asyncio
import json

import orbit.peft.rewards.ultra_longtail as lt
from orbit.peft.rewards.ultra_agents import grade_structured_output

# ---------------------------------------------------------------------------
# Boxed answers
# ---------------------------------------------------------------------------


def test_boxed_answer_exact_and_numeric():
    assert lt.grade_boxed_answer("reasoning \\boxed{129}", "129") == 1.0
    assert lt.grade_boxed_answer("\\boxed{1.0}", "1") == 1.0
    assert lt.grade_boxed_answer("\\boxed{130}", "129") == 0.0


def test_boxed_answer_last_line_fallback_and_case():
    assert lt.grade_boxed_answer("thinking...\n129", "129") == 1.0
    assert lt.grade_boxed_answer("\\boxed{Benzene Ring}", "benzene ring") == 1.0
    assert lt.grade_boxed_answer("", "129") == 0.0


# ---------------------------------------------------------------------------
# NVARC
# ---------------------------------------------------------------------------

GRID = [[1, 2], [3, 4]]


def test_nvarc_transductive_grid_formats():
    assert lt.grade_nvarc_transductive("\\boxed{1 2\n3 4}", GRID) == 1.0
    assert lt.grade_nvarc_transductive("\\boxed{12\n34}", GRID) == 1.0  # digit runs
    assert lt.grade_nvarc_transductive("\\boxed{1 2\n3 5}", GRID) == 0.0
    assert lt.grade_nvarc_transductive("no box", GRID) == 0.0


def test_nvarc_inductive_executes_transform():
    resp = "```python\ndef transform(grid):\n    return [[v + 1 for v in row] for row in grid]\n```"
    assert asyncio.run(lt.grade_nvarc_inductive(resp, [[0, 1]], [[1, 2]])) == 1.0
    assert asyncio.run(lt.grade_nvarc_inductive(resp, [[0, 1]], [[9, 9]])) == 0.0
    assert asyncio.run(lt.grade_nvarc_inductive("no code", [[0]], [[0]])) == 0.0


def test_nvarc_inductive_crash_scores_zero():
    resp = "```python\ndef transform(grid):\n    raise RuntimeError('boom')\n```"
    assert asyncio.run(lt.grade_nvarc_inductive(resp, [[0]], [[0]])) == 0.0


# ---------------------------------------------------------------------------
# Verifier specs
# ---------------------------------------------------------------------------


def test_verifier_string_match():
    v = {"type": "string_match", "patterns": [r"\(ref\ 1\)"]}
    assert lt.grade_verifier_spec("as shown (ref 1) here", v) == 1.0
    assert lt.grade_verifier_spec("no citation", v) == 0.0


def test_verifier_regex_min_matches():
    v = {"type": "regex", "verify_regex": [r"^===.+===\s*$"], "verify_min_matches": 2}
    assert lt.grade_verifier_spec("=== A ===\ntext\n=== B ===", v) == 1.0
    assert lt.grade_verifier_spec("=== only one ===", v) == 0.0


def test_verifier_unknown_type_scores_zero():
    assert lt.grade_verifier_spec("text", {"type": "quantum"}) == 0.0
    assert lt.grade_verifier_spec("text", None) == 0.0


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

CAL_STATE = {
    "0": {"event_id": 0, "duration": 45, "constraint": "after 10:15am", "min_time": "10:00", "max_time": "16:00"},
    "1": {"event_id": 1, "duration": 30, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
}


def _cal_response(events):
    return "Here is your calendar:\n" + json.dumps(events)


def test_calendar_valid_schedule():
    events = [
        {"event_id": 0, "event_name": "A", "start_time": "10:30", "duration": 45},
        {"event_id": 1, "event_name": "B", "start_time": "11:30", "duration": 30},
    ]
    assert lt.grade_calendar(_cal_response(events), CAL_STATE) == 1.0


def test_calendar_violations():
    base = [
        {"event_id": 0, "event_name": "A", "start_time": "10:30", "duration": 45},
        {"event_id": 1, "event_name": "B", "start_time": "11:30", "duration": 30},
    ]
    early = [dict(base[0], start_time="10:00"), base[1]]  # violates "after 10:15am"
    assert lt.grade_calendar(_cal_response(early), CAL_STATE) == 0.0
    overlap = [base[0], dict(base[1], start_time="10:45")]
    assert lt.grade_calendar(_cal_response(overlap), CAL_STATE) == 0.0
    wrong_dur = [dict(base[0], duration=60), base[1]]
    assert lt.grade_calendar(_cal_response(wrong_dur), CAL_STATE) == 0.0
    missing = [base[0]]
    assert lt.grade_calendar(_cal_response(missing), CAL_STATE) == 0.0
    assert lt.grade_calendar("no json here", CAL_STATE) == 0.0


# ---------------------------------------------------------------------------
# Judge-backed graders (scripted judge)
# ---------------------------------------------------------------------------


class _Args:
    judge_base_url = "http://scripted"
    judge_model = "default"
    judge_max_tokens = 128
    judge_timeout_secs = None


def _script_judge(monkeypatch, replies):
    queue = list(replies)

    async def fake(base_url, messages, **kw):
        return queue.pop(0)

    monkeypatch.setattr(lt, "post_chat_completions", fake)


def test_rubric_judge_all_must_pass(monkeypatch):
    rubric = [{"question": "q1", "pass_criteria": "c1"}, {"question": "q2", "pass_criteria": "c2"}]
    _script_judge(monkeypatch, ["reasoning\nYES", "sure\nYES"])
    assert asyncio.run(lt.grade_rubric_judge(_Args(), "ctx", "resp", rubric)) == 1.0
    _script_judge(monkeypatch, ["YES", "NO"])
    assert asyncio.run(lt.grade_rubric_judge(_Args(), "ctx", "resp", rubric)) == 0.0
    assert asyncio.run(lt.grade_rubric_judge(_Args(), "ctx", "resp", [])) == 0.0


def test_policy_judge(monkeypatch):
    _script_judge(monkeypatch, ["analysis...\nYES"])
    assert asyncio.run(lt.grade_policy_judge(_Args(), "attack", "I can't help with that because...", "refusal_with_explanation")) == 1.0
    _script_judge(monkeypatch, ["NO"])
    assert asyncio.run(lt.grade_policy_judge(_Args(), "attack", "Sure, here's how", "refusal_with_explanation")) == 0.0
    assert asyncio.run(lt.grade_policy_judge(_Args(), "x", "y", "unknown_policy")) == 0.0


# ---------------------------------------------------------------------------
# YAML structured outputs (v3 extension in ultra_agents)
# ---------------------------------------------------------------------------

YAML_SCHEMA = json.dumps(
    {"type": "array", "items": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}}
)


def test_structured_yaml_valid_and_invalid():
    good = "```yaml\n- name: Alice\n- name: Bob\n```"
    assert grade_structured_output(good, YAML_SCHEMA, "yaml") == 1.0
    bad = "```yaml\n- nickname: Alice\n```"
    assert grade_structured_output(bad, YAML_SCHEMA, "yaml") == 0.0
    assert grade_structured_output("not: [valid: yaml: {", YAML_SCHEMA, "yaml") == 0.0
