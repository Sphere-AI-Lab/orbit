"""LLM-judge reward hooks (orbit/rollout/llm_judge.py).

An external judge model (any instruct model served by sglang) grades each
sample via the OpenAI-compatible chat endpoint, wired through orbit's
--custom-rm-path. Two modes:
- equivalence: binary verdict vs the reference label (the NeMo-RL
  equivalence_llm_judge analog) -> reward 1.0 / 0.0.
- score: pointwise 0-10 grade -> reward normalized to [0, 1].
"""

import argparse

import pytest

from orbit.rollout import llm_judge
from orbit.utils.types import Sample


def _args(**overrides):
    defaults = dict(
        judge_base_url="http://judge:30600",
        judge_mode="equivalence",
        judge_model="default",
        judge_max_tokens=1024,
        judge_timeout_secs=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _sample(**overrides):
    defaults = dict(
        index=0,
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        response="The answer is 4.",
        response_length=5,
        label="4",
    )
    defaults.update(overrides)
    return Sample(**defaults)


# --- question extraction ---


def test_extract_question_from_messages_takes_last_user_turn():
    prompt = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "second"},
    ]
    assert llm_judge._extract_question(prompt) == "second"


def test_extract_question_from_plain_string():
    assert llm_judge._extract_question("plain question") == "plain question"


# --- verdict parsing ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reasoning...\nVERDICT: EQUIVALENT", 1.0),
        ("reasoning...\nVERDICT: DIFFERENT", 0.0),
        ("verdict: equivalent", 1.0),
        ("VERDICT: DIFFERENT\nwait no\nVERDICT: EQUIVALENT", 1.0),  # last wins
        ("no verdict here", None),
        ("", None),
    ],
)
def test_parse_equivalence(text, expected):
    assert llm_judge._parse_equivalence(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("thoughts\nSCORE: 7", 0.7),
        ("SCORE: 10", 1.0),
        ("SCORE: 0", 0.0),
        ("score: 8.5", 0.85),
        ("SCORE: 15", 1.0),  # clamped
        ("nothing", None),
    ],
)
def test_parse_score(text, expected):
    assert llm_judge._parse_score(text) == expected


# --- judge message construction ---


def test_equivalence_messages_contain_question_label_and_response():
    msgs = llm_judge._build_judge_messages("equivalence", "Q?", "resp", "ref")
    joined = " ".join(m["content"] for m in msgs)
    assert "Q?" in joined and "resp" in joined and "ref" in joined
    assert "VERDICT" in joined


def test_score_messages_work_without_label():
    msgs = llm_judge._build_judge_messages("score", "Q?", "resp", None)
    joined = " ".join(m["content"] for m in msgs)
    assert "SCORE" in joined


def test_equivalence_requires_label():
    with pytest.raises(ValueError, match="label"):
        llm_judge._build_judge_messages("equivalence", "Q?", "resp", None)


# --- reward_func (judge server monkeypatched) ---


async def test_reward_func_equivalence_positive(monkeypatch):
    async def fake_chat(base_url, messages, **kwargs):
        assert base_url == "http://judge:30600"
        return "The candidate matches.\nVERDICT: EQUIVALENT"

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    reward = await llm_judge.reward_func(_args(), _sample())
    assert reward == 1.0


async def test_reward_func_score_mode(monkeypatch):
    async def fake_chat(base_url, messages, **kwargs):
        return "Decent.\nSCORE: 6"

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    reward = await llm_judge.reward_func(_args(judge_mode="score"), _sample())
    assert reward == 0.6


async def test_reward_func_unparseable_verdict_returns_zero(monkeypatch):
    async def fake_chat(base_url, messages, **kwargs):
        return "I refuse to answer in the requested format."

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    reward = await llm_judge.reward_func(_args(), _sample())
    assert reward == 0.0


# --- startup validation ---

from orbit.utils.arguments import _validate_judge_args  # noqa: E402


def test_validate_judge_requires_base_url():
    args = argparse.Namespace(
        custom_rm_path="orbit.rollout.llm_judge.reward_func", judge_base_url=None, judge_mode="equivalence"
    )
    with pytest.raises(ValueError, match="judge-base-url"):
        _validate_judge_args(args)


def test_validate_judge_noop_for_other_rm():
    args = argparse.Namespace(custom_rm_path="orbit.rollout.opd_sglang.reward_func", judge_base_url=None)
    _validate_judge_args(args)


def test_validate_judge_passes_when_configured():
    args = argparse.Namespace(
        custom_rm_path="orbit.rollout.llm_judge.reward_func",
        judge_base_url="http://judge:30600",
        judge_mode="score",
    )
    _validate_judge_args(args)
