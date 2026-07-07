"""Unit tests for group-wise pairwise GenRM rewards (orbit/rollout/genrm_judge.py).

The hook shape: ``--group-rm`` routes the whole n-samples-per-prompt group into
``batched_async_rm``, which calls ``reward_func(args, samples)`` -> list of
rewards. The judge compares responses pairwise (round-robin, single order)
under the row's rubric (``sample.metadata["principle"]``); rewards are
win-rates in [0, 1].
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import orbit.rollout.genrm_judge as genrm
from orbit.utils.types import Sample


def _args(**overrides):
    values = {
        "judge_base_url": "http://judge:30801",
        "judge_model": "default",
        "judge_max_tokens": 512,
        "judge_timeout_secs": 60,
        "group_rm": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(response: str, principle: str | None = "Prefer concise, correct answers.") -> Sample:
    metadata = {"principle": principle} if principle is not None else {}
    return Sample(
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        response=response,
        label=None,
        metadata=metadata,
    )


def _run(coro):
    return asyncio.run(coro)


def _scripted_judge(script):
    """Fake post_chat_completions: looks up the verdict by (A-text, B-text)."""

    calls = []

    async def fake(base_url, messages, **kwargs):
        user = messages[-1]["content"]
        calls.append(user)
        for (a, b), verdict in script.items():
            if f"Response A:\n{a}" in user and f"Response B:\n{b}" in user:
                return f"thinking...\nWINNER: {verdict}"
        raise AssertionError(f"unexpected pair in judge call:\n{user}")

    fake.calls = calls
    return fake


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def test_parse_winner_a_b_tie_and_last_match_wins():
    assert genrm._parse_winner("WINNER: A") == "A"
    assert genrm._parse_winner("winner: b") == "B"
    assert genrm._parse_winner("WINNER: TIE") == "TIE"
    assert genrm._parse_winner("WINNER: A\n... reconsidering ...\nWINNER: B") == "B"
    assert genrm._parse_winner("no verdict here") is None
    assert genrm._parse_winner("") is None


# ---------------------------------------------------------------------------
# Judge prompt construction
# ---------------------------------------------------------------------------


def test_pair_messages_include_rubric_question_and_both_responses():
    messages = genrm._build_pair_messages(
        rubric="Judge on correctness only.",
        question="What is 2+2?",
        response_a="4",
        response_b="5",
    )
    assert messages[0]["role"] == "system"
    user = messages[-1]["content"]
    assert "Judge on correctness only." in user
    assert "What is 2+2?" in user
    assert "Response A:\n4" in user
    assert "Response B:\n5" in user
    assert "WINNER:" in user


def test_pair_messages_without_rubric_use_generic_grading():
    messages = genrm._build_pair_messages(rubric=None, question="Q", response_a="x", response_b="y")
    assert "WINNER:" in messages[-1]["content"]


# ---------------------------------------------------------------------------
# Group reward computation
# ---------------------------------------------------------------------------


def test_round_robin_win_rates(monkeypatch):
    # 3 responses; r0 beats r1 and r2; r1 beats r2. Expect 1.0, 0.5, 0.0.
    samples = [_sample("r0"), _sample("r1"), _sample("r2")]
    script = {("r0", "r1"): "A", ("r0", "r2"): "A", ("r1", "r2"): "A"}
    fake = _scripted_judge(script)
    monkeypatch.setattr(genrm, "post_chat_completions", fake)

    rewards = _run(genrm.reward_func(_args(), samples))

    assert rewards == [1.0, 0.5, 0.0]
    assert len(fake.calls) == 3  # K*(K-1)/2 single-order pairs


def test_tie_and_unparseable_split_the_point(monkeypatch):
    samples = [_sample("r0"), _sample("r1")]

    async def tie_judge(base_url, messages, **kwargs):
        return "WINNER: TIE"

    monkeypatch.setattr(genrm, "post_chat_completions", tie_judge)
    assert _run(genrm.reward_func(_args(), samples)) == [0.5, 0.5]

    async def broken_judge(base_url, messages, **kwargs):
        return "I refuse to answer in the required format."

    monkeypatch.setattr(genrm, "post_chat_completions", broken_judge)
    assert _run(genrm.reward_func(_args(), samples)) == [0.5, 0.5]


def test_judge_error_counts_as_tie(monkeypatch):
    samples = [_sample("r0"), _sample("r1")]

    async def failing_judge(base_url, messages, **kwargs):
        raise RuntimeError("judge down")

    monkeypatch.setattr(genrm, "post_chat_completions", failing_judge)
    assert _run(genrm.reward_func(_args(), samples)) == [0.5, 0.5]


def test_empty_responses_lose_without_judge_calls(monkeypatch):
    samples = [_sample("real answer"), _sample(""), _sample("   ")]

    async def never_called(base_url, messages, **kwargs):
        raise AssertionError("judge should not be called for a single valid response")

    monkeypatch.setattr(genrm, "post_chat_completions", never_called)

    rewards = _run(genrm.reward_func(_args(), samples))

    # Sole valid response gets the neutral 0.5 (no opponents); empties get 0.
    assert rewards == [0.5, 0.0, 0.0]


def test_single_sample_group_is_neutral(monkeypatch):
    async def never_called(base_url, messages, **kwargs):
        raise AssertionError("no pairs to judge")

    monkeypatch.setattr(genrm, "post_chat_completions", never_called)
    assert _run(genrm.reward_func(_args(), [_sample("only")])) == [0.5]


def test_empty_group_returns_empty():
    assert _run(genrm.reward_func(_args(), [])) == []


def test_requires_judge_base_url():
    with pytest.raises(ValueError, match="judge-base-url"):
        _run(genrm.reward_func(_args(judge_base_url=None), [_sample("a"), _sample("b")]))


def test_rubric_read_from_first_sample_metadata(monkeypatch):
    seen = []

    async def capture(base_url, messages, **kwargs):
        seen.append(messages[-1]["content"])
        return "WINNER: TIE"

    monkeypatch.setattr(genrm, "post_chat_completions", capture)
    samples = [_sample("a", principle="My special rubric."), _sample("b", principle="My special rubric.")]
    _run(genrm.reward_func(_args(), samples))
    assert "My special rubric." in seen[0]
