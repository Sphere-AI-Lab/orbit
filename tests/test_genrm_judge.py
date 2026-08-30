"""Unit tests for group-wise pairwise GenRM rewards (miles/orbit/rewards/genrm_judge.py).

The hook shape: ``--group-rm`` routes the whole n-samples-per-prompt group into
``batched_async_rm``, which calls ``reward_func(args, samples)`` -> list of
rewards. The judge compares responses pairwise (round-robin, single order)
under the row's rubric (``sample.metadata["principle"]``); rewards are
win-rates in [0, 1].
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest

import miles.orbit.rewards.genrm_judge as genrm
from miles.orbit.rewards import scoring_client
from miles.orbit.rewards.grader_errors import GraderInfrastructureError, InfrastructureErrorCode
from miles.utils.types import Sample


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


def _mock_success_decode_failure(monkeypatch, failure):
    attempts = []

    async def fail(url, payload, timeout, *, headers, max_response_bytes=None):
        attempts.append(1)
        raise scoring_client.ScoringProtocolError(type(failure).__name__)

    monkeypatch.setattr(scoring_client, "_post_json_once", fail)
    return attempts


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


def test_parse_winner_requires_an_exact_final_line():
    assert genrm._parse_winner('{"winner":"A"}') == "A"
    assert genrm._parse_winner('{"winner":"B"}') == "B"
    assert genrm._parse_winner('{"winner":"TIE"}') == "TIE"
    assert genrm._parse_winner("WINNER: A") == "A"
    assert genrm._parse_winner("reasoning\nWINNER: B") == "B"
    assert genrm._parse_winner("WINNER: TIE") == "TIE"
    assert genrm._parse_winner('{"winner":"A","reason":"extra"}') is None
    assert genrm._parse_winner('{"winner":"A","winner":"B"}') is None
    assert genrm._parse_winner('{"winner":"C"}') is None
    assert genrm._parse_winner("winner: b") is None
    assert genrm._parse_winner("WINNER: A\n... reconsidering ...") is None
    assert genrm._parse_winner("WINNER: A because it is better") is None
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
    assert '{"winner":"A"}' in user
    assert '{"winner":"B"}' in user
    assert '{"winner":"TIE"}' in user


def test_pair_messages_without_rubric_use_generic_grading():
    messages = genrm._build_pair_messages(rubric=None, question="Q", response_a="x", response_b="y")
    assert '"winner"' in messages[-1]["content"]


def test_pairwise_requests_a_strict_json_winner_schema(monkeypatch):
    seen = []

    async def capture(base_url, messages, **kwargs):
        seen.append(kwargs)
        return '{"winner":"TIE"}'

    monkeypatch.setattr(genrm, "post_chat_completions", capture)

    assert _run(genrm.reward_func(_args(), [_sample("a"), _sample("b")])) == [0.5, 0.5]
    assert len(seen) == 1
    response_format = seen[0]["response_format"]
    assert response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "pairwise_winner",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "winner": {"type": "string", "enum": ["A", "B", "TIE"]}
                },
                "required": ["winner"],
                "additionalProperties": False,
            },
        },
    }


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


def test_explicit_tie_remains_semantic(monkeypatch):
    samples = [_sample("r0"), _sample("r1")]

    async def tie_judge(base_url, messages, **kwargs):
        return "WINNER: TIE"

    monkeypatch.setattr(genrm, "post_chat_completions", tie_judge)
    assert _run(genrm.reward_func(_args(), samples)) == [0.5, 0.5]


def test_unparseable_and_transport_failures_are_infrastructure(monkeypatch):
    samples = [_sample("r0"), _sample("r1")]

    async def no_verdict(base_url, messages, **kwargs):
        return "no verdict"

    monkeypatch.setattr(genrm, "post_chat_completions", no_verdict)
    with pytest.raises(GraderInfrastructureError) as protocol:
        _run(genrm.reward_func(_args(), samples))
    assert protocol.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert protocol.value.retryable is False

    async def down(base_url, messages, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(genrm, "post_chat_completions", down)
    with pytest.raises(GraderInfrastructureError) as transport:
        _run(genrm.reward_func(_args(), samples))
    assert transport.value.code is InfrastructureErrorCode.TRANSPORT_ERROR
    assert transport.value.retryable is True


@pytest.mark.parametrize(("status", "retryable"), [(400, True), (503, True)])
def test_http_transport_retryability_is_preserved(monkeypatch, status, retryable):
    failure = aiohttp.ClientResponseError(None, (), status=status, message="judge HTTP error")

    async def fail(base_url, messages, **kwargs):
        raise failure

    monkeypatch.setattr(genrm, "post_chat_completions", fail)
    with pytest.raises(GraderInfrastructureError) as caught:
        _run(genrm.reward_func(_args(), [_sample("a"), _sample("b")]))
    assert caught.value.code is InfrastructureErrorCode.TRANSPORT_ERROR
    assert caught.value.retryable is retryable


@pytest.mark.parametrize(
    "failure",
    [
        json.JSONDecodeError("invalid JSON", "not-json", 0),
        aiohttp.ContentTypeError(None, (), status=200, message="unexpected content type"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ValueError("integer string conversion limit exceeded for 5,000-digit integer"),
    ],
    ids=["json-decode", "content-type", "invalid-utf8", "integer-limit"],
)
def test_direct_decode_failures_are_protocol_errors(monkeypatch, failure):
    attempts = _mock_success_decode_failure(monkeypatch, failure)
    with pytest.raises(GraderInfrastructureError) as caught:
        _run(genrm.reward_func(_args(), [_sample("a"), _sample("b")]))
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False
    assert attempts == [1]


def test_invalid_chat_schema_is_protocol_error(monkeypatch):
    async def malformed(base_url, messages, **kwargs):
        raise scoring_client.ScoringProtocolError("bad choices")

    monkeypatch.setattr(genrm, "post_chat_completions", malformed)
    with pytest.raises(GraderInfrastructureError) as caught:
        _run(genrm.reward_func(_args(), [_sample("a"), _sample("b")]))
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False


@pytest.mark.parametrize("kind", ["infrastructure", "cancellation"])
def test_pairwise_preserves_infrastructure_and_cancellation_identity(monkeypatch, kind):
    failure = (
        GraderInfrastructureError(
            InfrastructureErrorCode.CONFIGURATION,
            grader="upstream",
            stage="setup",
            retryable=False,
            safe_detail="upstream configuration failed",
        )
        if kind == "infrastructure"
        else asyncio.CancelledError("stop")
    )

    async def fail(base_url, messages, **kwargs):
        raise failure

    monkeypatch.setattr(genrm, "post_chat_completions", fail)
    with pytest.raises(type(failure)) as caught:
        _run(genrm.reward_func(_args(), [_sample("a"), _sample("b")]))
    assert caught.value is failure


@pytest.mark.parametrize("failure_kind", ["infrastructure", "cancellation"])
async def test_pair_failure_cancels_and_drains_siblings_preserving_identity(monkeypatch, failure_kind):
    failure = (
        GraderInfrastructureError(
            InfrastructureErrorCode.TRANSPORT_ERROR,
            grader="upstream",
            stage="request",
            retryable=True,
            safe_detail="upstream request failed",
        )
        if failure_kind == "infrastructure"
        else asyncio.CancelledError("pair cancelled")
    )
    all_started = asyncio.Event()
    release = asyncio.Event()
    siblings_settled = asyncio.Event()
    started = 0
    cancelled = set()
    settled = set()

    async def fail_one_pair(base_url, messages, **kwargs):
        nonlocal started
        ordinal = started
        started += 1
        if started == 3:
            all_started.set()
        await all_started.wait()
        if ordinal == 0:
            raise failure
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.add(ordinal)
            raise
        finally:
            settled.add(ordinal)
            if len(settled) == 2:
                siblings_settled.set()

    monkeypatch.setattr(genrm, "post_chat_completions", fail_one_pair)
    try:
        with pytest.raises(type(failure)) as caught:
            await genrm.reward_func(_args(), [_sample("a"), _sample("b"), _sample("c")])
        assert caught.value is failure
        assert cancelled == {1, 2}
        assert settled == {1, 2}
    finally:
        release.set()
        if not siblings_settled.is_set():
            await asyncio.wait_for(siblings_settled.wait(), timeout=1)


async def test_external_cancellation_preserves_identity_and_settles_all_pairs(monkeypatch):
    all_started = asyncio.Event()
    blocker = asyncio.Event()
    all_settled = asyncio.Event()
    started = 0
    settled = set()
    observed = []

    async def block_pair(base_url, messages, **kwargs):
        nonlocal started
        ordinal = started
        started += 1
        if started == 3:
            all_started.set()
        try:
            await blocker.wait()
        finally:
            settled.add(ordinal)
            if len(settled) == 3:
                all_settled.set()

    async def invoke_reward():
        try:
            await genrm.reward_func(_args(), [_sample("a"), _sample("b"), _sample("c")])
        except asyncio.CancelledError as exc:
            observed.append(exc)
            raise

    monkeypatch.setattr(genrm, "post_chat_completions", block_pair)
    task = asyncio.create_task(invoke_reward())
    await asyncio.wait_for(all_started.wait(), timeout=1)
    task.cancel("external stop")
    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert observed == [caught.value]
        assert caught.value.args == ("external stop",)
        assert settled == {0, 1, 2}
    finally:
        blocker.set()
        if not all_settled.is_set():
            await asyncio.wait_for(all_settled.wait(), timeout=1)


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
    with pytest.raises(GraderInfrastructureError) as caught:
        _run(genrm.reward_func(_args(judge_base_url=None), [_sample("a"), _sample("b")]))
    assert caught.value.code is InfrastructureErrorCode.CONFIGURATION
    assert caught.value.retryable is False


def test_rubric_read_from_first_sample_metadata(monkeypatch):
    seen = []

    async def capture(base_url, messages, **kwargs):
        seen.append(messages[-1]["content"])
        return "WINNER: TIE"

    monkeypatch.setattr(genrm, "post_chat_completions", capture)
    samples = [_sample("a", principle="My special rubric."), _sample("b", principle="My special rubric.")]
    _run(genrm.reward_func(_args(), samples))
    assert "My special rubric." in seen[0]


def test_pairwise_disables_scoring_client_retries(monkeypatch):
    seen = []

    async def capture(base_url, messages, **kwargs):
        seen.append(kwargs["max_retries"])
        return "WINNER: TIE"

    monkeypatch.setattr(genrm, "post_chat_completions", capture)
    _run(genrm.reward_func(_args(), [_sample("a"), _sample("b")]))
    assert seen == [0]
