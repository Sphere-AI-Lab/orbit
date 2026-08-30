"""LLM-judge reward hooks (orbit/rewards/llm_judge.py).

An external judge model (any instruct model served by sglang) grades each
sample via the OpenAI-compatible chat endpoint, wired through orbit's
--custom-rm-path. Two modes:
- equivalence: binary verdict vs the reference label (the NeMo-RL
  equivalence_llm_judge analog) -> reward 1.0 / 0.0.
- score: pointwise 0-10 grade -> reward normalized to [0, 1].
"""

import argparse
import asyncio
import json

import aiohttp
import pytest

from orbit.rewards import llm_judge, scoring_client
from orbit.rewards.grader_errors import GraderInfrastructureError, InfrastructureErrorCode
from miles.utils.types import Sample


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


def _mock_success_decode_failure(monkeypatch, failure):
    attempts = []

    async def fail(url, payload, timeout, *, headers, max_response_bytes=None):
        attempts.append(1)
        raise scoring_client.ScoringProtocolError(type(failure).__name__)

    monkeypatch.setattr(scoring_client, "_post_json_once", fail)
    return attempts


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
        ("verdict: equivalent", None),
        ("VERDICT: DIFFERENT\nwait no", None),
        ("VERDICT: EQUIVALENT because it matches", None),
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
        ("SCORE: 8.5", 0.85),
        ("SCORE: 15", 1.0),  # clamped
        ("score: 8.5", None),
        ("SCORE: 7 points", None),
        ("SCORE: 7\nadditional text", None),
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
    with pytest.raises(GraderInfrastructureError) as caught:
        llm_judge._build_judge_messages("equivalence", "Q?", "resp", None)
    assert caught.value.code is InfrastructureErrorCode.INVALID_SOURCE


# --- reward_func (judge server monkeypatched) ---


async def test_reward_func_equivalence_positive(monkeypatch):
    async def fake_chat(base_url, messages, **kwargs):
        assert base_url == "http://judge:30600"
        assert kwargs["max_retries"] == 0
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


async def test_reward_func_explicit_negative_and_zero_score_remain_semantic(monkeypatch):
    replies = ["Reasoning\nVERDICT: DIFFERENT", "Reasoning\nSCORE: 0"]

    async def fake_chat(base_url, messages, **kwargs):
        return replies.pop(0)

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    assert await llm_judge.reward_func(_args(), _sample()) == 0.0
    assert await llm_judge.reward_func(_args(judge_mode="score"), _sample()) == 0.0


@pytest.mark.parametrize(
    ("mode", "repaired_reply", "expected"),
    [
        ("equivalence", "VERDICT: DIFFERENT", 0.0),
        ("score", "SCORE: 8", 0.8),
    ],
)
async def test_reward_func_repairs_one_malformed_reply_with_marker_only_request(
    monkeypatch,
    mode,
    repaired_reply,
    expected,
):
    calls = []

    async def fake_chat(base_url, messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) == 1:
            return "I evaluated the answer but omitted the required marker."
        return repaired_reply

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    reward = await llm_judge.reward_func(_args(judge_mode=mode), _sample())

    assert reward == expected
    assert len(calls) == 2
    repair_messages, repair_kwargs = calls[1]
    joined = " ".join(message["content"] for message in repair_messages)
    assert "What is 2+2?" in joined
    assert "The answer is 4." in joined
    assert "Return exactly one" in joined
    assert "nothing else" in joined
    assert repair_kwargs["max_tokens"] == llm_judge.JUDGE_REPAIR_MAX_TOKENS
    assert repair_kwargs["max_retries"] == 0


async def test_reward_func_unparseable_verdict_is_protocol_error(monkeypatch):
    calls = []

    async def fake_chat(base_url, messages, **kwargs):
        calls.append((messages, kwargs))
        return "I refuse to answer in the requested format."

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    with pytest.raises(GraderInfrastructureError) as caught:
        await llm_judge.reward_func(_args(), _sample())
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False
    assert len(calls) == 2


async def test_reward_func_repair_transport_failure_remains_transport_error(monkeypatch):
    calls = []

    async def fake_chat(base_url, messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) == 1:
            return "No marker."
        raise aiohttp.ClientConnectionError("judge repair disconnected")

    monkeypatch.setattr(llm_judge, "post_chat_completions", fake_chat)
    with pytest.raises(GraderInfrastructureError) as caught:
        await llm_judge.reward_func(_args(), _sample())

    assert caught.value.code is InfrastructureErrorCode.TRANSPORT_ERROR
    assert caught.value.retryable is True
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("failure_kind", "code", "retryable"),
    [
        ("transport", InfrastructureErrorCode.TRANSPORT_ERROR, True),
        ("protocol", InfrastructureErrorCode.PROTOCOL_ERROR, False),
    ],
)
async def test_reward_func_translates_judge_failures(monkeypatch, failure_kind, code, retryable):
    failure = (
        RuntimeError("down") if failure_kind == "transport" else scoring_client.ScoringProtocolError("bad choices")
    )

    async def fail(base_url, messages, **kwargs):
        raise failure

    monkeypatch.setattr(llm_judge, "post_chat_completions", fail)
    with pytest.raises(GraderInfrastructureError) as caught:
        await llm_judge.reward_func(_args(), _sample())
    assert caught.value.code is code
    assert caught.value.retryable is retryable


@pytest.mark.parametrize(("status", "retryable"), [(400, False), (503, True)])
async def test_reward_func_preserves_http_transport_retryability(monkeypatch, status, retryable):
    failure = aiohttp.ClientResponseError(None, (), status=status, message="judge HTTP error")

    async def fail(base_url, messages, **kwargs):
        raise failure

    monkeypatch.setattr(llm_judge, "post_chat_completions", fail)
    with pytest.raises(GraderInfrastructureError) as caught:
        await llm_judge.reward_func(_args(), _sample())
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
async def test_reward_func_maps_direct_decode_failures_to_protocol(monkeypatch, failure):
    attempts = _mock_success_decode_failure(monkeypatch, failure)
    with pytest.raises(GraderInfrastructureError) as caught:
        await llm_judge.reward_func(_args(), _sample())
    assert caught.value.code is InfrastructureErrorCode.PROTOCOL_ERROR
    assert caught.value.retryable is False
    assert attempts == [1]


@pytest.mark.parametrize("kind", ["infrastructure", "cancellation"])
async def test_reward_func_preserves_infrastructure_and_cancellation_identity(monkeypatch, kind):
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

    monkeypatch.setattr(llm_judge, "post_chat_completions", fail)
    with pytest.raises(type(failure)) as caught:
        await llm_judge.reward_func(_args(), _sample())
    assert caught.value is failure


async def test_reward_func_configuration_and_source_failures_are_typed():
    with pytest.raises(GraderInfrastructureError) as configuration:
        await llm_judge.reward_func(_args(judge_base_url=None), _sample())
    assert configuration.value.code is InfrastructureErrorCode.CONFIGURATION

    with pytest.raises(GraderInfrastructureError) as missing_label:
        await llm_judge.reward_func(_args(), _sample(label=None))
    assert missing_label.value.code is InfrastructureErrorCode.INVALID_SOURCE

    with pytest.raises(GraderInfrastructureError) as unknown_mode:
        await llm_judge.reward_func(_args(judge_mode="unknown"), _sample())
    assert unknown_mode.value.code is InfrastructureErrorCode.INVALID_SOURCE


# --- startup validation ---

from miles.utils.arguments import _validate_judge_args  # noqa: E402


def test_validate_judge_requires_base_url():
    args = argparse.Namespace(
        custom_rm_path="orbit.rewards.llm_judge.reward_func", judge_base_url=None, judge_mode="equivalence"
    )
    with pytest.raises(ValueError, match="judge-base-url"):
        _validate_judge_args(args)


def test_validate_judge_noop_for_other_rm():
    args = argparse.Namespace(custom_rm_path="orbit.opd.opd_sglang.reward_func", judge_base_url=None)
    _validate_judge_args(args)


def test_validate_judge_passes_when_configured():
    args = argparse.Namespace(
        custom_rm_path="orbit.rewards.llm_judge.reward_func",
        judge_base_url="http://judge:30600",
        judge_mode="score",
    )
    _validate_judge_args(args)
