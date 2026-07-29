from __future__ import annotations

import asyncio
import inspect

import aiohttp
import pytest

from orbit.rollout import scoring_client


def _run(coro):
    return asyncio.run(coro)


class _ByteStream:
    def __init__(self, body):
        self.body = body
        self.offset = 0

    async def read(self, size):
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class _HTTPResponse:
    def __init__(self, status, body):
        self.status = status
        self.content = _ByteStream(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

class _HTTPSession:
    def __init__(self, factory, session_kwargs):
        self.factory = factory
        self.factory.session_kwargs.append(session_kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.factory.post_calls.append((url, kwargs))
        status, body = self.factory.script.pop(0)
        return _HTTPResponse(status, body)


class _HTTPFactory:
    def __init__(self, *script):
        self.script = list(script)
        self.session_kwargs = []
        self.post_calls = []

    def __call__(self, *args, **kwargs):
        return _HTTPSession(self, kwargs)


def test_post_json_max_retries_zero_makes_one_attempt(monkeypatch):
    attempts = 0

    async def fail(url, payload, timeout, *, headers):
        nonlocal attempts
        attempts += 1
        raise aiohttp.ClientConnectionError("down")

    monkeypatch.setattr(scoring_client, "_post_json_once", fail)
    with pytest.raises(scoring_client.ScoringRequestError) as caught:
        _run(scoring_client.post_json("http://judge", {}, max_retries=0))
    assert attempts == 1
    assert caught.value.retryable is True
    assert "judge" not in str(caught.value)


def test_post_json_default_preserves_one_retry(monkeypatch):
    attempts = 0

    async def fail(url, payload, timeout, *, headers):
        nonlocal attempts
        attempts += 1
        raise aiohttp.ClientConnectionError("down")

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(scoring_client, "_post_json_once", fail)
    monkeypatch.setattr(scoring_client.asyncio, "sleep", no_sleep)
    with pytest.raises(scoring_client.ScoringRequestError) as caught:
        _run(scoring_client.post_json("http://teacher", {}))
    assert attempts == 2
    assert caught.value.retryable is True


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b'{"score":NaN}',
        b"\xff",
        b'{"score":1,"score":2}',
    ],
    ids=["json-decode", "nonfinite", "invalid-utf8", "duplicate-key"],
)
def test_malformed_success_json_is_terminal_without_retry(monkeypatch, body):
    factory = _HTTPFactory((200, body), (200, b'{"unused":true}'))
    monkeypatch.setattr(scoring_client.aiohttp, "ClientSession", factory)

    with pytest.raises(scoring_client.ScoringProtocolError):
        _run(scoring_client.post_chat_completions("http://judge", []))

    assert len(factory.post_calls) == 1


def test_post_json_retries_5xx_with_identical_bounded_request(monkeypatch):
    factory = _HTTPFactory((503, b"secret body"), (200, b'{"ok":true}'))

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(scoring_client.aiohttp, "ClientSession", factory)
    monkeypatch.setattr(scoring_client.asyncio, "sleep", no_sleep)

    response = _run(
        scoring_client.post_json_with_metadata(
            "https://10.0.0.5/generate",
            {"rid": "same-request"},
            max_retries=1,
            headers={"Host": "teacher.internal:443"},
        )
    )

    assert response == scoring_client.ScoringJSONResponse(
        body={"ok": True},
        retry_count=1,
    )
    assert len(factory.post_calls) == 2
    assert factory.post_calls[0] == factory.post_calls[1]
    _, request = factory.post_calls[0]
    assert request["allow_redirects"] is False
    assert request["headers"] == {"Host": "teacher.internal:443"}
    assert request["server_hostname"] == "teacher.internal"
    assert all(
        kwargs["raise_for_status"] is False
        for kwargs in factory.session_kwargs
    )


@pytest.mark.parametrize("status", (302, 400, 404, 600))
def test_post_json_does_not_retry_non_5xx_statuses(monkeypatch, status):
    factory = _HTTPFactory((status, b"secret body"), (200, b'{"unused":true}'))
    monkeypatch.setattr(scoring_client.aiohttp, "ClientSession", factory)

    with pytest.raises(scoring_client.ScoringRequestError) as caught:
        _run(
            scoring_client.post_json_with_metadata(
                "http://teacher/generate",
                {"secret": "payload"},
                max_retries=1,
            )
        )

    assert caught.value.retryable is False
    assert str(caught.value) == "scoring request failed"
    assert len(factory.post_calls) == 1


def test_post_json_rejects_response_over_byte_bound_without_retry(monkeypatch):
    monkeypatch.setattr(scoring_client, "SCORING_MAX_RESPONSE_BYTES", 8)
    factory = _HTTPFactory((200, b"123456789"), (200, b'{"ok":1}'))
    monkeypatch.setattr(scoring_client.aiohttp, "ClientSession", factory)

    with pytest.raises(scoring_client.ScoringProtocolError):
        _run(
            scoring_client.post_json_with_metadata(
                "http://teacher/generate",
                {},
                max_retries=1,
            )
        )

    assert len(factory.post_calls) == 1


def test_post_json_requires_exact_top_level_object(monkeypatch):
    factory = _HTTPFactory((200, b"[]"))
    monkeypatch.setattr(scoring_client.aiohttp, "ClientSession", factory)

    with pytest.raises(scoring_client.ScoringProtocolError):
        _run(scoring_client.post_json("http://teacher/generate", {}))


def test_max_retries_is_keyword_only_on_both_clients():
    assert inspect.signature(scoring_client.post_json).parameters["max_retries"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        inspect.signature(scoring_client.post_json_with_metadata).parameters[
            "max_retries"
        ].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(scoring_client.post_chat_completions).parameters["max_retries"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_post_json_rejects_non_exact_integer_max_retries(monkeypatch, value):
    async def unexpected(*args, **kwargs):
        raise AssertionError("invalid retry count must fail before the request")

    monkeypatch.setattr(scoring_client, "_post_json_once", unexpected)
    with pytest.raises(TypeError, match="max_retries must be an exact integer"):
        _run(scoring_client.post_json("http://judge", {}, max_retries=value))


def test_post_json_rejects_negative_max_retries(monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("invalid retry count must fail before the request")

    monkeypatch.setattr(scoring_client, "_post_json_once", unexpected)
    with pytest.raises(ValueError, match="max_retries must be nonnegative"):
        _run(scoring_client.post_json("http://judge", {}, max_retries=-1))


def test_post_json_with_metadata_returns_retry_count_and_copies_headers(
    monkeypatch,
):
    attempts = 0
    observed = []

    async def scripted(url, payload, timeout, *, headers):
        nonlocal attempts
        attempts += 1
        observed.append(headers)
        if attempts == 1:
            raise aiohttp.ClientConnectionError("down")
        return {"ok": True}

    async def no_sleep(delay):
        return None

    headers = {"Host": "teacher.internal", "Authorization": "Bearer secret"}
    monkeypatch.setattr(scoring_client, "_post_json_once", scripted)
    monkeypatch.setattr(scoring_client.asyncio, "sleep", no_sleep)

    response = _run(
        scoring_client.post_json_with_metadata(
            "http://10.0.0.5/generate",
            {"rid": "request-1"},
            max_retries=1,
            headers=headers,
        )
    )

    headers["Authorization"] = "changed"
    assert response == scoring_client.ScoringJSONResponse(
        body={"ok": True},
        retry_count=1,
    )
    assert observed == [
        {"Host": "teacher.internal", "Authorization": "Bearer secret"},
        {"Host": "teacher.internal", "Authorization": "Bearer secret"},
    ]


@pytest.mark.parametrize(
    "headers",
    (
        {"Bad Header": "value"},
        {"X-Test": "bad\nvalue"},
        {"X-Test": 7},
        [("X-Test", "value")],
    ),
)
def test_post_json_with_metadata_rejects_unsafe_headers_before_io(
    monkeypatch,
    headers,
):
    async def unexpected(*args, **kwargs):
        raise AssertionError("unsafe headers must fail before I/O")

    monkeypatch.setattr(scoring_client, "_post_json_once", unexpected)

    with pytest.raises((TypeError, ValueError), match="header"):
        _run(
            scoring_client.post_json_with_metadata(
                "http://teacher/generate",
                {},
                headers=headers,
            )
        )


@pytest.mark.parametrize("retryable", (True, False))
def test_scoring_request_error_is_url_free_and_exact(retryable):
    error = scoring_client.ScoringRequestError(retryable=retryable)

    assert error.retryable is retryable
    assert str(error) == "scoring request failed"

    with pytest.raises(TypeError, match="retryable"):
        scoring_client.ScoringRequestError(retryable=1)


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_post_chat_completions_rejects_non_exact_integer_max_retries(monkeypatch, value):
    async def unexpected(*args, **kwargs):
        raise AssertionError("invalid retry count must fail before post_json")

    monkeypatch.setattr(scoring_client, "post_json", unexpected)
    with pytest.raises(TypeError, match="max_retries must be an exact integer"):
        _run(scoring_client.post_chat_completions("http://judge", [], max_retries=value))


def test_post_chat_completions_rejects_negative_max_retries(monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("invalid retry count must fail before post_json")

    monkeypatch.setattr(scoring_client, "post_json", unexpected)
    with pytest.raises(ValueError, match="max_retries must be nonnegative"):
        _run(scoring_client.post_chat_completions("http://judge", [], max_retries=-1))


def test_post_chat_completions_forwards_retries_and_returns_content(monkeypatch):
    seen = {}

    async def fake(url, payload, timeout_secs=None, *, max_retries):
        seen.update(url=url, payload=payload, timeout_secs=timeout_secs, max_retries=max_retries)
        return {"choices": [{"message": {"content": "WINNER: TIE"}}]}

    monkeypatch.setattr(scoring_client, "post_json", fake)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": {"type": "object"}},
    }
    content = _run(
        scoring_client.post_chat_completions(
            "http://judge/",
            [{"role": "user", "content": "compare"}],
            timeout_secs=12,
            max_retries=0,
            response_format=response_format,
        )
    )
    assert content == "WINNER: TIE"
    assert seen["url"] == "http://judge/v1/chat/completions"
    assert seen["timeout_secs"] == 12
    assert seen["max_retries"] == 0
    assert seen["payload"]["response_format"] == response_format


@pytest.mark.parametrize("response_format", [[], "json", True, 1])
def test_post_chat_completions_rejects_non_object_response_format(
    monkeypatch, response_format
):
    async def unexpected(*args, **kwargs):
        raise AssertionError("invalid response format must fail before post_json")

    monkeypatch.setattr(scoring_client, "post_json", unexpected)
    with pytest.raises(TypeError, match="response_format must be an exact object"):
        _run(
            scoring_client.post_chat_completions(
                "http://judge",
                [],
                response_format=response_format,
            )
        )


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"choices": None},
        {"choices": []},
        {"choices": ["bad"]},
        {"choices": [{}]},
        {"choices": [{"message": "bad"}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": 1}}]},
    ],
)
def test_post_chat_completions_rejects_invalid_response_schema(monkeypatch, response):
    async def fake(*args, **kwargs):
        return response

    monkeypatch.setattr(scoring_client, "post_json", fake)
    with pytest.raises(scoring_client.ScoringProtocolError):
        _run(scoring_client.post_chat_completions("http://judge", []))
