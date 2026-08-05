"""Shared HTTP client for rollout-side scoring calls (OPD teachers, LLM judges).

Bounded retry on transient failures (timeout, connection error, HTTP 5xx) with
jitter; 4xx responses never retry.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import random
import re
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from orbit.ultra.strict_json import loads_strict

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - the locked sglang environment provides it
    _orjson = None

SCORING_MAX_RETRIES = 1
SCORING_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_SCORING_MAX_JSON_DEPTH = 32
_MAX_HEADER_COUNT = 64
_MAX_HEADER_NAME_BYTES = 256
_MAX_HEADER_VALUE_BYTES = 8192
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")


class ScoringProtocolError(RuntimeError):
    """An OpenAI-compatible scoring response violated the expected schema."""


@dataclass(frozen=True)
class ScoringJSONResponse:
    body: dict[str, Any]
    retry_count: int


class ScoringRequestError(RuntimeError):
    def __init__(self, *, retryable: bool) -> None:
        if type(retryable) is not bool:
            raise TypeError("retryable must be bool")
        self.retryable = retryable
        super().__init__("scoring request failed")


class _ScoringHTTPStatusError(RuntimeError):
    def __init__(self, *, retryable: bool) -> None:
        self.retryable = retryable
        super().__init__("scoring service returned a terminal HTTP status")


def _validate_max_retries(max_retries: int) -> None:
    if type(max_retries) is not int:
        raise TypeError("max_retries must be an exact integer")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")


def _validate_trusted_local_response(trusted_local_response: bool) -> None:
    if type(trusted_local_response) is not bool:
        raise TypeError("trusted_local_response must be an exact boolean")


def _is_retryable_scoring_error(exc: BaseException) -> bool:
    if isinstance(exc, (ScoringRequestError, _ScoringHTTPStatusError)):
        return exc.retryable
    if isinstance(exc, ScoringProtocolError):
        return False
    if isinstance(exc, aiohttp.ClientResponseError):
        return 500 <= exc.status < 600
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


def scoring_transport_error_retryable(exc: BaseException) -> bool:
    """Preserve terminal HTTP status semantics; other request failures retry."""
    if isinstance(exc, (ScoringRequestError, _ScoringHTTPStatusError)):
        return exc.retryable
    if isinstance(exc, ScoringProtocolError):
        return False
    if isinstance(exc, aiohttp.ClientResponseError):
        return 500 <= exc.status < 600
    return True


def _validate_scoring_headers(
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping) or len(headers) > _MAX_HEADER_COUNT:
        raise TypeError("scoring headers must be a bounded mapping")
    copied: dict[str, str] = {}
    for name, value in headers.items():
        if (
            type(name) is not str
            or _HEADER_NAME.fullmatch(name) is None
            or len(name.encode("ascii")) > _MAX_HEADER_NAME_BYTES
        ):
            raise ValueError("scoring header name is invalid")
        if (
            type(value) is not str
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in value
            )
        ):
            raise ValueError("scoring header value is invalid")
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("scoring header value is invalid") from None
        if len(encoded) > _MAX_HEADER_VALUE_BYTES:
            raise ValueError("scoring header value is too large")
        copied[name] = value
    return copied


async def _post_json_once(
    url: str,
    payload: dict[str, Any],
    timeout: aiohttp.ClientTimeout,
    *,
    headers: Mapping[str, str],
    max_response_bytes: int = SCORING_MAX_RESPONSE_BYTES,
    trusted_local_response: bool = False,
) -> dict[str, Any]:
    request_options: dict[str, Any] = {
        "json": payload,
        "headers": dict(headers),
        "allow_redirects": False,
    }
    if urlsplit(url).scheme.lower() == "https":
        host_header = next(
            (
                value
                for name, value in headers.items()
                if name.lower() == "host"
            ),
            None,
        )
        if host_header is not None:
            server_hostname = urlsplit(f"//{host_header}").hostname
            if server_hostname is not None:
                request_options["server_hostname"] = server_hostname
    async with aiohttp.ClientSession(
        timeout=timeout,
        raise_for_status=False,
    ) as session:
        async with session.post(url, **request_options) as response:
            status = response.status
            if type(status) is not int or not 200 <= status < 300:
                raise _ScoringHTTPStatusError(
                    retryable=(
                        type(status) is int and 500 <= status < 600
                    ),
                )
            encoded = bytearray()
            while True:
                chunk = await response.content.read(
                    min(
                        1024 * 1024,
                        max_response_bytes + 1 - len(encoded),
                    )
                )
                if not chunk:
                    break
                encoded.extend(chunk)
                if len(encoded) > max_response_bytes:
                    raise ScoringProtocolError(
                        "scoring response exceeds its byte limit"
                    )
    try:
        # Managed in-job services are trusted producers. Their full-vocab OPD
        # responses contain tens of MB of base64 hidden states, so running the
        # strict decoder's Python character-by-character validation on the one
        # asyncio loop serializes an otherwise concurrent request batch. orjson
        # retains JSON syntax/UTF-8 validation while avoiding that scan. Keep the
        # strict decoder as the default for every external scoring endpoint.
        if trusted_local_response and _orjson is not None:
            body = _orjson.loads(encoded)
        else:
            body = loads_strict(
                bytes(encoded),
                max_bytes=max_response_bytes,
                max_depth=_SCORING_MAX_JSON_DEPTH,
            )
    except (TypeError, ValueError):
        raise ScoringProtocolError(
            "scoring service returned invalid strict JSON"
        ) from None
    if type(body) is not dict:
        raise ScoringProtocolError("scoring response must be an exact object")
    return body


async def post_json_with_metadata(
    url: str,
    payload: dict[str, Any],
    timeout_secs: int | float | None = None,
    *,
    max_retries: int = SCORING_MAX_RETRIES,
    headers: Mapping[str, str] | None = None,
    max_response_bytes: int | None = None,
    trusted_local_response: bool = False,
) -> ScoringJSONResponse:
    _validate_max_retries(max_retries)
    _validate_trusted_local_response(trusted_local_response)
    if max_response_bytes is None:
        max_response_bytes = SCORING_MAX_RESPONSE_BYTES
    safe_headers = _validate_scoring_headers(headers)
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    for attempt in range(max_retries + 1):
        try:
            request_options = {
                "headers": safe_headers,
                "max_response_bytes": max_response_bytes,
            }
            # Preserve compatibility with tests/callers that monkeypatch the
            # private helper using its original default-path signature.
            if trusted_local_response:
                request_options["trusted_local_response"] = True
            body = await _post_json_once(url, payload, timeout, **request_options)
            return ScoringJSONResponse(body=body, retry_count=attempt)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if isinstance(error, ScoringProtocolError):
                raise
            retryable = _is_retryable_scoring_error(error)
            if attempt >= max_retries or not retryable:
                raise ScoringRequestError(retryable=retryable) from None
            delay = min(2**attempt, 4) * (0.5 + 0.5 * random.random())
            await asyncio.sleep(delay)
    raise AssertionError("retry loop exhausted without returning or raising")


async def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_secs: int | float | None = None,
    *,
    max_retries: int = SCORING_MAX_RETRIES,
    max_response_bytes: int | None = None,
    trusted_local_response: bool = False,
) -> dict[str, Any]:
    response = await post_json_with_metadata(
        url,
        payload,
        timeout_secs,
        max_retries=max_retries,
        max_response_bytes=max_response_bytes,
        trusted_local_response=trusted_local_response,
    )
    return response.body


async def post_chat_completions(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    model: str = "default",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_secs: int | float | None = None,
    max_retries: int = SCORING_MAX_RETRIES,
    response_format: dict[str, Any] | None = None,
) -> str:
    """POST to an OpenAI-compatible ``/v1/chat/completions`` endpoint (e.g. an
    sglang server) and return the assistant message content."""
    _validate_max_retries(max_retries)
    if response_format is not None and type(response_format) is not dict:
        raise TypeError("response_format must be an exact object")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    response = await post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        payload,
        timeout_secs=timeout_secs,
        max_retries=max_retries,
    )
    if not isinstance(response, dict):
        raise ScoringProtocolError("chat-completion response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ScoringProtocolError("chat-completion choices must be a nonempty list")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ScoringProtocolError("chat-completion first choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ScoringProtocolError("chat-completion message must be an object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ScoringProtocolError("chat-completion content must be a string")
    return content
