"""Shared HTTP client for rollout-side scoring calls (OPD teachers, LLM judges).

Bounded retry on transient failures (timeout, connection error, HTTP 5xx) with
jitter; 4xx responses never retry.
"""

import asyncio
import random
from typing import Any

import aiohttp

SCORING_MAX_RETRIES = 1


def _is_retryable_scoring_error(exc: BaseException) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


async def _post_json_once(url: str, payload: dict[str, Any], timeout: aiohttp.ClientTimeout) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def post_json(url: str, payload: dict[str, Any], timeout_secs: int | float | None = None) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_secs)
    for attempt in range(SCORING_MAX_RETRIES + 1):
        try:
            return await _post_json_once(url, payload, timeout)
        except BaseException as exc:
            if attempt >= SCORING_MAX_RETRIES or not _is_retryable_scoring_error(exc):
                raise
            await asyncio.sleep(min(2**attempt, 4) * (0.5 + 0.5 * random.random()))
    raise AssertionError("unreachable")


async def post_chat_completions(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    model: str = "default",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_secs: int | float | None = None,
) -> str:
    """POST to an OpenAI-compatible ``/v1/chat/completions`` endpoint (e.g. an
    sglang server) and return the assistant message content."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = await post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout_secs=timeout_secs)
    return response["choices"][0]["message"]["content"]
