"""Local retrieval client for Search-R1 style rollouts."""

import asyncio
from typing import Any

import httpx


async def local_search(
    search_url: str,
    query: str,
    top_k: int = 5,
    timeout: int = 60,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,
    }

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if proxy:
        client_kwargs["proxy"] = proxy

    if proxy is None:
        client_kwargs["trust_env"] = False

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(search_url, json=payload)
                response.raise_for_status()
                result = response.json()
            break
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt == 2:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError("unreachable") from last_error

    retrieval_results = result.get("result", [[]])[0]
    contexts = []
    for item in retrieval_results:
        if not isinstance(item, dict):
            continue
        document = item.get("document", item)
        content = document.get("contents", "") if isinstance(document, dict) else ""
        if not content:
            content = '"No title."\nNo snippet available.'
        contexts.append({"document": {"contents": content}})
    return contexts
