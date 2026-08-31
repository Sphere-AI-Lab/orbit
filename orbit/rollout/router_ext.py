"""Orbit's added / rewritten ``MilesRouter`` methods.

Home mixin for the miles-router endpoints and proxy behaviour lifted out of
miles/router/router.py. ``MilesRouter`` in the miles file lists
``OrbitRouterExtensions`` as its first (and only) base:

    class MilesRouter(OrbitRouterExtensions):

so every method here runs with ``self`` bound to a live router and reaches the
vendored state/methods (``self.client``, ``self.worker_request_counts``,
``self.worker_failure_counts``, ``self.dead_workers``, ``self._use_url``,
``self._finish_url``) the normal attribute-lookup way.

What lives here and why:

* ``remove_worker`` / ``workers`` -- orbit-ADDED endpoints. ``remove_worker``
  retires an engine mid-run (base can only ADD workers, so a decommissioned
  engine kept receiving traffic forever); ``workers`` answers in the
  sglang-router response shape for tooling that expects it. Both are registered
  as routes by the vendored ``_setup_routes``, which reaches them through
  ``self``.
* ``do_proxy`` -- a FULL-REPLACE override. Orbit's two additions (the opt-in
  PEFT-request diagnostic and the upstream-error -> 502 translation) are, in
  lines, about the size of upstream's own body, so this is carried whole rather
  than stamped into the vendored file. The method MUST be deleted from
  ``MilesRouter``'s body for this to run at all -- a class's own ``__dict__``
  beats every base, so a retained upstream body would silently shadow this one
  and the 502 translation would just stop happening.

Why the 502 matters: without it an ``httpx`` connection/timeout error escapes
``do_proxy`` into FastAPI, which answers a bare 500 with no worker attribution,
and the caller cannot tell "the engine is gone" from "the engine rejected the
request". The ``finally`` still releases the active-request slot either way --
that is upstream's, and is preserved verbatim.

``_debug_peft_request_count`` is a CLASS attribute here, not something the
vendored ``__init__`` sets. ``self._debug_peft_request_count += 1`` reads the
class default and writes an instance attribute on first use, which is
behaviourally identical to initialising it in ``__init__`` and leaves the
vendored ``__init__`` byte-pristine.

Plain mixin: no ``__init__``, no ``super()`` call (from here ``super()`` is
``object``; there is nothing upstream left to delegate to once the vendored body
is deleted).
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class OrbitRouterExtensions:
    # Rate limit for the ORBIT_DEBUG_PEFT_REQUEST diagnostic below. Class-level
    # default; the first `+= 1` shadows it with an instance attribute.
    _debug_peft_request_count: int = 0

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        # Opt-in, rate-limited: which PEFT adapter (if any) a generate request
        # selected. Off unless ORBIT_DEBUG_PEFT_REQUEST is set, because it
        # decodes the request body.
        if os.environ.get("ORBIT_DEBUG_PEFT_REQUEST") and path == "generate":
            limit = int(os.environ.get("ORBIT_DEBUG_PEFT_REQUEST_LIMIT", "16"))
            if self._debug_peft_request_count < limit:
                body_text = body.decode("utf-8", errors="replace") if body else ""
                logger.info(
                    "[miles-router] generate payload has_lora_path=%s has_oft_path=%s body_bytes=%d content_type=%s",
                    '"lora_path"' in body_text,
                    '"oft_path"' in body_text,
                    len(body or b""),
                    headers.get("content-type") or headers.get("Content-Type"),
                )
                self._debug_peft_request_count += 1

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        except httpx.HTTPError as exc:
            # Answer 502 rather than letting the connection error escape into
            # FastAPI as an unattributed 500. The `finally` below still runs, so
            # the worker's active-request slot is released either way.
            logger.warning(
                "[miles-router] Upstream request failed path=%s worker_url=%s error=%s",
                path,
                worker_url,
                repr(exc),
            )
            return {
                "request_body": body,
                "response_body": json.dumps(
                    {"error": f"upstream request failed: {type(exc).__name__}"}
                ).encode(),
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        finally:
            self._finish_url(worker_url)

    async def remove_worker(self, request: Request):
        """Remove a worker from the router."""
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self.worker_request_counts.pop(worker_url, None)
        self.worker_failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def workers(self, request: Request):
        """SGLang-router compatible worker listing."""
        workers = [
            {"id": str(i), "url": url, "worker_type": "regular"}
            for i, url in enumerate(self.worker_request_counts)
        ]
        return {"workers": workers, "urls": [worker["url"] for worker in workers]}


__all__ = ["OrbitRouterExtensions"]
