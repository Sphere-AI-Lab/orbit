import argparse
import asyncio
import json
# ORBIT-SEAM: os backs the ORBIT_DEBUG_PEFT_REQUEST diagnostic in do_proxy below
import logging
import os

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)


def run_router(args):
    """
    Run the Miles router with the specified configuration.
    """
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("miles_router")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(args, verbose=False)

    # Start the server
    uvicorn.run(miles_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class MilesRouter:
    def __init__(self, args, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()
        # ORBIT-SEAM: rate-limit counter for the ORBIT_DEBUG_PEFT_REQUEST diagnostic in do_proxy below
        self._debug_peft_request_count = 0

        max_connections = getattr(args, "miles_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = getattr(args, "miles_router_timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        # ORBIT-SEAM: expose remove_worker (see below) as a route, matching add_worker
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        # ORBIT-SEAM: sglang-router-compatible /workers GET (see workers() below) and POST alias to
        # add_worker, for tooling that expects the sglang-router worker-management API shape
        self.app.get("/workers")(self.workers)
        self.app.post("/workers")(self.add_worker)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.args.rollout_health_check_interval
        threshold = self.args.miles_router_health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                # ORBIT-SEAM: only health-check idle workers (active_requests == 0), not every non-dead
                # worker -- see the comment below for why
                # SGLang's HTTP health endpoint can be starved while long
                # generations occupy a worker.  A timeout is therefore only
                # actionable while the worker is idle; in-flight request
                # failures are handled by the rollout engine health monitor.
                urls = [
                    url
                    for url, active_requests in self.worker_request_counts.items()
                    if url not in self.dead_workers and active_requests == 0
                ]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # ORBIT-SEAM: TODO wording normalized (repo-wide comment style pass, no functional change)
                            # Follow-up (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

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

        # ORBIT-SEAM: opt-in rate-limited diagnostic logging of PEFT adapter selection on generate
        # requests passing through the router (LoRA lora_path vs OFT oft_path presence)
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
        # ORBIT-SEAM: return a synthesized 502 instead of letting an upstream connection error escape
        # do_proxy uncaught (the finally below still runs to release the active-request slot)
        except httpx.HTTPError as exc:
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

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "server", "date")
        }
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    # ORBIT-SEAM: new endpoint (registered above) to remove a worker mid-run, clearing its request/
    # failure counts and dead-worker quarantine so a decommissioned engine stops being routed to
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

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    # ORBIT-SEAM: sglang-router-compatible worker listing (registered as GET /workers above), for
    # tooling that expects that response shape instead of list_workers' plain URL list
    async def workers(self, request: Request):
        """SGLang-router compatible worker listing."""
        workers = [
            {"id": str(i), "url": url, "worker_type": "regular"}
            for i, url in enumerate(self.worker_request_counts)
        ]
        return {"workers": workers, "urls": [worker["url"] for worker in workers]}

    def _use_url(self):
        """Select worker URL with minimal active requests."""

        if not self.dead_workers:
            # Healthy path: select from all workers
            url = min(self.worker_request_counts, key=self.worker_request_counts.get)
        else:
            # Degraded path: select from workers not in dead_workers
            valid_workers = (w for w in self.worker_request_counts if w not in self.dead_workers)
            try:
                url = min(valid_workers, key=self.worker_request_counts.get)
            except ValueError:
                raise RuntimeError("No healthy workers available in the pool") from None

        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        assert url in self.worker_request_counts, f"URL {url} not recognized"
        self.worker_request_counts[url] -= 1
        assert self.worker_request_counts[url] >= 0, f"URL {url} count went negative"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
