"""Lightweight rollout timing instrumentation.

Home for the phase-stats subsystem lifted out of miles/rollout/sglang_rollout.py:
per-phase (train/eval) request timing/throughput counters, the active-phase
switch consulted by each generated request, and a background poller that logs
sglang scheduler stats (running batch, queue, KV usage) during a phase.

All counters are process-local; the eval/rollout coroutines run on a single
rank so this is safe.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from argparse import Namespace


logger = logging.getLogger(__name__)


class _PhaseStats:
    __slots__ = (
        "name",
        "tokenize_s",
        "http_post_s",
        "server_e2e_s",
        "n_completed",
        "completion_tokens",
        "prompt_tokens",
        "cached_tokens",
        "n_started",
        "first_completion_t",
        "started_t",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.tokenize_s = 0.0
        self.http_post_s = 0.0
        self.server_e2e_s = 0.0
        self.n_completed = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.n_started = 0
        self.first_completion_t: float | None = None
        self.started_t: float | None = None

    def reset(self) -> None:
        self.tokenize_s = 0.0
        self.http_post_s = 0.0
        self.server_e2e_s = 0.0
        self.n_completed = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.n_started = 0
        self.first_completion_t = None
        self.started_t = None


_EVAL_PHASE = _PhaseStats("eval")
_TRAIN_PHASE = _PhaseStats("train")
_ACTIVE_PHASE: _PhaseStats | None = None


def _set_active_phase(phase: _PhaseStats | None) -> None:
    global _ACTIVE_PHASE
    _ACTIVE_PHASE = phase
    if phase is not None:
        phase.reset()
        phase.started_t = time.perf_counter()


def _phase_record_request(
    tokenize_s: float,
    http_post_s: float,
    meta_info: dict | None,
) -> None:
    phase = _ACTIVE_PHASE
    if phase is None:
        return
    phase.n_completed += 1
    phase.tokenize_s += tokenize_s
    phase.http_post_s += http_post_s
    if phase.first_completion_t is None:
        phase.first_completion_t = time.perf_counter()
    if meta_info is None:
        return
    phase.completion_tokens += int(meta_info.get("completion_tokens", 0) or 0)
    phase.prompt_tokens += int(meta_info.get("prompt_tokens", 0) or 0)
    phase.cached_tokens += int(meta_info.get("cached_tokens", 0) or 0)
    e2e = meta_info.get("e2e_latency")
    if e2e is None:
        e2e = meta_info.get("finish_time")
    if isinstance(e2e, (int, float)):
        phase.server_e2e_s += float(e2e)


def _phase_log_progress(phase: _PhaseStats, total: int, prefix: str) -> None:
    # [eval-prof] per-checkpoint progress line disabled — the SUMMARY at end of phase
    # carries the same numbers, and the tqdm bar already shows live progress.
    # Re-enable by removing the early return below.
    return
    if phase.n_completed == 0 or phase.started_t is None:
        return
    elapsed = time.perf_counter() - phase.started_t
    if phase.first_completion_t is not None:
        ttft = phase.first_completion_t - phase.started_t
    else:
        ttft = float("nan")
    rate = phase.n_completed / elapsed if elapsed > 0 else 0.0
    tps = phase.completion_tokens / elapsed if elapsed > 0 else 0.0
    cache_hit = (
        phase.cached_tokens / phase.prompt_tokens if phase.prompt_tokens > 0 else 0.0
    )
    avg_tokenize_ms = 1000 * phase.tokenize_s / max(1, phase.n_completed)
    avg_http_ms = 1000 * phase.http_post_s / max(1, phase.n_completed)
    avg_server_ms = 1000 * phase.server_e2e_s / max(1, phase.n_completed)
    logger.info(
        "%s progress: completed=%d/%d (%.1f%%) elapsed=%.1fs ttft=%.1fs "
        "throughput=%.1f req/s decode=%.0f tok/s avg_tokenize=%.1fms "
        "avg_http=%.0fms avg_server_e2e=%.0fms cache_hit=%.2f%% "
        "completion_tokens=%d prompt_tokens=%d",
        prefix,
        phase.n_completed,
        total,
        100.0 * phase.n_completed / max(1, total),
        elapsed,
        ttft,
        rate,
        tps,
        avg_tokenize_ms,
        avg_http_ms,
        avg_server_ms,
        100.0 * cache_hit,
        phase.completion_tokens,
        phase.prompt_tokens,
    )


def _phase_log_summary(phase: _PhaseStats, total: int, prefix: str) -> None:
    if phase.started_t is None:
        return
    elapsed = time.perf_counter() - phase.started_t
    rate = phase.n_completed / elapsed if elapsed > 0 else 0.0
    tps = phase.completion_tokens / elapsed if elapsed > 0 else 0.0
    cache_hit = (
        phase.cached_tokens / phase.prompt_tokens if phase.prompt_tokens > 0 else 0.0
    )
    ttft = (
        (phase.first_completion_t - phase.started_t)
        if phase.first_completion_t is not None
        else float("nan")
    )
    logger.info(
        "%s SUMMARY: total=%d completed=%d wall=%.1fs ttft=%.1fs "
        "throughput=%.2f req/s decode=%.1f tok/s tokenize_total=%.2fs "
        "http_total=%.2fs server_e2e_total=%.2fs (sum across reqs) "
        "cache_hit=%.2f%% completion_tokens=%d prompt_tokens=%d",
        prefix,
        total,
        phase.n_completed,
        elapsed,
        ttft,
        rate,
        tps,
        phase.tokenize_s,
        phase.http_post_s,
        phase.server_e2e_s,
        100.0 * cache_hit,
        phase.completion_tokens,
        phase.prompt_tokens,
    )


_PROGRESS_LOG_EVERY = int(os.environ.get("ORBIT_ROLLOUT_PROGRESS_EVERY", "100") or "0")
_SERVER_POLL_INTERVAL = float(
    os.environ.get("ORBIT_ROLLOUT_SERVER_POLL_S", "5") or "0"
)


async def _server_info_poller(args: Namespace, prefix: str, stop_event: asyncio.Event) -> None:
    """Periodically log sglang scheduler stats (running batch, queue, KV usage)."""
    from miles.utils.http_utils import get

    if _SERVER_POLL_INTERVAL <= 0:
        return
    base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    try:
        workers_resp = await get(f"{base}/workers")
        worker_urls = [w["url"] for w in workers_resp.get("workers", [])]
    except Exception:
        try:
            workers_resp = await get(f"{base}/list_workers")
            worker_urls = workers_resp.get("urls", [])
        except Exception as e:
            logger.warning("[server-poll] cannot list workers: %r", e)
            return
    if not worker_urls:
        logger.warning("[server-poll] no worker urls found, disabling polling")
        return

    poll_t0 = time.perf_counter()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_SERVER_POLL_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass
        for url in worker_urls:
            try:
                info = await get(f"{url}/server_info")
            except Exception as e:
                logger.warning("[server-poll] %s failed: %r", url, e)
                continue
            states = info.get("internal_states") or []
            for i, st in enumerate(states):
                gen_tps = st.get("last_gen_throughput") or st.get("gen_throughput")
                max_running = st.get(
                    "effective_max_running_requests_per_dp"
                ) or st.get("max_running_requests")
                memu = st.get("memory_usage") or {}
                logger.info(
                    "[server-poll] %s t=%.0fs worker=%s dp=%d "
                    "last_gen_throughput=%s max_running_per_dp=%s "
                    "kv_token_capacity=%s kvcache_mem=%s graph_mem=%s",
                    prefix,
                    time.perf_counter() - poll_t0,
                    url.split("//")[-1],
                    i,
                    gen_tps,
                    max_running,
                    memu.get("token_capacity"),
                    memu.get("kvcache"),
                    memu.get("graph"),
                )
