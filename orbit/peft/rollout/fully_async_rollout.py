from __future__ import annotations

import asyncio
import atexit
import logging
import os
import queue
import threading
import time
from argparse import Namespace
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx

from orbit.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from orbit.utils.async_utils import run
from orbit.utils.misc import load_function
from orbit.utils.types import Sample

if TYPE_CHECKING:
    from orbit.rollout.base_types import RolloutFnTrainOutput
    from orbit.rollout.data_source import DataSource

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def group_oldest_weight_version(group: list[Sample]) -> int | None:
    """Return the minimum rollout-engine weight version used by a sample group."""
    versions = [sample.oldest_weight_version for sample in group if sample.oldest_weight_version is not None]
    return min(versions) if versions else None


def group_weight_staleness(group: list[Sample], current_weight_version: int | None) -> int | None:
    if current_weight_version is None:
        return None
    oldest = group_oldest_weight_version(group)
    if oldest is None:
        return None
    return max(0, current_weight_version - oldest)


class _CachedWeightVersion:
    """Throttled query for the current SGLang router weight version."""

    def __init__(
        self,
        ttl: float = 1.0,
        timeout: float = 2.0,
        fetch_json: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    ):
        self._ttl = ttl
        self._timeout = timeout
        self._fetch_json = fetch_json or self._http_get_json
        self._value: int | None = None
        self._last_query: float = 0.0

    async def get(self, args: Namespace) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value

        self._last_query = now
        try:
            value = await self._query_weight_version(args)
            if value is not None:
                self._value = value
        except Exception as exc:
            logger.debug("Failed to query rollout engine weight version: %r", exc)
        return self._value

    async def _http_get_json(self, url: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def _query_weight_version(self, args: Namespace) -> int | None:
        router_base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        value = await self._query_weight_version_from_url(f"{router_base_url}/model_info")
        if value is not None:
            return value

        worker_versions = []
        for worker_url in await self._query_worker_urls(router_base_url):
            value = await self._query_weight_version_from_url(f"{worker_url.rstrip('/')}/model_info")
            if value is not None:
                worker_versions.append(value)
        return max(worker_versions) if worker_versions else None

    async def _query_weight_version_from_url(self, url: str) -> int | None:
        try:
            return _extract_adapter_version(await self._fetch_json(url))
        except Exception as exc:
            logger.debug("Failed to query rollout engine adapter version from %s: %r", url, exc)
            return None

    async def _query_worker_urls(self, router_base_url: str) -> list[str]:
        for endpoint in ("workers", "list_workers"):
            url = f"{router_base_url}/{endpoint}"
            try:
                data = await self._fetch_json(url)
            except Exception as exc:
                logger.debug("Failed to query rollout workers from %s: %r", url, exc)
                continue

            urls = []
            for worker in data.get("workers", []) or []:
                if isinstance(worker, dict) and worker.get("url"):
                    urls.append(worker["url"])
                elif isinstance(worker, str):
                    urls.append(worker)
            urls.extend(data.get("urls", []) or [])
            if urls:
                return urls
        return []


def _version_value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None and isinstance(data.get("model_info"), dict):
        value = data["model_info"].get(key)
    return value


def _parse_version(value: Any) -> int | None:
    if value is None:
        return None
    return int(value) if str(value).isdigit() else None


def _extract_adapter_version(data: dict[str, Any]) -> int | None:
    adapter_version = _parse_version(_version_value(data, "adapter_version"))
    weight_version = _parse_version(_version_value(data, "weight_version"))

    if adapter_version is not None and weight_version is not None and adapter_version != weight_version:
        raise ValueError(
            f"adapter_version and weight_version disagree in /model_info: "
            f"adapter_version={adapter_version!r} weight_version={weight_version!r}"
        )
    return adapter_version if adapter_version is not None else weight_version


_cached_version = _CachedWeightVersion(
    ttl=_env_float("ORBIT_FULLY_ASYNC_WEIGHT_VERSION_CACHE_TTL_S", 1.0),
    timeout=_env_float("ORBIT_FULLY_ASYNC_WEIGHT_VERSION_TIMEOUT_S", 2.0),
)


class AsyncRolloutWorker:
    """Persistent producer that keeps rollout requests in flight between train steps."""

    def __init__(
        self,
        args: Namespace,
        data_source: DataSource,
        *,
        max_active_groups: int | None = None,
        output_queue_size: int | None = None,
        poll_interval_s: float | None = None,
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.max_active_groups = max(
            1,
            max_active_groups
            or _env_int("ORBIT_FULLY_ASYNC_MAX_ACTIVE_GROUPS", args.rollout_batch_size),
        )
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(
            maxsize=max(1, output_queue_size or _env_int("ORBIT_FULLY_ASYNC_OUTPUT_QUEUE_SIZE", 1000))
        )
        self.poll_interval_s = poll_interval_s or _env_float("ORBIT_FULLY_ASYNC_WORKER_POLL_S", 0.01)
        self.worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._data_source_lock = threading.Lock()
        self._next_group_id = 0

    def start(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        self._stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_thread_func,
            name="orbit-fully-async-rollout",
            daemon=True,
        )
        self.worker_thread.start()
        logger.info(
            "Started fully async rollout worker: max_active_groups=%d output_queue_size=%d",
            self.max_active_groups,
            self.output_queue.maxsize,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def recycle_groups(self, groups: list[list[Sample]]) -> None:
        if not groups:
            return
        with self._data_source_lock:
            self.data_source.add_samples(groups)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()

    def _worker_thread_func(self) -> None:
        asyncio.run(self._continuous_worker_loop())

    async def _continuous_worker_loop(self) -> None:
        from orbit.rollout.sglang_rollout import GenerateState, generate_and_rm_group

        state = GenerateState(self.args)
        active_tasks: dict[asyncio.Task, tuple[int, list[Sample]]] = {}
        logger.info("Fully async rollout worker loop started")

        while not self._stop_event.is_set():
            self._collect_finished_tasks(active_tasks)

            while len(active_tasks) < self.max_active_groups and not self._stop_event.is_set():
                group = self._get_next_group()
                if group is None:
                    break

                group_id = self._next_group_id
                self._next_group_id += 1
                task = asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=state.sampling_params.copy(),
                        evaluation=False,
                    )
                )
                active_tasks[task] = (group_id, group)

            await asyncio.sleep(self.poll_interval_s)

        if active_tasks:
            logger.info("Stopping fully async rollout worker with %d active groups", len(active_tasks))
            done, pending = await asyncio.wait(active_tasks.keys(), timeout=5)
            for task in done:
                self._finish_task(task, active_tasks)
            for task in pending:
                task.cancel()
                _group_id, original_group = active_tasks.pop(task)
                _reset_groups_for_retry([original_group])
                self.recycle_groups([original_group])

        logger.info("Fully async rollout worker loop stopped")

    def _collect_finished_tasks(self, active_tasks: dict[asyncio.Task, tuple[int, list[Sample]]]) -> None:
        done_tasks = [task for task in active_tasks if task.done()]
        for task in done_tasks:
            self._finish_task(task, active_tasks)

    def _finish_task(self, task: asyncio.Task, active_tasks: dict[asyncio.Task, tuple[int, list[Sample]]]) -> None:
        group_id, original_group = active_tasks.pop(task)
        try:
            group = task.result()
        except Exception:
            logger.exception("Fully async rollout group %d failed; recycling the original prompt group", group_id)
            _reset_groups_for_retry([original_group])
            self.recycle_groups([original_group])
            return
        self.output_queue.put((group_id, group))

    def _get_next_group(self) -> list[Sample] | None:
        with self._data_source_lock:
            groups = self.data_source.get_samples(1)
        return groups[0] if groups else None


_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def get_global_worker(args: Namespace, data_source: DataSource) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if (
            _global_worker is None
            or _global_worker.worker_thread is None
            or not _global_worker.worker_thread.is_alive()
            or _global_worker.data_source is not data_source
        ):
            if _global_worker is not None:
                _global_worker.stop()
            _global_worker = AsyncRolloutWorker(args, data_source)
            _global_worker.start()
        return _global_worker


def stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


def _reset_groups_for_retry(groups: list[list[Sample]]) -> None:
    for group in groups:
        for sample in group:
            sample.reset_for_retry()


def _recycle_groups(worker: Any, data_source: DataSource, groups: list[list[Sample]]) -> None:
    _reset_groups_for_retry(groups)
    if hasattr(worker, "recycle_groups"):
        worker.recycle_groups(groups)
    else:
        data_source.add_samples(groups)


def _any_aborted(group: list[Sample]) -> bool:
    return any(sample.status == Sample.Status.ABORTED for sample in group)


def _sort_group_key(group: list[Sample]) -> int:
    first = group[0]
    return -1 if first.index is None else first.index


def _build_fully_async_metrics(
    *,
    staleness_values: list[int],
    current_weight_version: int | None,
    accepted_groups: int,
    processed_completed_groups: int,
    recycled_stale_groups: int,
    recycled_aborted_groups: int,
    output_queue_size: int,
    elapsed_s: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "fully_async/accepted_groups": accepted_groups,
        "fully_async/processed_completed_groups": processed_completed_groups,
        "fully_async/recycled_stale_groups": recycled_stale_groups,
        "fully_async/recycled_aborted_groups": recycled_aborted_groups,
        "fully_async/output_queue_size": output_queue_size,
        "fully_async/collection_time": elapsed_s,
        "fully_async/staleness/count": len(staleness_values),
    }
    if current_weight_version is not None:
        metrics["fully_async/current_weight_version"] = current_weight_version
    if staleness_values:
        metrics["fully_async/staleness/mean"] = sum(staleness_values) / len(staleness_values)
        metrics["fully_async/staleness/max"] = max(staleness_values)
        metrics["fully_async/staleness/min"] = min(staleness_values)
    return metrics


async def generate_rollout_async(args: Namespace, rollout_id: int, data_source: DataSource) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_source)
    target_data_size = args.rollout_batch_size
    max_weight_staleness = getattr(args, "max_weight_staleness", None)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    data: list[list[Sample]] = []
    completed_groups: dict[int, list[Sample]] = {}
    staleness_values: list[int] = []
    recycled_stale_groups = 0
    recycled_aborted_groups = 0
    processed_completed_groups = 0
    current_weight_version: int | None = None

    start_time = time.monotonic()
    last_progress_time = start_time
    no_progress_timeout_s = _env_float("ORBIT_FULLY_ASYNC_NO_PROGRESS_TIMEOUT_S", 30.0)
    logged_first_sample = False

    logger.info(
        "Collecting fully async rollout_id=%d target_groups=%d max_weight_staleness=%s queue_size=%d",
        rollout_id,
        target_data_size,
        max_weight_staleness,
        worker.get_queue_size(),
    )

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        if completed:
            last_progress_time = time.monotonic()
            for group_id, group in completed:
                completed_groups[group_id] = group

        if completed_groups:
            queried_weight_version = await _cached_version.get(args)
            if queried_weight_version is not None:
                current_weight_version = queried_weight_version

        processed_in_iteration = False
        for group_id in sorted(list(completed_groups.keys())):
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)
            processed_completed_groups += 1
            processed_in_iteration = True

            if _any_aborted(group):
                _recycle_groups(worker, data_source, [group])
                recycled_aborted_groups += 1
                continue

            staleness = group_weight_staleness(group, current_weight_version)
            if staleness is not None:
                staleness_values.append(staleness)
                if max_weight_staleness is not None and staleness > max_weight_staleness:
                    _recycle_groups(worker, data_source, [group])
                    recycled_stale_groups += 1
                    logger.info(
                        "Recycled stale rollout group: rollout_id=%d group_id=%d staleness=%d max=%d current=%s oldest=%s",
                        rollout_id,
                        group_id,
                        staleness,
                        max_weight_staleness,
                        current_weight_version,
                        group_oldest_weight_version(group),
                    )
                    continue

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(dynamic_filter_output.reason)
                continue

            if not logged_first_sample and group:
                sample = group[0]
                logger.info(
                    "First fully async rollout sample: %s label=%s reward=%s",
                    [str(sample.prompt) + sample.response],
                    str(sample.label)[:100],
                    sample.reward,
                )
                logged_first_sample = True

            data.append(group)

        if not processed_in_iteration:
            await asyncio.sleep(0.01)

        now = time.monotonic()
        if now - last_progress_time > no_progress_timeout_s:
            logger.warning(
                "No fully async rollout progress for %.1fs: rollout_id=%d collected=%d/%d queue_size=%d",
                no_progress_timeout_s,
                rollout_id,
                len(data),
                target_data_size,
                worker.get_queue_size(),
            )
            last_progress_time = now

    data = sorted(data, key=_sort_group_key)

    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, data, data_source)

    if data:
        sample = data[-1][0]
        logger.info(
            "Finished fully async rollout_id=%d last_sample=%s label=%s reward=%s",
            rollout_id,
            [str(sample.prompt) + sample.response],
            str(sample.label)[:100],
            sample.reward,
        )

    metrics = metric_gatherer.collect()
    metrics.update(
        _build_fully_async_metrics(
            staleness_values=staleness_values,
            current_weight_version=current_weight_version,
            accepted_groups=len(data),
            processed_completed_groups=processed_completed_groups,
            recycled_stale_groups=recycled_stale_groups,
            recycled_aborted_groups=recycled_aborted_groups,
            output_queue_size=worker.get_queue_size(),
            elapsed_s=time.monotonic() - start_time,
        )
    )

    from orbit.rollout.base_types import RolloutFnTrainOutput

    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def generate_rollout_fully_async(
    args: Namespace,
    rollout_id: int,
    data_source: DataSource,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    if evaluation:
        raise ValueError("Evaluation mode is not supported by fully async rollout")
    return run(generate_rollout_async(args, rollout_id, data_source))


atexit.register(stop_global_worker)
