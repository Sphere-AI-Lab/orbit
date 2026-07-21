import asyncio
import atexit
import logging
import queue
import threading
import time

import aiohttp

from miles.rollout.data_source import DataSource
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def group_oldest_weight_version(group: list[Sample]) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [s.oldest_weight_version for s in group if s.oldest_weight_version is not None]
    return min(versions) if versions else None


class _CachedWeightVersion:
    """Throttled query for the current engine weight version via /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query: float = 0.0

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._value = int(data["weight_version"])
                        self._last_query = now
        except Exception as e:
            logger.debug(f"Failed to query engine weight version: {e}")
        return self._value


_cached_version = _CachedWeightVersion()


# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer: DataSource):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


async def _close_opd_scoring_transport(args):
    if not getattr(args, "use_opd", False):
        return

    from miles.rollout.on_policy_distillation import close_scoring_transport

    await close_scoring_transport()


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer: DataSource, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        self.prefetch_batches = max(1, int(getattr(args, "fully_async_prefetch_batches", 1)))
        self.max_concurrent_tasks = args.rollout_batch_size * self.prefetch_batches
        self.max_completed_queue_groups = max(1, int(getattr(args, "fully_async_max_completed_queue_groups", 2048)))
        # Continuous output queue for completed-but-not-yet-consumed groups.
        # Keep this unbounded so task callbacks never block the worker event loop.
        # max_completed_queue_groups is enforced as a soft launch cap below.
        self.output_queue = queue.Queue()
        self.worker_thread = None
        self._loop = None
        self._active_tasks = set()
        self.state = GenerateState(args)

        self._warn_if_prefetch_is_likely_wasteful()

    def _warn_if_prefetch_is_likely_wasteful(self):
        max_staleness = getattr(self.args, "max_weight_staleness", None)
        if max_staleness is not None and self.prefetch_batches > max_staleness + 1:
            logger.warning(
                "fully_async_prefetch_batches=%s is larger than max_weight_staleness+1=%s. "
                "Later prefetched groups are likely to become stale and be recycled before training.",
                self.prefetch_batches,
                max_staleness + 1,
            )

        request_window = self.max_concurrent_tasks * self.args.n_samples_per_prompt
        request_limit = (
            self.args.sglang_server_concurrency * self.args.rollout_num_gpus // self.args.rollout_num_gpus_per_engine
        )
        if request_window > request_limit:
            logger.warning(
                "Fully async prefetch window can launch up to %s sample requests "
                "(%s groups * %s samples/group), above the SGLang client semaphore limit %s. "
                "The semaphore will backpressure requests, but extra group tasks can increase memory and stale-work pressure.",
                request_window,
                self.max_concurrent_tasks,
                self.args.n_samples_per_prompt,
                request_limit,
            )

        if self.max_completed_queue_groups < self.max_concurrent_tasks:
            logger.warning(
                "fully_async_max_completed_queue_groups=%s is smaller than max active prompt groups=%s. "
                "The soft queue cap can trigger quickly if the trainer falls behind; consider increasing it.",
                self.max_completed_queue_groups,
                self.max_concurrent_tasks,
            )

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        active_tasks = set()
        self._active_tasks = active_tasks
        max_concurrent_tasks = self.max_concurrent_tasks
        group_id_counter = 0
        print(
            "Fully async worker prefetch: "
            f"rollout_batch_size={self.args.rollout_batch_size}, "
            f"prefetch_batches={self.prefetch_batches}, "
            f"max_active_groups={max_concurrent_tasks}, "
            f"max_completed_queue_groups={self.max_completed_queue_groups}",
            flush=True,
        )

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                # Keep sampler-side generation saturated. Completed queued groups
                # do not count against this active task window; the queue maxsize
                # is only a safety cap for stalled trainers.
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    if self.output_queue.qsize() >= self.max_completed_queue_groups:
                        break
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                try:
                                    result = done_task.result()
                                except asyncio.CancelledError:
                                    return
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to stop...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def _cancel_active_tasks(self):
        for task in tuple(self._active_tasks):
            task.cancel()

    def worker_thread_func(self):
        """Worker function running in independent thread"""

        async def run_worker():
            self._loop = asyncio.get_running_loop()
            try:
                await self.continuous_worker_loop()
            finally:
                try:
                    await _close_opd_scoring_transport(self.args)
                finally:
                    self._active_tasks = set()
                    self._loop = None

        asyncio.run(run_worker())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._cancel_active_tasks)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            if self.worker_thread.is_alive():
                logger.warning("Fully async rollout worker did not stop within 5 seconds")
        print("Stopped async worker thread")

    def get_completed_groups(self, max_items: int | None = None) -> list[tuple]:
        """Get completed sample groups without draining more than the caller can consume."""
        completed = []
        while max_items is None or len(completed) < max_items:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()

    def get_max_concurrent_tasks(self) -> int:
        """Get the configured in-flight prompt-group window."""
        return self.max_concurrent_tasks

    def get_max_completed_queue_groups(self) -> int:
        """Get the completed-group queue safety cap."""
        return self.max_completed_queue_groups


async def generate_rollout_async(args, rollout_id: int, data_buffer: DataSource) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True
    stale_groups_recycled = 0
    staleness_values = []

    use_staleness_filter = getattr(args, "max_weight_staleness", None) is not None

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(
        f"Global worker queue size: {worker.get_queue_size()}, "
        f"max active groups: {worker.get_max_concurrent_tasks()}, "
        f"max completed queue groups: {worker.get_max_completed_queue_groups()}"
    )
    if use_staleness_filter:
        print(f"Staleness filter enabled: max_weight_staleness={args.max_weight_staleness}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        # Do not drain more completed groups than this rollout call can consume.
        # Stale or aborted groups are reset and returned to the data buffer below;
        # if that leaves the batch short, the next loop drains more completed groups.
        # This avoids over-accepting valid groups and needing an accepted-overflow buffer.
        completed = worker.get_completed_groups(max_items=target_data_size - len(data))

        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Query current engine version once per collection batch (cached/throttled)
        current_engine_version = None
        if use_staleness_filter:
            current_engine_version = await _cached_version.get(args)

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    for s in group:
                        s.reset_for_retry()
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                continue

            # Staleness filter: discard groups whose oldest weight version is too far behind
            oldest = group_oldest_weight_version(group)
            if oldest is not None and current_engine_version is not None:
                staleness = current_engine_version - oldest
                staleness_values.append(staleness)
                if staleness > args.max_weight_staleness:
                    try:
                        for s in group:
                            s.reset_for_retry()
                        data_buffer.add_samples([group])
                    except Exception as e:
                        logger.warning(f"Failed to recycle stale group {group_id}: {e}")
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group {group_id} "
                        f"(oldest_version={oldest}, current={current_engine_version}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                    # don't count as processed for training
                    continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
    if stale_groups_recycled > 0 or staleness_values:
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
        print(
            f"Staleness stats: recycled={stale_groups_recycled}, "
            f"avg_staleness={avg_staleness:.1f}, "
            f"max_staleness={max(staleness_values) if staleness_values else 0}"
        )

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer: DataSource, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# RolloutManager calls this hook before closing the default background loop.
generate_rollout_fully_async.dispose = stop_global_worker

# Register exit cleanup function
atexit.register(stop_global_worker)
