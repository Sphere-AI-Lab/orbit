import asyncio
import logging
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_DEFAULT_PENDING_EVAL_LOG_LIMIT = 20


@dataclass
class _EvalTaskProgress:
    stage: str
    sample: Sample
    started_at: float
    updated_at: float


def _sample_status(sample: Sample) -> str:
    status = sample.status
    return status.value if isinstance(status, Sample.Status) else str(status)


def _update_eval_task_progress(
    progress_by_index: MutableMapping[int, _EvalTaskProgress],
    stage: str,
    sample: Sample,
    *,
    now: float | None = None,
) -> None:
    if sample.index is None:
        return

    timestamp = time.monotonic() if now is None else now
    previous = progress_by_index.get(sample.index)
    started_at = previous.started_at if previous is not None else timestamp
    progress_by_index[sample.index] = _EvalTaskProgress(
        stage=stage,
        sample=sample,
        started_at=started_at,
        updated_at=timestamp,
    )


def _format_pending_eval_tasks(
    pending_indices: Sequence[int],
    progress_by_index: Mapping[int, _EvalTaskProgress],
    *,
    now: float | None = None,
    max_items: int = _DEFAULT_PENDING_EVAL_LOG_LIMIT,
) -> str:
    timestamp = time.monotonic() if now is None else now
    formatted = []
    visible_indices = list(pending_indices[:max_items])
    for index in visible_indices:
        progress = progress_by_index.get(index)
        if progress is None:
            formatted.append(f"{index}(stage=not-started,status=unknown,resp_len=?,idle=?,elapsed=?)")
            continue

        idle_s = max(0.0, timestamp - progress.updated_at)
        elapsed_s = max(0.0, timestamp - progress.started_at)
        formatted.append(
            f"{index}(stage={progress.stage},"
            f"status={_sample_status(progress.sample)},"
            f"resp_len={progress.sample.response_length},"
            f"idle={idle_s:.1f}s,"
            f"elapsed={elapsed_s:.1f}s)"
        )
    remaining = len(pending_indices) - len(visible_indices)
    if remaining > 0:
        formatted.append(f"... +{remaining} more")
    return ", ".join(formatted)


async def _log_pending_eval_tasks(
    *,
    dataset_name: str,
    tasks_by_index: Mapping[int, asyncio.Task[Any]],
    progress_by_index: Mapping[int, _EvalTaskProgress],
    interval_s: float = 60.0,
    now_fn: Callable[[], float] = time.monotonic,
    max_items: int = _DEFAULT_PENDING_EVAL_LOG_LIMIT,
) -> None:
    while True:
        pending_tasks = [task for task in tasks_by_index.values() if not task.done()]
        if not pending_tasks:
            return

        await asyncio.wait(pending_tasks, timeout=interval_s)
        pending_indices = [index for index, task in tasks_by_index.items() if not task.done()]
        if not pending_indices:
            return

        logger.warning(
            "Eval %s pending tasks: %d/%d unfinished: %s",
            dataset_name,
            len(pending_indices),
            len(tasks_by_index),
            _format_pending_eval_tasks(
                pending_indices=pending_indices,
                progress_by_index=progress_by_index,
                now=now_fn(),
                max_items=max_items,
            ),
        )
