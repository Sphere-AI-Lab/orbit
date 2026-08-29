"""Weight-sync instrumentation: payload byte accounting, engine-pause timing,
and rollout-timeline event markers.

Metric emission rides the existing perf-metrics flow (``Timer`` singleton ->
``log_perf_data_raw`` -> ``tracking_utils.log``):

- durations enter ``Timer().add(name, seconds)`` and surface as
  ``perf/<name>_time`` (e.g. ``perf/update_weights_pause_time``);
- non-time scalars enter ``Timer().perf_scalars`` and surface as
  ``perf/<name>`` (e.g. ``perf/update_weights_payload_bytes``).

Timeline events (consumed by ``tools/rollout_timeline``) are appended as JSONL
to the file named by the ``ORBIT_TIMELINE_EVENTS_FILE`` env var; when it is
unset the emitters are no-ops.

Nothing in this module may raise into the weight-update path: every entry
point swallows exceptions and logs them loudly instead. The only exception is
``sum_metrics_across_ranks``, which is a collective and therefore must run in
lockstep on every rank (same contract as the barriers in the update path).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from typing import Any

from miles.utils.timer import Timer

logger = logging.getLogger(__name__)

TIMELINE_EVENTS_ENV_VAR = "ORBIT_TIMELINE_EVENTS_FILE"

# Timer key for the engine pause window; surfaces as
# perf/update_weights_pause_time via log_perf_data_raw's "<key>_time" naming.
PAUSE_TIMER_KEY = "update_weights_pause"
# Non-time scalar keys; surface as perf/<key> (no suffix).
PAYLOAD_BYTES_KEY = "update_weights_payload_bytes"
PAYLOAD_NUM_TENSORS_KEY = "update_weights_payload_num_tensors"
NUM_CHUNKS_KEY = "update_weights_num_chunks"

_METRICS_FAILURE_MSG = "weight-sync metric instrumentation failed (metrics only; the sync itself is unaffected)"


def tensor_num_bytes(tensor: Any) -> int:
    """Bytes occupied by one tensor's elements (numel * element_size)."""
    return int(tensor.numel()) * int(tensor.element_size())


def named_tensors_num_bytes(named_tensors: Iterable[Any]) -> int:
    """Sum of numel*element_size over tensors.

    Accepts an iterable of ``(name, tensor)`` pairs or of bare tensors;
    ``None`` entries are skipped.
    """
    total = 0
    for item in named_tensors:
        tensor = item[1] if isinstance(item, (tuple, list)) else item
        if tensor is None:
            continue
        total += tensor_num_bytes(tensor)
    return total


class WeightSyncPayloadTracker:
    """Per-update accumulator of locally shipped payload bytes/tensors.

    The orchestrator (``UpdateWeightFromTensor.update_weights``) resets it at
    update start and snapshots it at update end; the actual send sites
    (colocated IPC flattening, distributed NCCL broadcast, PEFT transports)
    call :meth:`record` with what the local rank actually puts on the wire.
    Anything recorded outside an update window is discarded by the next reset.
    """

    def __init__(self) -> None:
        self.payload_bytes = 0
        self.num_tensors = 0
        # One record per send call (a broadcast bucket / a flat PEFT payload);
        # the distributed broadcast path reports it as perf/update_weights_num_chunks.
        self.num_records = 0

    def reset(self) -> None:
        self.payload_bytes = 0
        self.num_tensors = 0
        self.num_records = 0

    def record(
        self,
        named_tensors: Iterable[Any] | None = None,
        *,
        num_bytes: int | None = None,
        num_tensors: int | None = None,
    ) -> None:
        """Accumulate shipped payload. Never raises into the update path."""
        try:
            if named_tensors is not None:
                tensors = list(named_tensors)
                if num_bytes is None:
                    num_bytes = named_tensors_num_bytes(tensors)
                if num_tensors is None:
                    num_tensors = len(tensors)
            self.payload_bytes += int(num_bytes or 0)
            self.num_tensors += int(num_tensors or 0)
            self.num_records += 1
        except Exception:
            logger.exception(_METRICS_FAILURE_MSG)


_PAYLOAD_TRACKER = WeightSyncPayloadTracker()


def get_payload_tracker() -> WeightSyncPayloadTracker:
    return _PAYLOAD_TRACKER


def record_perf_scalar(name: str, value: float) -> None:
    """Stage a non-time scalar for the next perf flush as ``perf/<name>``.

    Values accumulate (sum) across repeated calls within one flush window,
    mirroring ``Timer.add``. ``log_perf_data_raw`` snapshots and clears the
    staged dict on every flush. Never raises.
    """
    try:
        timer = Timer()
        scalars = getattr(timer, "perf_scalars", None)
        if scalars is None:
            scalars = {}
            timer.perf_scalars = scalars
        scalars[name] = scalars.get(name, 0) + value
    except Exception:
        logger.exception(_METRICS_FAILURE_MSG)


def emit_update_weights_metrics(
    *,
    pause_seconds: float | None,
    payload_bytes: float,
    num_tensors: float,
    num_chunks: float,
) -> None:
    """Stage one weight update's metrics into the existing perf flow.

    Emits:
    - ``perf/update_weights_pause_time`` (seconds; skipped when
      ``pause_seconds`` is None so non-pausing paths report absent),
    - ``perf/update_weights_payload_bytes``,
    - ``perf/update_weights_payload_num_tensors``,
    - ``perf/update_weights_num_chunks``.

    Never raises into the update path.
    """
    try:
        if pause_seconds is not None:
            Timer().add(PAUSE_TIMER_KEY, float(pause_seconds))
        record_perf_scalar(PAYLOAD_BYTES_KEY, int(payload_bytes))
        record_perf_scalar(PAYLOAD_NUM_TENSORS_KEY, int(num_tensors))
        record_perf_scalar(NUM_CHUNKS_KEY, int(num_chunks))
    except Exception:
        logger.exception(_METRICS_FAILURE_MSG)


def sum_metrics_across_ranks(values: list[float], group=None) -> list[float]:
    """SUM a short vector of local metric values across all ranks.

    This is a collective when torch.distributed is initialized with a world
    size > 1: every rank must call it in lockstep (the caller sits right after
    a barrier on the same group, which provides that guarantee). Returns the
    local values unchanged when distributed is not initialized, so CPU tests
    and single-process runs need no mocking.
    """
    import torch
    import torch.distributed as dist

    local = [float(v) for v in values]
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    # float64 keeps byte counts exact up to 2**53 (~8 PiB); CPU tensor works
    # over the gloo group the update path already uses for its barriers.
    tensor = torch.tensor(local, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return [float(v) for v in tensor.tolist()]


def timeline_events_enabled() -> bool:
    return bool(os.environ.get(TIMELINE_EVENTS_ENV_VAR))


def emit_timeline_event(
    event: str,
    *,
    weight_version: int | str | None = None,
    mode: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one JSONL timeline marker to ``ORBIT_TIMELINE_EVENTS_FILE``.

    Record shape: ``{"t_wall": <unix seconds>, "event": <str>,
    "weight_version": ..., "mode": ...}``. No-op when the env var is unset;
    never raises into the update path.
    """
    path = os.environ.get(TIMELINE_EVENTS_ENV_VAR)
    if not path:
        return
    try:
        record: dict[str, Any] = {"t_wall": time.time(), "event": str(event)}
        if weight_version is not None:
            record["weight_version"] = weight_version
        if mode is not None:
            record["mode"] = str(mode)
        if extra:
            record.update(extra)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        logger.exception(_METRICS_FAILURE_MSG)
