"""Rollout-logging hook that writes viewer traces and the compact response log.

Wire it up with::

    --custom-rollout-log-function-path examples.model_response_trace_viewer.hook.log_rollout_data
    --save-model-response-trace-dir <run-dir>/traces
    --model-response-trace-max-samples-per-step <positive-count>

Both writers no-op unless their own flag is set, so the hook is safe to leave
configured on a recipe that has tracing switched off.

The hook returns ``False`` so Miles still emits its default rollout metrics --
this observes rollouts, it does not replace the built-in logging. Note that
Miles supports a single ``--custom-rollout-log-function-path``: a recipe that
also needs its own logger should call both from one wrapper function rather
than setting the flag twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from examples.model_response_trace_viewer.response_log import save_model_response_log
from examples.model_response_trace_viewer.response_trace import save_model_response_trace

from miles.utils.types import Sample


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: Sequence[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Persist accepted-sample traces, then let default logging run."""
    save_model_response_log(args, samples, rollout_id=rollout_id)
    save_model_response_trace(args, samples, rollout_id=rollout_id)
    return False
