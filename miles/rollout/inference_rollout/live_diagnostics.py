from __future__ import annotations

import logging
from typing import Any

from miles.rollout.inference_rollout.hook_utils import call_all_samples_process_fn
from miles.utils.misc import load_function
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


def initial_live_log_at(args, target_groups: int) -> int | None:
    """Return the first completed candidate-group count that should be logged.

    By default live diagnostics are enabled only for dynamic-sampling rollouts.
    Static GRPO rollouts already complete one fixed batch before logging, so the
    mid-rollout signal is mainly useful when DAPO/refill can stall before a
    normal rollout log is emitted.
    """

    interval = _live_log_interval(args, target_groups)
    return interval if interval > 0 else None


def maybe_log_all_samples_live_diagnostics(
    args,
    rollout_id: int,
    all_samples: list[list[Any]],
    data_source: Any,
    *,
    kept_groups: int,
    target_groups: int,
    pending_groups: int,
    next_log_at: int | None,
    extra_metrics: dict[str, Any] | None = None,
) -> int | None:
    """Emit pre-filter diagnostics while a dynamic rollout is still refilling.

    The normal all-samples hook runs only after enough prompt groups survived
    filtering.  During cold start that may never happen, so this live path calls
    the same hook with ``live=True`` and logs returned metrics without touching
    training samples or dumping artifacts.
    """

    completed_groups = len(all_samples)
    if next_log_at is None or completed_groups < next_log_at:
        return next_log_at

    process_path = getattr(args, "rollout_all_samples_process_path", None)
    if not process_path:
        return None

    process_func = load_function(process_path)
    if process_func is None:
        return None

    try:
        maybe_metrics = call_all_samples_process_fn(
            process_func,
            args,
            all_samples,
            data_source,
            is_eval=False,
            live=True,
            rollout_id=rollout_id,
            n_samples_per_group=getattr(args, "n_samples_per_prompt", None),
        )
    except Exception:
        logger.exception("Failed to compute live all-samples rollout diagnostics")
        return _advance(next_log_at, args, target_groups, completed_groups)

    if isinstance(maybe_metrics, dict) and maybe_metrics:
        metrics = dict(maybe_metrics)
        if extra_metrics:
            metrics.update(extra_metrics)
        metrics["rollout/step"] = _compute_rollout_step(args, rollout_id)
        metrics["rollout/refill/completed_prompt_groups"] = completed_groups
        metrics["rollout/refill/kept_prompt_groups"] = kept_groups
        metrics["rollout/refill/target_prompt_groups"] = target_groups
        metrics["rollout/refill/pending_prompt_groups"] = pending_groups
        metrics["rollout/refill/keep_prompt_group_frac"] = kept_groups / completed_groups
        tracking.log(args, metrics, step_key="rollout/step")
        logger.info(
            "live rollout diagnostics %s: completed=%s kept=%s target=%s pending=%s metrics=%s",
            rollout_id,
            completed_groups,
            kept_groups,
            target_groups,
            pending_groups,
            metrics,
        )

    return _advance(next_log_at, args, target_groups, completed_groups)


def _advance(next_log_at: int, args, target_groups: int, completed_groups: int) -> int | None:
    interval = _live_log_interval(args, target_groups)
    if interval <= 0:
        return None
    while next_log_at <= completed_groups:
        next_log_at += interval
    return next_log_at


def _live_log_interval(args, target_groups: int) -> int:
    configured = getattr(args, "rollout_all_samples_live_log_interval", None)
    if configured is None:
        if not getattr(args, "dynamic_sampling_filter_path", None):
            return 0
        return max(1, target_groups)
    return max(0, int(configured))


def _compute_rollout_step(args, rollout_id: int) -> int:
    if getattr(args, "wandb_always_use_train_step", False):
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
