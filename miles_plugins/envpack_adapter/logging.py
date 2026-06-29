"""Envpack-specific rollout logging helpers.

The custom rollout/eval hooks are intentionally additive and return ``False``
so Miles' default logging remains unchanged. The train all-samples hook also
returns pre-filter prompt-group diagnostics for Miles to merge into the normal
rollout log because DAPO's kept samples are post-filter by design.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


ALL_SAMPLES_PROCESS_PATH = "miles_plugins.envpack_adapter.logging.process_all_samples"


def log_rollout_data(
    rollout_id,
    args,
    samples: Iterable[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time,
) -> bool:
    """Add train metrics from samples kept for optimization.

    Under DAPO, ``samples`` contains only groups kept after dynamic filtering.
    Therefore ``rollout/solve_rate`` keeps its training-data semantics and can
    be biased upward. ``rollout/pre_filter_solve_rate`` from
    ``process_all_samples`` is the main distribution-level progress signal.
    The maintained recipes log pre-filter prompt-group diagnostics separately
    from ``process_all_samples``.
    """

    if rollout_extra_metrics is not None:
        add_bucket_solve_rate_metrics(
            samples,
            rollout_extra_metrics,
            prefix="envpack_rollout_bucket",
        )
        add_overall_solve_rate_metric(samples, rollout_extra_metrics, key="rollout/solve_rate")
        add_overall_solve_rate_metric(samples, rollout_extra_metrics, key="rollout/kept_solve_rate")
        if not _uses_envpack_all_samples_hook(args):
            group_counts = add_prompt_group_distribution_metrics(
                samples,
                rollout_extra_metrics,
                prefix="envpack_prompt_groups",
            )
            add_rollout_prompt_group_summary_metrics(rollout_extra_metrics, group_counts)
    return False


def process_all_samples(
    args,
    all_samples,
    data_source,
    *,
    is_eval: bool = False,
    live: bool = False,
    eval_dataset_name: str | None = None,
    rollout_id: int | None = None,
    n_samples_per_group: int | None = None,
) -> dict[str, Any] | None:
    """Persist debug dumps and return train metrics from all samples.

    Miles calls ``--custom-rollout-log-function-path`` with the final
    ``RolloutFnTrainOutput.samples``. Under DAPO that is the post-filter kept
    set, so every group is mixed by construction and prompt-group diagnostics
    become meaningless. This hook is called through
    ``--rollout-all-samples-process-path`` before DAPO-filtered groups are
    discarded, so train prompt-group metrics are computed from the full
    oversampled set and merged into Miles' normal rollout log. During long
    DAPO refill loops Miles may call this hook with ``live=True``; that path
    returns metrics only and deliberately skips debug dumping.
    """

    if live:
        return None if is_eval else build_all_sample_rollout_metrics(all_samples)

    from examples.vagen.debug_dump import dump_samples

    dump_samples(
        args,
        all_samples,
        data_source,
        is_eval=is_eval,
        eval_dataset_name=eval_dataset_name,
        rollout_id=rollout_id,
        n_samples_per_group=n_samples_per_group,
    )
    if is_eval:
        return None
    return build_all_sample_rollout_metrics(all_samples)


def log_eval_rollout_data(
    rollout_id,
    args,
    data: dict[str, Any],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Add envpack eval bucket metrics under a separate namespace."""

    if extra_metrics is None:
        return False

    samples = []
    for split_data in data.values():
        split_samples = split_data.get("samples") if isinstance(split_data, dict) else None
        if split_samples:
            samples.extend(split_samples)
    add_bucket_solve_rate_metrics(
        samples,
        extra_metrics,
        prefix="envpack_eval_bucket",
    )
    # Total solve rate over ALL eval samples (solved / total), surfaced in the
    # eval/ panel next to the mean-reward metric so true solving is visible
    # (mean reward includes a shaping/format term that stays nonzero even at a
    # ~0 solve rate, so it can mask whether anything is actually being solved).
    add_overall_solve_rate_metric(samples, extra_metrics, key="eval/solve_rate")
    return False


def build_all_sample_rollout_metrics(samples: Iterable[Any]) -> dict[str, Any]:
    """Return train-rollout envpack metrics computed from pre-filter samples."""

    log_dict: dict[str, Any] = {}
    add_all_sample_rollout_metrics(samples, log_dict)
    return log_dict


def add_all_sample_rollout_metrics(samples: Iterable[Any], log_dict: dict[str, Any]) -> None:
    """Append train-rollout envpack metrics computed from pre-filter samples."""

    flat_samples = _flatten_samples(samples)
    add_bucket_solve_rate_metrics(
        flat_samples,
        log_dict,
        prefix="envpack_rollout_pre_filter_bucket",
    )
    group_counts = add_prompt_group_distribution_metrics(
        flat_samples,
        log_dict,
        prefix="envpack_prompt_groups",
    )
    add_rollout_prompt_group_summary_metrics(log_dict, group_counts)
    add_overall_solve_rate_metric(flat_samples, log_dict, key="rollout/pre_filter_solve_rate")


def add_bucket_solve_rate_metrics(
    samples: Iterable[Any],
    log_dict: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    """Append solve-rate metrics grouped by env name and envpack bucket.

    Returns ``{env_name: overall_rate}`` for callers that need per-env summary
    values. The top-level rollout/eval metrics should remain env-agnostic.
    """

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample in samples:
        meta = _envpack_metadata(sample)
        bucket_name = _bucket_name(meta)
        if not bucket_name:
            continue
        solved = _solved(sample, meta)
        if solved is None:
            continue
        env_name = _metric_component(str(meta.get("env_name") or "unknown_env"))
        bucket = _metric_component(str(bucket_name))
        grouped[(env_name, bucket)].append(int(solved))

    per_env: dict[str, list[int]] = defaultdict(list)
    for (env_name, bucket), values in sorted(grouped.items()):
        if not values:
            continue
        base = f"{prefix}/{env_name}/{bucket}"
        log_dict[f"{base}/solve_rate"] = sum(values) / len(values)
        log_dict[f"{base}/count"] = len(values)
        per_env[env_name].extend(values)

    overall: dict[str, float] = {}
    for env_name, values in sorted(per_env.items()):
        if not values:
            continue
        rate = sum(values) / len(values)
        # leading "_overall" sorts above the per-bucket keys in the dashboard
        log_dict[f"{prefix}/{env_name}/_overall/solve_rate"] = rate
        log_dict[f"{prefix}/{env_name}/_overall/count"] = len(values)
        overall[env_name] = rate
    return overall


def add_overall_solve_rate_metric(samples: Iterable[Any], log_dict: dict[str, Any], *, key: str) -> None:
    """Append a task-agnostic solve rate over samples with explicit success signals."""

    values: list[int] = []
    for sample in _flatten_samples(samples):
        meta = _envpack_metadata(sample)
        solved = _solved(sample, meta)
        if solved is not None:
            values.append(int(solved))
    if values:
        log_dict[key] = sum(values) / len(values)


def add_prompt_group_distribution_metrics(
    samples: Iterable[Any],
    log_dict: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, int]:
    """Append prompt-group outcome distribution metrics.

    A prompt group contains multiple rollouts for the same prompt. If all
    rollouts are unsolved or all are solved, the group has no within-group
    success variance. Mixed groups contain both solved and unsolved samples.
    """

    grouped: dict[tuple[str, str, Any], list[int]] = defaultdict(list)
    for sample in samples:
        group_index = getattr(sample, "group_index", None)
        if group_index is None:
            continue
        meta = _envpack_metadata(sample)
        bucket_name = _bucket_name(meta)
        if not bucket_name:
            continue
        solved = _solved(sample, meta)
        if solved is None:
            continue
        env_name = _metric_component(str(meta.get("env_name") or "unknown_env"))
        bucket = _metric_component(str(bucket_name))
        grouped[(env_name, bucket, group_index)].append(int(solved))

    outcomes_by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    outcomes_by_env: dict[str, list[str]] = defaultdict(list)
    for (env_name, bucket, _group_index), values in grouped.items():
        if not values:
            continue
        outcome = _group_outcome(values)
        outcomes_by_bucket[(env_name, bucket)].append(outcome)
        outcomes_by_env[env_name].append(outcome)

    for (env_name, bucket), outcomes in sorted(outcomes_by_bucket.items()):
        _append_group_outcome_metrics(log_dict, f"{prefix}/{env_name}/{bucket}", outcomes)
    global_counts = {"none_solved": 0, "mixed": 0, "all_solved": 0, "total": 0}
    for env_name, outcomes in sorted(outcomes_by_env.items()):
        _append_group_outcome_metrics(log_dict, f"{prefix}/{env_name}/_overall", outcomes)
        counts = _group_outcome_counts(outcomes)
        for key, value in counts.items():
            global_counts[key] += value
    return global_counts


def add_rollout_prompt_group_summary_metrics(log_dict: dict[str, Any], counts: dict[str, int]) -> None:
    """Write the two primary prompt-group diagnostics into the rollout panel."""

    total = int(counts.get("total", 0))
    if total <= 0:
        return
    all_unsolved = int(counts.get("none_solved", 0))
    all_solved = int(counts.get("all_solved", 0))
    log_dict["rollout/all_unsolved_prompt_frac"] = all_unsolved / total
    log_dict["rollout/all_solved_prompt_frac"] = all_solved / total
    log_dict["rollout/all_unsolved_prompts"] = all_unsolved
    log_dict["rollout/all_solved_prompts"] = all_solved


def _group_outcome(values: list[int]) -> str:
    solved = sum(values)
    if solved <= 0:
        return "none_solved"
    if solved >= len(values):
        return "all_solved"
    return "mixed"


def _append_group_outcome_metrics(log_dict: dict[str, Any], base: str, outcomes: list[str]) -> None:
    counts = _group_outcome_counts(outcomes)
    total = counts["total"]
    if total <= 0:
        return
    none_solved = counts["none_solved"]
    mixed = counts["mixed"]
    all_solved = counts["all_solved"]
    log_dict[f"{base}/groups"] = total
    log_dict[f"{base}/none_solved_groups"] = none_solved
    log_dict[f"{base}/mixed_groups"] = mixed
    log_dict[f"{base}/all_solved_groups"] = all_solved
    log_dict[f"{base}/none_solved_frac"] = none_solved / total
    log_dict[f"{base}/mixed_frac"] = mixed / total
    log_dict[f"{base}/all_solved_frac"] = all_solved / total


def _group_outcome_counts(outcomes: list[str]) -> dict[str, int]:
    return {
        "total": len(outcomes),
        "none_solved": sum(1 for outcome in outcomes if outcome == "none_solved"),
        "mixed": sum(1 for outcome in outcomes if outcome == "mixed"),
        "all_solved": sum(1 for outcome in outcomes if outcome == "all_solved"),
    }


def _envpack_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None) or {}
    envpack = metadata.get("envpack") if isinstance(metadata, dict) else None
    legacy = metadata.get("vagen") if isinstance(metadata, dict) else None

    merged: dict[str, Any] = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(envpack, dict):
        merged.update(envpack)
    if not merged and isinstance(metadata, dict):
        merged.update(metadata)
    return merged


def _uses_envpack_all_samples_hook(args) -> bool:
    return getattr(args, "rollout_all_samples_process_path", None) == ALL_SAMPLES_PROCESS_PATH


def _flatten_samples(samples: Iterable[Any]) -> list[Any]:
    flat: list[Any] = []
    for sample in samples:
        if isinstance(sample, list):
            flat.extend(_flatten_samples(sample))
        else:
            flat.append(sample)
    return flat


def _bucket_name(meta: dict[str, Any]) -> str | None:
    bucket = meta.get("bucket_name")
    if bucket:
        return str(bucket)
    solver_metrics = meta.get("solver_metrics")
    if isinstance(solver_metrics, dict) and solver_metrics.get("bucket_name"):
        return str(solver_metrics["bucket_name"])
    return None


def _solved(sample: Any, meta: dict[str, Any]) -> bool | None:
    for key in ("traj_success", "success", "is_solved", "solved"):
        if key in meta:
            return bool(meta[key])

    reward_report = meta.get("reward_report")
    if isinstance(reward_report, dict):
        for section in ("components", "verifier_outputs", "raw_reward"):
            values = reward_report.get(section)
            if isinstance(values, dict):
                for key in ("success", "is_solved", "solved"):
                    if key in values:
                        return bool(values[key])

    # Deliberately avoid guessing from reward > 0. Bucket solve-rate should only
    # use explicit env success signals, since shaped rewards can be positive
    # without solving the task.
    return None


def _metric_component(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")
