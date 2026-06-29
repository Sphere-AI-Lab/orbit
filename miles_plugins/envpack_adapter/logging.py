"""Envpack-specific rollout logging helpers.

The hooks in this module are intentionally additive. They append bucket-level
envpack metrics into Miles' existing logging dictionaries and return ``False``
so the default Miles rollout/eval logging path remains unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def log_rollout_data(
    rollout_id,
    args,
    samples: Iterable[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time,
) -> bool:
    """Add envpack train-rollout bucket metrics under a separate namespace."""

    if rollout_extra_metrics is not None:
        overall = add_bucket_solve_rate_metrics(
            samples,
            rollout_extra_metrics,
            prefix="envpack_rollout_bucket",
        )
        # Total solve rate over ALL train-rollout samples, in the rollout/ panel.
        for env_name, rate in overall.items():
            rollout_extra_metrics[f"rollout/{env_name}_solve_rate"] = rate
    return False


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
    overall = add_bucket_solve_rate_metrics(
        samples,
        extra_metrics,
        prefix="envpack_eval_bucket",
    )
    # Total solve rate over ALL eval samples (solved / total), surfaced in the
    # eval/ panel next to the mean-reward metric so true solving is visible
    # (mean reward includes a shaping/format term that stays nonzero even at a
    # ~0 solve rate, so it can mask whether anything is actually being solved).
    for env_name, rate in overall.items():
        extra_metrics[f"eval/{env_name}_solve_rate"] = rate
    return False


def add_bucket_solve_rate_metrics(
    samples: Iterable[Any],
    log_dict: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    """Append solve-rate metrics grouped by env name and envpack bucket, plus a
    count-weighted OVERALL solve rate per env. Returns {env_name: overall_rate}
    so callers can also surface the total elsewhere (e.g. the eval/ panel)."""

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


def _envpack_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None) or {}
    envpack = metadata.get("envpack") if isinstance(metadata, dict) else None
    vagen = metadata.get("vagen") if isinstance(metadata, dict) else None

    merged: dict[str, Any] = {}
    if isinstance(vagen, dict):
        merged.update(vagen)
    if isinstance(envpack, dict):
        merged.update(envpack)
    if not merged and isinstance(metadata, dict):
        merged.update(metadata)
    return merged


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
