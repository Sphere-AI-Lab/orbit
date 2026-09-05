"""Dynamic-sampling filters for envpack rollouts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from orbit.rollout.filter_hub.base_types import DynamicFilterOutput

__all__ = ["check_envpack_success_nonzero_std"]


def check_envpack_success_nonzero_std(args, samples: list[Any], **kwargs) -> DynamicFilterOutput:
    """Keep only groups whose envpack solve rate is strictly between 0 and 1.

    This is the Sokoban/envpack analogue of DAPO dynamic sampling. It filters
    out prompts where every rollout solves the task and prompts where no rollout
    solves the task. The decision uses explicit envpack success metadata instead
    of shaped reward values, because non-solved Sokoban trajectories can still
    earn positive format/effective-action reward.
    """

    success = [_envpack_success(sample) for sample in _flatten_samples(samples)]
    if not success:
        raise ValueError("envpack dynamic filter received an empty sample group")

    solved = sum(success)
    total = len(success)
    keep = 0 < solved < total
    if keep:
        return DynamicFilterOutput(keep=True)
    reason = "all_solved" if solved == total else "none_solved"
    return DynamicFilterOutput(keep=False, reason=reason)


def _flatten_samples(samples: Iterable[Any]) -> Iterable[Any]:
    for sample in samples:
        if isinstance(sample, list):
            yield from _flatten_samples(sample)
        else:
            yield sample


def _envpack_success(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    envpack = metadata.get("envpack") if isinstance(metadata, dict) else None
    if isinstance(envpack, dict):
        for key in ("success", "traj_success", "is_solved", "solved"):
            if key in envpack:
                return bool(envpack[key])

        reward_report = envpack.get("reward_report")
        if isinstance(reward_report, dict):
            for section in ("components", "verifier_outputs", "raw_reward"):
                values = reward_report.get(section)
                if isinstance(values, dict):
                    for key in ("success", "is_solved", "solved"):
                        if key in values:
                            return bool(values[key])

    # Backward compatibility for older traces produced before envpack stored a
    # top-level success flag. New scripts should not depend on this path.
    legacy = metadata.get("vagen") if isinstance(metadata, dict) else None
    if isinstance(legacy, dict):
        for key in ("traj_success", "success", "is_solved", "solved"):
            if key in legacy:
                return bool(legacy[key])

    raise ValueError(
        "envpack dynamic filter requires explicit success metadata under "
        "sample.metadata['envpack']; shaped reward is not a valid fallback"
    )
