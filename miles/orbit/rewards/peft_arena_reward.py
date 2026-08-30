"""Bridge Orbit custom reward hooks to local PEFT-Arena release scorers."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from functools import lru_cache


logger = logging.getLogger(__name__)

SUPPORTED_SOURCES = {"openr1", "med_23k_think"}
SUPPORTED_MATH_EVAL_DATASETS = {"math500", "math250", "aime24", "amc23"}
_GENERIC_MATH_DATASET = "math500"
_DEFAULT_REWARD_TIMEOUT_S = 10.0
_REWARD_TIMEOUT_ENV = "ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S"
DATA_SOURCE_ALIASES = {
    "medthink-23k": "med_23k_think",
}


def _compute_math_rlvr_score(solution_str, ground_truth):
    from miles.orbit.rewards.math_alignment import extract_answer, math_equal

    answer = extract_answer(solution_str or "", _GENERIC_MATH_DATASET)
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    correct = math_equal(answer, ground_truth)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": answer,
    }


# Compatibility wrapper for tests and older imports. This used to load
# PEFT-Arena's math_rlvr.py from an external checkout; it is now self-contained.
@lru_cache(maxsize=1)
def _math_rlvr_compute_score():
    return _compute_math_rlvr_score


def _normalize_reward(result) -> dict[str, object]:
    if isinstance(result, dict):
        normalized = dict(result)
        normalized["score"] = float(normalized["score"])
        return normalized
    score = float(result)
    return {"score": score, "acc": bool(score)}


def _score_math_eval_sample(sample, metadata: dict) -> dict[str, object] | None:
    dataset_name = str(metadata.get("dataset_name") or "").strip()
    if dataset_name not in SUPPORTED_MATH_EVAL_DATASETS:
        return None

    from miles.orbit.rewards.math_alignment import extract_answer, grade_math_alignment

    response = sample.response or ""
    correct = grade_math_alignment(response, sample.label, metadata)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "pred": extract_answer(response, dataset_name),
    }


def _score_sample(sample) -> dict[str, object]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    math_eval_reward = _score_math_eval_sample(sample, metadata)
    if math_eval_reward is not None:
        return math_eval_reward

    data_source = metadata.get("data_source") or "openr1"
    data_source = DATA_SOURCE_ALIASES.get(data_source, data_source)
    if data_source not in SUPPORTED_SOURCES:
        raise NotImplementedError(
            f"Unsupported PEFT-Arena data_source={data_source!r}. Supported: {sorted(SUPPORTED_SOURCES)}"
        )
    compute_score = _math_rlvr_compute_score()
    response = sample.response or ""
    return _normalize_reward(compute_score(solution_str=response, ground_truth=sample.label))


def _failed_reward() -> dict[str, object]:
    return {"score": 0.0, "acc": False, "pred": ""}


def _reward_timeout_s(args) -> float | None:
    value = getattr(args, "peft_arena_reward_timeout_s", None)
    if value is None:
        value = os.getenv(_REWARD_TIMEOUT_ENV, str(_DEFAULT_REWARD_TIMEOUT_S))
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_REWARD_TIMEOUT_S
    if timeout <= 0:
        return None
    return timeout


async def _score_sample_async(args, sample) -> dict[str, object]:
    timeout = _reward_timeout_s(args)
    try:
        score_task = asyncio.to_thread(_score_sample, sample)
        if timeout is None:
            return await score_task
        return await asyncio.wait_for(score_task, timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("PEFT-Arena reward timed out after %.2fs; assigning zero reward", timeout)
        return _failed_reward()


async def peft_arena_reward(args, samples, **kwargs):
    del kwargs
    if isinstance(samples, Iterable) and not hasattr(samples, "response"):
        return await asyncio.gather(*(_score_sample_async(args, sample) for sample in samples))
    return await _score_sample_async(args, samples)
