"""SGLang external-teacher scoring for On-Policy Distillation (OPD).

A separate SGLang server hosts the teacher. We POST the student's rollout
token sequence for prefill-only *scoring* (``max_new_tokens=0,
return_logprob=True, temperature=0`` -- no generation) to ``args.opd_teacher_url``
and extract the teacher's per-response-token log-probs from the response,
storing them on ``sample.teacher_log_probs``.

With ``--opd-log-prob-top-k > 0`` (port of miles [2/N] af28a061d, following the
practical recipe from "Rethinking On-Policy Distillation"), scoring instead
forms a top-k token set per response position (strategy: only-student /
only-teacher / intersection / union / xor over the student's and teacher's
top-k), cross-scores both models on the selected tokens, and stores a
precomputed weighted reverse-KL estimate on ``sample.opd_reverse_kl`` that the
trainer consumes directly.

Wired via orbit's existing custom-reward hooks::

    --custom-rm-path orbit.rollout.opd_sglang.reward_func
    --custom-reward-post-process-path orbit.rollout.opd_sglang.post_process

Design note -- this differs from slime's ``slime/rollout/on_policy_distillation.py``:
slime's ``reward_func`` stores the raw sglang response dict directly on
``sample.reward``, and ``post_process_rewards`` reads it back via
``sample.get_reward_value(args)``. That does not carry over unmodified to
orbit: orbit computes zero-std-reward metrics from ``sample.reward``
(``orbit/ray/rollout.py::_compute_zero_std_metrics``, called from
``_log_rollout_data``) *before* ``_convert_samples_to_train_data``/
``post_process`` ever runs, and those metrics call
``round(sample.get_reward_value(args), 1)`` -- which raises on a dict. Orbit's
own ``--custom-rm-path`` docs also state the contract explicitly: "The
function should have the signature `def custom_rm(args, sample) -> float`"
(``orbit/utils/arguments.py``). So ``reward_func`` here keeps ``sample.reward``
numeric (``0.0`` -- pure distillation has no task reward) and stashes the raw
teacher response in ``sample.metadata`` instead, for ``post_process`` to read
back and discard.
"""

import logging
import math
from argparse import Namespace
from collections.abc import Iterable
from typing import Any

import aiohttp
import torch

from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

TEACHER_RESPONSE_METADATA_KEY = "opd_teacher_response"
STUDENT_TOP_LOGPROBS_METADATA_KEY = "opd_student_top_logprobs"

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}


def _get_opd_top_k(args: Namespace) -> int:
    return max(0, int(getattr(args, "opd_log_prob_top_k", 0) or 0))


def _get_top_k_strategy(args: Namespace) -> str:
    strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if strategy not in TOP_K_STRATEGIES:
        raise ValueError(f"Unknown OPD top-k strategy: {strategy}")
    return strategy


def _get_reward_weight_mode(args: Namespace) -> str:
    mode = getattr(args, "opd_reward_weight_mode", "student_p")
    if mode not in REWARD_WEIGHT_MODES:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")
    return mode


def _score_payload(input_ids: list[int], top_k: int = 0, token_ids: list[int] | None = None) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


async def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _top_entry_token_id(entry: list[Any]) -> int:
    return int(entry[1])


def _top_entry_logprob(entry: list[Any]) -> float:
    return float(entry[0])


def _top_entries_to_map(entries: Iterable[list[Any]] | None) -> dict[int, float]:
    if not entries:
        return {}
    return {_top_entry_token_id(entry): _top_entry_logprob(entry) for entry in entries if entry is not None}


def _trim_input_field(meta_info: dict[str, Any], field: str, response_length: int) -> list[Any]:
    values = meta_info.get(field)
    if values is None:
        raise ValueError(f"Teacher response is missing meta_info.{field}.")
    # SGLang's first input logprob/top-logprob position is a placeholder.
    return values[1:][-response_length:] if response_length > 0 else []


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _student_top_logprobs(sample: Sample, response_length: int) -> TopLogprobs:
    top_logprobs = sample.metadata.get(STUDENT_TOP_LOGPROBS_METADATA_KEY) if sample.metadata else None
    if top_logprobs is None:
        raise ValueError(
            "Top-k OPD requires student output_top_logprobs. "
            "Ensure --opd-log-prob-top-k is set before rollout generation starts."
        )
    top_logprobs = top_logprobs[-response_length:] if response_length > 0 else []
    if len(top_logprobs) != response_length:
        raise ValueError(
            f"Student top-k logprob length mismatch: got {len(top_logprobs)}, expected {response_length}."
        )
    return top_logprobs


def _unique_ids(top_logprobs: Iterable[Iterable[list[Any]]]) -> list[int]:
    ids = set()
    for entries in top_logprobs:
        for entry in entries or []:
            if entry is not None:
                ids.add(_top_entry_token_id(entry))
    return sorted(ids)


def _ordered_unique(ids: Iterable[int]) -> list[int]:
    seen = set()
    ordered = []
    for token_id in ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        ordered.append(token_id)
    return ordered


def _selected_token_ids(strategy: str, student_ids: list[int], teacher_ids: list[int]) -> list[int]:
    student_set = set(student_ids)
    teacher_set = set(teacher_ids)
    if strategy == "only-student":
        return student_ids
    if strategy == "only-teacher":
        return teacher_ids
    if strategy == "intersection":
        return [token_id for token_id in student_ids if token_id in teacher_set]
    if strategy == "union":
        return _ordered_unique([*student_ids, *teacher_ids])
    if strategy == "xor":
        return [
            token_id
            for token_id in [*student_ids, *teacher_ids]
            if (token_id in student_set) != (token_id in teacher_set)
        ]
    raise ValueError(f"Unknown OPD top-k strategy: {strategy}")


def _lookup_logprob(
    token_id: int,
    primary: dict[int, float],
    fallback: dict[int, float] | None,
    *,
    source: str,
) -> float:
    if token_id in primary:
        return primary[token_id]
    if fallback is not None and token_id in fallback:
        return fallback[token_id]
    raise ValueError(f"Missing {source} logprob for token id {token_id}.")


def _reward_weights(
    student_logps: list[float],
    teacher_logps: list[float],
    mode: str,
    *,
    normalize: bool,
) -> list[float]:
    if not student_logps:
        return []
    if mode == "student_p":
        logps = student_logps
    elif mode == "teacher_p":
        logps = teacher_logps
    elif mode == "none":
        logps = [0.0] * len(student_logps)
    else:
        raise ValueError(f"Unknown OPD reward weight mode: {mode}")

    if not normalize:
        return [math.exp(logp) for logp in logps]

    max_logp = max(logps)
    exp_vals = [math.exp(logp - max_logp) for logp in logps]
    denom = sum(exp_vals)
    if denom == 0.0:
        return [0.0] * len(logps)
    return [v / denom for v in exp_vals]


def _compute_topk_reverse_kl(
    args: Namespace,
    sample: Sample,
    reward_payload: dict[str, Any],
) -> torch.Tensor:
    response_length = sample.response_length
    if response_length == 0:
        return torch.zeros((0,), dtype=torch.float32)

    strategy = _get_top_k_strategy(args)
    weight_mode = _get_reward_weight_mode(args)

    student_top_maps = (
        [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_response = reward_payload["teacher"]
    teacher_top_maps = (
        _input_logprob_maps(teacher_response, "input_top_logprobs", response_length)
        if strategy in TEACHER_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    teacher_on_student_maps = (
        _input_logprob_maps(teacher_response, "input_token_ids_logprobs", response_length)
        if strategy in TEACHER_ON_STUDENT_STRATEGIES
        else [{} for _ in range(response_length)]
    )
    student_on_teacher_maps = (
        _input_logprob_maps(reward_payload["student_on_teacher"], "input_token_ids_logprobs", response_length)
        if strategy in STUDENT_ON_TEACHER_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    reverse_kls = []
    normalize_weights = strategy != "xor"
    for i in range(response_length):
        student_ids = list(student_top_maps[i].keys())
        teacher_ids = list(teacher_top_maps[i].keys())
        selected_ids = _selected_token_ids(strategy, student_ids, teacher_ids)

        student_logps = []
        teacher_logps = []
        for token_id in selected_ids:
            student_logps.append(
                _lookup_logprob(
                    token_id,
                    student_top_maps[i],
                    student_on_teacher_maps[i],
                    source="student",
                )
            )
            teacher_logps.append(
                _lookup_logprob(
                    token_id,
                    teacher_top_maps[i],
                    teacher_on_student_maps[i],
                    source="teacher",
                )
            )

        weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
        reverse_kl = sum(
            w * (s_logp - t_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
        )
        reverse_kls.append(reverse_kl)

    return torch.tensor(reverse_kls, dtype=torch.float32)


def _extract_teacher_log_probs(response: dict, response_length: int) -> list[float]:
    """Pure extraction/trim logic (no I/O) -- the unit-testable core.

    ``response`` is the JSON body of an sglang prefill-only scoring call
    (``max_new_tokens=0, return_logprob=True``): ``meta_info.input_token_logprobs``
    is a list of ``[logprob, token_id, ...]`` entries, one per input token
    (prompt followed by response). Trim to the last ``response_length``
    entries -- the response span -- and return their logprobs.
    """
    input_token_logprobs = response["meta_info"]["input_token_logprobs"]
    log_probs = [item[0] for item in input_token_logprobs]
    return log_probs[-response_length:]


async def _score_with_teacher(args, sample: Sample) -> dict:
    """POST the sample's full token sequence to the SGLang teacher server for
    prefill-only scoring (sampled-token path). Kept separate from
    ``reward_func`` so tests can monkeypatch it and never hit the network.
    """
    return await _post_json(args.opd_teacher_url, _score_payload(sample.tokens))


async def _score_top_k(args, sample: Sample) -> dict[str, Any]:
    """Orchestrate top-k cross-scoring: teacher scored with its own top-k and/or
    on the student's top-k token ids; optionally the student re-scored on the
    teacher's top-k ids (via the rollout router). Returns the reward payload
    consumed by ``_compute_topk_reverse_kl``.
    """
    top_k = _get_opd_top_k(args)
    strategy = _get_top_k_strategy(args)

    teacher_token_ids = None
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)

    teacher_payload = _score_payload(
        sample.tokens,
        top_k=top_k if strategy in TEACHER_TOP_STRATEGIES else 0,
        token_ids=teacher_token_ids,
    )
    teacher_response = await _post_json(args.opd_teacher_url, teacher_payload)

    reward_payload: dict[str, Any] = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        student_token_ids = _unique_ids(teacher_top)
        reward_payload["student_on_teacher"] = await _post_json(
            _student_score_url(args),
            _score_payload(sample.tokens, token_ids=student_token_ids),
        )
    return reward_payload


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """``--custom-rm-path`` hook.

    Scores ``sample`` against the external SGLang teacher and stashes the raw
    response (sampled-token path) or the top-k cross-scoring payload
    (``--opd-log-prob-top-k > 0``) on ``sample.metadata`` for ``post_process``
    to consume. Always returns ``0.0``: pure on-policy distillation has no
    task reward, and the learning signal comes entirely from the OPD
    (MOPD/blend) advantage term.
    """
    if _get_opd_top_k(args) > 0:
        sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = await _score_top_k(args, sample)
    else:
        sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = await _score_with_teacher(args, sample)
    return 0.0


def post_process(args, samples: list[Sample], **kwargs):
    """``--custom-reward-post-process-path`` hook.

    Sampled-token path (``--opd-log-prob-top-k=0``): extracts the teacher
    response stashed in each sample's metadata, trims it to the response span,
    and sets ``sample.teacher_log_probs`` (training computes
    ``student_logp - teacher_logp``).

    Top-k path (``--opd-log-prob-top-k>0``): computes the weighted top-k
    reverse-KL estimate per response position and stores it on
    ``sample.opd_reverse_kl`` for the trainer to consume directly.

    Returns ``(raw_rewards, rewards)`` -- both all-zero, matching
    ``reward_func``'s task-reward-free contract -- in the shape expected by
    ``RolloutManager._convert_samples_to_train_data``.
    """
    top_k = _get_opd_top_k(args)
    unscored = 0
    for sample in samples:
        payload = sample.metadata.pop(TEACHER_RESPONSE_METADATA_KEY, None)
        if payload is None:
            # e.g. aborted-then-recovered partial rollout whose reward was not
            # produced by reward_func. Keep the OPD fields None (honest
            # "not scored" state) instead of KeyError-ing the whole batch;
            # if such a sample reaches training, _convert_samples_to_train_data
            # rejects the mixed batch with an actionable error.
            unscored += 1
            continue
        if top_k > 0:
            sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, payload).tolist()
        else:
            sample.teacher_log_probs = _extract_teacher_log_probs(payload, sample.response_length)
    if unscored:
        logger.warning("OPD sglang post_process: %d/%d samples had no stashed teacher response.", unscored, len(samples))

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
