import asyncio
import logging
import math
import time
from argparse import Namespace
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import aiohttp
import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]

TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

OPD_SCORING_TELEMETRY_KEY = "opd_scoring_telemetry"


@dataclass(frozen=True)
class _PostJsonResult:
    response: dict[str, Any]
    request_body_bytes: int | None
    response_body_bytes: int
    body_read_s: float
    json_decode_s: float


@dataclass(frozen=True)
class _ScoringPostResult:
    response: dict[str, Any]
    telemetry: dict[str, Any]


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


def _score_payload(
    input_ids: list[int],
    response_length: int,
    top_k: int = 0,
    token_ids: list[int] | None = None,
) -> dict[str, Any]:
    if response_length < 0 or response_length > len(input_ids):
        raise ValueError(
            f"OPD scoring response_length must be between 0 and len(input_ids): "
            f"response_length={response_length}, len(input_ids)={len(input_ids)}."
        )

    prompt_length = len(input_ids) - response_length
    if prompt_length <= 0:
        raise ValueError(
            "OPD scoring requires at least one prompt token before the response: "
            f"response_length={response_length}, len(input_ids)={len(input_ids)}."
        )

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Keep the complete prefix in input_ids, but only materialize logprobs
        # from one token before the response so every response token is scored.
        "logprob_start_len": prompt_length - 1,
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k
    if token_ids:
        payload["token_ids_logprob"] = token_ids
    return payload


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _get_scoring_timeout(args: Namespace) -> float:
    return float(getattr(args, "opd_scoring_timeout", 600.0) or 600.0)


def _get_scoring_max_inflight(args: Namespace) -> int:
    value = getattr(args, "opd_scoring_max_inflight", 8)
    return int(8 if value is None else value)  # <= 0 disables the cap


def _get_scoring_retries(args: Namespace) -> int:
    value = getattr(args, "opd_scoring_retries", 1)
    return max(0, int(1 if value is None else value))


# One per process: reward_func runs on the RolloutManager's single event loop.
_SCORING_SEMAPHORE: asyncio.Semaphore | None = None


def _scoring_semaphore(args: Namespace) -> asyncio.Semaphore | None:
    """Return the shared in-flight semaphore, or None when the cap is disabled."""
    global _SCORING_SEMAPHORE
    max_inflight = _get_scoring_max_inflight(args)
    if max_inflight <= 0:
        return None
    if _SCORING_SEMAPHORE is None:
        _SCORING_SEMAPHORE = asyncio.Semaphore(max_inflight)
    return _SCORING_SEMAPHORE


async def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> _PostJsonResult:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            body_read_start = time.monotonic()
            body = await resp.read()
            body_read_s = time.monotonic() - body_read_start

            json_decode_start = time.monotonic()
            response = await resp.json()
            json_decode_s = time.monotonic() - json_decode_start

            request_body_bytes_header = resp.request_info.headers.get("Content-Length")
            request_body_bytes = int(request_body_bytes_header) if request_body_bytes_header is not None else None
            return _PostJsonResult(
                response=response,
                request_body_bytes=request_body_bytes,
                response_body_bytes=len(body),
                body_read_s=body_read_s,
                json_decode_s=json_decode_s,
            )


def _payload_input_token_count(payload: dict[str, Any]) -> int:
    input_ids = payload.get("input_ids") or []
    if input_ids and isinstance(input_ids[0], list):
        return sum(len(ids) for ids in input_ids)
    return len(input_ids)


def _returned_position_count(response: dict[str, Any]) -> int:
    meta_info = response.get("meta_info") or {}
    fields = ("input_token_logprobs", "input_top_logprobs", "input_token_ids_logprobs")
    return max((len(meta_info.get(field) or []) for field in fields), default=0)


async def _scoring_post(
    args: Namespace,
    url: str,
    payload: dict[str, Any],
    *,
    target: str,
    response_length: int,
) -> _ScoringPostResult:
    """One scoring RPC: bounded in-flight, explicit timeout, retries, contextual errors.

    A whole rollout batch finishes together, so without the semaphore every
    sample's request dogpiles the teacher at once and queue time burns each
    request's timeout; without an explicit timeout aiohttp's implicit 300s
    total applies and surfaces as a bare TimeoutError. Retries default to 1;
    --opd-scoring-retries 0 fails fast on the first error.
    """
    timeout_s = _get_scoring_timeout(args)
    semaphore = _scoring_semaphore(args)
    attempts = _get_scoring_retries(args) + 1
    n_tokens = _payload_input_token_count(payload)
    n_ids = len(payload.get("token_ids_logprob") or [])
    telemetry: dict[str, Any] = {
        "target": target,
        "attempts": 0,
        "input_tokens": n_tokens,
        "response_tokens": response_length,
        "requested_token_ids": n_ids,
        "top_k": int(payload.get("top_logprobs_num", 0) or 0),
        "semaphore_wait_s": 0.0,
        "http_s": 0.0,
    }
    overall_start = time.monotonic()
    last_exc: Exception | None = None

    async def post_once() -> _PostJsonResult:
        if semaphore is None:
            http_start = time.monotonic()
            try:
                return await _post_json(url, payload, timeout_s)
            finally:
                telemetry["http_s"] += time.monotonic() - http_start

        wait_start = time.monotonic()
        async with semaphore:
            telemetry["semaphore_wait_s"] += time.monotonic() - wait_start
            http_start = time.monotonic()
            try:
                return await _post_json(url, payload, timeout_s)
            finally:
                telemetry["http_s"] += time.monotonic() - http_start

    for attempt in range(1, attempts + 1):
        start = time.monotonic()
        telemetry["attempts"] = attempt
        try:
            result = await post_once()
            telemetry.update(
                {
                    "e2e_latency_s": time.monotonic() - overall_start,
                    "request_body_bytes": result.request_body_bytes,
                    "response_body_bytes": result.response_body_bytes,
                    "body_read_s": result.body_read_s,
                    "json_decode_s": result.json_decode_s,
                    "returned_positions": _returned_position_count(result.response),
                }
            )
            return _ScoringPostResult(response=result.response, telemetry=telemetry)
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError) as exc:
            last_exc = exc
            logger.warning(
                "OPD scoring POST %s attempt %d/%d failed after %.0fs "
                "(timeout %.0fs, %d input tokens, %d requested token ids): %r",
                url,
                attempt,
                attempts,
                time.monotonic() - start,
                timeout_s,
                n_tokens,
                n_ids,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(2.0)
    raise RuntimeError(
        f"OPD scoring request to {url} failed {attempts} time(s) (timeout {timeout_s:.0f}s/attempt, "
        f"{n_tokens} input tokens, {n_ids} requested token ids); last error: {last_exc!r}"
    ) from last_exc


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


def _validate_scoring_token_alignment(response: dict[str, Any], sample: Sample, *, source: str) -> list[Any]:
    response_length = sample.response_length
    if response_length == 0:
        return []

    if len(sample.tokens) < response_length:
        raise ValueError(
            f"OPD {source} scoring cannot align {response_length} response tokens against "
            f"only {len(sample.tokens)} total sample tokens."
        )

    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError(f"OPD {source} scoring response is missing a valid meta_info object.")

    entries = _trim_input_field(meta_info, "input_token_logprobs", response_length)
    if len(entries) != response_length:
        raise ValueError(
            f"OPD {source} scoring token count mismatch: got {len(entries)} response logprob entries, "
            f"expected {response_length} (sample index={sample.index}, group_index={sample.group_index})."
        )

    returned_token_ids = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or entry[1] is None:
            raise ValueError(
                f"OPD {source} scoring returned a malformed input_token_logprobs entry at response "
                f"position {position}: {entry!r}."
            )
        returned_token_ids.append(int(entry[1]))

    expected_token_ids = sample.tokens[-response_length:]
    if returned_token_ids != expected_token_ids:
        mismatch_position = next(
            i
            for i, (returned, expected) in enumerate(zip(returned_token_ids, expected_token_ids, strict=True))
            if returned != expected
        )
        raise ValueError(
            f"OPD {source} scoring token alignment mismatch at response position {mismatch_position}: "
            f"got token id {returned_token_ids[mismatch_position]}, expected "
            f"{expected_token_ids[mismatch_position]} (sample index={sample.index}, "
            f"group_index={sample.group_index}, response_length={response_length})."
        )

    return entries


def _input_logprob_maps(response: dict[str, Any], field: str, response_length: int) -> LogprobMaps:
    return [
        _top_entries_to_map(entries) for entries in _trim_input_field(response["meta_info"], field, response_length)
    ]


def _teacher_sampled_log_probs(response: dict[str, Any], sample: Sample) -> torch.Tensor:
    input_token_logprobs = _validate_scoring_token_alignment(response, sample, source="teacher")
    return torch.tensor([item[0] for item in input_token_logprobs], dtype=torch.float32)


def _student_top_logprobs(sample: Sample, response_length: int) -> TopLogprobs:
    top_logprobs = sample.metadata.get("opd_student_top_logprobs")
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


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    top_k = _get_opd_top_k(args)
    if top_k == 0:
        result = await _scoring_post(
            args,
            args.rm_url,
            _score_payload(sample.tokens, response_length=sample.response_length),
            target="teacher",
            response_length=sample.response_length,
        )
        _record_scoring_telemetry(sample, result.telemetry)
        return result.response

    strategy = _get_top_k_strategy(args)

    teacher_token_ids = None
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)

    teacher_payload = _score_payload(
        sample.tokens,
        response_length=sample.response_length,
        top_k=top_k if strategy in TEACHER_TOP_STRATEGIES else 0,
        token_ids=teacher_token_ids,
    )
    teacher_result = await _scoring_post(
        args,
        args.rm_url,
        teacher_payload,
        target="teacher",
        response_length=sample.response_length,
    )
    _record_scoring_telemetry(sample, teacher_result.telemetry)
    teacher_response = teacher_result.response

    reward_payload = {"teacher": teacher_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        teacher_top = _trim_input_field(teacher_response["meta_info"], "input_top_logprobs", sample.response_length)
        student_token_ids = _unique_ids(teacher_top)
        student_result = await _scoring_post(
            args,
            _student_score_url(args),
            _score_payload(
                sample.tokens,
                response_length=sample.response_length,
                token_ids=student_token_ids,
            ),
            target="student",
            response_length=sample.response_length,
        )
        _record_scoring_telemetry(sample, student_result.telemetry)
        reward_payload["student_on_teacher"] = student_result.response

    return reward_payload


def _record_scoring_telemetry(sample: Sample, telemetry: dict[str, Any]) -> None:
    metadata = dict(sample.metadata or {})
    entries = metadata.get(OPD_SCORING_TELEMETRY_KEY, [])
    if not isinstance(entries, list):
        raise ValueError(f"sample.metadata[{OPD_SCORING_TELEMETRY_KEY!r}] must be a list.")
    metadata[OPD_SCORING_TELEMETRY_KEY] = [*entries, telemetry]
    sample.metadata = metadata


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    """Extract OPD signals from teacher responses.

    ``--opd-log-prob-top-k=0`` preserves the original sampled-token OPD path:
    store teacher log-probs and let training compute ``student_logp - teacher_logp``.

    ``--opd-log-prob-top-k>0`` follows the practical recipe from
    "Rethinking On-Policy Distillation" by forming a top-k token set per
    response position and storing a precomputed weighted reverse-KL estimate.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    if _get_opd_top_k(args) > 0:
        strategy = _get_top_k_strategy(args)
        for sample, reward in zip(samples, raw_rewards, strict=True):
            _validate_scoring_token_alignment(reward["teacher"], sample, source="teacher")
            if strategy in STUDENT_ON_TEACHER_STRATEGIES:
                _validate_scoring_token_alignment(
                    reward["student_on_teacher"],
                    sample,
                    source="student",
                )
            sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, reward)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    teacher_log_probs = [
        _teacher_sampled_log_probs(reward, sample) for reward, sample in zip(raw_rewards, samples, strict=True)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
