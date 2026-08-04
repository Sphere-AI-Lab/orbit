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

import asyncio
import logging
import math
from argparse import Namespace
from collections.abc import Iterable
from typing import Any

import numpy as np
import pybase64
import torch

from orbit.rollout.scoring_client import post_json
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

TEACHER_RESPONSE_METADATA_KEY = "opd_teacher_response"
STUDENT_TOP_LOGPROBS_METADATA_KEY = "opd_student_top_logprobs"

TopLogprobs = list[list[Any]]
LogprobMaps = list[dict[int, float]]
# One teacher endpoint inside a (possibly singleton) ensemble group: (url, mixture weight).
TeacherTarget = tuple[str, float]

# Floor for the teacher's tail probability mass in --opd-topk-tail-bucket mode. Scoring
# logprobs arrive as JSON doubles, so 1 - sum(p) only goes non-positive through genuine
# float64 rounding; the floor (log ~= -27.6) is a last-resort guard, not a working range.
TAIL_PROB_FLOOR = 1e-12

# Padding sentinels for --loss-type opd_topk_loss's retained teacher top-k rows (czy's
# scheme): _TOPK_PAD_LOGPROB is chosen so exp(_TOPK_PAD_LOGPROB) underflows to exactly
# 0.0 in fp32, which the loss uses directly as the pad-slot validity mask.
_TOPK_PAD_TOKEN_ID = 0
_TOPK_PAD_LOGPROB = -1e4


TOP_K_STRATEGIES = {"only-student", "only-teacher", "intersection", "union", "xor"}
REWARD_WEIGHT_MODES = {"student_p", "teacher_p", "none"}
KL_TYPES = {"reverse", "forward", "mixed"}

STUDENT_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-teacher"}
TEACHER_TOP_STRATEGIES = TOP_K_STRATEGIES - {"only-student"}
TEACHER_ON_STUDENT_STRATEGIES = {"only-student", "union", "xor"}
STUDENT_ON_TEACHER_STRATEGIES = {"only-teacher", "union", "xor"}

# Reserved teacher name in --opd-teacher-urls used as the fallback route.
DEFAULT_TEACHER_NAME = "default"

# Element type of the teacher hidden states sglang sends back (full-vocab mode).
_HIDDEN_STATE_DTYPE = np.dtype(np.float32)

_warned_legacy_hidden_states = False


def _warn_legacy_hidden_states_format() -> None:
    """Warn once per process that the teacher is on the slow nested-JSON path.

    Emitted per sample would be one line per request, so this fires a single
    time and then stays quiet.
    """
    global _warned_legacy_hidden_states
    if _warned_legacy_hidden_states:
        return
    _warned_legacy_hidden_states = True
    logger.warning(
        "Teacher returned hidden_states as nested JSON floats rather than a base64 buffer. "
        "This works but is the dominant cost of a full-vocab OPD step: the sglang server "
        "spends minutes per step materializing hundreds of millions of Python floats and "
        "serializing them to multi-GB JSON while its GPU idles."
    )


def teacher_score_mode(args: Namespace) -> str:
    return getattr(args, "teacher_score_mode", "sampled_token") or "sampled_token"


_TEACHER_HIDDEN_SIZE_CACHE: dict[str, int] = {}


def _teacher_hidden_size(checkpoint_path: str) -> int:
    if checkpoint_path not in _TEACHER_HIDDEN_SIZE_CACHE:
        import json as _json
        import os as _os

        with open(_os.path.join(checkpoint_path, "config.json")) as f:
            _TEACHER_HIDDEN_SIZE_CACHE[checkpoint_path] = int(_json.load(f)["hidden_size"])
    return _TEACHER_HIDDEN_SIZE_CACHE[checkpoint_path]


def _full_vocab_response_byte_limit(args: Namespace, num_tokens: int) -> int:
    """Response cap for one full-vocab scoring call, sized to its actual payload.

    The dominant field is base64 fp32 hidden states: num_tokens x hidden_size x 4
    bytes x 4/3 encoding overhead. A 7B teacher (hidden 3584) scoring an eval-length
    sample legitimately exceeds the generic SCORING_MAX_RESPONSE_BYTES, so the cap
    scales with the request instead; the generic cap stays as the floor.
    """
    from orbit.rollout.scoring_client import SCORING_MAX_RESPONSE_BYTES

    hidden = _teacher_hidden_size(args.teacher_hf_checkpoint)
    payload = (num_tokens * hidden * 4 * 4) // 3 + 1024 * 1024
    return max(payload, SCORING_MAX_RESPONSE_BYTES)


# Conservative per-entry byte estimate for one JSON-serialized top-k logprob
# triple, ``[logprob, token_id, token_text]`` -- as returned in the sglang
# response's ``input_top_logprobs`` rows. Measuring a realistic entry (a
# full-precision negative float64, a 6-digit token id, and an 8-character
# token text -- worst case multi-byte/CJK, which json.dumps's default
# ensure_ascii escapes to ~6 bytes/char) gives 45-70 bytes; rounded up for
# headroom and JSON array punctuation.
_TOPK_LOGPROB_ENTRY_BYTES = 64


def _topk_response_byte_limit(args: Namespace, num_tokens: int) -> int:
    """Response cap for one top-k scoring call, sized to its actual payload.

    The dominant field is ``input_top_logprobs``: one entry per input position
    for the always-present observed-token logprob, plus up to
    ``--opd-log-prob-top-k`` entries for the top-k itself, so
    ``num_tokens * (top_k + 1)`` JSON array entries at
    ``_TOPK_LOGPROB_ENTRY_BYTES`` bytes each, doubled for safety margin. A
    large ``--opd-log-prob-top-k`` legitimately exceeds the generic
    SCORING_MAX_RESPONSE_BYTES, so the cap scales with the request instead;
    the generic cap stays as the floor.
    """
    from orbit.rollout.scoring_client import SCORING_MAX_RESPONSE_BYTES

    top_k = _get_opd_top_k(args)
    payload = num_tokens * (top_k + 1) * _TOPK_LOGPROB_ENTRY_BYTES * 2
    return max(payload, SCORING_MAX_RESPONSE_BYTES)


def _parse_teacher_target(part: str, entry: str) -> TeacherTarget:
    """Parse one ``URL[@WEIGHT]`` group member.

    Splits on the last ``@`` so userinfo URLs (``user:pass@host``) survive; a
    suffix that does not parse as a float is treated as part of the URL.
    """
    url, sep, weight_str = part.rpartition("@")
    if sep:
        try:
            weight = float(weight_str)
        except ValueError:
            return part, 1.0
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Teacher weight must be a positive finite number in --opd-teacher-urls entry {entry!r}.")
        if not url.strip():
            raise ValueError(f"Empty teacher URL in --opd-teacher-urls entry {entry!r}.")
        return url.strip(), weight
    return part, 1.0


def parse_teacher_urls(values: Iterable[str] | None) -> dict[str, list[TeacherTarget]]:
    """Parse ``NAME=URL[@WEIGHT][,URL[@WEIGHT]...]`` entries from ``--opd-teacher-urls``.

    Each name maps to a group of one or more teacher endpoints. A group with a
    single URL is plain routing; a group with several URLs is an ensemble —
    every member scores the sample and the targets are combined as a weighted
    mixture in probability space. Weights default to 1.0 (uniform mixture).

    Splits ``NAME=`` on the first ``=`` only, so URLs containing ``=`` (e.g.
    query strings) survive intact; group members are comma-separated. Raises
    on malformed entries, duplicate names, and duplicate URLs within a group
    so misconfiguration fails at startup, not mid-rollout.
    """
    url_map: dict[str, list[TeacherTarget]] = {}
    for value in values or []:
        name, sep, spec = value.partition("=")
        name, spec = name.strip(), spec.strip()
        if not sep or not name or not spec:
            raise ValueError(f"Invalid --opd-teacher-urls entry {value!r}; expected NAME=URL[@WEIGHT][,...].")
        if name in url_map:
            raise ValueError(f"Duplicate teacher name {name!r} in --opd-teacher-urls.")
        targets = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                raise ValueError(f"Empty teacher URL in --opd-teacher-urls entry {value!r}.")
            target = _parse_teacher_target(part, value)
            if not target[0].startswith(("http://", "https://")):
                # Catches comma-split fragments of a single URL (commas separate group
                # members and cannot appear inside member URLs) at startup instead of
                # mid-rollout as an invalid scoring endpoint.
                raise ValueError(
                    f"Teacher URL {target[0]!r} in --opd-teacher-urls entry {value!r} must start with "
                    "http:// or https://. Note: ',' separates ensemble group members and a trailing "
                    "'@<float>' is parsed as the member's mixture weight."
                )
            targets.append(target)
        if len({url for url, _ in targets}) != len(targets):
            raise ValueError(f"Duplicate URL within teacher group {name!r} in --opd-teacher-urls.")
        url_map[name] = targets
    return url_map


def _teacher_targets_for_sample(args: Namespace, sample: Sample) -> list[TeacherTarget]:
    """Resolve the teacher scoring endpoint group for one sample.

    Without ``--opd-teacher-urls`` every sample goes to ``--opd-teacher-url``
    (the original single-teacher path, unchanged). With it, the sample is
    routed by the teacher name in ``sample.metadata[--opd-teacher-key]``;
    samples whose name is missing or unknown fall back to the reserved
    ``default`` entry, and raise if no default is configured — silently
    distilling from the wrong teacher is worse than failing the rollout. The
    resolved group has one member for routing and several for an ensemble.
    """
    url_map = parse_teacher_urls(getattr(args, "opd_teacher_urls", None))
    if not url_map:
        return [(args.opd_teacher_url, 1.0)]

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    key = getattr(args, "opd_teacher_key", "opd_teacher")
    name = metadata.get(key)
    if name is not None:
        targets = url_map.get(str(name))
        if targets is not None:
            return targets
        if DEFAULT_TEACHER_NAME in url_map:
            return url_map[DEFAULT_TEACHER_NAME]
        raise ValueError(
            f"Sample metadata[{key!r}]={name!r} matches no --opd-teacher-urls name "
            f"(known: {sorted(url_map)}) and no 'default' entry is configured."
        )
    if DEFAULT_TEACHER_NAME in url_map:
        return url_map[DEFAULT_TEACHER_NAME]
    raise ValueError(f"Sample metadata is missing teacher key {key!r} and --opd-teacher-urls has no 'default' entry.")


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


def _get_kl_type(args: Namespace) -> tuple[str, float]:
    """Resolve the KL direction and the forward weight for ``mixed``.

    Mirrors NeMo-RL's DistillationLossFn ``kl_type``/``mixed_kl_weight``:
    ``reverse`` (default) keeps the original student-weighted estimate,
    ``forward`` weights by the teacher distribution, ``mixed`` is the convex
    combination with ``--opd-mixed-kl-weight`` on the forward term.
    """
    kl_type = getattr(args, "opd_kl_type", "reverse") or "reverse"
    if kl_type not in KL_TYPES:
        raise ValueError(f"Unknown OPD KL type: {kl_type}")
    mixed_weight = float(getattr(args, "opd_mixed_kl_weight", 0.5))
    if not (0.0 <= mixed_weight <= 1.0):
        raise ValueError(f"--opd-mixed-kl-weight must be in [0, 1], got {mixed_weight}.")
    return kl_type, mixed_weight


def _score_payload(
    input_ids: list[int], top_k: int = 0, token_ids: list[int] | None = None, lora_path: str | None = None
) -> dict[str, Any]:
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
    if lora_path is not None:
        payload["lora_path"] = lora_path
    return payload


def _student_score_url(args: Namespace) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _scoring_timeout(args: Namespace) -> int | float | None:
    """Per-request timeout for teacher/student scoring.

    ``--opd-scoring-timeout-secs`` when set; otherwise falls back to the shared
    router request timeout when available. Kept separate because teachers
    (often much larger than the student) need a different bound than
    generation requests.
    """
    timeout = getattr(args, "opd_scoring_timeout_secs", None)
    if timeout is not None:
        return timeout
    return getattr(args, "sglang_router_request_timeout_secs", None)


async def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_secs: int | float | None = None,
    max_response_bytes: int | None = None,
) -> dict[str, Any]:
    # Thin module-level wrapper around the shared scoring client so tests can
    # monkeypatch opd_sglang._post_json.
    return await post_json(url, payload, timeout_secs=timeout_secs, max_response_bytes=max_response_bytes)


def _mixture_log_probs(per_teacher: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Combine per-teacher logprob tensors into the weighted-mixture logprob.

    log p_bar = log(sum_m w_m * p_m / sum_m w_m), computed stably in log space
    (float64) so the result is the log-probability of the mixture distribution
    — not a mean of logprobs, which would be an unnormalized geometric mean.
    """
    stacked = torch.stack([t.double() for t in per_teacher])
    log_w = torch.tensor(weights, dtype=torch.float64).log()
    log_w = log_w.view(-1, *([1] * (stacked.dim() - 1)))
    return (torch.logsumexp(stacked + log_w, dim=0) - math.log(sum(weights))).float()


def _mixture_logprob_maps(maps_per_teacher: list[LogprobMaps], weights: list[float]) -> LogprobMaps:
    """Mix per-position ``{token_id: logprob}`` maps across teachers.

    All teachers are scored at the same requested token ids per position, so
    every map must contain every id — a missing id means a malformed teacher
    response and raises rather than silently treating the probability as 0.
    """
    log_total_w = math.log(sum(weights))
    log_weights = [math.log(w) for w in weights]
    mixed: LogprobMaps = []
    for position_maps in zip(*maps_per_teacher, strict=True):
        ids = set().union(*(m.keys() for m in position_maps))
        out: dict[int, float] = {}
        for token_id in ids:
            terms = []
            for log_w, teacher_map in zip(log_weights, position_maps, strict=True):
                if token_id not in teacher_map:
                    raise ValueError(
                        f"Ensemble teacher response is missing logprob for token id {token_id}; "
                        "all group members must be scored at the same token ids."
                    )
                terms.append(log_w + teacher_map[token_id])
            max_term = max(terms)
            out[token_id] = max_term + math.log(sum(math.exp(t - max_term) for t in terms)) - log_total_w
        mixed.append(out)
    return mixed


def _teacher_responses_from_payload(reward_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:
    """Normalize single-teacher and ensemble reward payloads to (responses, weights)."""
    if "teachers" in reward_payload:
        return reward_payload["teachers"], reward_payload["teacher_weights"]
    return [reward_payload["teacher"]], [1.0]


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


def _extract_teacher_topk(
    reward_payload: dict[str, Any], response_length: int, top_k: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Build per-position ``(ids, logprobs)`` rows of width exactly ``top_k`` for
    --loss-type opd_topk_loss's retained transport, from the single-teacher
    payload's ``input_top_logprobs`` maps. Rows are sorted by descending
    logprob and padded with (_TOPK_PAD_TOKEN_ID, _TOPK_PAD_LOGPROB) when a
    position has fewer than ``top_k`` entries.

    Raises ``ValueError`` on ensemble payloads (``"teachers" in reward_payload``)
    -- validation (Task 4) makes this unreachable in production.
    """
    if "teachers" in reward_payload:
        raise ValueError("--loss-type opd_topk_loss does not support teacher ensembles.")
    if response_length == 0:
        return [], []

    position_maps = _input_logprob_maps(reward_payload["teacher"], "input_top_logprobs", response_length)
    ids_rows: list[list[int]] = []
    logprobs_rows: list[list[float]] = []
    for position_map in position_maps:
        entries = sorted(position_map.items(), key=lambda item: item[1], reverse=True)[:top_k]
        pad_count = top_k - len(entries)
        ids_rows.append([token_id for token_id, _ in entries] + [_TOPK_PAD_TOKEN_ID] * pad_count)
        logprobs_rows.append([logprob for _, logprob in entries] + [_TOPK_PAD_LOGPROB] * pad_count)
    return ids_rows, logprobs_rows


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


def _tail_bucket_reverse_kl(student_logps: list[float], teacher_logps: list[float]) -> float:
    """Exact reverse KL over the (k+1)-bucket partition: selected ids + one tail bucket.

    Both logprob lists are exact full-softmax logprobs at the same token ids, so
    {p(v) for v in S} + tail with tail = 1 - sum_v p(v) is a proper distribution
    and KL = sum_v p_s(v)(s_v - t_v) + tail_s * (log tail_s - log tail_t) needs
    no renormalization — the truncated estimate stays sensitive to mass the
    student pushes outside S. Python floats keep the math in float64; the tail
    uses log1p for accuracy and a student tail rounded to <= 0 contributes 0
    (the x*log(x) -> 0 limit).
    """
    student_probs = [math.exp(logp) for logp in student_logps]
    kl = sum(
        p * (s_logp - t_logp) for p, s_logp, t_logp in zip(student_probs, student_logps, teacher_logps, strict=True)
    )
    student_mass = sum(student_probs)
    if student_mass >= 1.0:
        return kl
    teacher_mass = sum(math.exp(logp) for logp in teacher_logps)
    log_tail_s = math.log1p(-student_mass)
    log_tail_t = math.log1p(-min(teacher_mass, 1.0 - TAIL_PROB_FLOOR))
    return kl + (1.0 - student_mass) * (log_tail_s - log_tail_t)


def _tail_bucket_forward_kl(student_logps: list[float], teacher_logps: list[float]) -> float:
    """Exact forward KL over the (k+1)-bucket partition — the mirror of
    :func:`_tail_bucket_reverse_kl` with teacher and student roles swapped:
    KL(p_T || p_s) = sum_v p_T(v)(t_v - s_v) + tail_T * (log tail_T - log tail_s).
    A teacher tail rounded to <= 0 contributes 0 (the x*log(x) -> 0 limit);
    the student tail is floored to keep log finite.
    """
    teacher_probs = [math.exp(logp) for logp in teacher_logps]
    kl = sum(
        p * (t_logp - s_logp) for p, t_logp, s_logp in zip(teacher_probs, teacher_logps, student_logps, strict=True)
    )
    teacher_mass = sum(teacher_probs)
    if teacher_mass >= 1.0:
        return kl
    student_mass = sum(math.exp(logp) for logp in student_logps)
    log_tail_t = math.log1p(-teacher_mass)
    log_tail_s = math.log1p(-min(student_mass, 1.0 - TAIL_PROB_FLOOR))
    return kl + (1.0 - teacher_mass) * (log_tail_t - log_tail_s)


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
    kl_type, mixed_weight = _get_kl_type(args)
    tail_bucket = bool(getattr(args, "opd_topk_tail_bucket", False))
    if tail_bucket and strategy not in ("only-student", "intersection"):
        # The k+1 partition is only exact when all student logprobs come from one
        # softmax; only-teacher/union/xor mix the rollout harvest with a
        # separate rescoring pass (also validated at startup).
        raise ValueError(
            "--opd-topk-tail-bucket requires --opd-top-k-strategy only-student "
            f"or intersection, got {strategy!r}."
        )

    student_top_maps = (
        [_top_entries_to_map(entries) for entries in _student_top_logprobs(sample, response_length)]
        if strategy in STUDENT_TOP_STRATEGIES
        else [{} for _ in range(response_length)]
    )

    teacher_responses, teacher_weights = _teacher_responses_from_payload(reward_payload)
    if len(teacher_responses) == 1:
        teacher_response = teacher_responses[0]
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
    else:
        # Ensemble: every group member was scored at the student's per-position
        # top-k ids (strategy validated to only-student at startup); mix raw teacher
        # probabilities per token id BEFORE any weighting — a mixture of
        # renormalized truncations is not the truncation of the mixture.
        if strategy != "only-student":
            raise ValueError(f"Teacher ensembles require --opd-top-k-strategy only-student, got {strategy!r}.")
        teacher_top_maps = [{} for _ in range(response_length)]
        teacher_on_student_maps = _mixture_logprob_maps(
            [
                _input_logprob_maps(response, "input_token_ids_logprobs", response_length)
                for response in teacher_responses
            ],
            teacher_weights,
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

        def _reverse_term():
            if tail_bucket:
                return _tail_bucket_reverse_kl(student_logps, teacher_logps)
            weights = _reward_weights(student_logps, teacher_logps, weight_mode, normalize=normalize_weights)
            return sum(
                w * (s_logp - t_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
            )

        def _forward_term():
            # Forward KL weights by the teacher distribution (its natural
            # measure); --opd-reward-weight-mode applies to the reverse term only.
            if tail_bucket:
                return _tail_bucket_forward_kl(student_logps, teacher_logps)
            weights = _reward_weights(student_logps, teacher_logps, "teacher_p", normalize=normalize_weights)
            return sum(
                w * (t_logp - s_logp) for w, s_logp, t_logp in zip(weights, student_logps, teacher_logps, strict=True)
            )

        if kl_type == "reverse":
            value = _reverse_term()
        elif kl_type == "forward":
            value = _forward_term()
        else:
            value = mixed_weight * _forward_term() + (1.0 - mixed_weight) * _reverse_term()
        reverse_kls.append(value)

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


async def _post_teacher_group(
    targets: list[TeacherTarget],
    payload: dict[str, Any],
    timeout_secs: int | float | None,
    max_response_bytes: int | None = None,
) -> dict[str, Any]:
    """Score one payload against a teacher group.

    A singleton group returns the raw response (routing/single-teacher path,
    unchanged shape). An ensemble group fans the same payload out to every
    member in parallel — wall clock is max(teacher latencies), not the sum —
    and returns the responses with their mixture weights.
    """
    responses = await asyncio.gather(
        *[
            _post_json(url, payload, timeout_secs=timeout_secs, max_response_bytes=max_response_bytes)
            for url, _ in targets
        ]
    )
    if len(responses) == 1:
        return responses[0]
    return {"teachers": list(responses), "teacher_weights": [weight for _, weight in targets]}


def _sampled_teacher_log_probs(payload: dict[str, Any], response_length: int) -> list[float]:
    """Sampled-token teacher logprobs from a single-teacher or ensemble payload.

    For an ensemble, per-teacher sampled-token logprobs are combined as the
    weighted probability-space mixture (the logprob of the mixture teacher).
    """
    if isinstance(payload, dict) and "teachers" in payload:
        per_teacher = [
            torch.tensor(_extract_teacher_log_probs(response, response_length), dtype=torch.float32)
            for response in payload["teachers"]
        ]
        return _mixture_log_probs(per_teacher, payload["teacher_weights"]).tolist()
    return _extract_teacher_log_probs(payload, response_length)


def _full_vocab_payload(input_ids: list[int]) -> dict[str, Any]:
    """Prefill-only scoring request for the teacher's per-position hidden states.

    No ``return_logprob``: full-vocab mode reconstructs the whole teacher
    distribution trainer-side from the hidden states and the teacher's LM head
    (orbit/backends/training_utils/teacher_lm_head.py). The teacher server must
    run with ``--enable-return-hidden-states``, ``--disable-radix-cache`` (a
    cache hit skips the forward pass, so no hidden states for the matched
    prefix) and ``--chunked-prefill-size -1`` (only the last chunk of a chunked
    prefill returns hidden states -- sgl-project/sglang#8066).
    """
    return {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_hidden_states": True,
    }


def _teacher_hidden_states_from_payload(payload: dict[str, Any], num_tokens: int, response_length: int) -> np.ndarray:
    """Pure decode/slice logic for a full-vocab scoring response (no I/O).

    ``meta_info["hidden_states"]`` is wrapped in a request-batch dimension
    (always length 1 here: each HTTP call scores exactly one sample); the inner
    payload is one vector per input token, base64-encoded fp32 on current
    sglang, nested JSON floats on the legacy path. ``hidden_states[t]`` is the
    state after consuming token ``t`` -- what predicts token ``t+1`` -- so the
    rows that score the response span are ``[num_tokens - response_length - 1,
    num_tokens - 1)``.
    """
    outer_hidden_states = payload["meta_info"].get("hidden_states") or []
    if len(outer_hidden_states) != 1:
        raise ValueError(
            f"expected meta_info['hidden_states'] to have exactly 1 (batch) entry, got "
            f"{len(outer_hidden_states)} -- sglang's return_hidden_states response shape "
            "changed. Check the teacher server flags in _full_vocab_payload's docstring."
        )
    raw_hidden_states = outer_hidden_states[0]
    if isinstance(raw_hidden_states, str):
        raw = pybase64.b64decode(raw_hidden_states.encode("ascii"))
        if len(raw) % (num_tokens * _HIDDEN_STATE_DTYPE.itemsize) != 0:
            meta_info = payload["meta_info"]
            raise ValueError(
                f"teacher hidden_states buffer of {len(raw)} bytes is not a whole number of "
                f"{_HIDDEN_STATE_DTYPE.name} vectors over {num_tokens} input positions "
                f"(response_length={response_length}, cached_tokens={meta_info.get('cached_tokens')}, "
                f"prompt_tokens={meta_info.get('prompt_tokens')}). The teacher likely served fewer "
                "positions than sent -- check the server flags in _full_vocab_payload's docstring."
            )
        hidden_states = np.frombuffer(raw, dtype=_HIDDEN_STATE_DTYPE).reshape(num_tokens, -1)
    else:
        _warn_legacy_hidden_states_format()
        hidden_states = np.asarray(raw_hidden_states, dtype=_HIDDEN_STATE_DTYPE)
        if hidden_states.ndim != 2 or hidden_states.shape[0] != num_tokens:
            raise ValueError(
                f"teacher hidden_states has shape {hidden_states.shape}, expected "
                f"({num_tokens}, hidden_size) -- the teacher likely served fewer positions than "
                "sent. Check the server flags in _full_vocab_payload's docstring."
            )
    hidden_start = num_tokens - response_length - 1
    if hidden_start < 0:
        raise ValueError(
            "full-vocab teacher scoring needs at least one prompt token before the response "
            f"(num_tokens={num_tokens}, response_length={response_length})"
        )
    return np.array(hidden_states[hidden_start : hidden_start + response_length])


async def _score_with_teacher(
    args, sample: Sample, targets: list[TeacherTarget] | None = None, lora_path: str | None = None
) -> dict:
    """POST the sample's full token sequence to the SGLang teacher group for
    prefill-only scoring (sampled-token path). Kept separate from
    ``reward_func`` so tests can monkeypatch it and never hit the network.
    """
    targets = targets or [(args.opd_teacher_url, 1.0)]
    return await _post_teacher_group(
        targets, _score_payload(sample.tokens, lora_path=lora_path), _scoring_timeout(args)
    )


async def _score_top_k(
    args, sample: Sample, targets: list[TeacherTarget] | None = None, lora_path: str | None = None
) -> dict[str, Any]:
    """Orchestrate top-k cross-scoring: the teacher group scored with its own
    top-k and/or on the student's top-k token ids; optionally the student
    re-scored on the teacher's top-k ids (via the rollout router). Returns the
    reward payload consumed by ``_compute_topk_reverse_kl``.
    """
    top_k = _get_opd_top_k(args)
    strategy = _get_top_k_strategy(args)
    targets = targets or [(args.opd_teacher_url, 1.0)]
    request_timeout = _scoring_timeout(args)
    response_byte_limit = _topk_response_byte_limit(args, len(sample.tokens))

    teacher_token_ids = None
    if strategy in TEACHER_ON_STUDENT_STRATEGIES:
        student_top = _student_top_logprobs(sample, sample.response_length)
        teacher_token_ids = _unique_ids(student_top)

    teacher_payload = _score_payload(
        sample.tokens,
        top_k=top_k if strategy in TEACHER_TOP_STRATEGIES else 0,
        token_ids=teacher_token_ids,
        lora_path=lora_path,
    )
    group_response = await _post_teacher_group(
        targets, teacher_payload, request_timeout, max_response_bytes=response_byte_limit
    )

    reward_payload = group_response if "teachers" in group_response else {"teacher": group_response}
    if strategy in STUDENT_ON_TEACHER_STRATEGIES:
        if "teachers" in reward_payload:
            raise ValueError(f"Teacher ensembles require --opd-top-k-strategy only-student, got {strategy!r}.")
        teacher_top = _trim_input_field(
            reward_payload["teacher"]["meta_info"], "input_top_logprobs", sample.response_length
        )
        student_token_ids = _unique_ids(teacher_top)
        reward_payload["student_on_teacher"] = await _post_json(
            _student_score_url(args),
            _score_payload(sample.tokens, token_ids=student_token_ids),
            timeout_secs=request_timeout,
            max_response_bytes=response_byte_limit,
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
    # Evaluation samples want the real task reward, not teacher scoring: this hook
    # occupies the reward slot as a transport, and its 0.0 returns would zero every
    # eval pass-rate (and, in full_vocab mode, ship eval-length hidden states for
    # nothing). Delegate to the rule-based RM dispatch instead.
    if kwargs.get("evaluation"):
        from orbit.rollout.rm_hub import default_async_rm

        return await default_async_rm(args, sample)

    # Multi-teacher routing/ensemble: pick this sample's teacher group (falls
    # back to --opd-teacher-url when --opd-teacher-urls is unset).
    teacher_targets = _teacher_targets_for_sample(args, sample)
    if teacher_score_mode(args) == "full_vocab":
        if len(teacher_targets) != 1:
            raise ValueError(
                "--teacher-score-mode full_vocab supports a single teacher only: mixing "
                "reconstructed distributions across ensemble members needs per-member LM heads "
                f"trainer-side, got {len(teacher_targets)} targets."
            )
        if sample.response_length == 0:
            sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = {"empty_response": True}
            return 0.0
        url, _ = teacher_targets[0]
        sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = await _post_json(
            url,
            _full_vocab_payload(sample.tokens),
            timeout_secs=_scoring_timeout(args),
            max_response_bytes=_full_vocab_response_byte_limit(args, len(sample.tokens)),
        )
    elif _get_opd_top_k(args) > 0:
        sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = await _score_top_k(args, sample, teacher_targets)
    else:
        sample.metadata[TEACHER_RESPONSE_METADATA_KEY] = await _score_with_teacher(args, sample, teacher_targets)
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
    full_vocab = teacher_score_mode(args) == "full_vocab"
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
        if full_vocab:
            if payload.get("empty_response"):
                sample.teacher_hidden_states = np.zeros((0, 0), dtype=_HIDDEN_STATE_DTYPE)
            else:
                sample.teacher_hidden_states = _teacher_hidden_states_from_payload(
                    payload, len(sample.tokens), sample.response_length
                )
        elif top_k > 0:
            sample.opd_reverse_kl = _compute_topk_reverse_kl(args, sample, payload).tolist()
            if getattr(args, "loss_type", None) == "opd_topk_loss":
                sample.teacher_topk_ids, sample.teacher_topk_logprobs = _extract_teacher_topk(
                    payload, sample.response_length, top_k
                )
            # The harvested per-position top-logprob lists are large (O(R*k) Python
            # objects); once the KL is computed they only bloat Ray transfers.
            sample.metadata.pop(STUDENT_TOP_LOGPROBS_METADATA_KEY, None)
        else:
            sample.teacher_log_probs = _sampled_teacher_log_probs(payload, sample.response_length)
    if unscored:
        logger.warning("OPD sglang post_process: %d/%d samples had no stashed teacher response.", unscored, len(samples))

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
