from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Turns are recorded on Sample.metadata rather than a core Sample field, so this
# stays a customization-layer feature with no miles/ surface of its own.
RESPONSE_TURNS_KEY = "response_turns"
_OMIT = object()
_MEDIA_TYPES = {
    "audio",
    "audio_url",
    "image",
    "image_url",
    "input_audio",
    "input_image",
    "input_video",
    "video",
    "video_url",
}
_MEDIA_KEYS = {
    "audio",
    "audios",
    "audio_url",
    "base64",
    "bytes",
    "image",
    "images",
    "image_url",
    "input_audio",
    "input_image",
    "input_video",
    "pixel_values",
    "video",
    "videos",
    "video_url",
}
_OMITTED_PAYLOAD_KEYS = {
    "accumulated_token_ids",
    "attention_mask",
    "attention_masks",
    "indexer_topk",
    "input_ids",
    "loss_mask",
    "loss_masks",
    "mask",
    "masks",
    "multimodal_inputs",
    "multimodal_train_inputs",
    "opd_student_top_logprobs",
    "output_ids",
    "rollout_indexer_topk",
    "rollout_log_probs",
    "rollout_routed_experts",
    "routed_experts",
    "teacher_log_probs",
    "teacher_topk_log_probs",
    "teacher_topk_token_ids",
    "teacher_topk_valid_mask",
    "top_logprobs",
    "log_probs",
    "logprobs",
    "token_ids",
    "tokens",
}
_OMITTED_PAYLOAD_SUFFIXES = (
    "_log_probs",
    "_logprobs",
    "_mask",
    "_masks",
    "_token_ids",
)


def _is_omitted_payload_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in _OMITTED_PAYLOAD_KEYS or normalized.endswith(_OMITTED_PAYLOAD_SUFFIXES)


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        media_type = value.get("type")
        if isinstance(media_type, str) and media_type.lower() in _MEDIA_TYPES:
            return {"type": media_type, "omitted": True}

        normalized = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key
            if _is_omitted_payload_key(key):
                continue
            if key.lower() in _MEDIA_KEYS:
                normalized[key] = {"omitted": True}
                continue
            normalized_item = _json_native(item)
            if normalized_item is not _OMIT:
                normalized[key] = normalized_item
        return normalized
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            normalized_item = _json_native(item)
            if normalized_item is not _OMIT:
                normalized.append(normalized_item)
        return normalized
    return _OMIT


def _value_or_none(value: Any) -> Any:
    normalized = _json_native(value)
    return None if normalized is _OMIT else normalized


def _conversation_turns(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    turns = []
    turn_number = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            turn_number += 1
            entry = {
                "turn": turn_number,
                "role": "assistant",
                "content": message.get("content", ""),
            }
            for key in ("finish_reason", "weight_version", "name", "tool_calls"):
                if key in message:
                    entry[key] = message[key]
        elif turn_number > 0 and role in {"user", "tool", "environment"}:
            entry = {
                "turn": turn_number,
                "role": "environment",
                "model_input_role": role,
                "content": message.get("content", ""),
            }
            for key in ("name", "tool_call_id"):
                if key in message:
                    entry[key] = message[key]
        else:
            continue

        normalized = _json_native(entry)
        if isinstance(normalized, dict):
            turns.append(normalized)
    return turns


def _response_turns(sample: Sample) -> list[dict[str, Any]]:
    turns = sample.metadata.get(RESPONSE_TURNS_KEY) if sample.metadata else None
    if turns is not None:
        normalized = _json_native(turns)
        return normalized if isinstance(normalized, list) else []

    messages = sample.metadata.get("messages") if sample.metadata else None
    if messages is not None:
        return _conversation_turns(messages)

    return [{"turn": 1, "role": "assistant", "content": sample.response}]


def model_response_row(sample: Sample, *, rollout_id: int) -> dict[str, Any]:
    # "messages" and the turn list are both raw conversation payloads that the
    # row already surfaces under "turns"; keep them out of the metadata copy.
    raw_metadata = {
        key: value for key, value in (sample.metadata or {}).items() if key not in ("messages", RESPONSE_TURNS_KEY)
    }
    metadata = _json_native(raw_metadata)
    assert isinstance(metadata, dict)

    return {
        "schema_version": SCHEMA_VERSION,
        "rollout_id": rollout_id,
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "status": sample.status.value,
        "reward": _value_or_none(sample.reward),
        "label": _value_or_none(sample.label),
        "response_length": sample.response_length,
        "weight_versions": _value_or_none(sample.weight_versions),
        "prompt": _value_or_none(sample.prompt),
        "metadata": metadata,
        "turns": _response_turns(sample),
    }


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                json.dump(
                    row,
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Failed to clean temporary model response log %s",
                    temporary_path,
                    exc_info=True,
                )


def save_model_response_log(
    args: Any,
    samples: Sequence[Sample],
    *,
    rollout_id: int,
) -> None:
    template = getattr(args, "save_model_response_log", None)
    if template is None:
        return

    destination: str | Path = template
    try:
        path = Path(template.format(rollout_id=rollout_id))
        destination = path
        rows = (model_response_row(sample, rollout_id=rollout_id) for sample in samples)
        _atomic_write_jsonl(path, rows)
        logger.info("Saved accepted model response log to %s", path)
    except Exception:
        logger.exception(
            "Failed to save model response log for rollout %s to %s; training will continue",
            rollout_id,
            destination,
        )
