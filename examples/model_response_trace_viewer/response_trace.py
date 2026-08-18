from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from examples.model_response_trace_viewer.response_log import model_response_row
from PIL import Image

from miles.utils.types import Sample

_MESSAGE_ATTRIBUTES = (
    "name",
    "tool_call_id",
    "tool_calls",
    "finish_reason",
    "weight_version",
    "model_input_role",
)
_VIEWER_ROLES = {
    "system",
    "developer",
    "prompt",
    "user",
    "assistant",
    "tool",
    "environment",
    "unknown",
}

logger = logging.getLogger(__name__)


def _display_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _text_message(role: str, value: Any) -> dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "text", "text": _display_text(value)}],
    }


def _turn_message(turn: dict[str, Any]) -> dict[str, Any]:
    role = turn.get("role")
    if role not in _VIEWER_ROLES:
        raise ValueError(f"unsupported trace role: {role!r}")
    message = _text_message(role, turn.get("content", ""))
    for key in _MESSAGE_ATTRIBUTES:
        if key in turn:
            message[key] = turn[key]
    return message


def _media_rows(image_count: int) -> list[dict[str, Any]]:
    if type(image_count) is not int or image_count < 0:
        raise ValueError("image_count must be a non-negative integer")
    return [
        {
            "id": f"image-{index}",
            "type": "image",
            "path": f"turn{index}_obs.png",
            "message_index": None,
            "content_index": None,
        }
        for index in range(image_count)
    ]


def model_response_trace_record(
    sample: Sample,
    *,
    rollout_id: int,
    group_index: int,
    image_count: int,
) -> dict[str, Any]:
    if type(rollout_id) is not int or rollout_id < 0:
        raise ValueError("rollout_id must be a non-negative integer")
    if type(group_index) is not int or group_index < 0:
        raise ValueError("group_index must be a non-negative integer")
    if type(sample.remove_sample) is not bool:
        raise ValueError("sample.remove_sample must be a Boolean")
    if not isinstance(sample.response, str):
        raise ValueError("sample.response must be a string")

    row = model_response_row(sample, rollout_id=rollout_id)
    messages = [_text_message("prompt", row["prompt"])]
    messages.extend(_turn_message(turn) for turn in row["turns"])
    if not any(message["role"] == "assistant" for message in messages):
        messages.append(_text_message("assistant", sample.response))

    media = _media_rows(image_count)
    sample_index = sample.index
    if type(sample_index) is not int or sample_index < 0:
        sample_index = None

    return {
        "trace_schema_version": 1,
        "ids": {
            "step": rollout_id,
            "group_index": group_index,
            "sample_index": sample_index,
        },
        "env": {
            "name": None,
            "seed": None,
            "max_turns": None,
            "config": {},
        },
        "outcome": {
            "status": row["status"],
            "reward": row["reward"],
            "num_turns": sum(message["role"] == "assistant" for message in messages),
            "remove_sample": sample.remove_sample,
        },
        "counts": {
            "response_length": row["response_length"],
            "n_images": len(media),
            "n_messages": len(messages),
            "n_tools": 0,
        },
        "trajectory": {
            "prompt": row["prompt"] if isinstance(row["prompt"], str) else None,
            "response": sample.response,
        },
        "conversation": {
            "source_format": "rendered_prompt_plus_turns",
            "tools": [],
            "messages": messages,
        },
        "media": media,
        "metadata": {"producer": "miles"},
    }


def _sample_images(sample: Sample) -> list[Image.Image]:
    inputs = sample.multimodal_inputs
    if inputs is None:
        return []
    if not isinstance(inputs, dict):
        raise TypeError("sample.multimodal_inputs must be a mapping or None")
    images = inputs.get("images")
    if images is None:
        return []
    if type(images) is not list:
        raise TypeError("sample.multimodal_inputs['images'] must be a list")
    if not all(isinstance(image, Image.Image) for image in images):
        raise TypeError("every trace image must be an in-memory PIL image")
    return images


def _selected_samples(
    samples: Sequence[Sample],
    cap: int | None,
) -> list[tuple[Sample, int, int, str]]:
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError("model response trace sample cap must be a positive integer")
    next_ordinal: dict[int, int] = defaultdict(int)
    selected = []
    names = set()
    for position, sample in enumerate(samples if cap is None else samples[:cap]):
        group_index = sample.group_index
        if type(group_index) is not int or group_index < 0:
            group_index = position
        ordinal = next_ordinal[group_index]
        next_ordinal[group_index] += 1
        name = f"prompt{group_index:05d}_rollout{ordinal:02d}"
        if name in names:
            raise ValueError(f"duplicate trace destination: {name}")
        names.add(name)
        selected.append((sample, group_index, ordinal, name))
    return selected


def _write_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            record,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_png(path: Path, image: Image.Image) -> None:
    rgba = image.convert("RGBA")
    clean = Image.frombytes("RGBA", rgba.size, rgba.tobytes())
    try:
        with path.open("xb") as handle:
            clean.save(
                handle,
                format="PNG",
                optimize=False,
                compress_level=6,
            )
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        clean.close()
        if rgba is not image:
            rgba.close()


def save_model_response_trace(
    args: Any,
    samples: Sequence[Sample],
    *,
    rollout_id: int,
) -> None:
    trace_dir = getattr(args, "save_model_response_trace_dir", None)
    if trace_dir is None or not samples:
        return

    staging: Path | None = None
    final_step: Path | None = None
    try:
        cap = getattr(args, "model_response_trace_max_samples_per_step", None)
        selected = _selected_samples(samples, cap)
        train_dir = Path(trace_dir) / "train"
        final_step = train_dir / f"step{rollout_id:04d}"
        if final_step.exists():
            logger.warning(
                "Skipping model response trace rollout %s because %s already exists",
                rollout_id,
                final_step,
            )
            return

        train_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".step{rollout_id:04d}.",
                dir=train_dir,
            )
        )
        for sample, group_index, _ordinal, name in selected:
            sample_dir = staging / name
            sample_dir.mkdir()
            images = _sample_images(sample)
            for image_index, image in enumerate(images):
                _write_png(sample_dir / f"turn{image_index}_obs.png", image)
            record = model_response_trace_record(
                sample,
                rollout_id=rollout_id,
                group_index=group_index,
                image_count=len(images),
            )
            _write_json(sample_dir / "record.json", record)
        staging.rename(final_step)
        staging = None
    except Exception:
        failure_info = sys.exc_info()
        if staging is not None:
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
        logger.warning(
            "Failed to save model response trace for rollout %s at %s",
            rollout_id,
            final_step if final_step is not None else trace_dir,
            exc_info=failure_info,
        )
