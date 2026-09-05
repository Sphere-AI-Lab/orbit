"""Orbit-side rendering helpers for envpack observations."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any


class EnvpackRendererError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RenderedObservation:
    message: dict[str, Any]
    images: list[Any]
    videos: list[Any]
    media_hashes: list[str]
    artifacts: list[Any]


def observation_to_chat_message(observation, *, role: str = "user") -> RenderedObservation:
    """Convert envpack SemanticObservation to Orbit/VLM chat-message pieces.

    Envpack media bytes are the source of truth. Artifact refs are preserved for
    audit only and are not used as the model input source.
    """

    try:
        from envpack.client import prepare_observation_message
    except Exception as exc:
        raise EnvpackRendererError(
            "envpack is not importable. Run `pip install -e thirdparty/envpack` "
            "or add the envpack repo to PYTHONPATH on every Orbit worker."
        ) from exc

    prepared = prepare_observation_message(observation, role=role, strict_placeholders=True)
    images = [_decode_image(media.bytes) for media in prepared.media if media.kind == "image"]
    videos = [media.bytes for media in prepared.media if media.kind == "video"]
    return RenderedObservation(
        message=prepared.message,
        images=images,
        videos=videos,
        media_hashes=prepared.media_hashes,
        artifacts=prepared.artifacts,
    )


def _decode_image(data: bytes):
    try:
        from PIL import Image
    except Exception as exc:
        raise EnvpackRendererError("Pillow is required to decode envpack image bytes for Orbit processors") from exc
    return Image.open(io.BytesIO(data)).convert("RGB")
