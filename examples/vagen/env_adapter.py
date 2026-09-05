"""Thin bridge between VAGEN `GymImageEnv` and the orbit rollout path.

`vagen.agent_loop.gym_agent_loop` imports VERL at the top, which is not in
the orbit env, so we keep a local copy of the three helpers we need
(`_normalize_images`, `convert_obs_to_content`, `extract_success`). See
`examples/vagen/docs/rollout.md` (env_adapter local copy).
"""

import re
from typing import Any

from PIL import Image as _PIL_Image

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.registry import get_env_cls

# --- Begin local copy (do NOT import gym_agent_loop: it pulls in VERL) ---


def _normalize_images(imgs):
    """Ensure PIL RGB and drop Nones. Ported from VAGEN
    `gym_agent_loop._normalize_images`."""
    out = []
    for im in imgs or []:
        if im is None:
            continue
        out.append(im.convert("RGB") if isinstance(im, _PIL_Image.Image) else im)
    return out


def convert_obs_to_content(
    obs: dict[str, Any],
    obs_text_key: str = "obs_str",
    image_placeholder: str = "<image>",
    video_placeholder: str = "<video>",
    multi_modal_key: str = "multi_modal_input",
    **kwargs,
) -> list[dict[str, Any]]:
    """Split `obs_str` by placeholders into an interleaved content list.
    Ported from VAGEN `gym_agent_loop.convert_obs_to_content`; placeholder
    order MUST be preserved for processor alignment."""
    text = obs[obs_text_key]
    mmi = obs.get(multi_modal_key, {}) or {}
    n_img_tok = text.count(image_placeholder)
    n_vid_tok = text.count(video_placeholder)
    n_imgs = len(mmi.get(image_placeholder, []) or [])
    n_vids = len(mmi.get(video_placeholder, []) or [])
    assert n_img_tok == n_imgs, f"#images ({n_imgs}) != #{image_placeholder} ({n_img_tok})"
    assert n_vid_tok == n_vids, f"#videos ({n_vids}) != #{video_placeholder} ({n_vid_tok})"
    pattern = f"({re.escape(image_placeholder)}|{re.escape(video_placeholder)})"
    out: list[dict[str, Any]] = []
    for seg in re.split(pattern, text):
        if not seg:
            continue
        if seg == image_placeholder:
            out.append({"type": "image"})
        elif seg == video_placeholder:
            out.append({"type": "video"})
        else:
            out.append({"type": "text", "text": seg})
    return out


def extract_success(info: dict[str, Any], success_keys: str = "success|is_success") -> bool:
    """Overwrite semantics (not sticky-OR); supports `success|is_success` dual
    keys. Ported from VAGEN `gym_agent_loop.extract_success`."""
    for key in success_keys.split("|"):
        if key in info:
            return bool(info[key])
    return False


# --- End local copy ---


def build_env(meta: dict) -> GymImageEnv:
    """Build a fresh GymImageEnv from a sample's `metadata['vagen']` dict.

    Note: Sokoban's `__init__` immediately constructs the underlying gym env;
    FrozenLake constructs lazily in `reset()`. Either way the call is sync.
    """
    env_cls = get_env_cls(meta["env_name"])
    return env_cls(env_config=meta["config"])


def vagen_obs_to_chat_message(obs: dict) -> tuple[dict, list]:
    """Convert VAGEN obs `{obs_str, multi_modal_input}` to
    `({role, content[]}, pil_list)`. PIL order matches `{"type": "image"}`
    blocks in `content`. FrozenLake/Sokoban vision mode emit `<image>`."""
    content = convert_obs_to_content(obs)
    pils = _normalize_images((obs.get("multi_modal_input") or {}).get("<image>", []) or [])
    return {"role": "user", "content": content}, pils


async def safe_close(env: GymImageEnv) -> None:
    try:
        await env.close()
    except Exception:
        pass
