"""VAGEN env dynamics probe.

Asserts seed determinism (FrozenLake same/different seeds), valid action
→ reward ≥ format_reward AND state changes, invalid action → reward == 0
AND no change. Vision mode mirrors text mode. Run via:

    env -u LD_LIBRARY_PATH conda run -n miles python -m examples.vagen.tests.env_dynamics_probe
"""

import asyncio
import hashlib
import re
from typing import Any

from vagen.envs.registry import get_env_cls


VALID_FL = "<think>analyze</think><answer>Down</answer>"
VALID_SK = "<think>push box right</think><answer>Right</answer>"
INVALID = "go down\n<answer>Down</answer>"  # missing <think> wrapper

# obs_str's templated wrapper differs valid vs invalid even when grid is
# identical — compare grid rows only.
_GRID_ROW_RE = re.compile(r"^\s*[_OGPXV# ]+\s*$")


def grid_of(obs: dict) -> str:
    return "\n".join(line.rstrip() for line in obs["obs_str"].splitlines() if _GRID_ROW_RE.fullmatch(line))


def image_digest(obs: dict) -> str | None:
    pils = (obs.get("multi_modal_input") or {}).get("<image>") or []
    if not pils:
        return None
    h = hashlib.blake2b(digest_size=16)
    for img in pils:
        h.update(img.convert("RGB").tobytes())
    return h.hexdigest()


def initial_signature(obs: dict, mode: str) -> str:
    if mode == "text":
        return grid_of(obs)
    digest = image_digest(obs)
    assert digest, "vision mode must emit PIL images"
    return digest


async def _initial_signature(cls, env_cfg: dict, seed: int) -> str:
    env = cls(env_config=env_cfg)
    try:
        obs, _ = await env.reset(seed)
        return initial_signature(obs, env_cfg["render_mode"])
    finally:
        await env.close()


async def _assert_same_seed_reset(cls, env_cfg: dict, seed: int, label: str) -> None:
    sig1 = await _initial_signature(cls, env_cfg, seed)
    sig2 = await _initial_signature(cls, env_cfg, seed)
    assert sig1 == sig2, f"{label}: same seed produced different initial states"
    print(f"  {label:30s} same-seed reset: stable")


async def _assert_frozenlake_seed_affects_map(cls, env_cfg: dict, seed: int, alt_seeds: list[int], label: str) -> None:
    base = await _initial_signature(cls, env_cfg, seed)
    alts = [await _initial_signature(cls, env_cfg, alt_seed) for alt_seed in alt_seeds]
    assert any(base != alt for alt in alts), f"{label}: changing seeds did not change any initial map"
    print(f"  {label:30s} different-seed reset: changes map")


async def _trace(
    cls, env_cfg: dict, seed: int, valid_action: str, invalid_action: str, format_reward_min: float, label: str
) -> None:
    env = cls(env_config=env_cfg)
    try:
        obs0, _ = await env.reset(seed)
        obs1, r1, _, _ = await env.step(valid_action)
        obs2, r2, _, _ = await env.step(invalid_action)
    finally:
        await env.close()

    mode = env_cfg["render_mode"]
    if mode == "text":
        s0, s1, s2 = grid_of(obs0), grid_of(obs1), grid_of(obs2)
        changed_valid = s0 != s1
        changed_invalid = s1 != s2
        assert s0 != s1, f"{label}: valid action must transition state"
        assert s1 == s2, f"{label}: invalid action must leave grid unchanged"
    else:  # vision
        d0, d1, d2 = image_digest(obs0), image_digest(obs1), image_digest(obs2)
        assert d0 and d1 and d2, f"{label}: vision mode must emit PIL images"
        changed_valid = d0 != d1
        changed_invalid = d1 != d2
        assert d0 != d1, f"{label}: valid action must repaint the image"
        assert d1 == d2, f"{label}: invalid action must leave pixels identical"

    assert (
        r1 >= format_reward_min - 1e-6
    ), f"{label}: valid action should earn >= format_reward({format_reward_min}), got {r1}"
    assert r2 == 0.0, f"{label}: invalid format must earn 0 reward, got {r2}"

    print(
        f"  {label:30s} valid: reward={r1:.3f} changed={changed_valid}  |  "
        f"invalid: reward={r2:.3f} changed={changed_invalid}"
    )


async def main() -> None:
    fl_cls = get_env_cls("FrozenLake")
    sk_cls = get_env_cls("Sokoban")

    fl_common: dict[str, Any] = {
        "size": 4,
        "is_slippery": False,
        "max_actions_per_step": 1,
        "use_example_in_sys_prompt": False,
        "prompt_format": "free_think",
    }
    sk_common: dict[str, Any] = {
        "dim_room": [6, 6],
        "num_boxes": 1,
        "max_steps": 50,
        "prompt_format": "free_think",
    }

    print("Seed determinism:")
    await _assert_same_seed_reset(fl_cls, {**fl_common, "render_mode": "text"}, seed=0, label="FrozenLake text")
    await _assert_same_seed_reset(fl_cls, {**fl_common, "render_mode": "vision"}, seed=0, label="FrozenLake vision")
    await _assert_frozenlake_seed_affects_map(
        fl_cls,
        {**fl_common, "render_mode": "text"},
        seed=0,
        alt_seeds=[1, 2, 3, 4],
        label="FrozenLake text",
    )
    # FrozenLake seed=0: Down moves P to a safe cell. format_reward=0.02.
    print("FrozenLake seed=0  (format_reward=0.02):")
    await _trace(
        fl_cls,
        {**fl_common, "render_mode": "text"},
        seed=0,
        valid_action=VALID_FL,
        invalid_action=INVALID,
        format_reward_min=0.02,
        label="text mode",
    )
    await _trace(
        fl_cls,
        {**fl_common, "render_mode": "vision"},
        seed=0,
        valid_action=VALID_FL,
        invalid_action=INVALID,
        format_reward_min=0.02,
        label="vision mode",
    )

    # Sokoban seed=42: Right pushes the box one cell. format_reward=0.10.
    print("Sokoban    seed=42 (format_reward=0.10):")
    await _trace(
        sk_cls,
        {**sk_common, "render_mode": "text"},
        seed=42,
        valid_action=VALID_SK,
        invalid_action=INVALID,
        format_reward_min=0.10,
        label="text mode",
    )
    await _trace(
        sk_cls,
        {**sk_common, "render_mode": "vision"},
        seed=42,
        valid_action=VALID_SK,
        invalid_action=INVALID,
        format_reward_min=0.10,
        label="vision mode",
    )

    print("dynamics-probe OK")


if __name__ == "__main__":
    asyncio.run(main())
