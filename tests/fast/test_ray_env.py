"""Orbit's Ray visible-device switch still works now that miles/ray/utils.py is pristine.

``build_noset_visible_devices_env_vars`` was a function orbit added inside the
vendored file; it now lives in orbit/utils/ray_env.py and the two vendored call
sites import it from there. A lift that stopped being called looks exactly like
one that is called, so the properties below are asserted against the returned
env dict and against the vendored source, not against the code's presence.
"""

import ast
from pathlib import Path

import pytest

pytest.importorskip("ray")
pytest.importorskip("torch")

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST  # noqa: E402
from orbit.utils.ray_env import (  # noqa: E402
    CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR,
    RESPECT_CUDA_VISIBLE_DEVICES_ENV_VAR,
    build_noset_visible_devices_env_vars,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_vendored_module_no_longer_carries_the_orbit_function():
    """The point of the move. If this fails the file went dirty again."""
    src = (REPO / "miles" / "ray" / "utils.py").read_text()
    assert "build_noset_visible_devices_env_vars" not in src
    assert "ORBIT" not in src


def test_default_disables_masking_for_every_accelerator():
    """Orbit's default: Ray must not rewrite any visible-device variable, because
    colocated rollout spawns children that need the whole node's GPUs."""
    env = build_noset_visible_devices_env_vars({})
    assert env == {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}
    assert CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR in env


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "y", "on"])
def test_the_opt_in_flag_hands_cuda_masking_back_to_ray(value):
    """The case the function exists for: a parity run pins the launcher to a GPU
    subset, and only Ray's masking carries that pinning into the actors."""
    env = build_noset_visible_devices_env_vars({RESPECT_CUDA_VISIBLE_DEVICES_ENV_VAR: value})
    assert CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR not in env
    # ...and nothing else is affected: only CUDA is ever held back.
    assert set(env) == set(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST) - {
        CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR
    }


@pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
def test_anything_that_is_not_a_yes_keeps_the_default(value):
    env = build_noset_visible_devices_env_vars({RESPECT_CUDA_VISIBLE_DEVICES_ENV_VAR: value})
    assert CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR in env


def test_both_vendored_call_sites_import_it_from_orbit():
    """A lift is only finished when the callers actually reach the new home; an
    import left pointing at miles/ would fail at import time, but one silently
    re-added to the vendored file would not."""
    for rel in ("miles/ray/actor_group.py", "miles/ray/rollout.py"):
        tree = ast.parse((REPO / rel).read_text())
        sources = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(a.name == "build_noset_visible_devices_env_vars" for a in node.names)
        }
        assert sources == {"orbit.utils.ray_env"}, f"{rel} imports it from {sources}"
