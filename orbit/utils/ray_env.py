"""Ray accelerator visible-device env vars, with orbit's opt-back-in switch.

``miles/ray/utils.py`` ships ``NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`` and nothing
else: upstream's actor launches splat the whole list into the worker env, which
tells Ray not to rewrite the visible-device variable for any accelerator. Orbit
wants that by default -- colocated rollout launches child processes that must
see every GPU on the node -- but not unconditionally: a parity or debug run pins
the launcher to a physical GPU subset, and Ray's masking is exactly what carries
that pinning into the actors.

This is a LIFT, not a patch. There is no upstream function to replace; orbit
added one next to upstream's list, and the whole edit was orbit's. Moving it
here is what lets ``miles/ray/utils.py`` be byte-pristine again, and the two
vendored call sites (miles/ray/actor_group.py, miles/ray/rollout.py) import it
from orbit instead.

``NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`` is read inside the function rather than
imported at module scope, because ``miles.ray.utils`` pulls in ray and torch and
nothing here should pay for that at import time.
"""

from __future__ import annotations

import os

# The one entry of the vendored list orbit ever holds back.
CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
RESPECT_CUDA_VISIBLE_DEVICES_ENV_VAR = "ORBIT_RESPECT_CUDA_VISIBLE_DEVICES"


def _env_flag_enabled(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "y", "on"}


def build_noset_visible_devices_env_vars(env_vars=os.environ) -> dict[str, str]:
    """Build Ray env vars that disable accelerator visible-device masking.

    Orbit normally asks Ray not to rewrite CUDA_VISIBLE_DEVICES because some
    colocated rollout setups launch child processes that need visibility into
    multiple GPUs. Parity/debug runs often pin the launcher to a physical GPU
    subset, though. ORBIT_RESPECT_CUDA_VISIBLE_DEVICES=1 keeps that pinning
    effective by letting Ray manage CUDA_VISIBLE_DEVICES for CUDA actors.
    """
    from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

    excluded = set()
    if _env_flag_enabled(env_vars.get(RESPECT_CUDA_VISIBLE_DEVICES_ENV_VAR)):
        excluded.add(CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR)
    return {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST if name not in excluded}
