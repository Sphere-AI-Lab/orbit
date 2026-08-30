"""Ray accelerator visible-device masking, orbit's policy.

Orbit normally asks Ray not to rewrite ``CUDA_VISIBLE_DEVICES`` because some
colocated rollout setups launch child processes that need visibility into
multiple GPUs. ``ORBIT_RESPECT_CUDA_VISIBLE_DEVICES=1`` opts back into Ray's
masking so a parity/debug run pinned to a physical GPU subset keeps that pinning.

Lifted out of ``miles/ray/utils.py`` (now byte-pristine again): orbit-authored
policy that only orbit's actor factory consumes.
"""

from __future__ import annotations

import os

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

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

    excluded = set()
    if _env_flag_enabled(env_vars.get(RESPECT_CUDA_VISIBLE_DEVICES_ENV_VAR)):
        excluded.add(CUDA_NOSET_VISIBLE_DEVICES_ENV_VAR)
    return {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST if name not in excluded}
