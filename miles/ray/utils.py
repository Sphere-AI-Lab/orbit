# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import os

import ray
import torch
from miles.ray.ray_actor import RayActor


# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

# ORBIT-SEAM: ORBIT_RESPECT_CUDA_VISIBLE_DEVICES opts back into Ray device masking (parity/debug GPU pinning)
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


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False  # False: unlocked, True: locked

    def acquire(self):
        """
        Try to acquire the lock. Returns True if acquired, False otherwise.
        Caller should retry until it returns True.
        """
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        """Release the lock, allowing others to acquire."""
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
