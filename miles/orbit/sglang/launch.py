"""Launch-env setup for the SGLang rollout server child process.

Home for the launch-time PYTHONPATH/env preparation helpers lifted out of
miles/backends/sglang_utils/sglang_engine.py (Phase 3 isolation, slice 3c):
native-ops compat-site injection and PEFT-rollout radix-cache env handling,
consumed by ``launch_server_process`` / ``_init_normal`` in the miles file.
"""

import logging
import os
from pathlib import Path

from sglang.srt.server_args import ServerArgs

from miles.orbit import sglang as _peft_sglang
from miles.orbit.sglang.native_ops import patch_sglang_native_ops

logger = logging.getLogger(__name__)

_COMPAT_SITE_DIR = Path(_peft_sglang.__file__).resolve().parent / "compat_site"


def _prepend_pythonpath(path: Path):
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    path_str = str(path)
    if path_str not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *entries])


def _prepare_child_native_ops_env(force_native_ops: bool):
    if not force_native_ops:
        return

    os.environ["ORBIT_SGLANG_FORCE_NATIVE_OPS"] = "1"
    _prepend_pythonpath(_COMPAT_SITE_DIR)


def _server_args_enable_peft(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "enable_lora", False) or getattr(server_args, "enable_oft", False))


def _prepare_child_peft_cache_env(server_args: ServerArgs):
    if not _server_args_enable_peft(server_args):
        return

    # PEFT rollout requests rely on SGLang's adapter/version extra_key when
    # matching prefix cache entries. In the tested SGLang build, the Python
    # radix cache honors it while the experimental C++ radix tree drops it.
    previous = os.environ.get("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE")
    if previous not in (None, "", "0", "false", "False"):
        logger.warning(
            "Disabling SGLang experimental C++ radix tree for PEFT rollout; "
            "the Python radix cache preserves adapter-specific prefix keys."
        )
    os.environ["SGLANG_EXPERIMENTAL_CPP_RADIX_TREE"] = "0"


def _configure_peft_cache_kwargs(kwargs: dict, peft_method: str | None):
    if peft_method not in {"lora", "oft"}:
        return

    if kwargs.get("disable_radix_cache") is not True:
        logger.warning(
            "Disabling SGLang radix cache for PEFT rollout; cached prefixes can "
            "produce stale adapter activations and train-inference mismatch."
        )
    kwargs["disable_radix_cache"] = True


def _launch_server_with_orbit_compat(server_args: ServerArgs, force_native_ops: bool):
    _prepare_child_peft_cache_env(server_args)

    if force_native_ops:
        patch_sglang_native_ops()

    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)
