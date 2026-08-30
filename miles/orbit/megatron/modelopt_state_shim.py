"""Supply `megatron.post_training.checkpointing` when Megatron-LM is not installed.

Megatron's `get_model()` does this, unconditionally when nvidia-modelopt is
present (`megatron/training/training.py:1324`)::

    if has_nvidia_modelopt:
        from megatron.post_training.checkpointing import has_modelopt_state

`has_nvidia_modelopt` is True in this environment -- nvidia-modelopt 0.44.0 is
installed for the NVFP4/INT4 work -- but the installed `megatron` namespace
package carries only `bridge`, `core` and `training`. `post_training` lives in
the full Megatron-LM distribution, which is not a dependency here.

The result was that **every full fine-tuning run** died with
`ModuleNotFoundError: No module named 'megatron.post_training'`, while every
PEFT run was fine: `_build_model` routes PEFT through
`_setup_peft_model_via_bridge` and never reaches Megatron's `get_model()`.
Found by the coverage probe on 2026-07-31, on the first FullFT arm ever run.

**Why a shim rather than installing Megatron-LM.** The only Megatron-LM checkout
on this filesystem is core 0.16.0rc0 against the installed 0.18.0rc0. Both
`megatron/` trees are PEP-420 namespace packages, so putting that checkout on
PYTHONPATH merges them -- and since PYTHONPATH precedes site-packages,
`megatron.core` would silently resolve to 0.16. Downgrading the entire core to
supply one predicate is a worse trade than the bug.

**Why this is not a stub.** `has_modelopt_state` answers a question about the
checkpoint on disk, and for a sharded checkpoint that question is "is there a
`modelopt_state/` directory in the load dir" -- which this file answers exactly.
Where it cannot answer correctly it raises rather than guessing; see the
function's docstring.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

# The name Megatron imports. Pinned against the upstream source by
# test_the_module_name_is_the_one_megatron_actually_imports, so a rename there
# cannot leave this registering a module nobody looks for.
MODULE_NAME = "megatron.post_training.checkpointing"
_PACKAGE_NAME = "megatron.post_training"

_MISSING_PACKAGE_ERROR = (
    "checkpoint {path} carries ModelOpt state, but `megatron.post_training` is "
    "not installed in this environment -- only megatron.{{bridge,core,training}} "
    "are. orbit ships a shim for the common case (a checkpoint with no ModelOpt "
    "state, which is every checkpoint in the lora-regret campaign); loading a "
    "real ModelOpt checkpoint needs the genuine package from Megatron-LM, at a "
    "version matching the installed megatron-core."
)


def _load_dir(checkpoint_path: Path) -> Path | None:
    """The directory a sharded Megatron checkpoint actually loads from.

    Mirrors upstream's `get_sharded_load_dir`: the iteration named by
    `latest_checkpointed_iteration.txt`, or the checkpoint root itself when
    there is no such file (a bare dist-checkpoint directory).
    """
    marker = checkpoint_path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return checkpoint_path if checkpoint_path.is_dir() else None
    tag = marker.read_text(encoding="utf-8").strip()
    if not tag:
        return checkpoint_path
    # Megatron writes either an integer iteration or the literal "release".
    if tag == "release":
        candidate = checkpoint_path / "release"
    else:
        try:
            candidate = checkpoint_path / f"iter_{int(tag):07d}"
        except ValueError:
            return checkpoint_path
    return candidate if candidate.is_dir() else checkpoint_path


def has_modelopt_state(checkpoint_path) -> bool:
    """Whether `checkpoint_path` carries ModelOpt state.

    Returns `False` when it demonstrably does not -- which is the answer for
    every checkpoint in this campaign, verified against the real
    `Llama-3.1-8B_torch_dist` (0 `modelopt_state` directories).

    **Raises when it does.** Neither other answer is defensible: returning
    `False` would silently skip ModelOpt setup and train a model that is not the
    one on disk, and returning `True` sends Megatron into code that needs more of
    `megatron.post_training` than this file supplies, failing later and less
    clearly. Raising names the missing package while the checkpoint path is
    still in hand.
    """
    if checkpoint_path is None:
        return False
    path = Path(checkpoint_path)
    if not path.exists():
        # Megatron only calls this with args.load set, but a nonexistent path
        # is not a ModelOpt checkpoint, and raising inside a Ray actor for it
        # would obscure the real "checkpoint missing" error that follows.
        return False

    for candidate in {path, _load_dir(path)}:
        if candidate is not None and (candidate / "modelopt_state").is_dir():
            raise RuntimeError(_MISSING_PACKAGE_ERROR.format(path=path))
    return False


def install_if_missing() -> bool:
    """Register the shim unless the genuine package is importable.

    Returns True if it installed something. Never shadows a real installation:
    if Megatron-LM's `post_training` is ever added to this environment, that one
    wins and this becomes a no-op.
    """
    if MODULE_NAME in sys.modules:
        return False
    try:  # the real thing, if this env ever grows it
        __import__(MODULE_NAME)
    except ImportError:
        pass
    else:
        return False

    package = sys.modules.get(_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(_PACKAGE_NAME)
        package.__path__ = []  # a package, so submodule imports resolve
        sys.modules[_PACKAGE_NAME] = package

    module = types.ModuleType(MODULE_NAME)
    module.has_modelopt_state = has_modelopt_state
    module.__doc__ = __doc__
    sys.modules[MODULE_NAME] = module
    package.checkpointing = module
    logger.debug(
        "installed orbit's %s shim; megatron.post_training is not available in "
        "this environment",
        MODULE_NAME,
    )
    return True
