"""Regression test: --lora-a-init-method must actually reach Megatron-Bridge's
adapter constructor under the exact keyword name Bridge expects.

Why this file exists (and is not tests/fast/backends/megatron_utils/test_lora_utils.py):
importing `miles.backends.megatron_utils` for real runs its `__init__.py`, which
unconditionally `import deep_ep` (CUDA-only, raises AssertionError via
find_cuda_home() on this box), and `lora_utils.py` depends on `.peft_utils`,
which does `from megatron.core import mpu` at module scope (needs a sourced
CUDA env). Both make the real package uncollectable in this bare CPU venv --
that's exactly why tests/fast/backends/megatron_utils/test_lora_utils.py is a
pre-existing collection error here.

Instead of importing the package, this test stubs those two dependencies in
sys.modules and loads the real miles/backends/megatron_utils/lora_utils.py
straight from disk by file path, so create_lora_instance's actual logic
executes completely unmodified -- only its imports are faked. Megatron-Bridge's
`megatron.bridge.peft.lora.LoRA` / `canonical_lora.CanonicalLoRA` are stubbed
too (create_lora_instance imports them lazily inside the function), recording
whatever kwargs they're constructed with.
"""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LORA_UTILS_PATH = _REPO_ROOT / "miles" / "backends" / "megatron_utils" / "lora_utils.py"


def _install_stub_miles_backends_megatron_utils_package(monkeypatch):
    # Prevent the real miles/backends/megatron_utils/__init__.py from running --
    # it unconditionally imports deep_ep, which raises AssertionError
    # (find_cuda_home) on a box with no CUDA toolchain.
    pkg = ModuleType("miles.backends.megatron_utils")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils", pkg)


def _install_stub_peft_utils(monkeypatch):
    # The real peft_utils.py does `from megatron.core import mpu` at module
    # scope, which needs a sourced CUDA env. Fake just the names lora_utils.py
    # imports from it; create_lora_instance only actually calls the first two.
    stub = ModuleType("miles.backends.megatron_utils.peft_utils")

    # lora_utils imports this for a type annotation only
    # (checkpoint_preflight: PeftCheckpointPreflight | None). The name still has
    # to exist at import time, and the stub package's __path__ is empty, so a
    # miss surfaces as "cannot import name ... (unknown location)" rather than
    # anything pointing at the annotation.
    class _PeftCheckpointPreflight:
        pass

    stub.PeftCheckpointPreflight = _PeftCheckpointPreflight
    stub.convert_target_modules_to_hf = lambda *a, **k: None
    stub.convert_target_modules_to_megatron = lambda target_modules, variant=None: list(target_modules)
    stub.get_peft_method = lambda args: getattr(args, "peft_method", "none")
    stub.is_adapter_param_name = lambda *a, **k: False
    stub.load_peft_adapter_checkpoint = lambda *a, **k: None
    stub.parse_exclude_modules = lambda *a, **k: None
    stub.resolve_target_modules_hf = lambda *a, **k: []
    stub.save_peft_adapter_checkpoint = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.peft_utils", stub)


class _RecordingLoRA:
    """Stand-in for megatron.bridge.peft.lora.LoRA; records constructor kwargs."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _RecordingCanonicalLoRA:
    """Stand-in for megatron.bridge.peft.canonical_lora.CanonicalLoRA."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


def _install_stub_bridge_peft(monkeypatch):
    _RecordingLoRA.last_kwargs = None
    _RecordingCanonicalLoRA.last_kwargs = None

    lora_module = ModuleType("megatron.bridge.peft.lora")
    lora_module.LoRA = _RecordingLoRA
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.lora", lora_module)

    canonical_module = ModuleType("megatron.bridge.peft.canonical_lora")
    canonical_module.CanonicalLoRA = _RecordingCanonicalLoRA
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.canonical_lora", canonical_module)


def _load_real_lora_utils(monkeypatch):
    _install_stub_miles_backends_megatron_utils_package(monkeypatch)
    _install_stub_peft_utils(monkeypatch)
    spec = importlib.util.spec_from_file_location("miles.backends.megatron_utils.lora_utils", _LORA_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.lora_utils", module)
    spec.loader.exec_module(module)
    return module


def _make_args(**overrides):
    args = {
        "target_modules": ["q_proj"],
        "exclude_modules": None,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_type": "lora",
        "multi_latent_attention": False,
    }
    args.update(overrides)
    return Namespace(**args)


def test_lora_a_init_method_reaches_bridge_as_capital_a_kwarg(monkeypatch):
    """The exact silent-failure mode this task exists to prevent: Miles's CLI
    landing attribute is lowercase (`lora_a_init_method`, produced by argparse
    from `--lora-a-init-method`), but Megatron-Bridge's LoRA dataclass field is
    capital-A (`lora_A_init_method`). If lora_utils.py's getattr key ever drifts
    back to the capital-A spelling (the original bug), or a future refactor
    typos the CLI-facing key, the value silently stops reaching Bridge and every
    run falls back to Bridge's own "xavier" default -- with no error anywhere.
    This test fails loudly instead.
    """
    _install_stub_bridge_peft(monkeypatch)
    lora_utils = _load_real_lora_utils(monkeypatch)

    args = _make_args(lora_a_init_method="kaiming")
    lora_utils.create_lora_instance(args)

    assert _RecordingLoRA.last_kwargs is not None, "LoRA() was never constructed"
    assert (
        "lora_A_init_method" in _RecordingLoRA.last_kwargs
    ), "create_lora_instance did not pass lora_A_init_method (capital A) to Bridge's LoRA"
    assert _RecordingLoRA.last_kwargs["lora_A_init_method"] == "kaiming"
    # The lowercase CLI attribute name must never leak through as the kwarg name --
    # that mismatch is the exact silent-failure mode this test guards against.
    assert "lora_a_init_method" not in _RecordingLoRA.last_kwargs


def test_lora_a_init_method_falls_back_to_xavier_when_unset(monkeypatch):
    _install_stub_bridge_peft(monkeypatch)
    lora_utils = _load_real_lora_utils(monkeypatch)

    args = _make_args()  # no lora_a_init_method attribute at all
    lora_utils.create_lora_instance(args)

    assert _RecordingLoRA.last_kwargs["lora_A_init_method"] == "xavier"
