import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[4]
_MULTI_LORA_UTILS_PATH = _REPO_ROOT / "orbit" / "backends" / "megatron_utils" / "multi_lora_utils.py"


class _RecordingMultiLoRA:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _LoRA:
    pass


def _load_multi_lora_utils(monkeypatch):
    stubs = {
        "orbit.backends.training_utils.parallel": {"get_parallel_state": lambda: None},
        "orbit.ray.multi_lora.controller": {"get_multi_lora_controller": lambda: None},
        "orbit.utils.adapter_config": {"AdapterRun": object},
        "orbit.utils.distributed_utils": {"get_gloo_group": lambda: None},
        "orbit.backends.megatron_utils.lora_utils": {
            "convert_target_modules_to_megatron": lambda modules, lora_type=None: list(modules)
        },
        "megatron.bridge.peft.multi_lora": {"MultiLoRA": _RecordingMultiLoRA},
        "megatron.bridge.peft.lora": {"LoRA": _LoRA},
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "orbit.backends.megatron_utils.multi_lora_utils",
        _MULTI_LORA_UTILS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_multi_lora_uses_lowercase_cli_init_attribute(monkeypatch):
    module = _load_multi_lora_utils(monkeypatch)
    args = Namespace(
        target_modules=["q_proj"],
        multi_lora_n_adapters=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_type="lora",
        lora_a_init_method="uniform",
    )

    module.create_multi_lora_instance(args)

    assert _RecordingMultiLoRA.last_kwargs["lora_A_init_method"] == "uniform"
