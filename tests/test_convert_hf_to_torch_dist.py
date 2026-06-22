import sys
import types
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_hf_to_megatron_disables_gradient_accumulation_fusion(monkeypatch):
    calls = {}

    class FakeProvider:
        gradient_accumulation_fusion = True

        def finalize(self):
            calls["finalized"] = True

        def provide_distributed_model(self, **kwargs):
            calls["provide_kwargs"] = kwargs
            calls["gradient_accumulation_fusion"] = self.gradient_accumulation_fusion
            return ["model"]

    class FakeModelBridge:
        def get_hf_tokenizer_kwargs(self):
            return {"existing": True}

    class FakeBridge:
        _model_bridge = FakeModelBridge()

        def to_megatron_provider(self):
            calls["provider_created"] = True
            return FakeProvider()

        def save_megatron_model(self, model, path, **kwargs):
            calls["saved"] = (model, path, kwargs)

    class FakeAutoBridge:
        @staticmethod
        def from_hf_pretrained(path, **kwargs):
            calls["from_hf_pretrained"] = (path, kwargs)
            return FakeBridge()

    fake_megatron = types.ModuleType("megatron")
    fake_bridge_module = types.ModuleType("megatron.bridge")
    fake_bridge_module.AutoBridge = FakeAutoBridge
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.bridge", fake_bridge_module)

    from orbit_plugins.megatron_bridge.patches.conversion.convert_checkpoints import import_hf_to_megatron

    import_hf_to_megatron(
        hf_model="/tmp/hf",
        megatron_path="/tmp/megatron",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )

    assert calls["provider_created"] is True
    assert calls["finalized"] is True
    assert calls["provide_kwargs"] == {"wrap_with_ddp": False, "use_cpu_initialization": True}
    assert calls["gradient_accumulation_fusion"] is False
    assert calls["saved"] == (
        ["model"],
        "/tmp/megatron",
        {
            "hf_tokenizer_path": "/tmp/hf",
            "hf_tokenizer_kwargs": {"existing": True, "trust_remote_code": True},
            "low_memory_save": True,
        },
    )


def test_legacy_cli_prefers_repo_root_when_executed_from_tools(monkeypatch):
    script = REPO_ROOT / "tools" / "convert_hf_to_torch_dist.py"
    original_path = [p for p in sys.path if p not in {str(REPO_ROOT), str(REPO_ROOT / "tools")}]
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT / "tools"), *original_path])

    spec = importlib.util.spec_from_file_location("legacy_convert_hf_to_torch_dist_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert sys.path[0] == str(REPO_ROOT)
