import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools import orthomerge_bridge as bridge


def _adapter(path: Path, *, seed: int, name: str = "adapter", block_size: int = 4) -> Path:
    path.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.oft_R.weight": torch.randn(2, 6, generator=g),
            "extra.scalar": torch.tensor([float(seed)]),
        },
        str(path / "adapter_model.safetensors"),
    )
    (path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "OFT",
        "oft_type": "canonical_oft",
        "oft_block_size": block_size,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "base/model",
        "name": name,
    }))
    return path


def test_summarize_adapter_reports_keys_shapes_and_finiteness(tmp_path):
    adapter = _adapter(tmp_path / "a", seed=1)
    summary = bridge.summarize_adapter(adapter)
    assert summary["path"] == str(adapter)
    assert summary["num_tensors"] == 2
    assert summary["num_oft_tensors"] == 1
    assert summary["all_finite"] is True
    assert summary["tensors"]["base_model.model.layers.0.self_attn.q_proj.oft_R.weight"]["shape"] == [2, 6]


def test_compare_adapters_reports_zero_diff_for_identical_dirs(tmp_path):
    adapter = _adapter(tmp_path / "a", seed=2)
    report = bridge.compare_adapters(adapter, adapter)
    assert report["same_keys"] is True
    assert report["global_max_abs_diff"] == 0.0
    assert report["num_different_tensors"] == 0


def test_compare_adapters_reports_nonzero_diff_for_different_dirs(tmp_path):
    reference = _adapter(tmp_path / "a", seed=21)
    candidate = _adapter(tmp_path / "b", seed=22)
    report = bridge.compare_adapters(reference, candidate)
    assert report["same_keys"] is True
    assert report["global_max_abs_diff"] > 0.0
    assert report["num_different_tensors"] > 0


def test_write_manifest_uses_stable_adapter_order(tmp_path):
    a = _adapter(tmp_path / "b_task", seed=3)
    b = _adapter(tmp_path / "a_task", seed=4)
    out = tmp_path / "manifest.json"
    manifest = bridge.write_manifest([a, b], out)
    assert [item["name"] for item in manifest["adapters"]] == ["a_task", "b_task"]
    loaded = json.loads(out.read_text())
    assert loaded == manifest


def test_build_reference_command_uses_original_script_and_just_merge_adapter(tmp_path):
    original_repo = Path("/fast/zqiu/NeckariumAI/clthegoat/dev/OrthoMerge")
    adapters = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    cmd = bridge.build_reference_command(
        original_repo=original_repo,
        base_model="base/model",
        adapters=adapters,
        output_dir=tmp_path / "out",
        gpu=0,
    )
    assert str(original_repo / "merge" / "OrthoMerge_OFT_models.py") in cmd
    assert "--just_merge_adapter" in cmd
    assert "--adapter_paths" in cmd
    assert str(tmp_path / "out") in cmd


def test_bin_loading_uses_weights_only_when_supported(tmp_path, monkeypatch):
    adapter = tmp_path / "bin_adapter"
    adapter.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"placeholder")
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"x.oft_R.weight": torch.ones(1)}

    monkeypatch.setattr(bridge.torch, "load", fake_load)
    state = bridge.load_adapter_state(adapter)
    assert list(state) == ["x.oft_R.weight"]
    assert calls == [(str(adapter / "adapter_model.bin"), {"map_location": "cpu", "weights_only": True})]


def test_non_finite_summary_writes_strict_json(tmp_path):
    adapter = tmp_path / "nonfinite"
    adapter.mkdir()
    save_file(
        {"x.oft_R.weight": torch.tensor([float("nan"), float("inf")])},
        str(adapter / "adapter_model.safetensors"),
    )
    summary = bridge.summarize_adapter(adapter)
    out = tmp_path / "summary.json"
    bridge._write_json(summary, out)
    text = out.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    loaded = json.loads(text)
    assert loaded["all_finite"] is False
    assert loaded["tensors"]["x.oft_R.weight"]["max_abs_finite"] is False


def test_run_reference_rejects_non_32_block_size_before_subprocess(tmp_path, monkeypatch):
    a = _adapter(tmp_path / "a", seed=31, block_size=4)
    b = _adapter(tmp_path / "b", seed=32, block_size=4)
    manifest = bridge.write_manifest([a, b], tmp_path / "manifest.json")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess should not be invoked")

    monkeypatch.setattr(bridge.subprocess, "run", fail_if_called)
    with pytest.raises(ValueError, match=r"block_size=32.*a"):
        bridge.main([
            "run-reference",
            "--manifest", str(tmp_path / "manifest.json"),
            "--original-repo", str(tmp_path / "OrthoMerge"),
            "--output", str(tmp_path / "out"),
        ])
    assert manifest["base_model"] == "base/model"
