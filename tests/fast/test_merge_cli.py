import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

import tools.merge_oft_adapters as cli
from orbit.merge.oft_merge import magnitude_corrected_merge, orthomerge_original_merge


def _write_adapter(
    path,
    *,
    block_size=4,
    num_blocks=3,
    base="base/X",
    seed=0,
    targets=None,
    state_dict=None,
):
    path.mkdir(parents=True, exist_ok=True)
    P = block_size * (block_size - 1) // 2
    g = torch.Generator().manual_seed(seed)
    sd = state_dict or {
        "base_model.model.layers.0.self_attn.q_proj.oft_R.weight": torch.randn(num_blocks, P, generator=g)
    }
    save_file(sd, str(path / "adapter_model.safetensors"))
    cfg = {
        "peft_type": "OFT", "oft_type": "canonical_oft", "oft_block_size": block_size,
        "target_modules": targets or ["q_proj"], "base_model_name_or_path": base,
    }
    (path / "adapter_config.json").write_text(json.dumps(cfg))
    return path


def test_validate_accepts_matching(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=1)
    b = _write_adapter(tmp_path / "b", seed=2)
    cfg = cli.validate_adapters([str(a), str(b)])
    assert cfg["oft_block_size"] == 4


def test_validate_accepts_reordered_target_modules(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=1, targets=["q_proj", "k_proj"])
    b = _write_adapter(tmp_path / "b", seed=2, targets=["k_proj", "q_proj"])
    cfg = cli.validate_adapters([str(a), str(b)])
    assert cfg["target_modules"] == ["q_proj", "k_proj"]


def test_validate_rejects_different_target_modules(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=1, targets=["q_proj", "k_proj"])
    b = _write_adapter(tmp_path / "b", seed=2, targets=["q_proj", "v_proj"])
    with pytest.raises(ValueError, match="target_modules"):
        cli.validate_adapters([str(a), str(b)])


def test_validate_rejects_block_size_mismatch(tmp_path):
    a = _write_adapter(tmp_path / "a", block_size=4, seed=1)
    b = _write_adapter(tmp_path / "b", block_size=8, seed=2)
    with pytest.raises(ValueError, match="oft_block_size"):
        cli.validate_adapters([str(a), str(b)])


def test_validate_rejects_single_adapter(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=1)
    with pytest.raises(ValueError, match="at least 2"):
        cli.validate_adapters([str(a)])


def test_main_end_to_end_writes_valid_merged_adapter(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=1)
    b = _write_adapter(tmp_path / "b", seed=2)
    out = tmp_path / "out"
    rc = cli.main([
        "--adapters", str(a), str(b),
        "--output", str(out),
    ])
    assert rc == 0
    merged_dir = out / "merged_adapter"
    assert (merged_dir / "adapter_model.safetensors").exists()
    assert (merged_dir / "adapter_config.json").exists()
    # output is a valid OFT adapter
    cli.read_oft_config(str(merged_dir))
    # merged tensor equals the magnitude-corrected merge of inputs
    k = "base_model.model.layers.0.self_attn.q_proj.oft_R.weight"
    got = load_file(str(merged_dir / "adapter_model.safetensors"))[k]
    exp = magnitude_corrected_merge([load_file(str(a / "adapter_model.safetensors"))[k],
                                     load_file(str(b / "adapter_model.safetensors"))[k]])
    assert torch.allclose(got, exp, atol=1e-6)


def test_main_save_megatron_writes_shard(tmp_path):
    # _write_adapter (defined earlier in this file) writes adapter_model.safetensors + config.
    # Add a Megatron-native shard to each input so --save-megatron has something to merge.
    a = _write_adapter(tmp_path / "a", seed=1)
    b = _write_adapter(tmp_path / "b", seed=2)
    key = "module.module.decoder.layers.0.self_attention.linear_proj.adapter.oft_r"
    for d, s in ((a, 1), (b, 2)):
        g = torch.Generator().manual_seed(s)
        torch.save({key: torch.randn(3, 6, generator=g)}, d / "adapter_megatron_tp0_pp0.pt")
    out = tmp_path / "out"
    rc = cli.main(["--adapters", str(a), str(b), "--output", str(out), "--save-megatron"])
    assert rc == 0
    assert (out / "merged_megatron" / "adapter_megatron_tp0_pp0.pt").exists()
    assert (out / "merged_megatron" / "adapter_config.json").exists()


def test_main_end_to_end_with_oft_original_method_writes_reference_merge(tmp_path):
    k = "decoder.layers.0.mlp.experts.w1_oft_r"
    inputs = []
    adapters = []
    for name, seed in (("a", 31), ("b", 32), ("c", 33)):
        tensor = torch.randn(2, 3, 6, generator=torch.Generator().manual_seed(seed))
        inputs.append(tensor)
        adapters.append(_write_adapter(tmp_path / name, state_dict={k: tensor}))
    out = tmp_path / "out"
    rc = cli.main([
        "--adapters", *(str(adapter) for adapter in adapters),
        "--output", str(out),
        "--method", "oft-original",
    ])
    assert rc == 0
    merged_dir = out / "merged_adapter"
    got = load_file(str(merged_dir / "adapter_model.safetensors"))
    expected = orthomerge_original_merge(inputs)
    default_oft_for_dsv4_key = torch.stack([tensor.float() for tensor in inputs]).mean(0)
    assert torch.allclose(got[k], expected, atol=1e-6)
    assert not torch.allclose(got[k], default_oft_for_dsv4_key, atol=1e-6)


def test_script_uses_worktree_orbit_package_for_oft_original(tmp_path):
    k = "decoder.layers.0.mlp.experts.w1_oft_r"
    inputs = []
    adapters = []
    for name, seed in (("a", 41), ("b", 42)):
        tensor = torch.randn(2, 3, 6, generator=torch.Generator().manual_seed(seed))
        inputs.append(tensor)
        adapters.append(_write_adapter(tmp_path / name, state_dict={k: tensor}))

    stale_site = tmp_path / "stale_site"
    (stale_site / "orbit" / "merge").mkdir(parents=True)
    (stale_site / "miles" / "utils").mkdir(parents=True)
    (stale_site / "orbit" / "__init__.py").write_text("")
    (stale_site / "miles" / "__init__.py").write_text("")
    (stale_site / "orbit" / "merge" / "__init__.py").write_text(
        """
class _Strategy:
    def merge(self, state_dicts, weights=None):
        return state_dicts[0]

def get_strategy(method):
    if method in ("oft", "oft-naive"):
        return _Strategy()
    raise KeyError(
        f"unknown merge method {method!r}; available: ['oft', 'oft-naive']"
    )
"""
    )
    (stale_site / "miles" / "utils" / "__init__.py").write_text("")
    (stale_site / "miles" / "utils" / "logging_utils.py").write_text(
        "def configure_logger(*args, **kwargs):\n    return None\n"
    )

    out = tmp_path / "out"
    cwd = tmp_path / "run_from_elsewhere"
    cwd.mkdir()
    env = os.environ.copy()
    repo_root = Path(cli.__file__).resolve().parents[1]
    pythonpath_entries = [str(stale_site), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(cli.__file__).resolve()),
            "--adapters",
            *(str(adapter) for adapter in adapters),
            "--output",
            str(out),
            "--method",
            "oft-original",
        ],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    merged_dir = out / "merged_adapter"
    assert (merged_dir / "adapter_model.safetensors").exists()
    got = load_file(str(merged_dir / "adapter_model.safetensors"))
    expected = orthomerge_original_merge(inputs)
    default_oft_for_dsv4_key = torch.stack([tensor.float() for tensor in inputs]).mean(0)
    assert torch.allclose(got[k], expected, atol=1e-6)
    assert not torch.allclose(got[k], default_oft_for_dsv4_key, atol=1e-6)
