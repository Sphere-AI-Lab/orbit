from pathlib import Path

import pytest
import torch

import miles.merge  # noqa: F401  (registers strategies)
from miles.merge.megatron_io import (
    list_megatron_shards,
    merge_megatron_adapters,
    write_megatron_adapter,
)
from miles.merge.oft_merge import magnitude_corrected_merge

_KEY = "module.module.decoder.layers.0.self_attention.linear_proj.adapter.oft_r"


def _write_meg_adapter(path, seed, nblocks=2, block_size=4):
    path.mkdir(parents=True, exist_ok=True)
    P = block_size * (block_size - 1) // 2
    g = torch.Generator().manual_seed(seed)
    sd = {_KEY: torch.randn(nblocks, P, generator=g)}
    torch.save(sd, path / "adapter_megatron_tp0_pp0.pt")
    (path / "adapter_config.json").write_text('{"peft_type":"OFT","oft_block_size":4}')
    return path


def test_list_shards(tmp_path):
    a = _write_meg_adapter(tmp_path / "a", 1)
    assert list_megatron_shards(str(a)) == ["adapter_megatron_tp0_pp0.pt"]


def test_list_shards_missing_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        list_megatron_shards(str(tmp_path / "empty"))


def test_merge_matches_core(tmp_path):
    a = _write_meg_adapter(tmp_path / "a", 1)
    b = _write_meg_adapter(tmp_path / "b", 2)
    merged = merge_megatron_adapters([str(a), str(b)])
    shard = "adapter_megatron_tp0_pp0.pt"
    va = torch.load(a / shard, weights_only=True)[_KEY]
    vb = torch.load(b / shard, weights_only=True)[_KEY]
    assert torch.allclose(merged[shard][_KEY], magnitude_corrected_merge([va, vb]), atol=1e-6)


def test_merge_preserves_vpp_chunk_aware_native_keys(tmp_path):
    key = (1, _KEY)
    a = _write_meg_adapter(tmp_path / "a", 1)
    b = _write_meg_adapter(tmp_path / "b", 2)
    shard = "adapter_megatron_tp0_pp0.pt"
    state_a = torch.load(a / shard, weights_only=True)
    state_b = torch.load(b / shard, weights_only=True)
    torch.save({key: state_a[_KEY]}, a / shard)
    torch.save({key: state_b[_KEY]}, b / shard)

    merged = merge_megatron_adapters([str(a), str(b)])

    assert set(merged[shard]) == {key}
    assert torch.allclose(
        merged[shard][key],
        magnitude_corrected_merge([state_a[_KEY], state_b[_KEY]]),
        atol=1e-6,
    )


def test_merge_rejects_shard_set_mismatch(tmp_path):
    a = _write_meg_adapter(tmp_path / "a", 1)
    b = tmp_path / "b"
    b.mkdir()
    torch.save({_KEY: torch.randn(2, 6)}, b / "adapter_megatron_tp0_pp1.pt")  # different (tp,pp)
    (b / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="shard"):
        merge_megatron_adapters([str(a), str(b)])


def test_write_roundtrip(tmp_path):
    a = _write_meg_adapter(tmp_path / "a", 1)
    b = _write_meg_adapter(tmp_path / "b", 2)
    merged = merge_megatron_adapters([str(a), str(b)])
    out = write_megatron_adapter(merged, str(a), str(tmp_path / "out"))
    assert (Path(out) / "adapter_megatron_tp0_pp0.pt").exists()
    assert (Path(out) / "adapter_config.json").exists()
