import json
from argparse import Namespace

import pytest

from orbit.arguments import _normalize_peft_args


def _write_adapter_config(adapter_dir, **overrides) -> None:
    config = {"peft_type": "OFT", "oft_block_size": 64}
    config.update(overrides)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))


def _oft_args(adapter_dir, **overrides) -> Namespace:
    values = {
        "peft_method": "oft",
        "target_modules": "q_proj",
        "exclude_modules": None,
        "peft_adapter_path": None,
        "lora_rank": 0,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_type": "lora",
        "lora_adapter_path": None,
        "lora_sync_from_tensor": False,
        "lora_a_init_method": "xavier",
        "oft_type": "canonical_oft",
        "oft_block_size": 0,
        "oft_coft": False,
        "oft_eps": 1e-5,
        "oft_block_share": False,
        "oft_adapter_path": str(adapter_dir),
        "adapter_double_buffer": False,
        "peft_distributed_transport": "nccl",
        "colocate": True,
        "megatron_to_hf_mode": "bridge",
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("path_argument", ["oft_adapter_path", "peft_adapter_path"])
def test_oft_adapter_hydrates_block_size_for_direct_and_alias_paths(tmp_path, path_argument):
    _write_adapter_config(tmp_path)
    args = _oft_args(tmp_path, oft_adapter_path=None)
    setattr(args, path_argument, str(tmp_path))

    result = _normalize_peft_args(args)

    assert result.oft_adapter_path == str(tmp_path)
    assert result.oft_block_size == 64


def test_oft_adapter_accepts_matching_explicit_block_size(tmp_path):
    _write_adapter_config(tmp_path)

    result = _normalize_peft_args(_oft_args(tmp_path, oft_block_size=64))

    assert result.oft_block_size == 64


def test_oft_adapter_rejects_conflicting_explicit_block_size(tmp_path):
    _write_adapter_config(tmp_path)

    with pytest.raises(ValueError, match="conflicts"):
        _normalize_peft_args(_oft_args(tmp_path, oft_block_size=32))


@pytest.mark.parametrize("config_value", [pytest.param(None, id="missing"), "64", 0, -1, True])
def test_oft_adapter_rejects_missing_non_integer_or_nonpositive_block_size(tmp_path, config_value):
    config = {"peft_type": "OFT"}
    if config_value is not None:
        config["oft_block_size"] = config_value
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="oft_block_size"):
        _normalize_peft_args(_oft_args(tmp_path))


def test_oft_adapter_preserves_peft_type_validation(tmp_path):
    _write_adapter_config(tmp_path, peft_type="LORA")

    with pytest.raises(ValueError, match="peft_type=LORA, expected OFT"):
        _normalize_peft_args(_oft_args(tmp_path))
