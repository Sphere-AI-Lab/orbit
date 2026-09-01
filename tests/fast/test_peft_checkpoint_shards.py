import sys
from argparse import Namespace
from contextlib import nullcontext
from types import ModuleType

import pytest
import torch

import orbit.megatron.peft_utils as peft_utils


class _LoRAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(1))


class _OFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1_oft_r = torch.nn.Parameter(torch.zeros(1))


def _patch_distributed_rank(
    monkeypatch,
    *,
    global_rank=6,
    tp_rank=1,
    tp_size=2,
    pp_rank=2,
    ep_rank=3,
    ep_size=4,
    etp_rank=1,
    etp_size=2,
):
    monkeypatch.setattr(peft_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(peft_utils.dist, "get_rank", lambda: global_rank)
    monkeypatch.setattr(peft_utils.dist, "barrier", lambda: None)
    monkeypatch.setattr(peft_utils.mpu, "get_tensor_model_parallel_rank", lambda: tp_rank, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_pipeline_model_parallel_rank", lambda: pp_rank, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_expert_model_parallel_rank", lambda: ep_rank, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_expert_model_parallel_world_size", lambda: ep_size, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_expert_tensor_parallel_rank", lambda: etp_rank, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_expert_tensor_parallel_world_size", lambda: etp_size, raising=False)
    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", lambda value: [value])


def _patch_bridge_export(monkeypatch):
    class _Bridge:
        @classmethod
        def from_hf_pretrained(cls, *_args, **_kwargs):
            return cls()

        def export_adapter_weights(self, *_args, **_kwargs):
            return ()

        def export_oft_adapter_weights(self, *_args, **_kwargs):
            return ()

    bridge_module = ModuleType("megatron.bridge")
    bridge_module.AutoBridge = _Bridge
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_module)
    conversion_module = ModuleType("megatron.bridge.orbit.conversion")
    conversion_module.__path__ = []
    oft_export_module = ModuleType("megatron.bridge.orbit.conversion.oft_export")
    oft_export_module.export_oft_adapter_weights = lambda *_args, **_kwargs: ()
    monkeypatch.setitem(sys.modules, "megatron.bridge.orbit.conversion", conversion_module)
    monkeypatch.setitem(sys.modules, "megatron.bridge.orbit.conversion.oft_export", oft_export_module)

    from miles.utils import megatron_bridge_utils

    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", lambda _model: nullcontext())
    monkeypatch.setattr(peft_utils, "_save_peft_hf_artifacts", lambda *_args, **_kwargs: None)


def _args(method):
    return Namespace(
        peft_method=method,
        hf_checkpoint="base",
        no_save_optim=True,
        target_modules=["linear_qkv"],
        peft_variant="standard",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        oft_type="oft",
        oft_block_size=4,
        oft_coft=False,
        oft_eps=1e-5,
        oft_block_share=False,
    )


@pytest.mark.parametrize(
    ("tp_rank", "tp_size", "ep_rank", "ep_size", "etp_rank", "etp_size", "expected"),
    [
        (1, 2, 0, 1, 1, 2, "adapter_megatron_tp1_pp2.pt"),
        (1, 2, 3, 4, 1, 2, "adapter_megatron_tp1_pp2_ep3.pt"),
        (0, 1, 0, 1, 1, 2, "adapter_megatron_tp0_pp2_etp1.pt"),
        (0, 1, 3, 4, 1, 2, "adapter_megatron_tp0_pp2_ep3_etp1.pt"),
        (1, 2, 0, 1, 0, 1, "adapter_megatron_tp1_pp2.pt"),
        (3, 4, 0, 1, 1, 2, "adapter_megatron_tp3_pp2.pt"),
    ],
)
def test_native_shard_name_preserves_legacy_name_only_for_redundant_expert_axes(
    tp_rank,
    tp_size,
    ep_rank,
    ep_size,
    etp_rank,
    etp_size,
    expected,
):
    assert (
        peft_utils._native_adapter_shard_name(
            tp_rank,
            2,
            ep_rank,
            ep_size,
            etp_rank,
            etp_size,
            tp_size,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("global_rank", "ep_rank", "expected_writer"),
    [
        (4, 0, True),
        (7, 0, False),
        (5, 1, True),
        (8, 1, False),
    ],
)
def test_native_writer_is_unique_per_realized_tp_pp_ep_coordinate(
    monkeypatch,
    global_rank,
    ep_rank,
    expected_writer,
):
    _patch_distributed_rank(
        monkeypatch,
        global_rank=global_rank,
        tp_rank=0,
        etp_rank=0,
        pp_rank=0,
        ep_rank=ep_rank,
        ep_size=2,
    )
    gathered = [
        ((0, 0, 0, 0), 4),
        ((0, 0, 1, 0), 5),
        ((0, 0, 0, 0), 7),
        ((0, 0, 1, 0), 8),
    ]
    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", lambda _value: gathered)

    roles = peft_utils._resolve_peft_save_rank_roles()

    assert roles.native_writer is expected_writer


@pytest.mark.parametrize(
    ("global_rank", "etp_rank", "expected_writer"),
    [
        (4, 0, True),
        (5, 1, True),
        (7, 0, False),
        (8, 1, False),
    ],
)
def test_native_writer_is_unique_per_realized_etp_coordinate(
    monkeypatch,
    global_rank,
    etp_rank,
    expected_writer,
):
    _patch_distributed_rank(
        monkeypatch,
        global_rank=global_rank,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        ep_rank=0,
        ep_size=1,
        etp_rank=etp_rank,
        etp_size=2,
    )
    gathered = [
        ((0, 0, 0, 0), 4),
        ((0, 0, 0, 1), 5),
        ((0, 0, 0, 0), 7),
        ((0, 0, 0, 1), 8),
    ]
    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", lambda _value: gathered)

    roles = peft_utils._resolve_peft_save_rank_roles()

    assert roles.native_writer is expected_writer


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_saves_expert_parallel_coordinate_shard(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(monkeypatch)
    _patch_bridge_export(monkeypatch)

    peft_utils.save_peft_checkpoint([model()], _args(method), str(tmp_path))

    native_path = tmp_path / "adapter_megatron_tp1_pp2_ep3.pt"
    assert native_path.is_file()
    assert not (tmp_path / "adapter_megatron_tp1_pp2.pt").exists()
    assert set(torch.load(native_path, map_location="cpu", weights_only=True)) == {(0, adapter_key)}


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_saves_distinct_expert_tensor_parallel_coordinate_shard(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    _patch_bridge_export(monkeypatch)

    peft_utils.save_peft_checkpoint([model()], _args(method), str(tmp_path))

    native_path = tmp_path / "adapter_megatron_tp0_pp2_etp1.pt"
    assert native_path.is_file()
    assert not (tmp_path / "adapter_megatron_tp0_pp2.pt").exists()
    assert set(torch.load(native_path, map_location="cpu", weights_only=True)) == {(0, adapter_key)}


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_preserves_legacy_name_when_etp_divides_tp(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=1,
        tp_size=2,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )
    _patch_bridge_export(monkeypatch)

    peft_utils.save_peft_checkpoint([model()], _args(method), str(tmp_path))

    native_path = tmp_path / "adapter_megatron_tp1_pp2.pt"
    assert native_path.is_file()
    assert not (tmp_path / "adapter_megatron_tp1_pp2_etp0.pt").exists()
    assert set(torch.load(native_path, map_location="cpu", weights_only=True)) == {(0, adapter_key)}


def test_production_dispatch_elects_one_writer_per_coordinate(monkeypatch, tmp_path):
    _patch_distributed_rank(monkeypatch)
    _patch_bridge_export(monkeypatch)

    def gather_with_lower_rank_replica(value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], tuple):
            return [(value[0], 5), value]
        return [value, value]

    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", gather_with_lower_rank_replica)

    peft_utils.save_peft_checkpoint([_LoRAModel()], _args("lora"), str(tmp_path))

    assert not (tmp_path / "adapter_megatron_tp1_pp2_ep3.pt").exists()


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_loads_expert_parallel_coordinate_shard(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(monkeypatch)
    torch.save({(0, adapter_key): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2_ep3.pt")
    adapter_model = model()

    loaded, iteration = peft_utils.load_peft_adapter(
        [adapter_model],
        _args(method),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(getattr(adapter_model, adapter_key), torch.tensor([7.0]))


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_loads_distinct_expert_tensor_parallel_coordinate_shard(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    torch.save({(0, adapter_key): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp0_pp2_etp1.pt")
    adapter_model = model()

    loaded, iteration = peft_utils.load_peft_adapter(
        [adapter_model],
        _args(method),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(getattr(adapter_model, adapter_key), torch.tensor([7.0]))


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_loads_legacy_name_when_etp_divides_tp(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=1,
        tp_size=2,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )
    torch.save({(0, adapter_key): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")
    adapter_model = model()

    loaded, iteration = peft_utils.load_peft_adapter(
        [adapter_model],
        _args(method),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(getattr(adapter_model, adapter_key), torch.tensor([7.0]))


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_does_not_load_unsuffixed_shard_for_distinct_etp(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    torch.save({(0, adapter_key): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp0_pp2.pt")
    adapter_model = model()

    loaded, iteration = peft_utils.load_peft_adapter(
        [adapter_model],
        _args(method),
        str(tmp_path),
    )

    assert loaded is False
    assert iteration is None
    assert torch.equal(getattr(adapter_model, adapter_key), torch.zeros(1))


@pytest.mark.parametrize(
    ("method", "model", "adapter_key"),
    [
        ("lora", _LoRAModel, "lora_A"),
        ("oft", _OFTModel, "w1_oft_r"),
    ],
)
def test_production_dispatch_does_not_fall_back_to_unsuffixed_tp_pp_shard_with_ep(
    monkeypatch,
    tmp_path,
    method,
    model,
    adapter_key,
):
    _patch_distributed_rank(monkeypatch)
    torch.save({(0, adapter_key): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")
    adapter_model = model()

    loaded, iteration = peft_utils.load_peft_adapter(
        [adapter_model],
        _args(method),
        str(tmp_path),
    )

    assert loaded is False
    assert iteration is None
    assert torch.equal(getattr(adapter_model, adapter_key), torch.zeros(1))


@pytest.mark.parametrize(
    ("tp_rank", "tp_size", "ep_rank", "ep_size", "etp_rank", "etp_size"),
    [
        (1, 2, 0, 1, 1, 2),
        (1, 2, 3, 4, 1, 2),
        (0, 1, 0, 1, 1, 2),
        (1, 2, 0, 1, 0, 1),
    ],
)
def test_lora_production_dispatch_loads_legacy_global_rank_shard(
    monkeypatch,
    tmp_path,
    tp_rank,
    tp_size,
    ep_rank,
    ep_size,
    etp_rank,
    etp_size,
):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
        etp_rank=etp_rank,
        etp_size=etp_size,
    )
    torch.save({"lora_A": torch.tensor([7.0])}, tmp_path / "adapter_megatron_rank6.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter([model], _args("lora"), str(tmp_path))

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.lora_A, torch.tensor([7.0]))


def test_preflight_rejects_multiple_rank_local_native_candidates(monkeypatch, tmp_path):
    _patch_distributed_rank(monkeypatch)
    torch.save({(0, "lora_A"): torch.ones(1)}, tmp_path / "adapter_megatron_tp1_pp2_ep3.pt")
    torch.save({"lora_A": torch.ones(1)}, tmp_path / "adapter_megatron_rank6.pt")

    with pytest.raises(RuntimeError, match="multiple native adapter shards match this rank"):
        peft_utils.preflight_peft_adapter_checkpoint(tmp_path)


def test_preflight_rejects_mixed_native_layouts_across_ranks(monkeypatch, tmp_path):
    _patch_distributed_rank(monkeypatch)
    torch.save({"lora_A": torch.ones(1)}, tmp_path / "adapter_megatron_rank6.pt")

    def gather_with_mixed_layout(value):
        if isinstance(value, tuple) and len(value) == 5 and value[-1] == "global-rank":
            return [value, (*value[:-1], "coordinate")]
        return [value, value]

    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", gather_with_mixed_layout)

    with pytest.raises(RuntimeError, match="native adapter shard layouts differ across ranks"):
        peft_utils.preflight_peft_adapter_checkpoint(tmp_path)


def test_teacher_load_uses_expert_parallel_coordinate_shard(monkeypatch, tmp_path):
    _patch_distributed_rank(monkeypatch)
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2_ep3.pt")

    tensors = peft_utils.load_adapter_tensors_for_teacher([_LoRAModel()], str(tmp_path))

    assert torch.equal(tensors[(0, "lora_A")], torch.tensor([7.0]))


def test_teacher_load_uses_distinct_expert_tensor_parallel_coordinate_shard(monkeypatch, tmp_path):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp0_pp2_etp1.pt")

    tensors = peft_utils.load_adapter_tensors_for_teacher([_LoRAModel()], str(tmp_path))

    assert torch.equal(tensors[(0, "lora_A")], torch.tensor([7.0]))


def test_teacher_loads_legacy_name_when_etp_divides_tp(monkeypatch, tmp_path):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=1,
        tp_size=2,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")

    tensors = peft_utils.load_adapter_tensors_for_teacher([_LoRAModel()], str(tmp_path))

    assert torch.equal(tensors[(0, "lora_A")], torch.tensor([7.0]))


def test_teacher_load_does_not_use_unsuffixed_shard_for_distinct_etp(monkeypatch, tmp_path):
    _patch_distributed_rank(
        monkeypatch,
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp0_pp2.pt")

    with pytest.raises(FileNotFoundError, match="adapter_megatron_tp0_pp2_etp1.pt"):
        peft_utils.load_adapter_tensors_for_teacher([_LoRAModel()], str(tmp_path))


def test_teacher_load_does_not_fall_back_to_unsuffixed_tp_pp_shard_with_ep(monkeypatch, tmp_path):
    _patch_distributed_rank(monkeypatch)
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")

    with pytest.raises(FileNotFoundError, match="adapter_megatron_tp1_pp2_ep3.pt"):
        peft_utils.load_adapter_tensors_for_teacher([_LoRAModel()], str(tmp_path))
