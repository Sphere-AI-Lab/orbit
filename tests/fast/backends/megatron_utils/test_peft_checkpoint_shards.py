import sys
from argparse import Namespace
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

import orbit.backends.megatron_utils.peft_utils as peft_utils


def _parallel_state(
    *,
    tp_rank=0,
    tp_size=None,
    pp_rank=0,
    ep_rank=0,
    ep_size=1,
    etp_rank=None,
    etp_size=None,
    intra_dp_cp_rank=0,
):
    tp_size = max(tp_rank + 1, 1) if tp_size is None else tp_size
    etp_rank = tp_rank if etp_rank is None else etp_rank
    etp_size = tp_size if etp_size is None else etp_size
    return SimpleNamespace(
        intra_dp_cp=SimpleNamespace(rank=intra_dp_cp_rank),
        effective_dp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=tp_rank, size=tp_size),
        pp=SimpleNamespace(rank=pp_rank),
        ep=SimpleNamespace(rank=ep_rank, size=ep_size),
        etp=SimpleNamespace(rank=etp_rank, size=etp_size),
    )


def _patch_distributed_rank(monkeypatch, *, global_rank, parallel_state, gathered_coordinates):
    monkeypatch.setattr(peft_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(peft_utils.dist, "get_rank", lambda: global_rank)
    monkeypatch.setattr(peft_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", lambda _value: gathered_coordinates)


@pytest.mark.parametrize(
    ("global_rank", "ep_rank", "intra_dp_cp_rank", "expected_writer"),
    [
        (4, 0, 1, True),
        (7, 0, 0, False),
        (5, 1, 1, True),
        (8, 1, 0, False),
    ],
)
def test_native_writer_is_unique_per_realized_tp_pp_ep_coordinate(
    monkeypatch,
    global_rank,
    ep_rank,
    intra_dp_cp_rank,
    expected_writer,
):
    gathered = [
        ((0, 0, 0, 0), 4),
        ((0, 0, 1, 0), 5),
        ((0, 0, 0, 0), 7),
        ((0, 0, 1, 0), 8),
    ]
    state = _parallel_state(
        ep_rank=ep_rank,
        ep_size=2,
        intra_dp_cp_rank=intra_dp_cp_rank,
    )
    _patch_distributed_rank(
        monkeypatch,
        global_rank=global_rank,
        parallel_state=state,
        gathered_coordinates=gathered,
    )

    roles = peft_utils._resolve_peft_save_rank_roles()

    assert roles.native_writer is expected_writer


def test_native_save_names_expert_parallel_shard(monkeypatch, tmp_path):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=3, ep_size=4)
    _patch_distributed_rank(
        monkeypatch,
        global_rank=6,
        parallel_state=state,
        gathered_coordinates=[((1, 2, 3, 1), 6)],
    )
    monkeypatch.setattr(peft_utils, "native_adapter_state", lambda _model: {(0, "w1_oft_r"): torch.ones(1)})

    result = peft_utils._save_native_adapter_shard(
        [object()],
        tmp_path,
        peft_utils._resolve_peft_save_rank_roles(),
    )

    assert result is not None
    assert result[1] == tmp_path / "adapter_megatron_tp1_pp2_ep3.pt"
    assert result[1].is_file()


class _LoRAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(1))


def _patch_lora_dispatch(monkeypatch, state, *, global_rank=6):
    _patch_distributed_rank(
        monkeypatch,
        global_rank=global_rank,
        parallel_state=state,
        gathered_coordinates=None,
    )
    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", lambda value: [value])
    monkeypatch.setattr(peft_utils.dist, "barrier", lambda: None)


def _patch_lora_bridge(monkeypatch):
    class _Bridge:
        @classmethod
        def from_hf_pretrained(cls, *_args, **_kwargs):
            return cls()

        def export_adapter_weights(self, *_args, **_kwargs):
            return ()

    bridge_module = ModuleType("megatron.bridge")
    bridge_module.AutoBridge = _Bridge
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_module)

    from orbit.utils import megatron_bridge_utils

    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", lambda _model: nullcontext())


@pytest.mark.parametrize(
    ("ep_rank", "ep_size", "expected_name"),
    [
        (0, 1, "adapter_megatron_tp1_pp2.pt"),
        (3, 4, "adapter_megatron_tp1_pp2_ep3.pt"),
    ],
)
def test_lora_production_dispatch_saves_coordinate_shard(
    monkeypatch,
    tmp_path,
    ep_rank,
    ep_size,
    expected_name,
):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=ep_rank, ep_size=ep_size)
    _patch_lora_dispatch(monkeypatch, state)
    _patch_lora_bridge(monkeypatch)
    args = Namespace(peft_method="lora", hf_checkpoint="base", no_save_optim=True)

    peft_utils.save_peft_checkpoint(
        [_LoRAModel()],
        args,
        str(tmp_path),
    )

    native_path = tmp_path / expected_name
    assert native_path.is_file()
    assert not (tmp_path / "adapter_megatron_rank6.pt").exists()
    assert set(torch.load(native_path, map_location="cpu", weights_only=True)) == {(0, "lora_A")}


def test_lora_production_dispatch_saves_distinct_expert_tensor_parallel_shard(monkeypatch, tmp_path):
    state = _parallel_state(
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    _patch_lora_dispatch(monkeypatch, state)
    _patch_lora_bridge(monkeypatch)

    peft_utils.save_peft_checkpoint(
        [_LoRAModel()],
        Namespace(peft_method="lora", hf_checkpoint="base", no_save_optim=True),
        str(tmp_path),
    )

    native_path = tmp_path / "adapter_megatron_tp0_pp2_etp1.pt"
    assert native_path.is_file()
    assert set(torch.load(native_path, map_location="cpu", weights_only=True)) == {(0, "lora_A")}


def test_lora_production_dispatch_preserves_tp_shard_when_etp_is_redundant(monkeypatch, tmp_path):
    state = _parallel_state(
        tp_rank=1,
        tp_size=2,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )
    _patch_lora_dispatch(monkeypatch, state)
    _patch_lora_bridge(monkeypatch)

    peft_utils.save_peft_checkpoint(
        [_LoRAModel()],
        Namespace(peft_method="lora", hf_checkpoint="base", no_save_optim=True),
        str(tmp_path),
    )

    native_path = tmp_path / "adapter_megatron_tp1_pp2.pt"
    assert native_path.is_file()
    assert not (tmp_path / "adapter_megatron_tp1_pp2_etp0.pt").exists()


def test_lora_production_dispatch_elects_one_writer_per_coordinate(monkeypatch, tmp_path):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=3, ep_size=4)
    _patch_lora_dispatch(monkeypatch, state)
    _patch_lora_bridge(monkeypatch)

    def gather_with_lower_rank_replica(value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], tuple) and len(value[0]) == 4:
            return [(value[0], 5), value]
        return [value, value]

    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", gather_with_lower_rank_replica)

    peft_utils.save_peft_checkpoint(
        [_LoRAModel()],
        Namespace(peft_method="lora", hf_checkpoint="base", no_save_optim=True),
        str(tmp_path),
    )

    assert not (tmp_path / "adapter_megatron_tp1_pp2_ep3.pt").exists()
    assert not (tmp_path / "adapter_megatron_rank6.pt").exists()


def test_native_load_resolves_expert_parallel_shard(monkeypatch, tmp_path):
    class _AdapterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1_oft_r = torch.nn.Parameter(torch.zeros(1))

    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=3, ep_size=4)
    _patch_distributed_rank(
        monkeypatch,
        global_rank=6,
        parallel_state=state,
        gathered_coordinates=None,
    )
    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", lambda value: [value])
    torch.save({(0, "w1_oft_r"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2_ep3.pt")
    model = _AdapterModel()

    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    loaded, iteration = peft_utils.load_peft_adapter_checkpoint(
        [model],
        str(tmp_path),
        label="OFT",
        checkpoint_preflight=preflight,
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.w1_oft_r, torch.tensor([7.0]))


def test_lora_production_dispatch_loads_expert_parallel_shard(monkeypatch, tmp_path):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=3, ep_size=4)
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2_ep3.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter(
        [model],
        Namespace(peft_method="lora"),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.lora_A, torch.tensor([7.0]))


def test_lora_production_dispatch_loads_distinct_expert_tensor_parallel_shard(monkeypatch, tmp_path):
    state = _parallel_state(
        tp_rank=0,
        tp_size=1,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp0_pp2_etp1.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter(
        [model],
        Namespace(peft_method="lora"),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.lora_A, torch.tensor([7.0]))


def test_lora_production_dispatch_loads_tp_shard_when_etp_is_redundant(monkeypatch, tmp_path):
    state = _parallel_state(
        tp_rank=1,
        tp_size=2,
        pp_rank=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({(0, "lora_A"): torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter(
        [model],
        Namespace(peft_method="lora"),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.lora_A, torch.tensor([7.0]))


def test_lora_production_dispatch_rejects_unsuffixed_tp_pp_shard_with_ep(monkeypatch, tmp_path):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=3, ep_size=4)
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({"lora_A": torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter(
        [model],
        Namespace(peft_method="lora"),
        str(tmp_path),
    )

    assert loaded is False
    assert iteration is None
    assert torch.equal(model.lora_A, torch.zeros(1))


@pytest.mark.parametrize(("ep_rank", "ep_size"), [(0, 1), (3, 4)])
def test_lora_production_dispatch_loads_legacy_global_rank_shard(
    monkeypatch,
    tmp_path,
    ep_rank,
    ep_size,
):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=ep_rank, ep_size=ep_size)
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({"lora_A": torch.tensor([7.0])}, tmp_path / "adapter_megatron_rank6.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter(
        [model],
        Namespace(peft_method="lora"),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.lora_A, torch.tensor([7.0]))


def test_lora_production_dispatch_rejects_mixed_native_shard_layouts(monkeypatch, tmp_path):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=3, ep_size=4)
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({"lora_A": torch.tensor([7.0])}, tmp_path / "adapter_megatron_rank6.pt")

    def gather_with_mixed_layout(value):
        if isinstance(value, tuple) and len(value) == 5 and value[-1] == "global-rank":
            return [value, (*value[:-1], "coordinate")]
        return [value, value]

    monkeypatch.setattr(peft_utils, "_all_gather_checkpoint_object", gather_with_mixed_layout)

    with pytest.raises(RuntimeError, match="native adapter shard layouts differ across ranks"):
        peft_utils.load_peft_adapter(
            [_LoRAModel()],
            Namespace(peft_method="lora"),
            str(tmp_path),
        )


def test_lora_production_dispatch_loads_unsuffixed_tp_pp_shard_without_ep(monkeypatch, tmp_path):
    state = _parallel_state(tp_rank=1, pp_rank=2, ep_rank=0, ep_size=1)
    _patch_lora_dispatch(monkeypatch, state)
    torch.save({"lora_A": torch.tensor([7.0])}, tmp_path / "adapter_megatron_tp1_pp2.pt")
    model = _LoRAModel()

    loaded, iteration = peft_utils.load_peft_adapter(
        [model],
        Namespace(peft_method="lora"),
        str(tmp_path),
    )

    assert loaded is True
    assert iteration is None
    assert torch.equal(model.lora_A, torch.tensor([7.0]))
