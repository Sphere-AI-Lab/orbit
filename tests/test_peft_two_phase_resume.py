import argparse
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.checkpoint as checkpoint_mod
import miles.backends.megatron_utils.peft_utils as peft_utils
from miles.backends.megatron_utils.peft_utils import (
    load_training_state,
    restore_peft_training_state_after_optimizer_build,
    save_training_state,
)


class _ExternalStateOptimizer:
    def __init__(self, *, step=0, main=0.0, moment=0.0):
        self.step = step
        self.main = torch.tensor([main])
        self.moment = torch.tensor([moment])
        self.load_state_calls = 0
        self.load_parameter_state_calls = 0
        self.reload_model_params_calls = 0

    def state_dict(self):
        return {"optimizer": {"param_groups": [{"step": self.step}]}}

    def load_state_dict(self, state):
        self.load_state_calls += 1
        self.step = state["optimizer"]["param_groups"][0]["step"]

    def save_parameter_state(self, filename):
        torch.save({"main": self.main.clone(), "moment": self.moment.clone()}, filename)

    def load_parameter_state(self, filename):
        self.load_parameter_state_calls += 1
        state = torch.load(filename, weights_only=False)
        self.main.copy_(state["main"])
        self.moment.copy_(state["moment"])

    def reload_model_params(self):
        self.reload_model_params_calls += 1


class _Scheduler:
    def __init__(self, num_steps=0):
        self.num_steps = num_steps
        self.load_calls = 0

    def state_dict(self):
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.load_calls += 1
        self.num_steps = state["num_steps"]


class _FakeGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _FakeDistributedLeaf:
    def __init__(self, width=2):
        self.is_stub_optimizer = False
        self.data_parallel_group = _FakeGroup()
        self.data_parallel_group_gloo = _FakeGroup()
        self.model_param = torch.nn.Parameter(torch.zeros(width))
        self.main_param = torch.nn.Parameter(torch.zeros(width))
        self.optimizer = SimpleNamespace(
            param_groups=[{"params": [self.main_param]}],
            state={
                self.main_param: {
                    "exp_avg": torch.zeros(width),
                    "exp_avg_sq": torch.zeros(width),
                }
            },
        )
        self.model_param_group_index_map = {self.model_param: (0, 0)}
        local_range = SimpleNamespace(start=0, end=width)
        self.gbuf_ranges = [
            {
                torch.float32: [
                    {
                        "param_map": {
                            self.model_param: {"gbuf_local": local_range},
                        }
                    }
                ]
            }
        ]
        self.buffers = [
            SimpleNamespace(
                numel_unpadded=width,
                buckets=[SimpleNamespace(grad_data=torch.zeros(width), numel_unpadded=width)],
            )
        ]
        self.load_state_calls = 0
        self.lower_loads = []

    def state_dict(self):
        return {"optimizer": {"param_groups": []}}

    def load_state_dict(self, _state):
        self.load_state_calls += 1

    def get_parameter_state_dp_zero(self):
        return _valid_external_leaf_state(self)

    def save_parameter_state(self, filename):
        torch.save(_valid_external_leaf_state(self), filename)

    def load_parameter_state(self, _filename):
        raise AssertionError("the pinned Megatron filename loader must not be called")

    def load_parameter_state_from_dp_zero(self, state, *, update_legacy_format=False):
        assert update_legacy_format is False
        self.lower_loads.append(state)

    def split_state_dict_if_needed(self, _state):
        return None


class _WarmupDistributedLeaf(_FakeDistributedLeaf):
    def __init__(self, width=3, *, empty_state=True):
        super().__init__(width=width)
        self.main_param.data.copy_(torch.arange(width, dtype=torch.float32) + 4.0)
        self.optimizer.state = {}
        self.config = SimpleNamespace(name="test-config")
        self.init_state_calls = 0
        self.state_dict_calls = 0
        self.save_parameter_state_calls = 0
        self.source_overrides = {}

        def init_state_fn(inner_optimizer, config):
            assert config is self.config
            self.init_state_calls += 1
            for group in inner_optimizer.param_groups:
                for param in group["params"]:
                    state = inner_optimizer.state.setdefault(param, {})
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

        self.init_state_fn = init_state_fn
        if not empty_state:
            self.init_state_fn(self.optimizer, self.config)
            self.init_state_calls = 0

    def state_dict(self):
        self.state_dict_calls += 1
        state = self.optimizer.state[self.main_param]
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        return {"optimizer": {"param_groups": [{"step": 0}]}}

    def _get_main_param_and_optimizer_states(self, _model_param):
        state = self.optimizer.state[self.main_param]
        tensors = {
            "param": self.main_param,
            "exp_avg": state.get("exp_avg"),
            "exp_avg_sq": state.get("exp_avg_sq"),
        }
        tensors.update(self.source_overrides)
        return tensors

    def get_parameter_state_dp_zero(self):
        tensors = self._get_main_param_and_optimizer_states(self.model_param)
        return {
            "buckets_coalesced": True,
            0: {
                torch.float32: {
                    "numel_unpadded": self.main_param.numel(),
                    **{key: tensor.detach().cpu().clone() for key, tensor in tensors.items()},
                }
            },
        }

    def save_parameter_state(self, filename):
        self.save_parameter_state_calls += 1
        torch.save(self.get_parameter_state_dp_zero(), filename)


def _valid_external_leaf_state(leaf):
    width = leaf.buffers[0].numel_unpadded
    return {
        "buckets_coalesced": True,
        0: {
            torch.float32: {
                "numel_unpadded": width,
                "param": torch.ones(width),
                "exp_avg": torch.full((width,), 2.0),
                "exp_avg_sq": torch.full((width,), 3.0),
            }
        },
    }


def test_warmup_save_initializes_zero_adam_state_without_advancing_training(tmp_path):
    optimizer = _WarmupDistributedLeaf()
    scheduler = _Scheduler(num_steps=192)
    original_param = optimizer.main_param.detach().clone()
    original_scheduler_state = scheduler.state_dict().copy()

    save_training_state(tmp_path, optimizer, scheduler, iteration=0)

    assert optimizer.init_state_calls == 1
    assert optimizer.state_dict_calls == 1
    assert optimizer.save_parameter_state_calls == 1
    assert torch.equal(optimizer.main_param, original_param)
    assert scheduler.state_dict() == original_scheduler_state
    assert scheduler.load_calls == 0
    assert torch.count_nonzero(optimizer.optimizer.state[optimizer.main_param]["exp_avg"]) == 0
    assert torch.count_nonzero(optimizer.optimizer.state[optimizer.main_param]["exp_avg_sq"]) == 0

    parameter_state = torch.load(
        tmp_path / "optimizer_parameter_state_rank0.pt",
        map_location="cpu",
        weights_only=False,
    )
    saved_sources = parameter_state[0][torch.float32]
    assert torch.equal(saved_sources["param"], original_param)
    assert torch.count_nonzero(saved_sources["exp_avg"]) == 0
    assert torch.count_nonzero(saved_sources["exp_avg_sq"]) == 0
    training_state = torch.load(tmp_path / "training_state_rank0.pt", weights_only=False)
    assert training_state["iteration"] == 0
    assert training_state["optimizer_parameter_state"] is True
    assert training_state["opt_param_scheduler"] == original_scheduler_state


@pytest.mark.parametrize(
    "bad_index",
    [None, (1, 0), (0, 1), ("0", 0), (0,)],
)
def test_save_rejects_invalid_model_parameter_group_indices_before_serialization(tmp_path, bad_index):
    optimizer = _WarmupDistributedLeaf(empty_state=False)
    if bad_index is None:
        optimizer.model_param_group_index_map.clear()
    else:
        optimizer.model_param_group_index_map[optimizer.model_param] = bad_index

    with pytest.raises(RuntimeError, match="distributed optimizer state initialization"):
        save_training_state(tmp_path, optimizer, _Scheduler(), iteration=0)

    assert optimizer.state_dict_calls == 0
    assert optimizer.save_parameter_state_calls == 0
    assert not (tmp_path / "optimizer_parameter_state_rank0.pt").exists()


@pytest.mark.parametrize(
    "invalid_source",
    [
        "missing_exp_avg",
        "missing_exp_avg_sq",
        "wrong_width",
        "integer",
        "matrix",
        "sparse",
        "meta",
        "quantized",
    ],
)
def test_save_rejects_incompatible_live_optimizer_sources_before_materialization(
    tmp_path,
    invalid_source,
):
    optimizer = _WarmupDistributedLeaf(empty_state=False)
    width = optimizer.main_param.numel()
    replacements = {
        "wrong_width": lambda: torch.zeros(width + 1),
        "integer": lambda: torch.zeros(width, dtype=torch.int64),
        "matrix": lambda: torch.zeros(1, width),
        "sparse": lambda: torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            (width,),
        ),
        "meta": lambda: torch.empty(width, device="meta"),
        "quantized": lambda: torch.quantize_per_tensor(
            torch.ones(width),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        ),
    }
    if invalid_source == "missing_exp_avg":
        optimizer.source_overrides["exp_avg"] = None
    elif invalid_source == "missing_exp_avg_sq":
        optimizer.source_overrides["exp_avg_sq"] = None
    else:
        optimizer.source_overrides["exp_avg"] = replacements[invalid_source]()

    with pytest.raises(RuntimeError, match="distributed optimizer source validation"):
        save_training_state(tmp_path, optimizer, _Scheduler(), iteration=0)

    assert optimizer.state_dict_calls == 1
    assert optimizer.save_parameter_state_calls == 0
    assert not (tmp_path / "optimizer_parameter_state_rank0.pt").exists()


def test_low_precision_resume_discovers_iteration_then_restores_training_state(tmp_path):
    source_optimizer = _ExternalStateOptimizer(step=7, main=3.5, moment=9.0)
    source_scheduler = _Scheduler(num_steps=224)
    save_training_state(tmp_path, source_optimizer, source_scheduler, iteration=7)

    # Phase one runs while only model/adapter tensors exist.
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    assert load_training_state(tmp_path, None, None, checkpoint_preflight=preflight) == 7

    # Phase two runs immediately after optimizer/scheduler construction.
    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()
    args = argparse.Namespace(
        _peft_resume_adapter_dir=str(tmp_path),
        _peft_training_state_found=True,
        _peft_checkpoint_preflight=preflight,
    )
    assert restore_peft_training_state_after_optimizer_build(
        args,
        target_optimizer,
        target_scheduler,
        expected_iteration=7,
    )

    assert target_optimizer.step == 7
    assert target_optimizer.load_state_calls == 1
    assert target_optimizer.load_parameter_state_calls == 1
    assert torch.equal(target_optimizer.main, source_optimizer.main)
    assert torch.equal(target_optimizer.moment, source_optimizer.moment)
    assert target_scheduler.num_steps == 224
    assert target_scheduler.load_calls == 1


def test_second_phase_rejects_same_iteration_training_state_replacement(tmp_path):
    save_training_state(
        tmp_path,
        _ExternalStateOptimizer(step=7),
        _Scheduler(num_steps=224),
        iteration=7,
    )
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    discovered_iteration = load_training_state(
        tmp_path,
        None,
        None,
        checkpoint_preflight=preflight,
    )
    assert discovered_iteration == 7

    state_path = tmp_path / "training_state_rank0.pt"
    state = torch.load(state_path, weights_only=False)
    replacement_path = tmp_path / "training_state_replacement.pt"
    torch.save(state, replacement_path)
    replacement_path.replace(state_path)

    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()
    args = argparse.Namespace(
        _peft_resume_adapter_dir=str(tmp_path),
        _peft_training_state_found=True,
        _peft_checkpoint_preflight=preflight,
    )
    with pytest.raises(RuntimeError, match="checkpoint file changed after preflight"):
        restore_peft_training_state_after_optimizer_build(
            args,
            target_optimizer,
            target_scheduler,
            expected_iteration=discovered_iteration,
        )

    assert target_optimizer.load_state_calls == 0
    assert target_optimizer.load_parameter_state_calls == 0
    assert target_scheduler.load_calls == 0


def test_second_phase_rejects_same_payload_external_state_replacement(tmp_path):
    source_optimizer = _ExternalStateOptimizer(step=7, main=3.5, moment=9.0)
    source_scheduler = _Scheduler(num_steps=224)
    save_training_state(tmp_path, source_optimizer, source_scheduler, iteration=7)
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    assert load_training_state(tmp_path, None, None, checkpoint_preflight=preflight) == 7

    state_path = tmp_path / "optimizer_parameter_state_rank0.pt"
    state = torch.load(state_path, weights_only=False)
    replacement_path = tmp_path / "optimizer_parameter_state_replacement.pt"
    torch.save(state, replacement_path)
    replacement_path.replace(state_path)

    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()
    args = argparse.Namespace(
        _peft_resume_adapter_dir=str(tmp_path),
        _peft_training_state_found=True,
        _peft_checkpoint_preflight=preflight,
    )
    with pytest.raises(RuntimeError, match="checkpoint file changed after preflight"):
        restore_peft_training_state_after_optimizer_build(
            args,
            target_optimizer,
            target_scheduler,
            expected_iteration=7,
        )

    assert target_optimizer.load_state_calls == 0
    assert target_optimizer.load_parameter_state_calls == 0
    assert target_scheduler.load_calls == 0


def test_second_phase_is_a_noop_without_a_peft_resume_directory():
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()
    assert not restore_peft_training_state_after_optimizer_build(
        argparse.Namespace(),
        optimizer,
        scheduler,
        expected_iteration=0,
    )
    assert optimizer.load_state_calls == 0
    assert scheduler.load_calls == 0


def test_second_phase_requires_saved_preflight_before_optimizer_mutation(tmp_path):
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()
    args = argparse.Namespace(
        _peft_resume_adapter_dir=str(tmp_path),
        _peft_training_state_found=False,
    )

    with pytest.raises(RuntimeError, match="requires the saved checkpoint preflight"):
        restore_peft_training_state_after_optimizer_build(
            args,
            optimizer,
            scheduler,
            expected_iteration=0,
        )

    assert optimizer.load_state_calls == 0
    assert optimizer.load_parameter_state_calls == 0
    assert scheduler.load_calls == 0


def test_low_precision_weights_only_adapter_keeps_fresh_optimizer(tmp_path):
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    args = argparse.Namespace(
        _peft_resume_adapter_dir=str(tmp_path),
        _peft_adapter_weights_loaded=True,
        _peft_training_state_found=False,
        _peft_checkpoint_preflight=preflight,
    )

    assert not restore_peft_training_state_after_optimizer_build(
        args,
        optimizer,
        scheduler,
        expected_iteration=0,
    )
    assert optimizer.load_state_calls == 0
    assert optimizer.load_parameter_state_calls == 0
    assert scheduler.load_calls == 0


def test_low_precision_weights_only_adapter_rejects_sidecar_appearing_after_preflight(tmp_path):
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    args = argparse.Namespace(
        _peft_resume_adapter_dir=str(tmp_path),
        _peft_adapter_weights_loaded=True,
        _peft_training_state_found=False,
        _peft_checkpoint_preflight=preflight,
    )
    save_training_state(
        tmp_path,
        _ExternalStateOptimizer(step=1),
        _Scheduler(num_steps=32),
        iteration=1,
    )
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()

    with pytest.raises(RuntimeError, match="appeared"):
        restore_peft_training_state_after_optimizer_build(
            args,
            optimizer,
            scheduler,
            expected_iteration=0,
        )

    assert optimizer.load_state_calls == 0
    assert optimizer.load_parameter_state_calls == 0
    assert scheduler.load_calls == 0


@pytest.mark.parametrize(
    ("training_state_found", "loaded_iteration", "expected_reload_calls"),
    [(False, None, 1), (True, 7, 0)],
)
def test_normal_precision_adapter_load_syncs_main_params_only_without_training_state(
    monkeypatch,
    tmp_path,
    training_state_found,
    loaded_iteration,
    expected_reload_calls,
):
    (tmp_path / "payload").write_text("base")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_megatron_tp0_pp0.pt").write_bytes(b"native")
    if training_state_found:
        (adapter_dir / "training_state_rank0.pt").write_bytes(b"training")
    args = argparse.Namespace(
        load=str(tmp_path),
        megatron_to_hf_mode="raw",
        peft_method="lora",
        peft_adapter_path=str(adapter_dir),
        lora_adapter_path=None,
        oft_adapter_path=None,
        fp16=False,
        bf16=True,
    )
    optimizer = _ExternalStateOptimizer()

    monkeypatch.setattr(checkpoint_mod, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_mod, "is_distributed_checkpoint", lambda _path: True)
    monkeypatch.setattr(checkpoint_mod, "_resolve_selected_distributed_checkpoint", lambda _args: tmp_path)
    monkeypatch.setattr(checkpoint_mod, "_load_checkpoint_dist", lambda **_kwargs: (0, 0))
    monkeypatch.setattr(checkpoint_mod, "is_peft_enabled", lambda _args: True)
    monkeypatch.setattr(checkpoint_mod, "is_peft_model", lambda _model: True, raising=False)
    checkpoint_preflight = peft_utils.preflight_peft_adapter_checkpoint(adapter_dir)
    monkeypatch.setattr(
        checkpoint_mod,
        "preflight_peft_adapter_checkpoint",
        lambda _path: checkpoint_preflight,
    )
    monkeypatch.setattr(
        checkpoint_mod,
        "load_peft_adapter",
        lambda *args, **kwargs: (True, loaded_iteration),
    )

    iteration, _ = checkpoint_mod.load_checkpoint(
        [object()],
        optimizer,
        _Scheduler(),
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )

    assert iteration == (loaded_iteration if loaded_iteration is not None else 0)
    assert args._peft_adapter_weights_loaded is True
    assert args._peft_training_state_found is training_state_found
    assert args._peft_checkpoint_preflight is checkpoint_preflight
    assert optimizer.reload_model_params_calls == expected_reload_calls


def test_no_save_optim_removes_stale_peft_training_state(tmp_path):
    optimizer = _ExternalStateOptimizer(step=3, main=2.0, moment=4.0)
    scheduler = _Scheduler(num_steps=96)
    save_training_state(tmp_path, optimizer, scheduler, iteration=3)

    state_path = tmp_path / "training_state_rank0.pt"
    parameter_state_path = tmp_path / "optimizer_parameter_state_rank0.pt"
    assert state_path.is_file()
    assert parameter_state_path.is_file()

    save_training_state(
        tmp_path,
        optimizer,
        scheduler,
        iteration=3,
        no_save_optim=True,
    )
    assert not state_path.exists()
    assert not parameter_state_path.exists()
    assert load_training_state(tmp_path, None, None) is None


def test_peft_checkpoint_save_threads_no_save_optim_from_args(monkeypatch, tmp_path):
    optimizer = _ExternalStateOptimizer(step=3, main=2.0, moment=4.0)
    scheduler = _Scheduler(num_steps=96)
    save_training_state(tmp_path, optimizer, scheduler, iteration=3)

    class _Bridge:
        @classmethod
        def from_hf_pretrained(cls, *args, **kwargs):
            return cls()

        def export_adapter_weights(self, *args, **kwargs):
            return ()

    import megatron.bridge as bridge_module

    from miles.utils import megatron_bridge_utils

    monkeypatch.setattr(bridge_module, "AutoBridge", _Bridge, raising=False)
    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", lambda model: nullcontext())
    monkeypatch.setattr(
        peft_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0)),
    )
    monkeypatch.setattr(peft_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_pipeline_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(peft_utils, "native_adapter_state", lambda model: {"adapter": torch.ones(1)})
    monkeypatch.setattr(peft_utils, "_save_peft_hf_artifacts", lambda *args, **kwargs: None)

    args = argparse.Namespace(hf_checkpoint="base", no_save_optim=True)
    peft_utils.save_peft_adapter_checkpoint(
        [object()],
        args,
        str(tmp_path),
        method="lora",
        build_config=dict,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        iteration=3,
    )

    assert not (tmp_path / "training_state_rank0.pt").exists()
    assert not (tmp_path / "optimizer_parameter_state_rank0.pt").exists()


@pytest.mark.parametrize("files_present", [False, True])
def test_peft_checkpoint_preflight_all_rank_local_files_present_or_absent(
    monkeypatch,
    tmp_path,
    files_present,
):
    monkeypatch.setattr(peft_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_pipeline_model_parallel_rank", lambda: 0, raising=False)
    if files_present:
        (tmp_path / "adapter_megatron_tp0_pp0.pt").write_bytes(b"native")
        (tmp_path / "training_state_rank0.pt").write_bytes(b"training")

    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)

    assert preflight.adapter_dir == str(tmp_path)
    assert preflight.native_shards_present is files_present
    assert preflight.training_state_present is files_present
    assert (preflight.native_shard_binding is not None) is files_present
    assert (preflight.training_state_binding is not None) is files_present
    assert preflight.optimizer_parameter_state_binding is None


def test_preflight_rejects_external_state_without_training_sidecar(tmp_path):
    torch.save({"stale": True}, tmp_path / "optimizer_parameter_state_rank0.pt")

    with pytest.raises(RuntimeError, match="present without training state"):
        peft_utils.preflight_peft_adapter_checkpoint(tmp_path)


def test_false_external_state_marker_rejects_bound_stale_file_before_optimizer_mutation(tmp_path):
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()
    torch.save(
        {
            "iteration": 3,
            "active_student_version": None,
            "optimizer": optimizer.state_dict(),
            "optimizer_parameter_state": False,
            "opt_param_scheduler": scheduler.state_dict(),
        },
        tmp_path / "training_state_rank0.pt",
    )
    torch.save({"stale": True}, tmp_path / "optimizer_parameter_state_rank0.pt")
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)

    with pytest.raises(RuntimeError, match="marker is false"):
        load_training_state(
            tmp_path,
            optimizer,
            scheduler,
            checkpoint_preflight=preflight,
        )

    assert optimizer.load_state_calls == 0
    assert optimizer.load_parameter_state_calls == 0
    assert scheduler.load_calls == 0


def test_native_shard_replacement_is_rejected_before_adapter_mutation(monkeypatch, tmp_path):
    class _AdapterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(peft_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(peft_utils.mpu, "get_pipeline_model_parallel_rank", lambda: 0, raising=False)
    native_path = tmp_path / "adapter_megatron_tp0_pp0.pt"
    torch.save({(0, "lora_A"): torch.ones(1)}, native_path)
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    replacement_path = tmp_path / "native_replacement.pt"
    torch.save({(0, "lora_A"): torch.full((1,), 2.0)}, replacement_path)
    replacement_path.replace(native_path)
    model = _AdapterModel()

    with pytest.raises(RuntimeError, match="checkpoint file changed after preflight"):
        peft_utils.load_peft_adapter_checkpoint(
            [model],
            str(tmp_path),
            label="LoRA",
            checkpoint_preflight=preflight,
        )

    assert torch.equal(model.lora_A, torch.zeros(1))


@pytest.mark.parametrize("initially_present", [False, True])
def test_training_state_race_after_preflight_is_rejected_before_optimizer_mutation(
    tmp_path,
    initially_present,
):
    source_optimizer = _ExternalStateOptimizer(step=4)
    source_scheduler = _Scheduler(num_steps=128)
    if initially_present:
        save_training_state(tmp_path, source_optimizer, source_scheduler, iteration=4)

    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    state_path = tmp_path / "training_state_rank0.pt"
    if initially_present:
        state_path.unlink()
        expected_change = "disappeared"
    else:
        save_training_state(tmp_path, source_optimizer, source_scheduler, iteration=4)
        expected_change = "appeared"

    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()
    with pytest.raises(RuntimeError, match=expected_change):
        load_training_state(
            tmp_path,
            target_optimizer,
            target_scheduler,
            checkpoint_preflight=preflight,
        )

    assert target_optimizer.load_state_calls == 0
    assert target_optimizer.load_parameter_state_calls == 0
    assert target_scheduler.load_calls == 0


def test_corrupt_training_state_is_coordinated_before_optimizer_mutation(tmp_path):
    torch.save("not a training-state mapping", tmp_path / "training_state_rank0.pt")
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()

    with pytest.raises(RuntimeError, match="training-state parse/validation"):
        load_training_state(
            tmp_path,
            optimizer,
            scheduler,
            checkpoint_preflight=preflight,
        )

    assert optimizer.load_state_calls == 0
    assert optimizer.load_parameter_state_calls == 0
    assert scheduler.load_calls == 0


@pytest.mark.parametrize("embedded_key", ["param_state", "param_state_sharding_type"])
def test_embedded_distributed_parameter_state_precedes_optimizer_mutation(tmp_path, embedded_key):
    optimizer = _ExternalStateOptimizer()
    scheduler = _Scheduler()
    torch.save(
        {
            "iteration": 5,
            "active_student_version": None,
            "optimizer": {
                "optimizer": {"param_groups": [{"step": 5}]},
                "nested": [{embedded_key: {"foreign": torch.ones(1)}}],
            },
            "optimizer_parameter_state": True,
            "opt_param_scheduler": scheduler.state_dict(),
        },
        tmp_path / "training_state_rank0.pt",
    )
    torch.save({"main": torch.ones(1)}, tmp_path / "optimizer_parameter_state_rank0.pt")

    with pytest.raises(RuntimeError, match="embedded distributed parameter state"):
        load_training_state(tmp_path, optimizer, scheduler)

    assert optimizer.load_state_calls == 0
    assert optimizer.load_parameter_state_calls == 0
    assert scheduler.load_calls == 0


@pytest.mark.parametrize("corrupt_external_state", [False, True])
def test_external_parameter_state_is_preflighted_before_optimizer_mutation(
    tmp_path,
    corrupt_external_state,
):
    optimizer = _ExternalStateOptimizer(step=5)
    scheduler = _Scheduler(num_steps=160)
    torch.save(
        {
            "iteration": 5,
            "active_student_version": None,
            "optimizer": optimizer.state_dict(),
            "optimizer_parameter_state": True,
            "opt_param_scheduler": scheduler.state_dict(),
        },
        tmp_path / "training_state_rank0.pt",
    )
    if corrupt_external_state:
        (tmp_path / "optimizer_parameter_state_rank0.pt").write_bytes(b"not a torch checkpoint")
    preflight = peft_utils.preflight_peft_adapter_checkpoint(tmp_path)
    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()

    with pytest.raises(RuntimeError, match="optimizer parameter-state preflight"):
        load_training_state(
            tmp_path,
            target_optimizer,
            target_scheduler,
            checkpoint_preflight=preflight,
        )

    assert target_optimizer.load_state_calls == 0
    assert target_optimizer.load_parameter_state_calls == 0
    assert target_scheduler.load_calls == 0


def test_pinned_external_parameter_state_is_cached_before_collective_dispatch(tmp_path):
    optimizer = _FakeDistributedLeaf()
    parameter_state_path = tmp_path / "optimizer_parameter_state_rank0.pt"
    torch.save(_valid_external_leaf_state(optimizer), parameter_state_path)

    binding = peft_utils._capture_checkpoint_file_binding(parameter_state_path)
    plan = peft_utils._build_external_parameter_state_plan(optimizer, parameter_state_path, binding)
    assert plan is not None
    # A filename-based second load would now fail. Cached dispatch must not touch
    # the filesystem again after all-rank validation.
    parameter_state_path.unlink()
    peft_utils._validate_external_parameter_state_destinations(plan)
    peft_utils._dispatch_external_parameter_state(plan)

    assert len(optimizer.lower_loads) == 1
    assert optimizer.lower_loads[0]["buckets_coalesced"] is True


def test_structurally_invalid_external_state_precedes_optimizer_mutation(tmp_path):
    optimizer = _FakeDistributedLeaf()
    scheduler = _Scheduler()
    torch.save(
        {
            "iteration": 5,
            "active_student_version": None,
            "optimizer": optimizer.state_dict(),
            "optimizer_parameter_state": True,
            "opt_param_scheduler": scheduler.state_dict(),
        },
        tmp_path / "training_state_rank0.pt",
    )
    # Valid pickle, invalid Megatron distributed-optimizer layout.
    torch.save({}, tmp_path / "optimizer_parameter_state_rank0.pt")

    with pytest.raises(RuntimeError, match="optimizer parameter-state preflight"):
        load_training_state(tmp_path, optimizer, scheduler)

    assert optimizer.load_state_calls == 0
    assert scheduler.load_calls == 0
    assert optimizer.lower_loads == []


def test_multi_child_external_state_uses_indexed_cached_dispatch(tmp_path):
    first = _FakeDistributedLeaf(width=2)
    second = _FakeDistributedLeaf(width=3)
    optimizer = SimpleNamespace(chained_optimizers=[first, second])
    parameter_state_path = tmp_path / "optimizer_parameter_state_rank0.pt"
    torch.save(
        [_valid_external_leaf_state(first), _valid_external_leaf_state(second)],
        parameter_state_path,
    )

    binding = peft_utils._capture_checkpoint_file_binding(parameter_state_path)
    plan = peft_utils._build_external_parameter_state_plan(optimizer, parameter_state_path, binding)
    assert plan is not None
    peft_utils._validate_external_parameter_state_destinations(plan)
    peft_utils._dispatch_external_parameter_state(plan)

    assert first.lower_loads[0][0][torch.float32]["param"].numel() == 2
    assert second.lower_loads[0][0][torch.float32]["param"].numel() == 3


def test_multi_child_distributed_and_stub_layout_is_rejected():
    active = _FakeDistributedLeaf()
    stub = SimpleNamespace(
        is_stub_optimizer=True,
        get_parameter_state_dp_zero=lambda: None,
        load_parameter_state_from_dp_zero=lambda *_args, **_kwargs: None,
    )
    optimizer = SimpleNamespace(chained_optimizers=[active, stub])

    with pytest.raises(RuntimeError, match="stub distributed children"):
        peft_utils._megatron_external_parameter_state_layout(optimizer)


def test_training_metadata_consensus_allows_rank_local_external_state_marker(monkeypatch):
    monkeypatch.setattr(
        peft_utils,
        "_all_gather_checkpoint_object",
        lambda _value: [(7, "3"), (7, "3")],
    )
    peft_utils._validate_training_metadata_consensus(
        {
            "iteration": 7,
            "active_student_version": "3",
            "optimizer_parameter_state": True,
        }
    )
