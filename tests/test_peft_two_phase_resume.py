import argparse
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import orbit.backends.megatron_utils.peft_utils as peft_utils
from orbit.backends.megatron_utils.peft_utils import (
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


class _Scheduler:
    def __init__(self, num_steps=0):
        self.num_steps = num_steps
        self.load_calls = 0

    def state_dict(self):
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.load_calls += 1
        self.num_steps = state["num_steps"]


def test_low_precision_resume_discovers_iteration_then_restores_training_state(tmp_path):
    source_optimizer = _ExternalStateOptimizer(step=7, main=3.5, moment=9.0)
    source_scheduler = _Scheduler(num_steps=224)
    save_training_state(tmp_path, source_optimizer, source_scheduler, iteration=7)

    # Phase one runs while only model/adapter tensors exist.
    assert load_training_state(tmp_path, None, None) == 7

    # Phase two runs immediately after optimizer/scheduler construction.
    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()
    args = argparse.Namespace(_peft_resume_adapter_dir=str(tmp_path))
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


def test_second_phase_rejects_training_state_changed_since_model_load(tmp_path):
    save_training_state(
        tmp_path,
        _ExternalStateOptimizer(step=7),
        _Scheduler(num_steps=224),
        iteration=7,
    )
    discovered_iteration = load_training_state(tmp_path, None, None)
    assert discovered_iteration == 7

    state_path = tmp_path / "training_state_rank0.pt"
    state = torch.load(state_path, weights_only=False)
    state["iteration"] = 8
    torch.save(state, state_path)

    target_optimizer = _ExternalStateOptimizer()
    target_scheduler = _Scheduler()
    args = argparse.Namespace(_peft_resume_adapter_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="checkpoint iteration does not match"):
        restore_peft_training_state_after_optimizer_build(
            args,
            target_optimizer,
            target_scheduler,
            expected_iteration=discovered_iteration,
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
    from orbit.utils import megatron_bridge_utils

    monkeypatch.setattr(bridge_module, "AutoBridge", _Bridge, raising=False)
    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", lambda model: nullcontext())
    monkeypatch.setattr(peft_utils, "get_parallel_state", lambda: SimpleNamespace(intra_dp=SimpleNamespace(rank=0)))
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
