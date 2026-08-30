import argparse
from pathlib import Path

import pytest
import torch

import orbit.critic.critic_adapter as critic_adapter
from orbit.critic.critic_adapter import (
    _check_resume_iteration,
    _expected_critic_resume_iteration,
    load_critic_checkpoint,
    save_critic_checkpoint,
)


class _Chunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(4, 4, bias=False)
        self.trunk.weight.requires_grad_(False)
        self.adapter = torch.nn.Linear(4, 4, bias=False)
        self.output_layer = torch.nn.Linear(4, 1, bias=False)


def _args(tmp_path):
    root = str(tmp_path / "critic")
    return argparse.Namespace(critic_load=root, critic_save=root, no_save_optim=False)


class _Scheduler:
    def __init__(self, num_steps=0):
        self.num_steps = num_steps

    def state_dict(self):
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]


class _ExternalStateOptimizer:
    """Small stand-in for Megatron's DistributedOptimizer checkpoint API."""

    def __init__(self, model):
        self.model_params = [p for p in model[0].parameters() if p.requires_grad]
        self.main_params = [p.detach().clone() for p in self.model_params]
        self.moments = [torch.zeros_like(p) for p in self.model_params]
        self.step = 0
        self.reload_calls = 0
        self.load_state_calls = 0
        self.load_parameter_state_calls = 0

    def state_dict(self):
        # Deliberately no parameter-dependent tensors: Megatron saves those via
        # save_parameter_state instead.
        return {"optimizer": {"param_groups": [{"step": self.step}]}}

    def load_state_dict(self, state):
        self.load_state_calls += 1
        self.step = state["optimizer"]["param_groups"][0]["step"]

    def reload_model_params(self):
        self.reload_calls += 1
        for main, model in zip(self.main_params, self.model_params, strict=True):
            main.copy_(model)

    def save_parameter_state(self, filename):
        torch.save(
            {
                "main_params": [p.clone() for p in self.main_params],
                "moments": [m.clone() for m in self.moments],
            },
            filename,
        )

    def load_parameter_state(self, filename):
        self.load_parameter_state_calls += 1
        state = torch.load(filename, weights_only=False)
        for target, saved in zip(self.main_params, state["main_params"], strict=True):
            target.copy_(saved)
        for target, saved in zip(self.moments, state["moments"], strict=True):
            target.copy_(saved)


class _NonWriterExternalStateOptimizer(_ExternalStateOptimizer):
    """Simulate a nonzero DP rank, which participates without reading a file."""

    def load_parameter_state(self, filename):
        self.load_parameter_state_calls += 1


class _UnsteppedPlainOptimizer:
    """Mimic ChainedOptimizer wrapping an unstepped non-distributed child."""

    def state_dict(self):
        return {"state": {}, "param_groups": []}

    def save_parameter_state(self, filename):
        raise AssertionError("plain optimizer must not use external parameter state")

    def load_parameter_state(self, filename):
        raise AssertionError("plain optimizer must not use external parameter state")


class _PreparationRequiredOptimizer:
    def __init__(self):
        self.prepared = False
        self.state_dict_calls = 0

    def state_dict(self):
        assert self.prepared
        self.state_dict_calls += 1
        return {
            "state": {0: {"exp_avg": torch.zeros(1)}},
            "param_groups": [],
        }


class _DeletingExternalStateOptimizer(_ExternalStateOptimizer):
    def __init__(self, model, parameter_state_path):
        super().__init__(model)
        self.parameter_state_path = parameter_state_path

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.parameter_state_path.unlink()


def test_round_trip_restores_trainable_tensors_only(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    save_critic_checkpoint(args, 3, model)

    target = [_Chunk()]
    frozen_before = target[0].trunk.weight.clone()
    assert load_critic_checkpoint(args, target) is not None
    assert torch.equal(target[0].adapter.weight, model[0].adapter.weight)
    assert torch.equal(target[0].output_layer.weight, model[0].output_layer.weight)
    assert torch.equal(target[0].trunk.weight, frozen_before)  # frozen params untouched


def test_checkpoint_contains_only_adapter_and_head(tmp_path):
    args = _args(tmp_path)
    save_critic_checkpoint(args, 1, [_Chunk()])
    payload = torch.load(tmp_path / "critic" / "iter_0000001" / "critic_rank0.pt", weights_only=False)
    assert set(payload["tensors"]) == {"0:adapter.weight", "0:output_layer.weight"}


def test_fresh_start_returns_none(tmp_path):
    assert load_critic_checkpoint(argparse.Namespace(critic_load=None), [_Chunk()]) is None


def test_explicit_missing_load_root_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="--critic-load"):
        load_critic_checkpoint(_args(tmp_path), [_Chunk()])


def test_load_returns_saved_iteration(tmp_path):
    args = _args(tmp_path)
    save_critic_checkpoint(args, 3, [_Chunk()])

    target = [_Chunk()]
    assert load_critic_checkpoint(args, target) == 3


def test_mismatched_tensor_set_fails_loud(tmp_path):
    args = _args(tmp_path)
    save_critic_checkpoint(args, 1, [_Chunk()])
    target = [_Chunk()]
    target[0].extra = torch.nn.Linear(4, 4, bias=False)
    with pytest.raises(RuntimeError, match="critic checkpoint mismatch"):
        load_critic_checkpoint(args, target)


def test_optimizer_state_round_trips(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    opt = torch.optim.AdamW([p for p in model[0].parameters() if p.requires_grad], lr=1e-3)
    model[0].adapter.weight.grad = torch.ones_like(model[0].adapter.weight)
    model[0].output_layer.weight.grad = torch.ones_like(model[0].output_layer.weight)
    opt.step()
    save_critic_checkpoint(args, 2, model, optimizer=opt)

    target = [_Chunk()]
    target_opt = torch.optim.AdamW([p for p in target[0].parameters() if p.requires_grad], lr=1e-3)
    assert load_critic_checkpoint(args, target, optimizer=target_opt) is not None
    assert len(target_opt.state_dict()["state"]) == len(opt.state_dict()["state"])


def test_unstepped_plain_optimizer_is_not_misclassified_as_distributed(tmp_path):
    args = _args(tmp_path)
    save_critic_checkpoint(args, 1, [_Chunk()], optimizer=_UnsteppedPlainOptimizer())
    payload = torch.load(tmp_path / "critic" / "iter_0000001" / "critic_rank0.pt", weights_only=False)
    assert payload["optimizer_parameter_state"] is False


def test_save_prepares_distributed_state_before_optimizer_serialization(monkeypatch, tmp_path):
    optimizer = _PreparationRequiredOptimizer()

    def prepare(candidate):
        assert candidate is optimizer
        candidate.prepared = True

    monkeypatch.setattr(critic_adapter.peft_utils, "prepare_distributed_optimizer_state_for_save", prepare)
    save_critic_checkpoint(_args(tmp_path), 1, [_Chunk()], optimizer=optimizer)

    assert optimizer.state_dict_calls == 1


def test_distributed_optimizer_external_state_and_scheduler_round_trip(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    optimizer = _ExternalStateOptimizer(model)
    optimizer.step = 9
    for main in optimizer.main_params:
        main.add_(0.125)
    for moment in optimizer.moments:
        moment.fill_(7.0)
    scheduler = _Scheduler(num_steps=41)

    save_critic_checkpoint(
        args,
        9,
        model,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
    )
    parameter_state_path = tmp_path / "critic" / "iter_0000009" / "optimizer_parameter_state_rank0.pt"
    assert parameter_state_path.is_file()

    target = [_Chunk()]
    target_optimizer = _ExternalStateOptimizer(target)
    target_scheduler = _Scheduler()
    assert (
        load_critic_checkpoint(
            args,
            target,
            optimizer=target_optimizer,
            opt_param_scheduler=target_scheduler,
        )
        == 9
    )

    assert target_optimizer.reload_calls == 1
    assert target_optimizer.load_parameter_state_calls == 1
    assert target_optimizer.step == 9
    assert target_scheduler.num_steps == 41
    for loaded, saved in zip(target_optimizer.main_params, optimizer.main_params, strict=True):
        assert torch.equal(loaded, saved)
    for loaded, saved in zip(target_optimizer.moments, optimizer.moments, strict=True):
        assert torch.equal(loaded, saved)


def test_missing_distributed_optimizer_external_state_fails_loud(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    optimizer = _ExternalStateOptimizer(model)
    save_critic_checkpoint(args, 2, model, optimizer=optimizer, opt_param_scheduler=_Scheduler())
    (tmp_path / "critic" / "iter_0000002" / "optimizer_parameter_state_rank0.pt").unlink()

    target = [_Chunk()]
    with pytest.raises(RuntimeError, match="optimizer parameter state is missing"):
        load_critic_checkpoint(
            args,
            target,
            optimizer=_ExternalStateOptimizer(target),
            opt_param_scheduler=_Scheduler(),
        )


def test_custom_external_state_dispatch_uses_cached_snapshot(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    source_optimizer = _ExternalStateOptimizer(model)
    source_optimizer.step = 5
    save_critic_checkpoint(args, 5, model, optimizer=source_optimizer, opt_param_scheduler=_Scheduler())

    parameter_state_path = tmp_path / "critic" / "iter_0000005" / "optimizer_parameter_state_rank0.pt"
    target = [_Chunk()]
    target_optimizer = _DeletingExternalStateOptimizer(target, parameter_state_path)

    assert (
        load_critic_checkpoint(
            args,
            target,
            optimizer=target_optimizer,
            opt_param_scheduler=_Scheduler(),
        )
        == 5
    )
    assert target_optimizer.load_parameter_state_calls == 1
    assert not parameter_state_path.exists()


@pytest.mark.parametrize("replaced_file", ["marker", "payload", "external"])
def test_load_rejects_checkpoint_file_replacement_before_mutation(monkeypatch, tmp_path, replaced_file):
    args = _args(tmp_path)
    source = [_Chunk()]
    save_critic_checkpoint(
        args,
        8,
        source,
        optimizer=_ExternalStateOptimizer(source),
        opt_param_scheduler=_Scheduler(),
    )

    target_names = {
        "marker": "latest_checkpointed_iteration.txt",
        "payload": "critic_rank0.pt",
        "external": "optimizer_parameter_state_rank0.pt",
    }
    target_name = target_names[replaced_file]
    original_capture = critic_adapter.peft_utils._capture_checkpoint_file_binding
    replaced = False

    def capture_then_replace(path):
        nonlocal replaced
        binding = original_capture(path)
        path = Path(path)
        if path.name == target_name and not replaced:
            replacement = path.with_name(f".{path.name}.replacement")
            if replaced_file == "marker":
                replacement.write_text("8")
            else:
                torch.save(torch.load(path, map_location="cpu", weights_only=False), replacement)
            replacement.replace(path)
            replaced = True
        return binding

    monkeypatch.setattr(
        critic_adapter.peft_utils,
        "_capture_checkpoint_file_binding",
        capture_then_replace,
    )
    target = [_Chunk()]
    target_before = {name: param.detach().clone() for name, param in target[0].named_parameters()}
    target_optimizer = _ExternalStateOptimizer(target)
    target_scheduler = _Scheduler()

    with pytest.raises(RuntimeError, match="checkpoint file changed"):
        load_critic_checkpoint(
            args,
            target,
            optimizer=target_optimizer,
            opt_param_scheduler=target_scheduler,
        )

    assert replaced is True
    assert all(torch.equal(param, target_before[name]) for name, param in target[0].named_parameters())
    assert target_optimizer.reload_calls == 0
    assert target_optimizer.load_state_calls == 0
    assert target_optimizer.load_parameter_state_calls == 0
    assert target_scheduler.num_steps == 0


def test_custom_external_optimizer_cannot_simulate_non_writer_without_a_process_group(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    save_critic_checkpoint(
        args,
        2,
        model,
        optimizer=_ExternalStateOptimizer(model),
        opt_param_scheduler=_Scheduler(),
    )
    (tmp_path / "critic" / "iter_0000002" / "optimizer_parameter_state_rank0.pt").unlink()

    target = [_Chunk()]
    optimizer = _NonWriterExternalStateOptimizer(target)
    with pytest.raises(RuntimeError, match="optimizer parameter state is missing"):
        load_critic_checkpoint(args, target, optimizer=optimizer, opt_param_scheduler=_Scheduler())
    assert optimizer.load_parameter_state_calls == 0


def test_no_save_optim_omits_all_optimizer_training_state(tmp_path):
    args = _args(tmp_path)
    model = [_Chunk()]
    optimizer = _ExternalStateOptimizer(model)
    save_critic_checkpoint(args, 4, model, optimizer=optimizer, opt_param_scheduler=_Scheduler())

    args.no_save_optim = True
    save_critic_checkpoint(
        args,
        4,
        model,
        optimizer=optimizer,
        opt_param_scheduler=_Scheduler(num_steps=22),
    )

    checkpoint_dir = tmp_path / "critic" / "iter_0000004"
    payload = torch.load(checkpoint_dir / "critic_rank0.pt", weights_only=False)
    assert payload["optimizer"] is None
    assert payload["optimizer_parameter_state"] is False
    assert payload["opt_param_scheduler"] is None
    assert not (checkpoint_dir / "optimizer_parameter_state_rank0.pt").exists()

    target = [_Chunk()]
    with pytest.raises(RuntimeError, match="no optimizer state"):
        load_critic_checkpoint(args, target, optimizer=_ExternalStateOptimizer(target))


def test_load_uses_critic_load_and_save_uses_critic_save(tmp_path):
    source_args = argparse.Namespace(
        critic_load=None,
        critic_save=str(tmp_path / "input"),
        no_save_optim=False,
    )
    source = [_Chunk()]
    save_critic_checkpoint(source_args, 6, source)

    load_args = argparse.Namespace(
        critic_load=str(tmp_path / "input"),
        critic_save=str(tmp_path / "output"),
        no_save_optim=False,
    )
    target = [_Chunk()]
    assert load_critic_checkpoint(load_args, target) == 6
    assert torch.equal(target[0].adapter.weight, source[0].adapter.weight)
    assert not (tmp_path / "output").exists()

    save_critic_checkpoint(load_args, 7, target)
    assert (tmp_path / "output" / "iter_0000007" / "critic_rank0.pt").is_file()
    assert (tmp_path / "input" / "latest_checkpointed_iteration.txt").read_text() == "6"


def test_check_resume_iteration_noop_when_unknown():
    _check_resume_iteration(None, None)
    _check_resume_iteration(None, 5)


def test_check_resume_iteration_rejects_critic_checkpoint_on_fresh_actor():
    with pytest.raises(RuntimeError, match="actor loaded no training checkpoint"):
        _check_resume_iteration(0, None)


def test_check_resume_iteration_requires_critic_when_actor_resumed():
    with pytest.raises(RuntimeError, match="actor resumed.*no matching adapter critic"):
        _check_resume_iteration(None, 5, require_checkpoint=True)


def test_check_resume_iteration_noop_on_match():
    _check_resume_iteration(3, 3)


def test_check_resume_iteration_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="critic/actor checkpoint iteration mismatch"):
        _check_resume_iteration(3, 5)


def test_expected_critic_iteration_ignores_model_only_bootstrap_iteration_zero():
    args = argparse.Namespace(_orbit_training_checkpoint_loaded=False)
    assert _expected_critic_resume_iteration(args, 0) is None


def test_expected_critic_iteration_preserves_real_resume_including_iteration_zero():
    args = argparse.Namespace(_orbit_training_checkpoint_loaded=True)
    assert _expected_critic_resume_iteration(args, 0) == 0
    assert _expected_critic_resume_iteration(args, 7) == 7
