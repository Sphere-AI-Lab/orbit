import argparse

import pytest
import torch

from orbit.backends.megatron_utils.critic_adapter import (
    _check_resume_iteration,
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
    return argparse.Namespace(critic_save=str(tmp_path / "critic"))


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
    payload = torch.load(
        tmp_path / "critic" / "iter_0000001" / "critic_rank0.pt", weights_only=False
    )
    assert set(payload["tensors"]) == {"0:adapter.weight", "0:output_layer.weight"}


def test_fresh_start_returns_none(tmp_path):
    assert load_critic_checkpoint(_args(tmp_path), [_Chunk()]) is None
    assert load_critic_checkpoint(argparse.Namespace(critic_save=None), [_Chunk()]) is None


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


def test_check_resume_iteration_noop_when_unknown():
    _check_resume_iteration(None, None)
    _check_resume_iteration(None, 5)
    _check_resume_iteration(5, None)


def test_check_resume_iteration_noop_on_match():
    _check_resume_iteration(3, 3)


def test_check_resume_iteration_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="critic/actor checkpoint iteration mismatch"):
        _check_resume_iteration(3, 5)
