import argparse

import pytest
import torch

from orbit.backends.megatron_utils.critic_adapter import (
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
    assert load_critic_checkpoint(args, target)
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


def test_fresh_start_returns_false(tmp_path):
    assert not load_critic_checkpoint(_args(tmp_path), [_Chunk()])
    assert not load_critic_checkpoint(argparse.Namespace(critic_save=None), [_Chunk()])


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
    assert load_critic_checkpoint(args, target, optimizer=target_opt)
    assert len(target_opt.state_dict()["state"]) == len(opt.state_dict()["state"])
