import argparse

import pytest

from miles.orbit.critic.critic_adapter import _critic_build_args


def _args():
    return argparse.Namespace(
        load="/ckpt/base",
        save="/ckpt/actor",
        lr=1e-6,
        lr_warmup_iters=10,
        critic_save="/ckpt/critic",
        critic_lr=1e-5,
        critic_lr_warmup_iters=3,
    )


def test_override_applies_critic_view_and_restores():
    args = _args()
    with _critic_build_args(args):
        assert args.load is None  # trunk arrives via alias, never from checkpoint
        assert args.save == "/ckpt/critic"
        assert args.lr == 1e-5
        assert args.lr_warmup_iters == 3
    assert args.load == "/ckpt/base"
    assert args.save == "/ckpt/actor"
    assert args.lr == 1e-6
    assert args.lr_warmup_iters == 10


def test_override_restores_on_exception():
    args = _args()
    with pytest.raises(RuntimeError):
        with _critic_build_args(args):
            raise RuntimeError("boom")
    assert args.load == "/ckpt/base"
    assert args.lr == 1e-6
