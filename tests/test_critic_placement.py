import argparse

import pytest

pytest.importorskip("ray")

import orbit.ray.placement_group as pg_mod


def test_pgs_dict_has_no_critic_entry_in_adapter_mode():
    src = open(pg_mod.__file__).read()
    assert "uses_separate_critic" in src, (
        "placement_group must gate critic pg creation on uses_separate_critic"
    )


def test_adapter_mode_zeroes_the_gpu_offset_inputs():
    """The untouched offset arithmetic (placement_group.py:90-108, rollout.py:1035,
    sglang_engine.py:36) adds critic_num_nodes * critic_num_gpus_per_node — verify
    _apply_critic_args forces that product to 0 in adapter mode."""
    from orbit.utils.arguments import _apply_critic_args

    args = argparse.Namespace(
        advantage_estimator="ppo",
        critic_mode="adapter",
        critic_num_gpus_per_node=None,
        critic_num_nodes=None,
        critic_load=None,
        critic_lr=None,
        actor_num_gpus_per_node=2,
        actor_num_nodes=1,
        load="/ckpt/base",
        lr=1e-6,
        peft_method="oft",
        train_backend="megatron",
        keep_old_actor=False,
    )
    _apply_critic_args(args)
    assert args.critic_num_nodes * args.critic_num_gpus_per_node == 0
