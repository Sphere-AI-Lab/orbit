"""Managed OPD teacher serving (--opd-serve-teacher): teacher ModelConfig construction,
placement-group sizing, and validation coupling."""

import argparse

import pytest

from orbit.ray.placement_group import _opd_teacher_extra_gpus
from orbit.ray.rollout import OPD_TEACHER_MODEL_NAME, _opd_teacher_model_config
from orbit.utils.arguments import _validate_opd_args


def _serve_args(**overrides):
    defaults = dict(
        opd_serve_teacher=True,
        opd_teacher_num_gpus=1,
        opd_teacher_mem_fraction=None,
        teacher_hf_checkpoint="/fake/teacher",
        colocate=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_teacher_model_config_bakes_scoring_flags():
    cfg = _opd_teacher_model_config(_serve_args(opd_teacher_num_gpus=2))
    assert cfg.name == OPD_TEACHER_MODEL_NAME
    assert cfg.model_path == "/fake/teacher"
    assert cfg.update_weights is False
    assert cfg.num_gpus_per_engine == 2  # one engine, TP across the teacher GPUs
    (group,) = cfg.server_groups
    assert group.worker_type == "regular"
    assert group.num_gpus == 2
    assert group.overrides["enable_return_hidden_states"] is True
    assert group.overrides["disable_radix_cache"] is True
    assert group.overrides["chunked_prefill_size"] == -1
    assert "mem_fraction_static" not in group.overrides


def test_teacher_model_config_mem_fraction_override():
    cfg = _opd_teacher_model_config(_serve_args(opd_teacher_mem_fraction=0.25))
    (group,) = cfg.server_groups
    assert group.overrides["mem_fraction_static"] == 0.25


def test_teacher_model_config_none_when_not_serving():
    assert _opd_teacher_model_config(_serve_args(opd_serve_teacher=False)) is None


def test_placement_extra_gpus():
    assert _opd_teacher_extra_gpus(_serve_args(opd_teacher_num_gpus=2)) == 2
    # Colocate shares the actor/rollout GPUs -- no extra bundles.
    assert _opd_teacher_extra_gpus(_serve_args(colocate=True)) == 0
    assert _opd_teacher_extra_gpus(_serve_args(opd_serve_teacher=False)) == 0


def _validate_args(**overrides):
    defaults = dict(
        advantage_estimator="grpo",
        use_opd=False,
        opd_type="sglang",
        opd_kl_coef=1.0,
        opd_teacher_load=None,
        opd_teacher_ckpt_step=None,
        opd_teacher_url=None,
        opd_icepop=False,
        use_rollout_logprobs=False,
        peft_method="lora",
        opd_teacher=None,
        opd_teacher_urls=None,
        opd_ema_decay=0.999,
        opd_self_teacher_interval=1,
        opd_promote_interval=None,
        custom_rm_path="orbit.rollout.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.rollout.opd_sglang.post_process",
        loss_type="opd_jsd_loss",
        teacher_score_mode="full_vocab",
        teacher_hf_checkpoint="/fake/teacher",
        compute_advantages_and_returns=True,
        opd_serve_teacher=True,
        opd_teacher_num_gpus=1,
        opd_teacher_mem_fraction=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_validation_accepts_managed_full_vocab_without_url():
    args = _validate_args()
    _validate_opd_args(args)
    assert args.compute_advantages_and_returns is False


def test_validation_accepts_managed_sampled_token():
    _validate_opd_args(
        _validate_args(loss_type="policy_loss", teacher_score_mode="sampled_token", advantage_estimator="on_policy_distillation")
    )


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(opd_teacher_url="http://t:1/generate"), "mutually exclusive"),
        (dict(opd_teacher_urls=["default=http://t:1/generate"]), "mutually exclusive"),
        (dict(teacher_hf_checkpoint=None), "serves --teacher-hf-checkpoint"),
        (dict(opd_teacher_num_gpus=0), "must be >= 1"),
        (dict(opd_type="megatron"), "requires --opd-type sglang"),
        (dict(custom_rm_path=None), "custom-reward hooks"),
    ],
)
def test_validation_rejects_bad_serve_configs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_opd_args(_validate_args(**overrides))
