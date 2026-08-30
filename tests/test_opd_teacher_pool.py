"""Teacher pool manifests (--opd-teacher-pool): parse/validate, GPU accounting,
sglang model-entry construction, and routing-spec synthesis."""

import argparse
import json

import pytest

from miles.ray.placement_group import _opd_teacher_extra_gpus
from miles.orbit.opd.teacher_servers import _opd_teacher_pool, _opd_teacher_pool_model_configs
from miles.orbit.opd.opd_teacher_pool import TeacherPoolError, parse_teacher_pool


def _write_manifest(tmp_path, teachers):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"teachers": teachers}))
    return str(path)


def _two_teacher_manifest(tmp_path):
    return _write_manifest(
        tmp_path,
        [
            {"name": "math", "kind": "served", "model_path": "/ckpts/7B", "num_gpus": 2, "mem_fraction": 0.6},
            {"name": "default", "kind": "url", "url": "http://ext:30001/generate", "weight": 2.0},
        ],
    )


def test_parse_and_accounting(tmp_path):
    pool = parse_teacher_pool(_two_teacher_manifest(tmp_path))
    assert [e.name for e in pool.entries] == ["math", "default"]
    assert pool.served_num_gpus == 2
    assert pool.served[0].served_model_name == "opd_teacher_math"


@pytest.mark.parametrize(
    "teachers, match",
    [
        ([{"name": "a", "kind": "nope", "url": "u"}], "kind must be one of"),
        ([{"name": "a", "kind": "url"}], "url must be a non-empty string"),
        ([{"name": "a", "kind": "served"}], "model_path must be a non-empty string"),
        ([{"name": "a", "kind": "url", "url": "u", "num_gpus": 1}], "takes only"),
        ([{"name": "a", "kind": "served", "model_path": "m", "num_gpus": 0}], "positive integer"),
        ([{"name": "a", "kind": "served", "model_path": "m", "bogus": 1}], "unknown fields"),
        ([{"name": "a", "kind": "served", "model_path": "m"}, {"name": "a", "kind": "served", "model_path": "m"}], "unique"),
    ],
)
def test_parse_rejects_bad_manifests(tmp_path, teachers, match):
    with pytest.raises(TeacherPoolError, match=match):
        parse_teacher_pool(_write_manifest(tmp_path, teachers))


def test_model_configs_and_placement(tmp_path):
    args = argparse.Namespace(opd_teacher_pool=_two_teacher_manifest(tmp_path), opd_serve_teacher=False, colocate=False)
    cfgs = _opd_teacher_pool_model_configs(args)
    assert [c.name for c in cfgs] == ["opd_teacher_math"]
    assert cfgs[0].update_weights is False
    assert cfgs[0].num_gpus_per_engine == 2
    (group,) = cfgs[0].server_groups
    assert group.num_gpus == 2
    assert group.overrides["mem_fraction_static"] == 0.6
    assert group.overrides["disable_radix_cache"] is True
    assert _opd_teacher_extra_gpus(args) == 2
    assert _opd_teacher_extra_gpus(argparse.Namespace(opd_teacher_pool=args.opd_teacher_pool, opd_serve_teacher=False, colocate=True)) == 0


def test_routing_specs_after_serving(tmp_path):
    pool = parse_teacher_pool(_two_teacher_manifest(tmp_path))
    specs = pool.routing_specs({"opd_teacher_math": "http://10.0.0.1:3100/generate"})
    assert sorted(specs) == [
        "default=http://ext:30001/generate@2.0",
        "math=http://10.0.0.1:3100/generate@1.0",
    ]
    with pytest.raises(TeacherPoolError, match="no published endpoint"):
        pool.routing_specs({})


def test_routing_specs_feed_the_existing_router(tmp_path):
    from miles.orbit.opd.opd_sglang import parse_teacher_urls

    pool = parse_teacher_pool(_two_teacher_manifest(tmp_path))
    url_map = parse_teacher_urls(pool.routing_specs({"opd_teacher_math": "http://10.0.0.1:3100/generate"}))
    assert url_map["math"] == [("http://10.0.0.1:3100/generate", 1.0)]
    assert url_map["default"] == [("http://ext:30001/generate", 2.0)]


def _validate_args(tmp_path, **overrides):
    from miles.utils.arguments import _validate_opd_args

    defaults = dict(
        advantage_estimator="on_policy_distillation",
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
        custom_rm_path="miles.orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="miles.orbit.opd.opd_sglang.post_process",
        loss_type="policy_loss",
        teacher_score_mode="sampled_token",
        teacher_hf_checkpoint=None,
        opd_serve_teacher=False,
        opd_teacher_num_gpus=1,
        opd_teacher_mem_fraction=None,
        opd_teacher_pool=_two_teacher_manifest(tmp_path),
    )
    defaults.update(overrides)
    args = argparse.Namespace(**defaults)
    _validate_opd_args(args)
    return args


def test_validation_accepts_pool(tmp_path):
    _validate_args(tmp_path)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(opd_serve_teacher=True, teacher_hf_checkpoint="/x"), "subsumes"),
        (dict(opd_teacher_url="http://t/generate"), "subsumes"),
        (dict(teacher_score_mode="full_vocab", loss_type="opd_jsd_loss", teacher_hf_checkpoint="/x"), "sampled-token only"),
        (dict(opd_type="megatron"), "requires --opd-type sglang"),
        (dict(custom_rm_path=None), "custom-reward hooks|custom_rm"),
    ],
)
def test_validation_rejects_bad_pool_configs(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_args(tmp_path, **overrides)
