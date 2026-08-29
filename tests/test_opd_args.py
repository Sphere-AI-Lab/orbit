import argparse

import pytest

from miles.utils.arguments import (
    _validate_opd_args,
    add_on_policy_distillation_arguments,
    needs_opd_teacher,
)


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_on_policy_distillation_arguments(parser)
    return parser.parse_args(argv)


def _make_ckpt(tmp_path):
    ckpt = tmp_path / "teacher_ckpt"
    ckpt.mkdir()
    (ckpt / "latest_checkpointed_iteration.txt").write_text("10")
    return str(ckpt)


def _base_args(**overrides):
    defaults = dict(
        advantage_estimator="grpo",
        use_opd=False,
        opd_type=None,
        opd_kl_coef=1.0,
        opd_teacher_load=None,
        opd_teacher_ckpt_step=None,
        opd_teacher_url=None,
        opd_icepop=False,
        use_rollout_logprobs=False,
        peft_method="none",
        adapter_double_buffer=False,
        peft_distributed_transport="nccl",
        opd_teacher=None,
        opd_teacher_urls=None,
        opd_ema_decay=0.999,
        opd_self_teacher_interval=1,
        opd_promote_interval=None,
        custom_rm_path=None,
        custom_reward_post_process_path=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- Task 1.1 Step 1: the five attrs + defaults ---


def test_opd_args_defaults():
    args = _parse([])
    assert args.use_opd is False
    assert args.opd_type is None
    assert args.opd_kl_coef == 1.0
    assert args.opd_teacher_load is None
    assert args.opd_teacher_ckpt_step is None
    assert args.opd_teacher_url is None


def test_opd_args_parse_values():
    args = _parse(
        [
            "--use-opd",
            "--opd-type",
            "megatron",
            "--opd-kl-coef",
            "0.5",
            "--opd-teacher-load",
            "/some/ckpt",
            "--opd-teacher-ckpt-step",
            "100",
        ]
    )
    assert args.use_opd is True
    assert args.opd_type == "megatron"
    assert args.opd_kl_coef == 0.5
    assert args.opd_teacher_load == "/some/ckpt"
    assert args.opd_teacher_ckpt_step == 100


def test_opd_teacher_url_parses_value():
    args = _parse(["--opd-type", "sglang", "--opd-teacher-url", "http://host:1234/generate"])
    assert args.opd_teacher_url == "http://host:1234/generate"


def test_opd_icepop_defaults_false():
    args = _parse([])
    assert args.opd_icepop is False


def test_opd_icepop_parses_true():
    args = _parse(["--opd-icepop"])
    assert args.opd_icepop is True


# --- Phase 3 / Task 3.1: --opd-icepop validation ---


def test_validate_opd_icepop_requires_opd_enabled():
    args = _base_args(opd_icepop=True)  # no OPD estimator, no --use-opd
    with pytest.raises(ValueError, match="opd-icepop"):
        _validate_opd_args(args)


def test_validate_opd_icepop_incompatible_with_use_rollout_logprobs():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        opd_icepop=True,
        use_rollout_logprobs=True,
    )
    with pytest.raises(ValueError, match="use-rollout-logprobs"):
        _validate_opd_args(args)


def test_validate_opd_icepop_passes_with_pure_mopd(tmp_path):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="megatron",
        opd_teacher_load=_make_ckpt(tmp_path),
        opd_icepop=True,
        use_rollout_logprobs=False,
    )
    _validate_opd_args(args)


def test_opd_type_rejects_unknown_choice():
    with pytest.raises(SystemExit):
        _parse(["--opd-type", "vllm"])


# --- shared predicate ---


def test_needs_opd_teacher_true_for_pure_mopd():
    assert needs_opd_teacher(_base_args(advantage_estimator="on_policy_distillation")) is True


def test_needs_opd_teacher_true_for_blend():
    assert needs_opd_teacher(_base_args(use_opd=True)) is True


def test_needs_opd_teacher_false_by_default():
    assert needs_opd_teacher(_base_args()) is False


# --- Task 1.1 Step 1: validation raises ---


def test_validate_rejects_pure_mopd_and_blend_together():
    # (b) mutually exclusive
    args = _base_args(advantage_estimator="on_policy_distillation", use_opd=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_opd_args(args)


def test_validate_requires_opd_type_when_teacher_needed():
    # (c) needs_opd_teacher(args) and opd_type is None
    args = _base_args(advantage_estimator="on_policy_distillation", opd_type=None)
    with pytest.raises(ValueError, match="opd-type"):
        _validate_opd_args(args)


def test_validate_requires_opd_type_when_use_opd():
    args = _base_args(use_opd=True, opd_type=None)
    with pytest.raises(ValueError, match="opd-type"):
        _validate_opd_args(args)


def test_validate_megatron_requires_teacher_load():
    # (a) opd_type='megatron' and opd_teacher_load unset
    args = _base_args(use_opd=True, opd_type="megatron", opd_teacher_load=None)
    with pytest.raises(ValueError, match="opd-teacher-load"):
        _validate_opd_args(args)


def test_validate_megatron_teacher_load_missing_path(tmp_path):
    # (a) opd_type='megatron' and opd_teacher_load missing on disk
    args = _base_args(
        use_opd=True, opd_type="megatron", opd_teacher_load=str(tmp_path / "does_not_exist")
    )
    with pytest.raises(FileNotFoundError):
        _validate_opd_args(args)


def test_validate_sglang_rejects_teacher_load():
    # Pure MOPD (not blend) so this exercises the teacher_load check, not the
    # sglang+blend guard (see test_validate_rejects_sglang_blend below).
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_load="/some/ckpt",
        opd_teacher_url="http://host/generate",
    )
    with pytest.raises(ValueError, match="sglang"):
        _validate_opd_args(args)


def test_validate_sglang_requires_teacher_url():
    # Pure MOPD (not blend) so this exercises the teacher_url check, not the
    # sglang+blend guard (see test_validate_rejects_sglang_blend below).
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="sglang", opd_teacher_url=None
    )
    with pytest.raises(ValueError, match="opd-teacher-url"):
        _validate_opd_args(args)


def test_validate_rejects_peft_with_megatron_teacher(tmp_path):
    # I2: --opd-type megatron loads a full in-process teacher model (like ref),
    # which is incompatible with PEFT. The ref load is already guarded under PEFT;
    # the teacher load must be guarded too or it fails cryptically at load time.
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="megatron",
        opd_teacher_load=_make_ckpt(tmp_path),
        peft_method="lora",
    )
    with pytest.raises(ValueError, match="PEFT"):
        _validate_opd_args(args)


def test_validate_allows_peft_with_sglang_teacher():
    # Only the megatron teacher is rejected under PEFT; the external sglang
    # teacher server is fine.
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        peft_method="lora",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
    )
    _validate_opd_args(args)


def test_validate_rejects_sglang_blend():
    # Fix 2: --use-opd (blend) + --opd-type sglang must be rejected -- the
    # sglang teacher's reward_func occupies the single --custom-rm-path slot
    # and always returns 0.0, so blend would degrade to a KL-only signal with
    # ~0 base advantage. Blend requires --opd-type megatron instead.
    args = _base_args(use_opd=True, opd_type="sglang", opd_teacher_url="http://host/generate")
    with pytest.raises(ValueError, match="megatron"):
        _validate_opd_args(args)


# --- passing paths (no raise) ---


def test_validate_noop_when_opd_disabled():
    _validate_opd_args(_base_args())


def test_validate_megatron_passes_with_valid_ckpt(tmp_path):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="megatron",
        opd_teacher_load=_make_ckpt(tmp_path),
    )
    _validate_opd_args(args)


def test_validate_sglang_passes_without_teacher_load():
    # Pure MOPD (not blend) -- see test_validate_rejects_sglang_blend for the
    # blend+sglang rejection.
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_load=None,
        opd_teacher_url="http://host/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
    )
    _validate_opd_args(args)


def test_validate_sglang_requires_custom_reward_hooks():
    # The sglang teacher produces teacher_log_probs ONLY through its two
    # custom-reward hooks; forgetting them passes validation but dies after a
    # full (expensive) rollout in opd_mopd_advantages. Catch it at startup.
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
    )
    with pytest.raises(ValueError, match="custom-rm-path"):
        _validate_opd_args(args)


def test_validate_sglang_rejects_foreign_custom_rm():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        custom_rm_path="my_pkg.my_rm",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
    )
    with pytest.raises(ValueError, match="custom-rm-path"):
        _validate_opd_args(args)


def test_validate_topk_requires_sglang_teacher(tmp_path):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="megatron",
        opd_teacher_load=_make_ckpt(tmp_path),
        opd_log_prob_top_k=8,
    )
    with pytest.raises(ValueError, match="opd-log-prob-top-k"):
        _validate_opd_args(args)


def test_validate_topk_rejects_negative():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
        opd_log_prob_top_k=-1,
    )
    with pytest.raises(ValueError, match="non-negative"):
        _validate_opd_args(args)


def test_validate_topk_passes_with_sglang(tmp_path):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
        opd_log_prob_top_k=8,
    )
    _validate_opd_args(args)


def test_validate_teacher_urls_requires_sglang(tmp_path):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="megatron",
        opd_teacher_load=_make_ckpt(tmp_path),
        opd_teacher_urls=["math=http://h1/generate"],
    )
    with pytest.raises(ValueError, match="opd-teacher-urls"):
        _validate_opd_args(args)


def test_validate_teacher_urls_fail_fast_on_malformed():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url=None,
        opd_teacher_urls=["malformed-entry"],
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
    )
    with pytest.raises(ValueError, match="expected NAME=URL"):
        _validate_opd_args(args)


def test_validate_sglang_passes_with_teacher_urls_instead_of_url():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url=None,
        opd_teacher_urls=["default=http://h1/generate", "math=http://h2/generate"],
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
    )
    _validate_opd_args(args)


def test_validate_kl_type_forward_requires_topk():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
        opd_kl_type="forward",
        opd_log_prob_top_k=0,
    )
    with pytest.raises(ValueError, match="opd-kl-type"):
        _validate_opd_args(args)


def test_validate_mixed_kl_weight_range():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
        opd_log_prob_top_k=16,
        opd_kl_type="mixed",
        opd_mixed_kl_weight=1.5,
    )
    with pytest.raises(ValueError, match="mixed-kl-weight"):
        _validate_opd_args(args)


def test_validate_kl_type_mixed_passes_with_topk():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher_url="http://host/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
        opd_log_prob_top_k=16,
        opd_kl_type="mixed",
        opd_mixed_kl_weight=0.5,
    )
    _validate_opd_args(args)


# --- Teacher-as-Adapter-Slot: TeacherSpec validation matrix ---


def _make_adapter_dir(tmp_path, peft_type="LORA"):
    d = tmp_path / "teacher_adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(f'{{"peft_type": "{peft_type}"}}')
    return str(d)


def test_opd_teacher_arg_parses():
    args = _parse(["--opd-teacher", "adapter:/x/y"])
    assert args.opd_teacher == "adapter:/x/y"
    assert args.opd_ema_decay == 0.999
    assert args.opd_self_teacher_interval == 1
    assert args.opd_promote_interval is None


def test_legacy_load_still_validates(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    args = _base_args(advantage_estimator="on_policy_distillation", opd_type="megatron", opd_teacher_load=ckpt)
    _validate_opd_args(args)
    assert args.opd_teacher_spec.source == "load"
    assert args.opd_teacher_spec.path == ckpt


def test_load_spec_with_peft_rejected(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher=f"load:{ckpt}", peft_method="lora",
    )
    with pytest.raises(ValueError, match="incompatible with PEFT"):
        _validate_opd_args(args)


def test_same_base_spec_without_peft_rejected():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron", opd_teacher="base",
    )
    with pytest.raises(ValueError, match="adapter structure"):
        _validate_opd_args(args)


def test_base_spec_with_peft_accepted():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher="base", peft_method="lora",
    )
    _validate_opd_args(args)
    assert args.opd_teacher_spec.source == "base"


def test_adapter_spec_missing_dir_rejected():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher="adapter:/nonexistent/dir", peft_method="lora",
    )
    with pytest.raises(FileNotFoundError):
        _validate_opd_args(args)


def test_adapter_spec_method_mismatch_rejected(tmp_path):
    adapter = _make_adapter_dir(tmp_path, peft_type="OFT")
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher=f"adapter:{adapter}", peft_method="lora",
    )
    with pytest.raises(ValueError, match="peft_type"):
        _validate_opd_args(args)


def test_adapter_spec_method_match_accepted(tmp_path):
    adapter = _make_adapter_dir(tmp_path, peft_type="LORA")
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher=f"adapter:{adapter}", peft_method="lora",
    )
    _validate_opd_args(args)
    assert args.opd_teacher_spec.source == "adapter"


def test_self_ema_megatron_accepted_without_promote_interval():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher="self:ema", peft_method="lora",
    )
    _validate_opd_args(args)


def test_bad_ema_decay_rejected():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="megatron",
        opd_teacher="self:ema", peft_method="lora", opd_ema_decay=1.5,
    )
    with pytest.raises(ValueError, match="opd-ema-decay"):
        _validate_opd_args(args)


def test_megatron_without_any_teacher_rejected():
    args = _base_args(advantage_estimator="on_policy_distillation", opd_type="megatron")
    with pytest.raises(ValueError, match="--opd-teacher"):
        _validate_opd_args(args)


# --- Task 8: local-mode sglang + blend unlock ---


def test_sglang_local_mode_accepted_without_url_or_hooks():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="sglang",
        opd_teacher="base", peft_method="oft",
    )
    _validate_opd_args(args)


def test_sglang_local_mode_blend_allowed():
    args = _base_args(
        advantage_estimator="grpo", use_opd=True, opd_type="sglang",
        opd_teacher="self:ema", peft_method="oft", opd_promote_interval=1,
    )
    _validate_opd_args(args)  # custom-rm slot is free: real rewards compose


@pytest.mark.parametrize(
    ("teacher", "promote_interval"),
    (
        ("base", None),
        ("adapter:/teacher", None),
        ("self:ema", 1),
        ("self:lag", 1),
    ),
)
def test_sglang_local_mode_rejects_lora(teacher, promote_interval):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher=teacher,
        peft_method="lora",
        opd_promote_interval=promote_interval,
    )
    with pytest.raises(ValueError, match="single-active"):
        _validate_opd_args(args)


@pytest.mark.parametrize("teacher", ("self:ema", "self:lag"))
def test_sglang_local_mode_rejects_oft_self_teacher_with_double_buffer(teacher):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher=teacher,
        peft_method="oft",
        opd_promote_interval=1,
        adapter_double_buffer=True,
    )
    with pytest.raises(ValueError, match="fixed active OFT slot"):
        _validate_opd_args(args)


@pytest.mark.parametrize("teacher", ("self:ema", "self:lag"))
def test_sglang_local_mode_allows_oft_self_teacher_without_double_buffer(teacher):
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher=teacher,
        peft_method="oft",
        opd_promote_interval=1,
        peft_distributed_transport="ray",
    )
    _validate_opd_args(args)


def test_sglang_local_mode_allows_base_oft_teacher_with_double_buffer():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher="base",
        peft_method="oft",
        adapter_double_buffer=True,
    )
    _validate_opd_args(args)


def test_sglang_local_mode_allows_frozen_oft_teacher_with_double_buffer(tmp_path):
    adapter = _make_adapter_dir(tmp_path, peft_type="OFT")
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher=f"adapter:{adapter}",
        peft_method="oft",
        adapter_double_buffer=True,
    )
    _validate_opd_args(args)


def test_sglang_local_mode_rejects_missing_frozen_oft_teacher():
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher="adapter:/missing/teacher",
        peft_method="oft",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _validate_opd_args(args)


def test_sglang_local_mode_rejects_mismatched_frozen_oft_teacher(tmp_path):
    adapter = _make_adapter_dir(tmp_path, peft_type="LORA")
    args = _base_args(
        advantage_estimator="on_policy_distillation",
        opd_type="sglang",
        opd_teacher=f"adapter:{adapter}",
        peft_method="oft",
    )
    with pytest.raises(ValueError, match="peft_type"):
        _validate_opd_args(args)


def test_sglang_external_url_blend_still_rejected():
    args = _base_args(
        advantage_estimator="grpo", use_opd=True, opd_type="sglang",
        opd_teacher_url="http://h:1/generate",
        custom_rm_path="orbit.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.opd.opd_sglang.post_process",
    )
    with pytest.raises(ValueError, match="blend"):
        _validate_opd_args(args)


def test_sglang_no_teacher_at_all_still_rejected():
    args = _base_args(advantage_estimator="on_policy_distillation", opd_type="sglang")
    with pytest.raises(ValueError, match="opd-teacher"):
        _validate_opd_args(args)


def test_sglang_load_spec_still_rejected():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="sglang",
        opd_teacher="load:/ckpt",
    )
    with pytest.raises(ValueError, match="load"):
        _validate_opd_args(args)


def test_sglang_local_mode_without_peft_rejected():
    args = _base_args(
        advantage_estimator="on_policy_distillation", opd_type="sglang", opd_teacher="base",
    )
    with pytest.raises(ValueError, match="PEFT|adapter structure"):
        _validate_opd_args(args)
        force_on_policy_ratio=False,
        use_tis=False,
        tis_clip_low=0.0,
        tis_clip=2.0,
        num_steps_per_rollout=None,
def test_force_on_policy_ratio_defaults_false_and_parses_true():
    assert _parse([]).force_on_policy_ratio is False
    assert _parse(["--force-on-policy-ratio"]).force_on_policy_ratio is True


def _valid_force_ratio_args(**overrides):
    values = {
        "advantage_estimator": "on_policy_distillation",
        "force_on_policy_ratio": True,
        "use_tis": True,
        "tis_clip_low": 0.2,
        "tis_clip": 5.0,
        "num_steps_per_rollout": 1,
        "opd_type": "sglang",
        "opd_teacher_url": "http://host/generate",
        "custom_rm_path": "orbit.opd.opd_sglang.reward_func",
        "custom_reward_post_process_path": (
            "orbit.opd.opd_sglang.post_process"
        ),
    }
    values.update(overrides)
    return _base_args(**values)


def test_force_on_policy_ratio_accepts_pure_mopd_with_independent_tis():
    _validate_opd_args(_valid_force_ratio_args())


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"advantage_estimator": "grpo"}, "on_policy_distillation"),
        ({"use_opd": True}, "use-opd"),
        ({"use_rollout_logprobs": True}, "use-rollout-logprobs"),
        ({"num_steps_per_rollout": 2}, "one training step|num-steps"),
        ({"num_steps_per_rollout": True}, "one training step|num-steps"),
        ({"force_on_policy_ratio": 1}, "boolean"),
    ),
)
def test_force_on_policy_ratio_rejects_noncanonical_combinations(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_opd_args(_valid_force_ratio_args(**overrides))


@pytest.mark.parametrize(
    ("low", "high", "message"),
    (
        (True, 5.0, "float"),
        (0, 5.0, "float"),
        (0.2, 5, "float"),
        (float("nan"), 5.0, "finite"),
        (0.2, float("inf"), "finite"),
        (-0.1, 5.0, "0 <=|bounds"),
        (1.0, 1.0, "low|bounds"),
        (5.0, 1.0, "low|bounds"),
    ),
)
def test_tis_bounds_are_exact_finite_ordered_floats(
    low: object,
    high: object,
    message: str,
) -> None:
    args = _base_args(
        use_tis=True,
        tis_clip_low=low,
        tis_clip=high,
    )
    with pytest.raises(ValueError, match=message):
        _validate_opd_args(args)


def test_tis_bounds_accept_finite_ordered_floats_without_opd():
    _validate_opd_args(
        _base_args(use_tis=True, tis_clip_low=0.2, tis_clip=5.0)
    )
