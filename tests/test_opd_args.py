import argparse

import pytest

from orbit.utils.arguments import (
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
    args = _base_args(use_opd=True, opd_type="sglang", opd_teacher_load="/some/ckpt")
    with pytest.raises(ValueError, match="sglang"):
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
    args = _base_args(use_opd=True, opd_type="sglang", opd_teacher_load=None)
    _validate_opd_args(args)
