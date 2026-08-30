import pytest

from miles.orbit.opd.opd_teacher_spec import (
    OPD_TEACHER_ADAPTER_NAME,
    TeacherSpec,
    is_same_base,
    is_self_teacher,
    needs_engine_teacher_slot,
    parse_teacher_spec,
    teacher_forward_plan,
)


def test_parse_none_when_unset():
    assert parse_teacher_spec(None, None) is None


def test_parse_base():
    assert parse_teacher_spec("base", None) == TeacherSpec("base", None)


def test_parse_adapter_path():
    spec = parse_teacher_spec("adapter:/ckpts/sft_adapter", None)
    assert spec == TeacherSpec("adapter", "/ckpts/sft_adapter")


def test_parse_self_ema_and_lag():
    assert parse_teacher_spec("self:ema", None) == TeacherSpec("self_ema", None)
    assert parse_teacher_spec("self:lag", None) == TeacherSpec("self_lag", None)


def test_parse_load_prefix():
    assert parse_teacher_spec("load:/ckpts/teacher", None) == TeacherSpec("load", "/ckpts/teacher")


def test_legacy_teacher_load_maps_to_load():
    assert parse_teacher_spec(None, "/ckpts/teacher") == TeacherSpec("load", "/ckpts/teacher")


def test_both_args_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_teacher_spec("base", "/ckpts/teacher")


def test_unknown_spec_rejected():
    with pytest.raises(ValueError, match="Unknown --opd-teacher"):
        parse_teacher_spec("ema", None)


def test_empty_adapter_path_rejected():
    with pytest.raises(ValueError, match="empty path"):
        parse_teacher_spec("adapter:", None)


def test_same_base_predicate():
    assert is_same_base(TeacherSpec("base", None))
    assert is_same_base(TeacherSpec("adapter", "/x"))
    assert is_same_base(TeacherSpec("self_ema", None))
    assert is_same_base(TeacherSpec("self_lag", None))
    assert not is_same_base(TeacherSpec("load", "/x"))
    assert not is_same_base(None)


def test_self_teacher_predicate():
    assert is_self_teacher(TeacherSpec("self_ema", None))
    assert is_self_teacher(TeacherSpec("self_lag", None))
    assert not is_self_teacher(TeacherSpec("adapter", "/x"))
    assert not is_self_teacher(None)


def test_engine_slot_predicate():
    # base scores against the engine's base weights: no slot needed.
    assert not needs_engine_teacher_slot(TeacherSpec("base", None))
    assert needs_engine_teacher_slot(TeacherSpec("adapter", "/x"))
    assert needs_engine_teacher_slot(TeacherSpec("self_ema", None))
    assert needs_engine_teacher_slot(TeacherSpec("self_lag", None))
    assert not needs_engine_teacher_slot(TeacherSpec("load", "/x"))
    assert not needs_engine_teacher_slot(None)


def test_adapter_name_constant():
    assert OPD_TEACHER_ADAPTER_NAME == "orbit_teacher"


def test_plan_none_without_spec():
    assert teacher_forward_plan(None, peft_enabled=True, ref_available=True, opd_type="megatron") == "none"


def test_plan_load_is_switch_model():
    assert (
        teacher_forward_plan(TeacherSpec("load", "/x"), peft_enabled=False, ref_available=False, opd_type="megatron")
        == "switch_model"
    )


def test_plan_base_aliases_ref_when_available():
    assert (
        teacher_forward_plan(TeacherSpec("base", None), peft_enabled=True, ref_available=True, opd_type="megatron")
        == "alias_ref"
    )


def test_plan_base_disables_adapter_without_ref():
    assert (
        teacher_forward_plan(TeacherSpec("base", None), peft_enabled=True, ref_available=False, opd_type="megatron")
        == "adapter_off"
    )


def test_plan_adapter_and_self_swap():
    for source in ("adapter", "self_ema", "self_lag"):
        assert (
            teacher_forward_plan(TeacherSpec(source, "/x"), peft_enabled=True, ref_available=True, opd_type="megatron")
            == "adapter_swap"
        )


def test_plan_same_base_without_peft_raises():
    with pytest.raises(ValueError, match="PEFT"):
        teacher_forward_plan(TeacherSpec("base", None), peft_enabled=False, ref_available=False, opd_type="megatron")


_ALL_SOURCES = (
    None,
    TeacherSpec("base", None),
    TeacherSpec("adapter", "/x"),
    TeacherSpec("self_ema", None),
    TeacherSpec("self_lag", None),
    TeacherSpec("load", "/x"),
)


def test_plan_sglang_is_none_for_every_source():
    # sglang teachers are scored on the rollout engine; the trainer produces
    # nothing (engine-scored teacher_log_probs are authoritative).
    for spec in _ALL_SOURCES:
        for ref_available in (True, False):
            assert (
                teacher_forward_plan(spec, peft_enabled=True, ref_available=ref_available, opd_type="sglang")
                == "none"
            )


def test_plan_sglang_adapter_is_none_regression():
    # Regression for the crashed config: sglang + adapter:<path> used to reach
    # the trainer's adapter_swap branch and RuntimeError "has no tensors loaded"
    # because with_opd_teacher is megatron-only. Engine scoring is authoritative.
    for ref_available in (True, False):
        assert (
            teacher_forward_plan(
                TeacherSpec("adapter", "/x"),
                peft_enabled=True,
                ref_available=ref_available,
                opd_type="sglang",
            )
            == "none"
        )


def test_plan_megatron_routing_unchanged():
    # opd_type == "megatron" preserves the original per-source routing.
    assert teacher_forward_plan(None, peft_enabled=True, ref_available=True, opd_type="megatron") == "none"
    assert (
        teacher_forward_plan(TeacherSpec("load", "/x"), peft_enabled=False, ref_available=False, opd_type="megatron")
        == "switch_model"
    )
    assert (
        teacher_forward_plan(TeacherSpec("base", None), peft_enabled=True, ref_available=True, opd_type="megatron")
        == "alias_ref"
    )
    assert (
        teacher_forward_plan(TeacherSpec("base", None), peft_enabled=True, ref_available=False, opd_type="megatron")
        == "adapter_off"
    )
    for source in ("adapter", "self_ema", "self_lag"):
        assert (
            teacher_forward_plan(TeacherSpec(source, "/x"), peft_enabled=True, ref_available=True, opd_type="megatron")
            == "adapter_swap"
        )
