import pytest

from orbit.utils.opd_teacher_spec import (
    OPD_TEACHER_ADAPTER_NAME,
    TeacherSpec,
    is_same_base,
    is_self_teacher,
    needs_engine_teacher_slot,
    parse_teacher_spec,
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
