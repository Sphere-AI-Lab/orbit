import copy

import pytest
import torch

from orbit.utils.adapter_swap import swap_adapter_tensors
from orbit.utils.adapter_tensors import adapter_tensor_key_digest
from orbit.utils.self_teacher import (
    SELF_TEACHER_STATE_SCHEMA_VERSION,
    SelfTeacherBuffer,
)


_A = (0, "adapter.a")
_B = (1, "adapter.b")


def _params(value: float, *, dtype=torch.float32) -> dict[tuple[int, str], torch.Tensor]:
    return {
        _A: torch.full((2, 2), value, dtype=dtype),
        _B: torch.full((3,), value, dtype=dtype),
    }


def test_initializes_from_step0_state():
    buf = SelfTeacherBuffer(_params(2.0), mode="ema")
    torch.testing.assert_close(buf.tensors[_A], torch.full((2, 2), 2.0))
    assert buf._step == 0


def test_tensors_are_detached_contiguous_fp32_copies():
    live = {_A: torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16).T)}
    buf = SelfTeacherBuffer(live, mode="ema")
    live[_A].data.fill_(5.0)

    torch.testing.assert_close(buf.tensors[_A], torch.ones(2, 2))
    assert buf.tensors[_A].dtype is torch.float32
    assert buf.tensors[_A].is_contiguous()
    assert not buf.tensors[_A].requires_grad
    assert buf.tensors[_A].data_ptr() != live[_A].data_ptr()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_ema_and_lag_masters_remain_fp32(dtype):
    ema = SelfTeacherBuffer(_params(0.0, dtype=dtype), mode="ema", decay=0.9)
    lag = SelfTeacherBuffer(_params(0.0, dtype=dtype), mode="lag", interval=1)

    ema.update(_params(1.0, dtype=dtype))
    lag.update(_params(1.0, dtype=dtype))

    assert all(tensor.dtype is torch.float32 for tensor in ema.tensors.values())
    assert all(tensor.dtype is torch.float32 for tensor in lag.tensors.values())


def test_only_swap_casts_fp32_master_to_live_parameter_dtype():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    model = Toy()
    buf = SelfTeacherBuffer({(0, "adapter"): torch.full((2,), 1.25)}, mode="lag")

    assert buf.tensors[(0, "adapter")].dtype is torch.float32
    with swap_adapter_tensors([model], buf.tensors, lambda name: name == "adapter"):
        assert model.adapter.dtype is torch.bfloat16
        torch.testing.assert_close(model.adapter.float(), torch.full((2,), 1.25))
    assert buf.tensors[(0, "adapter")].dtype is torch.float32


def test_ema_update_math():
    buf = SelfTeacherBuffer(_params(0.0), mode="ema", decay=0.9)
    buf.update(_params(1.0))
    torch.testing.assert_close(buf.tensors[_A], torch.full((2, 2), 0.1))
    buf.update(_params(1.0))
    torch.testing.assert_close(buf.tensors[_A], torch.full((2, 2), 0.19))


def test_bf16_ema_matches_fp32_reference_without_bf16_accumulation_drift():
    decay = 0.997
    initial = torch.tensor([0.125], dtype=torch.bfloat16)
    buf = SelfTeacherBuffer({_A: initial}, mode="ema", decay=decay)
    reference = initial.float().clone()
    bf16_control = initial.clone()

    for step in range(2_000):
        live = torch.tensor([((step % 29) - 14) / 128.0], dtype=torch.bfloat16)
        buf.update({_A: live})
        reference.mul_(decay).add_(live.float(), alpha=1.0 - decay)
        bf16_control.mul_(decay).add_(live, alpha=1.0 - decay)

    torch.testing.assert_close(buf.tensors[_A], reference, rtol=0, atol=1e-6)
    assert not torch.allclose(buf.tensors[_A], bf16_control.float(), rtol=0, atol=1e-6)


def test_lag_updates_only_on_interval():
    buf = SelfTeacherBuffer(_params(0.0, dtype=torch.float16), mode="lag", interval=2)
    buf.update(_params(1.0, dtype=torch.float16))
    torch.testing.assert_close(buf.tensors[_A], torch.zeros(2, 2))
    buf.update(_params(2.0, dtype=torch.float16))
    torch.testing.assert_close(buf.tensors[_A], torch.full((2, 2), 2.0))
    buf.update(_params(3.0, dtype=torch.float16))
    torch.testing.assert_close(buf.tensors[_A], torch.full((2, 2), 2.0))


def test_state_dict_round_trip_is_exact_and_detached():
    buf = SelfTeacherBuffer(_params(0.5, dtype=torch.bfloat16), mode="ema", decay=0.9, interval=3)
    buf.update(_params(1.0, dtype=torch.bfloat16))
    buf.update(_params(2.0, dtype=torch.bfloat16))

    state = buf.state_dict()

    assert set(state) == {
        "schema_version",
        "mode",
        "decay",
        "interval",
        "step",
        "key_digest",
        "tensors",
    }
    assert state["schema_version"] == SELF_TEACHER_STATE_SCHEMA_VERSION == 1
    assert state["mode"] == "ema"
    assert state["decay"] == 0.9
    assert state["interval"] == 3
    assert state["step"] == 2
    assert state["key_digest"] == adapter_tensor_key_digest((_A, _B))
    assert set(state["tensors"]) == {_A, _B}
    assert state["tensors"][_A].shape == (2, 2)
    assert state["tensors"][_B].shape == (3,)
    assert all(tensor.device.type == "cpu" for tensor in state["tensors"].values())
    assert all(tensor.dtype is torch.float32 for tensor in state["tensors"].values())

    restored = SelfTeacherBuffer.from_state_dict(state)
    assert restored.mode == buf.mode
    assert restored.decay == buf.decay
    assert restored.interval == buf.interval
    assert restored._step == buf._step
    for key in buf.tensors:
        torch.testing.assert_close(restored.tensors[key], buf.tensors[key])

    state["tensors"][_A].zero_()
    assert not torch.equal(state["tensors"][_A], buf.tensors[_A])


def test_load_state_dict_restores_into_existing_devices_atomically():
    source = SelfTeacherBuffer(_params(1.0), mode="lag", interval=2)
    source.update(_params(2.0))
    source.update(_params(3.0))
    target = SelfTeacherBuffer(_params(0.0), mode="lag", interval=2)

    target.load_state_dict(source.state_dict())

    assert target._step == 2
    for key in source.tensors:
        torch.testing.assert_close(target.tensors[key], source.tensors[key])
        assert target.tensors[key].device == _params(0.0)[key].device


def _state_and_target():
    source = SelfTeacherBuffer(_params(1.0), mode="ema", decay=0.9, interval=2)
    source.update(_params(2.0))
    target = SelfTeacherBuffer(_params(7.0), mode="ema", decay=0.9, interval=2)
    return copy.deepcopy(source.state_dict()), target


def _assert_target_unchanged(target, before_step, before_tensors):
    assert target._step == before_step
    for key, tensor in before_tensors.items():
        torch.testing.assert_close(target.tensors[key], tensor)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", "lag"),
        ("decay", 0.8),
        ("interval", 3),
    ),
)
def test_load_rejects_changed_configuration_without_partial_mutation(field, value):
    state, target = _state_and_target()
    state[field] = value
    before_step = target._step
    before_tensors = {key: tensor.clone() for key, tensor in target.tensors.items()}

    with pytest.raises(ValueError, match="configured|match|state"):
        target.load_state_dict(state)

    _assert_target_unchanged(target, before_step, before_tensors)


def _unknown_field(state):
    state["future"] = 1


def _schema_change(state):
    state["schema_version"] = 2


def _negative_step(state):
    state["step"] = -1


def _digest_change(state):
    state["key_digest"] = "0" * 64


def _key_change(state):
    tensor = state["tensors"].pop(_B)
    state["tensors"][(2, _B[1])] = tensor
    state["key_digest"] = adapter_tensor_key_digest(state["tensors"])


def _local_name_change(state):
    tensor = state["tensors"].pop(_A)
    state["tensors"][(0, "adapter.changed")] = tensor
    state["key_digest"] = adapter_tensor_key_digest(state["tensors"])


def _shape_change(state):
    state["tensors"][_A] = torch.ones(1, dtype=torch.float32)


def _dtype_change(state):
    state["tensors"][_A] = state["tensors"][_A].to(torch.bfloat16)


def _nonfinite_change(state):
    state["tensors"][_A][0, 0] = torch.nan


@pytest.mark.parametrize(
    "mutate",
    (
        _unknown_field,
        _schema_change,
        _negative_step,
        _digest_change,
        _key_change,
        _local_name_change,
        _shape_change,
        _dtype_change,
        _nonfinite_change,
    ),
)
def test_invalid_state_is_rejected_without_partial_mutation(mutate):
    state, target = _state_and_target()
    mutate(state)
    before_step = target._step
    before_tensors = {key: tensor.clone() for key, tensor in target.tensors.items()}

    with pytest.raises((TypeError, ValueError), match="state|schema|step|digest|tensor|match|shape"):
        target.load_state_dict(state)

    _assert_target_unchanged(target, before_step, before_tensors)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        SelfTeacherBuffer(_params(0.0), mode="momentum")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"decay": True}, "decay"),
        ({"decay": float("nan")}, "decay"),
        ({"decay": 0.0}, "decay"),
        ({"decay": 1.0}, "decay"),
        ({"interval": 0}, "interval"),
        ({"interval": True}, "interval"),
    ),
)
def test_invalid_configuration_rejected(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        SelfTeacherBuffer(_params(0.0), mode="ema", **kwargs)


def test_update_key_mismatch_rejected_before_step_change():
    buf = SelfTeacherBuffer(_params(0.0), mode="ema")
    with pytest.raises(ValueError, match="keys"):
        buf.update({_A: torch.zeros(2, 2)})
    assert buf._step == 0


def test_update_shape_mismatch_is_rejected_before_mutation():
    buf = SelfTeacherBuffer(_params(0.0), mode="ema")
    before = {key: tensor.clone() for key, tensor in buf.tensors.items()}
    invalid = _params(1.0)
    invalid[_B] = torch.ones(2)

    with pytest.raises(ValueError, match="shape"):
        buf.update(invalid)

    assert buf._step == 0
    for key in before:
        torch.testing.assert_close(buf.tensors[key], before[key])
