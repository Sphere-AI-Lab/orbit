import pytest
import torch

from orbit.utils.self_teacher import SelfTeacherBuffer


def _params(value: float) -> dict[str, torch.Tensor]:
    return {"adapter.a": torch.full((2, 2), value), "adapter.b": torch.full((3,), value)}


def test_initializes_from_step0_state():
    buf = SelfTeacherBuffer(_params(2.0), mode="ema")
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.full((2, 2), 2.0))


def test_tensors_are_detached_copies():
    live = {"adapter.a": torch.nn.Parameter(torch.ones(2, 2))}
    buf = SelfTeacherBuffer(live, mode="ema")
    live["adapter.a"].data.fill_(5.0)
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.ones(2, 2))
    assert not buf.tensors["adapter.a"].requires_grad


def test_ema_update_math():
    buf = SelfTeacherBuffer(_params(0.0), mode="ema", decay=0.9)
    buf.update(_params(1.0))
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.full((2, 2), 0.1))
    buf.update(_params(1.0))
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.full((2, 2), 0.19))


def test_lag_updates_only_on_interval():
    buf = SelfTeacherBuffer(_params(0.0), mode="lag", interval=2)
    buf.update(_params(1.0))  # step 1: no refresh
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.zeros(2, 2))
    buf.update(_params(2.0))  # step 2: refresh
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.full((2, 2), 2.0))
    buf.update(_params(3.0))  # step 3: no refresh
    torch.testing.assert_close(buf.tensors["adapter.a"], torch.full((2, 2), 2.0))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        SelfTeacherBuffer(_params(0.0), mode="momentum")


def test_key_mismatch_rejected():
    buf = SelfTeacherBuffer(_params(0.0), mode="ema")
    with pytest.raises(ValueError, match="keys"):
        buf.update({"adapter.a": torch.zeros(2, 2)})
