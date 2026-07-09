import pytest
import torch

from orbit.utils.adapter_swap import swap_adapter_tensors


def _is_adapter(name: str) -> bool:
    return ".adapter." in name


class _Container(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.ParameterDict({"delta": torch.nn.Parameter(torch.zeros(4, 4))})


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4, bias=False)
        self.container = _Container()

    def forward(self, x):
        return x @ (self.base.weight + self.container.adapter["delta"]).T


def test_swap_changes_forward_and_restores():
    model = _Toy()
    x = torch.randn(2, 4)
    before = model(x)
    teacher = {"container.adapter.delta": torch.ones(4, 4)}
    with swap_adapter_tensors([model], teacher, _is_adapter):
        during = model(x)
    after = model(x)
    assert not torch.allclose(before, during)
    assert torch.allclose(before, after)
    torch.testing.assert_close(during, x @ (model.base.weight + torch.ones(4, 4)).T)


def test_restores_on_exception():
    model = _Toy()
    original = model.container.adapter["delta"].detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with swap_adapter_tensors([model], {"container.adapter.delta": torch.ones(4, 4)}, _is_adapter):
            raise RuntimeError("boom")
    torch.testing.assert_close(model.container.adapter["delta"], original)


def test_missing_teacher_tensor_rejected():
    model = _Toy()
    with pytest.raises(ValueError, match="missing"):
        with swap_adapter_tensors([model], {}, _is_adapter):
            pass


def test_extra_teacher_tensor_rejected():
    model = _Toy()
    teacher = {"container.adapter.delta": torch.ones(4, 4), "container.adapter.ghost": torch.ones(1)}
    with pytest.raises(ValueError, match="unknown"):
        with swap_adapter_tensors([model], teacher, _is_adapter):
            pass


def test_base_params_untouched():
    model = _Toy()
    base_before = model.base.weight.detach().clone()
    with swap_adapter_tensors([model], {"container.adapter.delta": torch.ones(4, 4)}, _is_adapter):
        torch.testing.assert_close(model.base.weight, base_before)
