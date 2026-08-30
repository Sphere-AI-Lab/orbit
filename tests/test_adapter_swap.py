import pytest
import torch

from miles.orbit.utils.adapter_swap import swap_adapter_tensors
from miles.orbit.utils.adapter_tensors import (
    adapter_named_parameters,
    adapter_tensor_key_digest,
)


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


_LOCAL_NAME = "container.adapter.delta"


def _key(chunk: int) -> tuple[int, str]:
    return (chunk, _LOCAL_NAME)


def test_enumeration_preserves_identical_local_names_across_chunks():
    chunks = [_Toy(), _Toy()]

    params = adapter_named_parameters(chunks, _is_adapter)

    assert set(params) == {_key(0), _key(1)}
    assert params[_key(0)] is chunks[0].container.adapter["delta"]
    assert params[_key(1)] is chunks[1].container.adapter["delta"]


def test_two_chunk_swap_is_independent_and_restores():
    chunks = [_Toy(), _Toy()]
    teacher = {
        _key(0): torch.ones(4, 4),
        _key(1): torch.full((4, 4), 2.0),
    }
    originals = [chunk.container.adapter["delta"].detach().clone() for chunk in chunks]

    with swap_adapter_tensors(chunks, teacher, _is_adapter):
        torch.testing.assert_close(chunks[0].container.adapter["delta"], teacher[_key(0)])
        torch.testing.assert_close(chunks[1].container.adapter["delta"], teacher[_key(1)])

    for chunk, original in zip(chunks, originals, strict=True):
        torch.testing.assert_close(chunk.container.adapter["delta"], original)


def test_swap_changes_forward_and_restores():
    model = _Toy()
    x = torch.randn(2, 4)
    before = model(x)
    teacher = {_key(0): torch.ones(4, 4)}
    with swap_adapter_tensors([model], teacher, _is_adapter):
        during = model(x)
    after = model(x)
    assert not torch.allclose(before, during)
    assert torch.allclose(before, after)
    torch.testing.assert_close(during, x @ (model.base.weight + torch.ones(4, 4)).T)


def test_restores_every_chunk_on_exception():
    chunks = [_Toy(), _Toy()]
    originals = [chunk.container.adapter["delta"].detach().clone() for chunk in chunks]
    teacher = {
        _key(0): torch.ones(4, 4),
        _key(1): torch.full((4, 4), 2.0),
    }

    with pytest.raises(RuntimeError, match="boom"):
        with swap_adapter_tensors(chunks, teacher, _is_adapter):
            raise RuntimeError("boom")

    for chunk, original in zip(chunks, originals, strict=True):
        torch.testing.assert_close(chunk.container.adapter["delta"], original)


def test_missing_teacher_tensor_rejected():
    chunks = [_Toy(), _Toy()]
    with pytest.raises(ValueError, match="missing"):
        with swap_adapter_tensors(chunks, {_key(0): torch.ones(4, 4)}, _is_adapter):
            pass


def test_extra_teacher_tensor_rejected():
    model = _Toy()
    teacher = {
        _key(0): torch.ones(4, 4),
        (0, "container.adapter.ghost"): torch.ones(1),
    }
    with pytest.raises(ValueError, match="unknown"):
        with swap_adapter_tensors([model], teacher, _is_adapter):
            pass


def test_shape_mismatch_is_rejected_before_any_chunk_is_mutated():
    chunks = [_Toy(), _Toy()]
    originals = [chunk.container.adapter["delta"].detach().clone() for chunk in chunks]
    teacher = {
        _key(0): torch.ones(4, 4),
        _key(1): torch.ones(3, 4),
    }

    with pytest.raises(ValueError, match="shape"):
        with swap_adapter_tensors(chunks, teacher, _is_adapter):
            pass

    for chunk, original in zip(chunks, originals, strict=True):
        torch.testing.assert_close(chunk.container.adapter["delta"], original)


def test_changed_chunk_count_is_rejected():
    teacher = {
        _key(0): torch.ones(4, 4),
        _key(1): torch.full((4, 4), 2.0),
    }
    with pytest.raises(ValueError, match="unknown"):
        with swap_adapter_tensors([_Toy()], teacher, _is_adapter):
            pass


def test_base_params_untouched():
    model = _Toy()
    base_before = model.base.weight.detach().clone()
    with swap_adapter_tensors([model], {_key(0): torch.ones(4, 4)}, _is_adapter):
        torch.testing.assert_close(model.base.weight, base_before)


def test_tensor_key_digest_is_canonical_and_sensitive():
    keys = [_key(1), _key(0)]
    digest = adapter_tensor_key_digest(keys)

    assert digest == adapter_tensor_key_digest(reversed(keys))
    assert len(digest) == 64
    assert digest != adapter_tensor_key_digest([_key(0)])
    assert digest != adapter_tensor_key_digest([(0, "container.adapter.other"), _key(1)])


@pytest.mark.parametrize(
    "keys",
    (
        [],
        [_key(0), _key(0)],
        [(True, _LOCAL_NAME)],
        [(-1, _LOCAL_NAME)],
        [(0, "")],
        [(0, "   ")],
        [[0, _LOCAL_NAME]],
    ),
)
def test_tensor_key_digest_rejects_invalid_keys(keys):
    with pytest.raises((TypeError, ValueError), match="key|nonempty|unique|invalid"):
        adapter_tensor_key_digest(keys)


def test_enumeration_rejects_duplicate_chunk_objects():
    chunk = _Toy()
    with pytest.raises(ValueError, match="distinct"):
        adapter_named_parameters([chunk, chunk], _is_adapter)


@pytest.mark.parametrize("container", ({_Toy()}, (chunk for chunk in [_Toy()])))
def test_enumeration_rejects_nondeterministic_or_one_shot_containers(container):
    with pytest.raises(TypeError, match="sequence"):
        adapter_named_parameters(container, _is_adapter)


def test_enumeration_requires_at_least_one_selected_parameter():
    with pytest.raises(ValueError, match="no adapter"):
        adapter_named_parameters([torch.nn.Linear(2, 2)], _is_adapter)
