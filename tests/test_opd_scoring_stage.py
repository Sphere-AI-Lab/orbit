from argparse import Namespace

import pytest

from miles.orbit.opd.opd_scoring import local_scoring_enabled, teacher_lora_path
from miles.orbit.opd.opd_sglang import _score_payload


def _args(**overrides):
    defaults = dict(
        opd_type="sglang",
        opd_teacher=None,
        opd_teacher_load=None,
        opd_teacher_url=None,
        opd_teacher_urls=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_local_scoring_needs_same_base_spec():
    assert local_scoring_enabled(_args(opd_teacher="base"))
    assert local_scoring_enabled(_args(opd_teacher="adapter:/x"))
    assert local_scoring_enabled(_args(opd_teacher="self:ema"))
    assert not local_scoring_enabled(_args())  # no teacher at all
    assert not local_scoring_enabled(_args(opd_teacher_load="/ckpt"))  # load: is not same-base


def test_external_url_wins_over_local():
    assert not local_scoring_enabled(_args(opd_teacher="base", opd_teacher_url="http://h:1/generate"))
    assert not local_scoring_enabled(_args(opd_teacher="base", opd_teacher_urls=["m=http://h:1/generate"]))


def test_local_scoring_requires_sglang_type():
    assert not local_scoring_enabled(_args(opd_type="megatron", opd_teacher="base"))


def test_teacher_lora_path_base_is_none():
    assert teacher_lora_path(_args(opd_teacher="base")) is None


def test_teacher_lora_path_adapter_and_self():
    assert teacher_lora_path(_args(opd_teacher="adapter:/x")) == "orbit_teacher"
    assert teacher_lora_path(_args(opd_teacher="self:ema")) == "orbit_teacher"


def test_score_payload_lora_path_threading():
    with_lora = _score_payload([1, 2, 3], lora_path="orbit_teacher")
    assert with_lora["lora_path"] == "orbit_teacher"
    without = _score_payload([1, 2, 3])
    assert "lora_path" not in without
    # existing fields unchanged
    assert without["sampling_params"]["max_new_tokens"] == 0
    assert without["return_logprob"] is True


def test_actor_teacher_state_keeps_vpp_chunk_identity(monkeypatch):
    import torch

    import miles.backends.megatron_utils.actor as actor_module

    class Chunk(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.container = torch.nn.Module()
            self.container.adapter = torch.nn.ParameterDict(
                {"delta": torch.nn.Parameter(torch.full((1,), value))}
            )

    monkeypatch.setattr(actor_module, "is_adapter_param_name", lambda name: ".adapter." in name)
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.model = [Chunk(1.0), Chunk(2.0)]

    params = actor._adapter_named_params()

    assert set(params) == {
        (0, "container.adapter.delta"),
        (1, "container.adapter.delta"),
    }
    assert params[(0, "container.adapter.delta")] is actor.model[0].container.adapter["delta"]
    assert params[(1, "container.adapter.delta")] is actor.model[1].container.adapter["delta"]
