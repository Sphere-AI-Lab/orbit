from types import SimpleNamespace

import pytest
from sglang.srt.managers.io_struct import (
    ActivateAdapterVersionReqInput,
    GenerateReqInput,
    UpdateAdapterFromDistributedReqInput,
)

from orbit.backends.sglang_utils.sglang_engine import SGLangEngine
from orbit.rollout.generate_utils.generate_endpoint_utils import attach_peft_request_payload


def test_oft_request_selects_adapter_in_submission_sglang(monkeypatch):
    monkeypatch.delenv("ORBIT_DSV4_DISABLE_OFT_REQUEST", raising=False)
    payload = attach_peft_request_payload(SimpleNamespace(peft_method="oft"), {"input_ids": [1, 2]})

    request = GenerateReqInput(**payload)

    assert request.oft_path == "orbit_oft"


def test_fullft_request_does_not_select_an_adapter():
    payload = attach_peft_request_payload(SimpleNamespace(peft_method="none"), {"input_ids": [1, 2]})

    request = GenerateReqInput(**payload)

    assert request.oft_path is None


@pytest.mark.parametrize("double_buffer", [False, True])
def test_adapter_update_rpc_matches_submission_schema(double_buffer):
    # Replace only HTTP I/O; construct the real SGLang request from Orbit's
    # actual payload, including the string-version contract.
    engine = SGLangEngine.__new__(SGLangEngine)
    engine._make_request = lambda endpoint, payload: (endpoint, payload)
    endpoint, payload = engine.update_adapter_from_distributed(
        names=["rotation"],
        dtypes=["torch.bfloat16"],
        shapes=[[1, 128, 128]],
        group_name="test",
        weight_version="2",
        adapter_version="2",
        load_format="oft_adapter",
        adapter_config={"oft_block_size": 128},
        adapter_name="orbit_oft",
        double_buffer=double_buffer,
    )

    request = UpdateAdapterFromDistributedReqInput(**payload)

    assert endpoint == "update_adapter_from_distributed"
    assert request.double_buffer is double_buffer
    assert request.adapter_version == request.weight_version == "2"
    assert request.dtypes == ["bfloat16"]


def test_adapter_activation_rpc_matches_submission_schema():
    engine = SGLangEngine.__new__(SGLangEngine)
    engine._make_request = lambda endpoint, payload: (endpoint, payload)
    endpoint, payload = engine.activate_adapter_version(
        adapter_name="orbit_oft", adapter_version="2", weight_version="2", load_format="oft_adapter"
    )

    request = ActivateAdapterVersionReqInput(**payload)

    assert endpoint == "activate_adapter_version"
    assert request.adapter_name == "orbit_oft"
    assert request.adapter_version == request.weight_version == "2"
