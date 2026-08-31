import time

import pytest
import requests


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


@pytest.mark.parametrize(
    ("method_name", "kwargs", "expected_endpoint", "expected_payload"),
    [
        (
            "update_adapter_from_distributed",
            {
                "names": ["adapter.layer.weight"],
                "dtypes": ["torch.float16"],
                "shapes": [[2, 4]],
                "group_name": "peft-update",
                "weight_version": "7",
                "adapter_version": "7",
                "load_format": "lora_adapter",
                "adapter_config": {"r": 8},
                "adapter_name": "policy",
                "payload_metadata": {"entries": [["adapter.layer.weight", 0]]},
                "double_buffer": True,
            },
            "update_adapter_from_distributed",
            {
                "names": ["adapter.layer.weight"],
                "dtypes": ["float16"],
                "shapes": [[2, 4]],
                "group_name": "peft-update",
                "weight_version": "7",
                "adapter_version": "7",
                "load_format": "lora_adapter",
                "adapter_config": {"r": 8},
                "adapter_name": "policy",
                "payload_metadata": {"entries": [["adapter.layer.weight", 0]]},
                "double_buffer": True,
            },
        ),
        (
            "activate_adapter_version",
            {
                "adapter_name": "policy",
                "adapter_version": "7",
                "weight_version": "7",
                "load_format": "lora_adapter",
            },
            "activate_adapter_version",
            {
                "adapter_name": "policy",
                "adapter_version": "7",
                "weight_version": "7",
                "load_format": "lora_adapter",
            },
        ),
    ],
)
def test_nccl_adapter_methods_post_the_sglang_endpoint_contract(
    monkeypatch, method_name, kwargs, expected_endpoint, expected_payload
):
    """NCCL sync must reach SGLang with the adapter versioning contract intact."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234
    requests_seen = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True}

    def post(url, json):
        requests_seen.append((url, json))
        return Response()

    monkeypatch.setattr(requests, "post", post)

    assert getattr(engine, method_name)(**kwargs) == {"success": True}
    assert requests_seen == [(f"http://fake-host:1234/{expected_endpoint}", expected_payload)]
