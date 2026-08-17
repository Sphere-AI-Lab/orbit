from argparse import Namespace

from orbit.rollout.generate_utils.generate_endpoint_utils import attach_peft_request_payload, compute_request_payload


def _args(peft_method: str) -> Namespace:
    return Namespace(
        peft_method=peft_method,
        rollout_max_context_len=None,
        rollout_max_response_len=8,
        use_rollout_routing_replay=False,
    )


def test_compute_request_payload_attaches_lora_adapter():
    payload, halt_status = compute_request_payload(
        _args("lora"),
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 4},
    )

    assert halt_status is None
    assert payload is not None
    # LoRA names NO adapter on the wire. It routes through the fork's
    # single-active peft/lora, which applies the index-0 adapter
    # unconditionally; sending an adapter key 400s in upstream's
    # _validate_and_resolve_lora when enable_lora is unset.
    assert "lora_path" not in payload
    assert "adapter_path" not in payload
    assert "oft_path" not in payload


def test_compute_request_payload_attaches_oft_adapter():
    payload, halt_status = compute_request_payload(
        _args("oft"),
        input_ids=[1, 2, 3],
        sampling_params={"max_new_tokens": 4},
    )

    assert halt_status is None
    assert payload is not None
    # OFT runs multi-slot (base slot 0 + adapter slot 1) and selects its trained
    # slot via adapter_path -- the v0.5.16 rename of oft_path.
    assert payload["adapter_path"] == "orbit_oft"
    assert "oft_path" not in payload
    assert "lora_path" not in payload


def test_attach_peft_request_payload_keeps_oft_disable_override(monkeypatch):
    monkeypatch.setenv("ORBIT_DSV4_DISABLE_OFT_REQUEST", "1")

    payload = attach_peft_request_payload(_args("oft"), {})

    assert "oft_path" not in payload


def test_attach_peft_request_payload_leaves_non_peft_requests_unchanged():
    payload = {"input_ids": [1, 2, 3]}

    assert attach_peft_request_payload(_args("none"), payload) == payload
    assert "lora_path" not in payload
    assert "oft_path" not in payload
