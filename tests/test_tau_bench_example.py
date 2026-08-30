import sys
import types
from argparse import Namespace
from types import SimpleNamespace

from miles_plugins.tau_bench.generate_with_tau import (
    _task_index_from_sample,
    append_environment_delta,
    build_generation_payload,
)
from miles_plugins.tau_bench.openai_tool_adapter import OpenAICompatibleToolCallAdapter
from miles.utils.types import Sample


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def _args(peft_method="none"):
    return Namespace(
        peft_method=peft_method,
        rollout_max_context_len=None,
        rollout_max_response_len=8,
        use_rollout_routing_replay=False,
        use_miles_router=False,
        miles_router_middleware_paths=[],
        eval_return_rollout_logprobs=False,
    )


def test_tau_task_index_prefers_metadata_index():
    assert _task_index_from_sample(Sample(prompt="7", metadata={"index": 3})) == 3
    assert _task_index_from_sample(Sample(prompt="7")) == 7


def test_tau_append_environment_delta_masks_new_tokens_and_aligns_logprobs():
    sample = Sample(
        tokens=[1, 2],
        response="ab",
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
    )

    ok = append_environment_delta(
        sample,
        [1, 2, ord("x"), ord("y")],
        FakeTokenizer(),
        has_rollout_logprobs=True,
    )

    assert ok is True
    assert sample.tokens == [1, 2, ord("x"), ord("y")]
    assert sample.loss_mask == [1, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0]
    sample.validate()


def test_tau_append_environment_delta_flags_non_append_mismatch():
    sample = Sample(tokens=[1, 2], metadata={})

    assert append_environment_delta(sample, [1, 3], FakeTokenizer(), has_rollout_logprobs=False) is False
    assert "tau_bench_token_mismatch" in sample.metadata


def test_tau_build_generation_payload_requests_logprobs(monkeypatch):
    module_name = "miles.rollout.generate_utils.generate_endpoint_utils"
    fake_module = types.ModuleType(module_name)
    captured = {}

    def fake_should_request_rollout_logprobs(args, evaluation=False):
        captured["evaluation"] = evaluation
        return True

    def fake_compute_request_payload(args, input_ids, sampling_params, return_logprob=True):
        captured["return_logprob"] = return_logprob
        return {"input_ids": input_ids, "return_logprob": return_logprob}, None

    fake_module.should_request_rollout_logprobs = fake_should_request_rollout_logprobs
    fake_module.compute_request_payload = fake_compute_request_payload
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    payload, halt_status = build_generation_payload(_args("oft"), [1, 2, 3], {"max_new_tokens": 4})

    assert halt_status is None
    assert payload["return_logprob"] is True
    assert captured == {"evaluation": False, "return_logprob": True}


def test_openai_tool_adapter_converts_call_to_tau_action(monkeypatch):
    tau_module = types.ModuleType("tau_bench")
    agents_module = types.ModuleType("tau_bench.agents")
    tool_calling_module = types.ModuleType("tau_bench.agents.tool_calling_agent")
    tool_calling_module.RESPOND_ACTION_NAME = "respond"

    types_module = types.ModuleType("tau_bench.types")

    class FakeAction(SimpleNamespace):
        pass

    types_module.Action = FakeAction
    monkeypatch.setitem(sys.modules, "tau_bench", tau_module)
    monkeypatch.setitem(sys.modules, "tau_bench.agents", agents_module)
    monkeypatch.setitem(sys.modules, "tau_bench.agents.tool_calling_agent", tool_calling_module)
    monkeypatch.setitem(sys.modules, "tau_bench.types", types_module)

    adapter = OpenAICompatibleToolCallAdapter([])
    action = adapter.call_to_action([{"name": "lookup", "parameters": '{"order_id": "123"}'}], "")

    assert action.name == "lookup"
    assert action.kwargs == {"order_id": "123"}
