import asyncio
import sys
import types
from argparse import Namespace

import pytest

from examples.search_r1.generate_with_search import (
    append_environment_observation,
    build_generation_payload,
    postprocess_predictions,
    reward_func,
)
from examples.search_r1.qa_em_format import compute_score_em, extract_information_blocks
from orbit.utils.types import Sample


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def _args(peft_method="none"):
    return Namespace(
        peft_method=peft_method,
        rollout_max_context_len=None,
        rollout_max_response_len=8,
        use_rollout_routing_replay=False,
        use_orbit_router=False,
        orbit_router_middleware_paths=[],
        eval_return_rollout_logprobs=False,
        search_r1_format_score=0.2,
    )


def _valid_solution(answer="Paris"):
    return (
        "example <answer>demo</answer>"
        "<|im_start|>assistant\n"
        "<think>I should search.</think>"
        "<search>capital of France</search>"
        "<information>Paris is the capital of France.</information>"
        "<think>The passage contains the answer.</think>"
        f"<answer>{answer}</answer>"
    )


def test_postprocess_predictions_extracts_first_search_action():
    action, content = postprocess_predictions("<think>x</think><search>capital of France</search>")

    assert action == "search"
    assert content == "capital of France"


def test_search_r1_reward_exact_match():
    score = compute_score_em(_valid_solution(), {"target": ["Paris"]}, format_score=0.2)

    assert score == 1.0


def test_search_r1_reward_func_reads_ground_truth_label():
    sample = Sample(
        prompt="example <answer>demo</answer><|im_start|>assistant\n",
        response=(
            "<think>I should search.</think>"
            "<search>capital of France</search>"
            "<information>Paris is the capital of France.</information>"
            "<think>The passage contains the answer.</think>"
            "<answer>Paris</answer>"
        ),
        label={"ground_truth": {"target": ["Paris"]}},
    )

    assert pytest.approx(asyncio.run(reward_func(_args(), sample))) == 1.0


def test_search_r1_reward_func_supports_batched_custom_rm_call():
    sample = Sample(
        prompt="example <answer>demo</answer><|im_start|>assistant\n",
        response=(
            "<think>I should search.</think>"
            "<search>capital of France</search>"
            "<information>Paris is the capital of France.</information>"
            "<think>The passage contains the answer.</think>"
            "<answer>Paris</answer>"
        ),
        label={"ground_truth": {"target": ["Paris"]}},
    )

    assert asyncio.run(reward_func(_args(), [sample, sample])) == [1.0, 1.0]


def test_extract_information_blocks():
    assert extract_information_blocks("a<information>doc one</information>b<information>doc two</information>") == [
        "doc one",
        "doc two",
    ]


def test_append_environment_observation_masks_tokens_and_aligns_logprobs():
    sample = Sample(
        prompt="prompt",
        tokens=[1, 2, 3],
        response="assistant",
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        status=Sample.Status.COMPLETED,
    )

    append_environment_observation(
        sample,
        "\n\n<information>doc</information>\n\n",
        FakeTokenizer(),
        has_rollout_logprobs=True,
    )

    assert sample.loss_mask[0] == 1
    assert set(sample.loss_mask[1:]) == {0}
    assert sample.rollout_log_probs[0] == -0.3
    assert set(sample.rollout_log_probs[1:]) == {0.0}
    sample.validate()


def test_build_generation_payload_requests_rollout_logprobs(monkeypatch):
    module_name = "orbit.rollout.generate_utils.generate_endpoint_utils"
    fake_module = types.ModuleType(module_name)
    captured = {}

    def fake_should_request_rollout_logprobs(args, evaluation=False):
        captured["evaluation"] = evaluation
        return True

    def fake_compute_request_payload(args, input_ids, sampling_params, return_logprob=True):
        captured["input_ids"] = input_ids
        captured["sampling_params"] = sampling_params
        captured["return_logprob"] = return_logprob
        return {"input_ids": input_ids, "return_logprob": return_logprob}, None

    fake_module.should_request_rollout_logprobs = fake_should_request_rollout_logprobs
    fake_module.compute_request_payload = fake_compute_request_payload
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    payload, halt_status = build_generation_payload(
        _args("oft"),
        [1, 2, 3],
        {"max_new_tokens": 4, "temperature": 1.0},
    )

    assert halt_status is None
    assert payload["return_logprob"] is True
    assert captured == {
        "evaluation": False,
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 4, "temperature": 1.0},
        "return_logprob": True,
    }
