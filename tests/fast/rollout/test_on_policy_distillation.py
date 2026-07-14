import asyncio
import math
from argparse import Namespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout import on_policy_distillation as opd
from miles.rollout.on_policy_distillation import _compute_topk_reverse_kl
from miles.utils.types import Sample

register_cpu_ci(est_time=60, suite="stage-a-cpu")


def _entry(prob: float, token_id: int):
    return [math.log(prob), token_id]


def _args(strategy: str, weight_mode: str = "student_p"):
    return Namespace(
        opd_top_k_strategy=strategy,
        opd_reward_weight_mode=weight_mode,
    )


def _sample():
    return Sample(
        tokens=[10, 11, 12],
        response_length=2,
        metadata={
            "opd_student_top_logprobs": [
                [_entry(0.6, 1), _entry(0.4, 2)],
                [_entry(0.7, 4), _entry(0.3, 5)],
            ]
        },
    )


def test_score_payload_materializes_only_the_response_window():
    input_ids = [10, 11, 12, 13, 14]

    payload = opd._score_payload(input_ids, response_length=2, top_k=4, token_ids=[21, 22])

    assert payload["input_ids"] is input_ids
    assert payload["logprob_start_len"] == 2
    assert payload["top_logprobs_num"] == 4
    assert payload["token_ids_logprob"] == [21, 22]


def test_score_payload_with_empty_response_starts_at_last_prompt_token():
    payload = opd._score_payload([10, 11, 12], response_length=0)

    assert payload["logprob_start_len"] == 2


@pytest.mark.parametrize("response_length", [-1, 4])
def test_score_payload_rejects_response_length_outside_input(response_length):
    with pytest.raises(ValueError, match="response_length must be between 0 and len"):
        opd._score_payload([10, 11, 12], response_length=response_length)


def test_score_payload_requires_at_least_one_prompt_token():
    with pytest.raises(ValueError, match="requires at least one prompt token"):
        opd._score_payload([10, 11, 12], response_length=3)


@pytest.mark.asyncio
async def test_scoring_post_retries_asyncio_timeout(monkeypatch):
    calls = 0

    async def fake_post_json(url, payload, timeout_s):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError
        return opd._PostJsonResult(
            response={"meta_info": {"input_token_logprobs": [None, [-0.5, 1]]}},
            request_body_bytes=128,
            response_body_bytes=256,
            body_read_s=0.01,
            json_decode_s=0.02,
        )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(opd, "_post_json", fake_post_json)
    monkeypatch.setattr(opd.asyncio, "sleep", no_sleep)
    args = Namespace(
        opd_scoring_timeout=1,
        opd_scoring_max_inflight=0,
        opd_scoring_retries=1,
    )

    result = await opd._scoring_post(
        args,
        "http://teacher/generate",
        {"input_ids": [1]},
        target="teacher",
        response_length=1,
    )

    assert result.response == {"meta_info": {"input_token_logprobs": [None, [-0.5, 1]]}}
    assert result.telemetry["attempts"] == 2
    assert result.telemetry["input_tokens"] == 1
    assert result.telemetry["response_tokens"] == 1
    assert result.telemetry["request_body_bytes"] == 128
    assert result.telemetry["response_body_bytes"] == 256
    assert result.telemetry["returned_positions"] == 2
    assert calls == 2


@pytest.mark.asyncio
async def test_reward_func_records_scoring_telemetry(monkeypatch):
    response = {"meta_info": {"input_token_logprobs": [None, [-0.5, 11], [-0.25, 12]]}}
    telemetry = {
        "target": "teacher",
        "attempts": 1,
        "input_tokens": 3,
        "response_tokens": 2,
    }

    async def fake_scoring_post(*args, **kwargs):
        return opd._ScoringPostResult(response=response, telemetry=telemetry)

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(tokens=[10, 11, 12], response_length=2, metadata={"dataset": "math"})
    args = Namespace(opd_log_prob_top_k=0, rm_url="http://teacher/generate")

    result = await opd.reward_func(args, sample)

    assert result is response
    assert sample.metadata["dataset"] == "math"
    assert sample.metadata[opd.OPD_SCORING_TELEMETRY_KEY] == [telemetry]


@pytest.mark.asyncio
async def test_teacher_topk_reward_func_uses_response_window_for_both_scoring_calls(monkeypatch):
    calls = []

    async def fake_scoring_post(args, url, payload, *, target, response_length):
        calls.append((target, response_length, payload))
        if target == "teacher":
            response = {
                "meta_info": {
                    "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
                    "input_top_logprobs": [None, [[-0.1, 21]], [[-0.2, 22]]],
                }
            }
        else:
            response = {
                "meta_info": {
                    "input_token_logprobs": [None, [-0.3, 11], [-0.4, 12]],
                    "input_token_ids_logprobs": [None, [[-0.5, 21]], [[-0.6, 22]]],
                }
            }
        return opd._ScoringPostResult(response=response, telemetry={"target": target})

    monkeypatch.setattr(opd, "_scoring_post", fake_scoring_post)
    sample = Sample(tokens=[7, 8, 9, 11, 12], response_length=2)
    args = Namespace(
        opd_log_prob_top_k=2,
        opd_top_k_strategy="only-teacher",
        rm_url="http://teacher/generate",
        sglang_router_ip="student",
        sglang_router_port=30000,
    )

    reward = await opd.reward_func(args, sample)

    assert set(reward) == {"teacher", "student_on_teacher"}
    assert [target for target, _, _ in calls] == ["teacher", "student"]
    assert all(response_length == 2 for _, response_length, _ in calls)
    assert all(payload["input_ids"] is sample.tokens for _, _, payload in calls)
    assert all(payload["logprob_start_len"] == 2 for _, _, payload in calls)
    assert calls[0][2]["top_logprobs_num"] == 2
    assert calls[1][2]["token_ids_logprob"] == [21, 22]


def _sampled_scoring_response(token_ids: list[int]) -> dict:
    return {
        "meta_info": {
            "input_token_logprobs": [None, *[[-0.1 * (i + 1), token_id] for i, token_id in enumerate(token_ids)]]
        }
    }


def _sampled_opd_args() -> Namespace:
    return Namespace(opd_log_prob_top_k=0, reward_key=None)


def test_sampled_token_post_process_extracts_same_values_from_full_and_response_windows():
    full_window_sample = Sample(tokens=[10, 11, 12], response_length=2)
    full_window_sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.1, 10], [-0.2, 11], [-0.3, 12]]}}
    response_window_sample = Sample(tokens=[10, 11, 12], response_length=2)
    response_window_sample.reward = {"meta_info": {"input_token_logprobs": [None, [-0.2, 11], [-0.3, 12]]}}

    raw_rewards, rewards = opd.post_process_rewards(
        _sampled_opd_args(),
        [full_window_sample, response_window_sample],
    )

    assert raw_rewards == [0.0, 0.0]
    assert rewards == [0.0, 0.0]
    assert full_window_sample.teacher_log_probs.tolist() == pytest.approx([-0.2, -0.3])
    assert response_window_sample.teacher_log_probs.tolist() == pytest.approx([-0.2, -0.3])


def test_sampled_token_post_process_rejects_token_alignment_mismatch():
    sample = Sample(tokens=[10, 11, 12], response_length=2, index=7, group_index=3)
    sample.reward = _sampled_scoring_response([10, 99, 12])

    with pytest.raises(
        ValueError,
        match=r"teacher scoring token alignment mismatch at response position 0.*got token id 99, expected 11",
    ):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def test_sampled_token_post_process_rejects_short_scoring_response():
    sample = Sample(tokens=[10, 11, 12], response_length=2)
    sample.reward = _sampled_scoring_response([12])

    with pytest.raises(ValueError, match=r"teacher scoring token count mismatch: got 1.*expected 2"):
        opd.post_process_rewards(_sampled_opd_args(), [sample])


def _teacher_payload():
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    [_entry(0.5, 2), _entry(0.5, 3)],
                    [_entry(0.8, 4), _entry(0.2, 6)],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.3, 1), _entry(0.7, 2)],
                    [_entry(0.4, 4), _entry(0.6, 5)],
                ],
            }
        },
        "student_on_teacher": {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [_entry(0.4, 2), _entry(0.2, 3)],
                    [_entry(0.7, 4), _entry(0.1, 6)],
                ]
            }
        },
    }


def test_topk_only_student_uses_student_probability_weights():
    reverse_kl = _compute_topk_reverse_kl(_args("only-student"), _sample(), _teacher_payload())

    expected_0 = 0.6 * math.log(0.6 / 0.3) + 0.4 * math.log(0.4 / 0.7)
    expected_1 = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_intersection_uses_overlap_only():
    reverse_kl = _compute_topk_reverse_kl(_args("intersection", "none"), _sample(), _teacher_payload())

    assert reverse_kl.tolist() == pytest.approx(
        [
            math.log(0.4 / 0.5),
            math.log(0.7 / 0.8),
        ]
    )


def test_topk_only_teacher_does_not_need_student_top_logprobs():
    sample = Sample(tokens=[10, 11, 12], response_length=2)

    reverse_kl = _compute_topk_reverse_kl(_args("only-teacher"), sample, _teacher_payload())

    expected_0 = (2 / 3) * math.log(0.4 / 0.5) + (1 / 3) * math.log(0.2 / 0.5)
    expected_1 = (7 / 8) * math.log(0.7 / 0.8) + (1 / 8) * math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])


def test_topk_xor_uses_symmetric_difference_without_normalization():
    reverse_kl = _compute_topk_reverse_kl(_args("xor", "none"), _sample(), _teacher_payload())

    expected_0 = math.log(0.6 / 0.3) + math.log(0.2 / 0.5)
    expected_1 = math.log(0.3 / 0.6) + math.log(0.1 / 0.2)

    assert reverse_kl.tolist() == pytest.approx([expected_0, expected_1])
