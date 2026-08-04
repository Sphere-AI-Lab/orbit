"""Unit tests for ``_extract_teacher_topk`` -- the payload -> per-position
(ids, logprobs) row builder that feeds the direct top-k OPD loss transport
(Task 1 of the opd-topk-direct-loss plan) -- and for the top-k scoring
response byte cap (gate-discovered: rollout-0 scoring died with
``ScoringProtocolError: scoring response exceeds its byte limit`` because
``_score_top_k`` passed no ``max_response_bytes``, defaulting to the 16MiB
generic cap even though ``input_top_logprobs`` legitimately exceeds it).

Fixture: R=3 response positions, k=2, 5-token prompt (8 input tokens total).
``meta_info.input_top_logprobs`` therefore has 8 entries: index 0 is SGLang's
placeholder (no logprob for the very first token), indices 1-4 are the
(irrelevant, dropped) prompt-token entries, and indices 5-7 are the three
response-position entries that ``_extract_teacher_topk`` must turn into rows.
"""

import argparse
import asyncio
import math

import pytest

import orbit.rollout.opd_sglang as opd_sglang
from orbit.rollout.opd_sglang import (
    _TOPK_PAD_LOGPROB,
    _TOPK_PAD_TOKEN_ID,
    _extract_teacher_topk,
)
from orbit.rollout.scoring_client import SCORING_MAX_RESPONSE_BYTES
from orbit.utils.types import Sample


def _entry(logprob: float, token_id: int) -> list:
    return [logprob, token_id]


def _payload() -> dict:
    return {
        "teacher": {
            "meta_info": {
                "input_top_logprobs": [
                    None,  # SGLang placeholder for input position 0
                    None,  # prompt token 1 (unused, dropped by response_length trim)
                    None,  # prompt token 2
                    None,  # prompt token 3
                    None,  # prompt token 4
                    # response position 0: exactly k=2 entries, unsorted on the wire
                    [_entry(math.log(0.3), 42), _entry(math.log(0.7), 7)],
                    # response position 1: only k-1=1 entry -> needs a trailing pad
                    [_entry(math.log(0.9), 3)],
                    # response position 2: exactly k=2 entries, unsorted on the wire
                    [_entry(math.log(0.2), 9), _entry(math.log(0.5), 15)],
                ]
            }
        }
    }


def test_full_position_sorted_by_descending_logprob_no_pad():
    ids_rows, logprobs_rows = _extract_teacher_topk(_payload(), response_length=3, top_k=2)

    # response position 0: token 7 (p=0.7) outranks token 42 (p=0.3).
    assert ids_rows[0] == [7, 42]
    assert logprobs_rows[0] == pytest.approx([math.log(0.7), math.log(0.3)])
    # response position 2: token 15 (p=0.5) outranks token 9 (p=0.2).
    assert ids_rows[2] == [15, 9]
    assert logprobs_rows[2] == pytest.approx([math.log(0.5), math.log(0.2)])


def test_short_position_gets_trailing_pad_sentinel():
    ids_rows, logprobs_rows = _extract_teacher_topk(_payload(), response_length=3, top_k=2)

    assert ids_rows[1] == [3, _TOPK_PAD_TOKEN_ID]
    assert logprobs_rows[1] == pytest.approx([math.log(0.9), _TOPK_PAD_LOGPROB])
    # czy's scheme: the pad logprob underflows to exactly 0.0 probability in fp32.
    assert math.exp(_TOPK_PAD_LOGPROB) == 0.0


def test_zero_response_length_returns_empty_lists():
    ids_rows, logprobs_rows = _extract_teacher_topk(_payload(), response_length=0, top_k=2)

    assert ids_rows == []
    assert logprobs_rows == []


def test_ensemble_payload_raises_value_error():
    ensemble_payload = {"teachers": [_payload()["teacher"]], "teacher_weights": [1.0]}

    with pytest.raises(ValueError):
        _extract_teacher_topk(ensemble_payload, response_length=3, top_k=2)


# --- _topk_response_byte_limit -----------------------------------------------


def _topk_limit_args(top_k: int) -> argparse.Namespace:
    return argparse.Namespace(opd_log_prob_top_k=top_k)


def test_topk_response_byte_limit_floors_at_generic_cap_for_tiny_k():
    # 5 tokens x (top_k=1 + 1) x 64 bytes/entry x 2 safety = 1280 bytes,
    # far below the generic 16MiB cap, which must win.
    args = _topk_limit_args(top_k=1)
    assert opd_sglang._topk_response_byte_limit(args, num_tokens=5) == SCORING_MAX_RESPONSE_BYTES


def test_topk_response_byte_limit_scales_with_num_tokens_times_k():
    # 2000 tokens x (top_k=100000 + 1) x 64 x 2 comfortably exceeds the
    # generic cap, so the scaled formula -- not the floor -- must win.
    args = _topk_limit_args(top_k=100000)
    num_tokens = 2000
    expected = num_tokens * (100000 + 1) * 64 * 2
    assert expected > SCORING_MAX_RESPONSE_BYTES
    assert opd_sglang._topk_response_byte_limit(args, num_tokens) == expected


# --- _score_top_k forwards the computed cap ----------------------------------


def _score_top_k_args() -> argparse.Namespace:
    return argparse.Namespace(
        opd_teacher_url="http://teacher:30001/generate",
        opd_log_prob_top_k=2,
        opd_top_k_strategy="only-teacher",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )


def _teacher_group_response(response_length: int) -> dict:
    # 7 placeholder/prompt positions (SGLang's index-0 placeholder + 6 prompt
    # tokens) followed by `response_length` real top-k rows.
    real_entry = [[math.log(0.6), 1], [math.log(0.4), 2]]
    return {"meta_info": {"input_top_logprobs": [None] * 7 + [real_entry] * response_length}}


def test_score_top_k_forwards_computed_byte_cap_to_both_posts(monkeypatch):
    seen = {}

    async def fake_post_teacher_group(targets, payload, timeout_secs, max_response_bytes=None):
        seen["teacher_max_response_bytes"] = max_response_bytes
        return _teacher_group_response(response_length=3)

    async def fake_post_json(url, payload, timeout_secs=None, max_response_bytes=None):
        seen["student_max_response_bytes"] = max_response_bytes
        return {"meta_info": {"input_token_ids_logprobs": []}}

    monkeypatch.setattr(opd_sglang, "_post_teacher_group", fake_post_teacher_group)
    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)

    args = _score_top_k_args()
    sample = Sample(tokens=list(range(10)), response_length=3)
    asyncio.run(opd_sglang._score_top_k(args, sample))

    expected = opd_sglang._topk_response_byte_limit(args, 10)
    assert seen["teacher_max_response_bytes"] == expected
    assert seen["student_max_response_bytes"] == expected
