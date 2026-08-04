"""Unit tests for ``_extract_teacher_topk`` -- the payload -> per-position
(ids, logprobs) row builder that feeds the direct top-k OPD loss transport
(Task 1 of the opd-topk-direct-loss plan).

Fixture: R=3 response positions, k=2, 5-token prompt (8 input tokens total).
``meta_info.input_top_logprobs`` therefore has 8 entries: index 0 is SGLang's
placeholder (no logprob for the very first token), indices 1-4 are the
(irrelevant, dropped) prompt-token entries, and indices 5-7 are the three
response-position entries that ``_extract_teacher_topk`` must turn into rows.
"""

import math

import pytest

from orbit.rollout.opd_sglang import (
    _TOPK_PAD_LOGPROB,
    _TOPK_PAD_TOKEN_ID,
    _extract_teacher_topk,
)


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
