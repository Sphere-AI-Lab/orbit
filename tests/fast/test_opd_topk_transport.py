"""Unit tests for ``_extract_teacher_topk`` -- the payload -> per-position
(ids, logprobs) row builder that feeds the direct top-k OPD loss transport
(Task 1 of the opd-topk-direct-loss plan) -- and for the top-k scoring
response byte cap (gate-discovered: rollout-0 scoring died with
``ScoringProtocolError: scoring response exceeds its byte limit`` because
``_score_top_k`` passed no ``max_response_bytes``, defaulting to the 16MiB
generic cap even though ``input_top_logprobs`` legitimately exceeds it).

Also covers a second gate-discovered defect: under ``--loss-type
opd_topk_loss`` + ``--opd-top-k-strategy only-teacher``, ``_score_top_k`` still
ran the PG rung's ``student_on_teacher`` rescore -- collecting the union of the
teacher's per-position top-k ids and re-scoring the student at all of them,
uncapped -- even though the direct loss never reads that rescore or the
``opd_reverse_kl`` estimate it feeds. The transport blowup is
positions x unique-ids response entries, which is exactly the field the first
defect already had to cap; here the fix is to skip the call entirely under the
direct loss, and to size the cap correctly (off the actual requested id count,
not ``top_k``) for the PG configurations that still need it.

Fixture: R=3 response positions, k=2, 5-token prompt (8 input tokens total).
``meta_info.input_top_logprobs`` therefore has 8 entries: index 0 is SGLang's
placeholder (no logprob for the very first token), indices 1-4 are the
(irrelevant, dropped) prompt-token entries, and indices 5-7 are the three
response-position entries that ``_extract_teacher_topk`` must turn into rows.
"""

import argparse
import asyncio
import math
from copy import deepcopy

import pytest

import orbit.peft.opd.opd_sglang as opd_sglang
from orbit.peft.opd.opd_sglang import _TOPK_PAD_LOGPROB, _TOPK_PAD_TOKEN_ID, _extract_teacher_topk
from orbit.peft.rewards.scoring_client import SCORING_MAX_RESPONSE_BYTES
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


@pytest.mark.parametrize("delta", [-1, 1])
def test_extract_teacher_topk_rejects_wrong_scored_position_count(delta):
    payload = deepcopy(_payload())
    rows = payload["teacher"]["meta_info"]["input_top_logprobs"]
    if delta < 0:
        rows.pop()
    else:
        rows.append([_entry(math.log(0.8), 99)])

    with pytest.raises(ValueError, match="position count does not match"):
        _extract_teacher_topk(payload, response_length=3, top_k=2, num_tokens=8)


def test_extract_teacher_topk_rejects_too_few_response_rows_without_num_tokens():
    payload = {"teacher": {"meta_info": {"input_top_logprobs": [None, [_entry(-0.1, 1)]]}}}

    with pytest.raises(ValueError, match="expected exactly 3"):
        _extract_teacher_topk(payload, response_length=3, top_k=2)


def test_ensemble_payload_raises_value_error():
    ensemble_payload = {"teachers": [_payload()["teacher"]], "teacher_weights": [1.0]}

    with pytest.raises(ValueError):
        _extract_teacher_topk(ensemble_payload, response_length=3, top_k=2)


@pytest.mark.parametrize(
    "bad_entry",
    [
        [-0.1, 1.9],
        ["-0.1", 1],
        [float("nan"), 1],
        [float("inf"), 1],
        [0.01, 1],
        [-0.1, -1],
        [-0.1, True],
    ],
)
def test_extract_teacher_topk_rejects_invalid_entry_values_without_coercion(bad_entry):
    payload = deepcopy(_payload())
    payload["teacher"]["meta_info"]["input_top_logprobs"][-1][0] = bad_entry

    with pytest.raises(ValueError, match="top-logprob"):
        _extract_teacher_topk(payload, response_length=3, top_k=2)


def test_extract_teacher_topk_rejects_duplicate_token_ids_per_position():
    payload = deepcopy(_payload())
    payload["teacher"]["meta_info"]["input_top_logprobs"][-1] = [
        _entry(-0.1, 9),
        _entry(-0.2, 9),
    ]

    with pytest.raises(ValueError, match="duplicate token id"):
        _extract_teacher_topk(payload, response_length=3, top_k=2)


# --- Sample-level pair/shape validation and truncation -----------------------


def _retained_sample(ids, logprobs, response_length: int = 2) -> Sample:
    return Sample(
        tokens=[0] * (response_length + 1),
        response_length=response_length,
        teacher_topk_ids=ids,
        teacher_topk_logprobs=logprobs,
    )


@pytest.mark.parametrize(
    ("ids", "logprobs"),
    [
        ([[1, 2], [3, 4]], None),
        (None, [[-0.1, -0.2], [-0.3, -0.4]]),
    ],
)
def test_sample_validate_rejects_unpaired_teacher_topk_fields(ids, logprobs):
    sample = _retained_sample(ids, logprobs)

    with pytest.raises(ValueError, match="must be present together"):
        sample.validate()


@pytest.mark.parametrize(
    ("ids", "logprobs", "message"),
    [
        ([[1, 2]], [[-0.1, -0.2]], "row count"),
        ([[1, 2], [3]], [[-0.1, -0.2], [-0.3]], "ragged"),
        ([[1, 2], [3, 4]], [[-0.1], [-0.3, -0.4]], "to match teacher_topk_ids"),
    ],
)
def test_sample_validate_rejects_malformed_teacher_topk_shape(ids, logprobs, message):
    sample = _retained_sample(ids, logprobs)

    with pytest.raises(ValueError, match=message):
        sample.validate()


def test_sample_validate_teacher_topk_checks_configured_width_when_known():
    sample = _retained_sample([[1, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]])

    assert sample.validate_teacher_topk(expected_top_k=2) == 2
    with pytest.raises(ValueError, match="configured top-k"):
        sample.validate_teacher_topk(expected_top_k=3)


@pytest.mark.parametrize(
    ("ids", "logprobs", "message"),
    [
        ([[1.5, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]], "exact integer"),
        ([[1, 2], [3, 4]], [[float("nan"), -0.2], [-0.3, -0.4]], "finite and <= 0"),
        ([[1, 1], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]], "duplicate token id"),
        ([[1, 2], [3, 4]], [[-1e4, -0.2], [-0.3, -0.4]], "padding"),
    ],
)
def test_sample_validate_rejects_invalid_teacher_topk_values(ids, logprobs, message):
    with pytest.raises(ValueError, match=message):
        _retained_sample(ids, logprobs).validate()


def test_strip_last_output_tokens_slices_both_teacher_topk_fields():
    class _Tokenizer:
        def decode(self, tokens):
            return ""

    sample = _retained_sample(
        [[1, 2], [3, 4], [5, 6]],
        [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
        response_length=3,
    )
    sample.strip_last_output_tokens(1, _Tokenizer())

    assert sample.teacher_topk_ids == [[1, 2], [3, 4]]
    assert sample.teacher_topk_logprobs == [[-0.1, -0.2], [-0.3, -0.4]]
    sample.validate()


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


# --- _topk_response_byte_limit's entries_per_token override -------------------


def test_topk_response_byte_limit_entries_per_token_override_ignores_top_k():
    # top_k=999 must be ignored once entries_per_token is given explicitly --
    # the student_on_teacher rescore uses this to size its cap off the actual
    # requested id count, not --opd-log-prob-top-k.
    args = _topk_limit_args(top_k=999)
    num_tokens = 2000
    entries_per_token = 100
    expected = num_tokens * entries_per_token * 64 * 2
    assert expected > SCORING_MAX_RESPONSE_BYTES
    assert opd_sglang._topk_response_byte_limit(args, num_tokens, entries_per_token=entries_per_token) == expected


# --- _score_top_k under --loss-type opd_topk_loss: skip the PG rescore -------


def test_score_top_k_direct_loss_performs_only_the_teacher_group_post(monkeypatch):
    calls = {"teacher_group": 0, "post_json": 0}

    async def fake_post_teacher_group(targets, payload, timeout_secs, max_response_bytes=None):
        calls["teacher_group"] += 1
        return _teacher_group_response(response_length=3)

    async def fake_post_json(url, payload, timeout_secs=None, max_response_bytes=None):
        calls["post_json"] += 1
        return {"meta_info": {"input_token_ids_logprobs": []}}

    monkeypatch.setattr(opd_sglang, "_post_teacher_group", fake_post_teacher_group)
    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)

    args = _score_top_k_args()
    args.loss_type = "opd_topk_loss"
    sample = Sample(tokens=list(range(10)), response_length=3)
    reward_payload = asyncio.run(opd_sglang._score_top_k(args, sample))

    assert calls["teacher_group"] == 1
    assert calls["post_json"] == 0
    assert "student_on_teacher" not in reward_payload


def test_score_top_k_pg_configuration_still_rescores_student_with_capped_bytes(monkeypatch):
    # No loss_type set -> the PG configuration (opd_reverse_kl consumer): the
    # student_on_teacher rescore must still happen, and its cap must be sized
    # off the teacher's actual reported unique ids (2, from the fixture's
    # repeated real_entry), not off --opd-log-prob-top-k (100000) -- proving
    # the fix uses the new formula rather than the pre-existing top_k-based one.
    seen = {}

    async def fake_post_teacher_group(targets, payload, timeout_secs, max_response_bytes=None):
        seen["teacher_max_response_bytes"] = max_response_bytes
        return _teacher_group_response(response_length=3)

    async def fake_post_json(url, payload, timeout_secs=None, max_response_bytes=None):
        seen["student_max_response_bytes"] = max_response_bytes
        seen["student_token_ids"] = payload["token_ids_logprob"]
        return {"meta_info": {"input_token_ids_logprobs": []}}

    monkeypatch.setattr(opd_sglang, "_post_teacher_group", fake_post_teacher_group)
    monkeypatch.setattr(opd_sglang, "_post_json", fake_post_json)

    args = _score_top_k_args()
    args.opd_log_prob_top_k = 100000
    sample = Sample(tokens=list(range(50000)), response_length=3)
    asyncio.run(opd_sglang._score_top_k(args, sample))

    assert seen["student_token_ids"] == [1, 2]
    teacher_expected = opd_sglang._topk_response_byte_limit(args, 50000)
    student_expected = opd_sglang._topk_response_byte_limit(args, 50000, entries_per_token=3)
    assert seen["teacher_max_response_bytes"] == teacher_expected
    assert seen["student_max_response_bytes"] == student_expected
    assert student_expected < teacher_expected


# --- post_process under --loss-type opd_topk_loss: skip opd_reverse_kl -------


def _post_process_args(**overrides) -> argparse.Namespace:
    args = _score_top_k_args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _student_on_teacher_response() -> dict:
    # Same 8-position shape as _payload()'s teacher fixture (placeholder + 4
    # dropped prompt positions + 3 response rows), reporting the student's own
    # logprob at each id the teacher reported at that position -- the ids
    # `_compute_topk_reverse_kl` looks up for strategy only-teacher.
    return {
        "meta_info": {
            "input_token_ids_logprobs": [
                None,
                None,
                None,
                None,
                None,
                [_entry(math.log(0.6), 42), _entry(math.log(0.4), 7)],
                [_entry(math.log(0.9), 3)],
                [_entry(math.log(0.55), 9), _entry(math.log(0.45), 15)],
            ]
        }
    }


def test_post_process_direct_loss_skips_reverse_kl_keeps_teacher_topk_extraction():
    args = _post_process_args(loss_type="opd_topk_loss")
    sample = Sample(tokens=[0] * 8, response_length=3)
    sample.metadata[opd_sglang.TEACHER_RESPONSE_METADATA_KEY] = _payload()

    opd_sglang.post_process(args, [sample])

    assert sample.opd_reverse_kl is None
    assert sample.teacher_topk_ids == [[7, 42], [3, _TOPK_PAD_TOKEN_ID], [15, 9]]
    expected_logprobs = [
        [math.log(0.7), math.log(0.3)],
        [math.log(0.9), _TOPK_PAD_LOGPROB],
        [math.log(0.5), math.log(0.2)],
    ]
    for got_row, expected_row in zip(sample.teacher_topk_logprobs, expected_logprobs, strict=True):
        assert got_row == pytest.approx(expected_row)
    assert opd_sglang.TEACHER_RESPONSE_METADATA_KEY not in sample.metadata


def test_post_process_keeps_malformed_teacher_payload_for_inspection_or_retry():
    args = _post_process_args(loss_type="opd_topk_loss")
    sample = Sample(tokens=[0] * 8, response_length=3)
    payload = _payload()
    payload["teacher"]["meta_info"]["input_top_logprobs"][-1][0] = [float("nan"), 9]
    sample.metadata[opd_sglang.TEACHER_RESPONSE_METADATA_KEY] = payload

    with pytest.raises(ValueError, match="finite"):
        opd_sglang.post_process(args, [sample])

    assert sample.metadata[opd_sglang.TEACHER_RESPONSE_METADATA_KEY] is payload
    assert sample.teacher_topk_ids is None
    assert sample.teacher_topk_logprobs is None


def test_post_process_pg_configuration_still_computes_reverse_kl():
    # No loss_type set -> existing behavior pinned: opd_reverse_kl still
    # computed, teacher_topk_ids/logprobs stay unset (only opd_topk_loss sets them).
    args = _post_process_args()
    sample = Sample(tokens=[0] * 8, response_length=3)
    payload = _payload()
    payload["student_on_teacher"] = _student_on_teacher_response()
    sample.metadata[opd_sglang.TEACHER_RESPONSE_METADATA_KEY] = payload

    opd_sglang.post_process(args, [sample])

    assert sample.opd_reverse_kl is not None
    assert len(sample.opd_reverse_kl) == 3
    assert sample.teacher_topk_ids is None
    assert sample.teacher_topk_logprobs is None
