"""Regression tests for threading teacher_log_probs (OPD) through sample merging.

merge_samples / _merge_sample_pair rebuild the merged sample via
_create_with_all_fields, which asserts exhaustiveness over ALL Sample fields.
Before the fix, the new teacher_log_probs field was not passed, so every
merge_samples call raised "Sample field mismatch. Missing: {'teacher_log_probs'}".
"""

import pytest

from orbit.rollout.generate_utils.sample_utils import merge_samples
from orbit.peft.opd.opd_sglang import _TOPK_PAD_LOGPROB, _TOPK_PAD_TOKEN_ID
from orbit.utils.types import Sample


class _FakeTokenizer:
    def decode(self, tokens):
        return "".join(str(t) for t in tokens)


def _make_pair(teacher_a, teacher_b):
    # a: first turn (prompt=[1], response=[2,3]); b extends a with one obs token
    # ([4]) then a second response ([5,6]). obs_len = 6 - 3 - 2 = 1.
    a = Sample(
        group_index=0,
        index=0,
        prompt="P",
        tokens=[1, 2, 3],
        response="AB",
        response_length=2,
        status=Sample.Status.COMPLETED,
        teacher_log_probs=teacher_a,
    )
    b = Sample(
        group_index=0,
        index=0,
        prompt="P",
        tokens=[1, 2, 3, 4, 5, 6],
        response="EF",
        response_length=2,
        status=Sample.Status.COMPLETED,
        teacher_log_probs=teacher_b,
    )
    return a, b


def test_merge_samples_concatenates_teacher_log_probs():
    # OPD sample: teacher_log_probs set on both halves. Merged value must be the
    # concatenation a + [0.0]*obs_len + b (same shape as rollout_log_probs).
    a, b = _make_pair([0.11, 0.22], [0.55, 0.66])
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_log_probs == [0.11, 0.22, 0.0, 0.55, 0.66]
    assert merged.response_length == 5
    merged.validate()


def test_merge_samples_teacher_log_probs_none_stays_none():
    # Non-OPD path: teacher_log_probs None on both halves must stay None (miles
    # 74198b45 semantics). Zero-filling here poisons non-OPD agentic batches: a
    # merged sample gets a fake non-None value while unmerged single-turn samples
    # keep None, and the mixed batch crashes train-side CP slicing.
    a, b = _make_pair(None, None)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_log_probs is None
    merged.validate()


def test_merge_samples_teacher_log_probs_one_sided_fills_missing_half_with_zeros():
    # OPD edge: only one half carries teacher log-probs -> missing half and the
    # observation span are zero-filled, matching rollout_log_probs shape.
    a, b = _make_pair([0.11, 0.22], None)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_log_probs == [0.11, 0.22, 0.0, 0.0, 0.0]
    merged.validate()


def _make_pair_kl(kl_a, kl_b, meta_a=None, meta_b=None):
    a, b = _make_pair(None, None)
    a.opd_reverse_kl, b.opd_reverse_kl = kl_a, kl_b
    if meta_a is not None:
        a.metadata = meta_a
    if meta_b is not None:
        b.metadata = meta_b
    return a, b


def test_merge_samples_opd_reverse_kl_none_stays_none():
    a, b = _make_pair_kl(None, None)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.opd_reverse_kl is None
    merged.validate()


def test_merge_samples_opd_reverse_kl_concatenates_with_zero_obs_span():
    a, b = _make_pair_kl([0.5, 0.6], [0.7, 0.8])
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.opd_reverse_kl == [0.5, 0.6, 0.0, 0.7, 0.8]
    merged.validate()


def test_merge_samples_student_top_logprobs_metadata_concatenates():
    # Top-k OPD: per-position student top-logprob lists must merge like the
    # other per-token fields, with empty entries over the observation span.
    a_top = [[[-0.1, 1]], [[-0.2, 2]]]
    b_top = [[[-0.3, 3]], [[-0.4, 4]]]
    a, b = _make_pair_kl(
        None, None, meta_a={"opd_student_top_logprobs": a_top}, meta_b={"opd_student_top_logprobs": b_top}
    )
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.metadata["opd_student_top_logprobs"] == a_top + [[]] + b_top


def test_merge_samples_teacher_hidden_states_one_sided_zero_fills():
    # Full-vocab OPD edge: hidden states are normally scored post-merge, but a
    # scored segment must survive a late merge -- missing half and observation
    # span become zero rows (loss-masked anyway), mirroring teacher_log_probs.
    import numpy as np

    a, b = _make_pair(None, None)
    a.teacher_hidden_states = np.ones((a.response_length, 4), dtype=np.float32)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_hidden_states.shape == (merged.response_length, 4)
    assert merged.teacher_hidden_states[: a.response_length].tolist() == np.ones((a.response_length, 4)).tolist()
    assert not merged.teacher_hidden_states[a.response_length :].any()
    merged.validate()


def test_merge_samples_teacher_hidden_states_none_stays_none():
    a, b = _make_pair(None, None)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_hidden_states is None


def _make_pair_topk(ids_a, ids_b, logprobs_a, logprobs_b):
    a, b = _make_pair(None, None)
    a.teacher_topk_ids, b.teacher_topk_ids = ids_a, ids_b
    a.teacher_topk_logprobs, b.teacher_topk_logprobs = logprobs_a, logprobs_b
    return a, b


def test_merge_samples_teacher_topk_none_stays_none():
    # Non-opd_topk_loss path: both halves None must stay None, same rationale as
    # teacher_log_probs -- zero/pad-filling here would poison non-OPD batches.
    a, b = _make_pair_topk(None, None, None, None)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_topk_ids is None
    assert merged.teacher_topk_logprobs is None
    merged.validate()


def test_merge_samples_teacher_topk_concatenates_with_pad_row_obs_span():
    # --loss-type opd_topk_loss: both halves scored -> merged rows are
    # a + [K-wide pad-sentinel row] (one obs position) + b, mirroring
    # opd_reverse_kl's shape but with a full row (not a scalar 0.0) over the gap.
    ids_a = [[7, 42], [3, _TOPK_PAD_TOKEN_ID]]
    logprobs_a = [[-0.35, -1.2], [-0.10, _TOPK_PAD_LOGPROB]]
    ids_b = [[15, 9], [5, 6]]
    logprobs_b = [[-0.5, -1.5], [-0.05, -0.9]]
    a, b = _make_pair_topk(ids_a, ids_b, logprobs_a, logprobs_b)
    merged = merge_samples([a, b], _FakeTokenizer())
    pad_row_ids = [_TOPK_PAD_TOKEN_ID, _TOPK_PAD_TOKEN_ID]
    pad_row_logprobs = [_TOPK_PAD_LOGPROB, _TOPK_PAD_LOGPROB]
    assert merged.teacher_topk_ids == ids_a + [pad_row_ids] + ids_b
    assert merged.teacher_topk_logprobs == logprobs_a + [pad_row_logprobs] + logprobs_b
    merged.validate()


def test_merge_samples_teacher_topk_normalizes_valid_tuple_containers():
    ids_a = ((7, 42), (3, _TOPK_PAD_TOKEN_ID))
    logprobs_a = ((-0.35, -1.2), (-0.10, _TOPK_PAD_LOGPROB))
    ids_b = ((15, 9), (5, 6))
    logprobs_b = ((-0.5, -1.5), (-0.05, -0.9))
    a, b = _make_pair_topk(ids_a, ids_b, logprobs_a, logprobs_b)

    merged = merge_samples([a, b], _FakeTokenizer())

    assert merged.teacher_topk_ids == [[7, 42], [3, 0], [0, 0], [15, 9], [5, 6]]
    assert merged.teacher_topk_logprobs == [
        [-0.35, -1.2],
        [-0.10, _TOPK_PAD_LOGPROB],
        [_TOPK_PAD_LOGPROB, _TOPK_PAD_LOGPROB],
        [-0.5, -1.5],
        [-0.05, -0.9],
    ]
    merged.validate()


@pytest.mark.parametrize("scored_side", ["a", "b"])
def test_merge_samples_teacher_topk_one_sided_requires_rescore(scored_side):
    # All-pad rows are safe only over the loss-masked observation gap. Filling a
    # generated, loss-live segment would create a large reverse/mixed outside-
    # support loss, so a partially scored merge must be rejected and re-scored.
    ids_a = [[7, 42], [3, _TOPK_PAD_TOKEN_ID]]
    logprobs_a = [[-0.35, -1.2], [-0.10, _TOPK_PAD_LOGPROB]]
    if scored_side == "a":
        a, b = _make_pair_topk(ids_a, None, logprobs_a, None)
    else:
        a, b = _make_pair_topk(None, ids_a, None, logprobs_a)

    with pytest.raises(ValueError, match="merge before teacher scoring or re-score"):
        merge_samples([a, b], _FakeTokenizer())


def test_merge_samples_teacher_topk_both_empty_scored_segments_require_rescore():
    # []/[] is a valid scored empty response, but it carries no row from which K
    # can be inferred for the injected observation span. Preserve the invariant
    # by requiring the merged trajectory to be scored after merging.
    a = Sample(
        group_index=0,
        index=0,
        prompt="P",
        tokens=[1],
        response="",
        response_length=0,
        status=Sample.Status.COMPLETED,
        teacher_topk_ids=[],
        teacher_topk_logprobs=[],
    )
    b = Sample(
        group_index=0,
        index=0,
        prompt="P",
        tokens=[1, 2, 3],
        response="",
        response_length=0,
        status=Sample.Status.COMPLETED,
        teacher_topk_ids=[],
        teacher_topk_logprobs=[],
    )
    with pytest.raises(ValueError, match="cannot infer K"):
        merge_samples([a, b], _FakeTokenizer())


def test_merge_samples_teacher_topk_rejects_different_widths():
    ids_a = [[7, 42], [3, 4]]
    logprobs_a = [[-0.35, -1.2], [-0.10, -0.2]]
    ids_b = [[15], [5]]
    logprobs_b = [[-0.5], [-0.05]]
    a, b = _make_pair_topk(ids_a, ids_b, logprobs_a, logprobs_b)

    with pytest.raises(ValueError, match="different K"):
        merge_samples([a, b], _FakeTokenizer())
