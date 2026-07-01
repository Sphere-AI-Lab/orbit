"""Regression tests for threading teacher_log_probs (OPD) through sample merging.

merge_samples / _merge_sample_pair rebuild the merged sample via
_create_with_all_fields, which asserts exhaustiveness over ALL Sample fields.
Before the fix, the new teacher_log_probs field was not passed, so every
merge_samples call raised "Sample field mismatch. Missing: {'teacher_log_probs'}".
"""

from orbit.rollout.generate_utils.sample_utils import merge_samples
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


def test_merge_samples_teacher_log_probs_none_does_not_raise():
    # Non-OPD path: teacher_log_probs None on both halves. _fill_defaults fills a
    # neutral [0.0]*response_length (mirroring rollout_log_probs) so the merge does
    # not raise the field-mismatch AssertionError.
    a, b = _make_pair(None, None)
    merged = merge_samples([a, b], _FakeTokenizer())
    assert merged.teacher_log_probs == [0.0, 0.0, 0.0, 0.0, 0.0]
    merged.validate()
