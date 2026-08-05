import pytest

from orbit.utils.types import Sample, collect_teacher_topk_data


def _sample(ids=None, logprobs=None, response_length: int = 2) -> Sample:
    return Sample(
        tokens=[0] * (response_length + 1),
        response_length=response_length,
        teacher_topk_ids=ids,
        teacher_topk_logprobs=logprobs,
    )


def test_collect_teacher_topk_data_keeps_valid_paired_batch():
    first = _sample([[1, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]])
    second = _sample([[5, 6], [7, 8]], [[-0.5, -0.6], [-0.7, -0.8]])

    data = collect_teacher_topk_data([first, second], expected_top_k=2)

    assert data == {
        "teacher_topk_ids": [first.teacher_topk_ids, second.teacher_topk_ids],
        "teacher_topk_logprobs": [first.teacher_topk_logprobs, second.teacher_topk_logprobs],
    }


def test_collect_teacher_topk_data_is_absent_for_unscored_batch():
    assert collect_teacher_topk_data([_sample(), _sample()], expected_top_k=2) is None


@pytest.mark.parametrize(
    "samples",
    [
        [_sample([[1, 2], [3, 4]], None)],
        [
            _sample([[1, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]]),
            _sample(),
        ],
    ],
)
def test_collect_teacher_topk_data_rejects_partial_pairs_or_samples(samples):
    with pytest.raises(ValueError, match="teacher top-k"):
        collect_teacher_topk_data(samples, expected_top_k=2)


def test_collect_teacher_topk_data_rejects_configured_width_mismatch():
    sample = _sample([[1, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]])

    with pytest.raises(ValueError, match="configured top-k"):
        collect_teacher_topk_data([sample], expected_top_k=3)


def test_collect_teacher_topk_data_rejects_cross_sample_width_mismatch_without_config():
    first = _sample([[1, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]])
    second = _sample([[5], [6]], [[-0.5], [-0.6]])

    with pytest.raises(ValueError, match="differs across samples"):
        collect_teacher_topk_data([first, second], expected_top_k=None)


def test_collect_teacher_topk_data_accepts_empty_response_when_k_is_configured():
    empty = _sample([], [], response_length=0)
    nonempty = _sample([[1, 2], [3, 4]], [[-0.1, -0.2], [-0.3, -0.4]])

    data = collect_teacher_topk_data([empty, nonempty], expected_top_k=2)

    assert data["teacher_topk_ids"] == [[], nonempty.teacher_topk_ids]
