from orbit.utils.types import Sample


def test_sample_declares_teacher_log_probs_default_none():
    s = Sample(index=0, prompt="p", response="r", response_length=3)
    assert s.teacher_log_probs is None
