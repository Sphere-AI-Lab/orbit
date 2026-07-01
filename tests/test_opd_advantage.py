import pytest
import torch

from orbit.utils.ppo_utils import apply_opd_kl_to_advantages, opd_mopd_advantages
from orbit.utils.types import Sample


def test_sample_declares_teacher_log_probs_default_none():
    s = Sample(index=0, prompt="p", response="r", response_length=3)
    assert s.teacher_log_probs is None


def test_sample_validate_raises_on_teacher_log_probs_length_mismatch():
    s = Sample(
        index=0, prompt="p", tokens=[1, 2, 3], response="r", response_length=3, teacher_log_probs=[0.1, 0.2]
    )
    with pytest.raises(AssertionError, match="teacher_log_probs"):
        s.validate()


def test_sample_validate_passes_with_correct_teacher_log_probs_length():
    s = Sample(
        index=0, prompt="p", tokens=[1, 2, 3], response="r", response_length=3, teacher_log_probs=[0.1, 0.2, 0.3]
    )
    s.validate()


def test_opd_mopd_advantages_raises_without_teacher_log_probs():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3])]
    response_lengths = [3]

    with pytest.raises(ValueError, match="--use-opd"):
        opd_mopd_advantages({"teacher_log_probs": None}, student_log_probs, response_lengths)


def test_opd_mopd_advantages_matches_teacher_minus_student():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([-0.5, -0.1])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3]), torch.tensor([-0.2, -0.3])]
    response_lengths = [3, 2]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    advantages = opd_mopd_advantages(rollout_data, student_log_probs, response_lengths)

    for adv, teacher, student in zip(advantages, teacher_log_probs, student_log_probs, strict=True):
        torch.testing.assert_close(adv, teacher - student)


def test_opd_mopd_advantages_raises_on_length_mismatch():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([-0.5, -0.1])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3])]
    response_lengths = [3, 2]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    with pytest.raises(ValueError):
        opd_mopd_advantages(rollout_data, student_log_probs, response_lengths)


def test_apply_opd_kl_to_advantages_blends_reverse_kl():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([-0.5, -0.1])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3]), torch.tensor([-0.2, -0.3])]
    advantages = [torch.ones(3), torch.ones(2)]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    apply_opd_kl_to_advantages(1.0, rollout_data, advantages, student_log_probs)

    for adv, teacher, student in zip(advantages, teacher_log_probs, student_log_probs, strict=True):
        torch.testing.assert_close(adv, torch.ones_like(student) - (student - teacher))
    assert "opd_reverse_kl" in rollout_data


def test_apply_opd_kl_to_advantages_raises_without_teacher_log_probs():
    advantages = [torch.ones(3)]
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3])]

    with pytest.raises(ValueError):
        apply_opd_kl_to_advantages(1.0, {"teacher_log_probs": None}, advantages, student_log_probs)


def test_apply_opd_kl_to_advantages_zero_coef_is_noop():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3])]
    advantages = [torch.ones(3)]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    apply_opd_kl_to_advantages(0.0, rollout_data, advantages, student_log_probs)

    torch.testing.assert_close(advantages[0], torch.ones(3))
