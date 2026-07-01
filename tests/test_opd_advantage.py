import pytest
import torch

from orbit.utils.ppo_utils import opd_mopd_advantages
from orbit.utils.types import Sample


def test_sample_declares_teacher_log_probs_default_none():
    s = Sample(index=0, prompt="p", response="r", response_length=3)
    assert s.teacher_log_probs is None


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
