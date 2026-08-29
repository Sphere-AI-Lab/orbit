"""The critic must not run OPD advantage adjustments.

teacher_log_probs is never broadcast to the critic (sync_actor_critic_data only
syncs values/log_probs/ref_log_probs), and the value loss consumes `returns`,
which is computed before the OPD blend rebinds `advantages`. So
compute_advantages_and_returns(role="critic") must skip both the --use-opd
blend and the --opd-icepop gate instead of crashing on the missing teacher.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import loss


class _Axis:
    def __init__(self, size=1, rank=0, group=None):
        self.size = size
        self.rank = rank
        self.group = group


@pytest.fixture
def patched(monkeypatch):
    fake = SimpleNamespace(cp=_Axis(), tp=_Axis(), intra_dp=_Axis())
    monkeypatch.setattr(loss, "get_parallel_state", lambda: fake)


def _args(**overrides):
    base = dict(
        use_rollout_logprobs=False,
        kl_coef=0,
        advantage_estimator="grpo",
        use_opd=True,
        opd_kl_coef=0.5,
        opd_icepop=False,
        normalize_advantages=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _rollout_data(with_teacher):
    data = {
        "log_probs": [torch.tensor([-0.1, -0.2, -0.3])],
        "rewards": [1.0],
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "total_lengths": [3],
    }
    if with_teacher:
        data["teacher_log_probs"] = [torch.tensor([-0.5, -0.5, -0.5])]
    return data


def test_critic_role_skips_opd_blend(patched):
    rollout_data = _rollout_data(with_teacher=False)

    loss.compute_advantages_and_returns(_args(), rollout_data, role="critic")

    assert "opd_reverse_kl" not in rollout_data
    # grpo advantages == returns == broadcast reward, untouched by the blend
    torch.testing.assert_close(rollout_data["advantages"][0], torch.ones(3))


def test_critic_role_skips_opd_icepop(patched):
    # No rollout_log_probs on the critic: icepop would raise if not gated.
    rollout_data = _rollout_data(with_teacher=False)

    loss.compute_advantages_and_returns(
        _args(opd_icepop=True, tis_clip_low=0.5, tis_clip=2.0), rollout_data, role="critic"
    )

    torch.testing.assert_close(rollout_data["advantages"][0], torch.ones(3))


def test_actor_role_still_applies_opd_blend(patched):
    rollout_data = _rollout_data(with_teacher=True)

    loss.compute_advantages_and_returns(_args(), rollout_data)

    assert "opd_reverse_kl" in rollout_data
    # adv = reward - coef * (student - teacher) = 1 - 0.5*((-0.1..-0.3) - (-0.5))
    expected = torch.ones(3) - 0.5 * (torch.tensor([-0.1, -0.2, -0.3]) - torch.tensor([-0.5, -0.5, -0.5]))
    torch.testing.assert_close(rollout_data["advantages"][0], expected)
