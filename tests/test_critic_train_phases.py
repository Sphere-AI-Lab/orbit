import argparse

import pytest

from orbit.peft.critic.critic_adapter import value_loss_phase


def test_value_loss_phase_toggles_and_restores():
    args = argparse.Namespace(loss_type="policy_loss")
    with value_loss_phase(args):
        assert args.loss_type == "value_loss"
    assert args.loss_type == "policy_loss"


def test_value_loss_phase_restores_on_exception():
    args = argparse.Namespace(loss_type="policy_loss")
    with pytest.raises(RuntimeError):
        with value_loss_phase(args):
            raise RuntimeError("boom")
    assert args.loss_type == "policy_loss"
