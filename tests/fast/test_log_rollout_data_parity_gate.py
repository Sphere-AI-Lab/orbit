"""Regression tests for the true-on-policy exact train/rollout parity CI gate.

orbit ported true-on-policy Phases 1-4 but not Phase 5 (SGLang kernels running
inside Megatron via the fork rebase; orbit/true_on_policy/contracts.py pins
``megatron_uses_sglang_backend: False`` until then). The inherited miles assert
in ``log_rollout_data`` compared ``log_probs`` and ``rollout_log_probs`` for
exact equality unconditionally, which is only valid once Phase 5 closes the
kernel gap; with Phase 5 absent, Megatron and SGLang legitimately run
different kernels and the arrays differ (measured, not asserted, via
``train_rollout_logprob_abs_diff{,_max}``). The assert is now gated on
``args.true_on_policy_megatron_uses_sglang_backend``, plumbed onto ``args`` by
``apply_true_on_policy_parse_defaults`` (orbit/true_on_policy/config.py) from
the contract's ``megatron_uses_sglang_backend`` kernel-policy field.

These tests simulate both states of that flag directly on a hand-built
``Namespace`` (mirroring tests/fast/test_log_rollout_data_topk_keys.py), since
no shipped contract sets it True yet.
"""

from argparse import Namespace

import pytest
import torch

from orbit.backends.training_utils import log_utils
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _single_process_state() -> None:
    # group=None short-circuits GroupInfo's post-init verification (no real process
    # group needed): gather_log_data is monkeypatched below, so no distributed calls
    # actually happen.
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(intra_dp=single, intra_dp_cp=single, cp=single, tp=single, is_pp_last_stage=True)
    )


def _identity_gather_log_data(metric_name, args, rollout_id, log_dict):
    # Simulates a single-DP-rank reduction: mean-over-1-rank is the identity.
    return {f"{metric_name}/{key}": value for key, value in log_dict.items()}


def _args(**overrides):
    values = dict(
        ci_test=True,
        ci_disable_logprobs_checker=False,
        true_on_policy_mode=True,
        true_on_policy_megatron_uses_sglang_backend=False,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
        qkv_format="thd",
    )
    values.update(overrides)
    return Namespace(**values)


def _rollout_data(*, matching: bool) -> dict:
    # Single sample, response length 3. rollout_log_probs differs from
    # log_probs by 0.01 at one token when matching=False -- small enough that
    # the pre-existing isclose(..., abs_tol=0.03) checker a few lines above
    # (gated only on ci_test, not true_on_policy_mode) still passes, isolating
    # this test to the exact-equality gate under test.
    last_rollout_value = -0.3 if matching else -0.29
    return {
        "response_lengths": [3],
        "total_lengths": [5],
        "loss_masks": [torch.ones(3)],
        "log_probs": [torch.tensor([-0.1, -0.2, -0.3])],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2, last_rollout_value])],
    }


def test_exact_assert_does_not_fire_without_phase5_backend_even_on_mismatch(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    args = _args(true_on_policy_megatron_uses_sglang_backend=False)
    rollout_data = _rollout_data(matching=False)

    # Must not raise: Phase 5 is not ported, so the gap is measured, not gated.
    log_utils.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)


def test_exact_assert_fires_with_phase5_backend_on_mismatch(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    args = _args(true_on_policy_megatron_uses_sglang_backend=True)
    rollout_data = _rollout_data(matching=False)

    with pytest.raises(AssertionError, match="CI check failed"):
        log_utils.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)


def test_exact_assert_passes_with_phase5_backend_on_match(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    args = _args(true_on_policy_megatron_uses_sglang_backend=True)
    rollout_data = _rollout_data(matching=True)

    # Must not raise: log_probs == rollout_log_probs exactly.
    log_utils.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)
