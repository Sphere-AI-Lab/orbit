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
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _single_process_state() -> None:
    # group=None short-circuits GroupInfo's post-init verification (no real process
    # group needed): gather_log_data is monkeypatched below, so no distributed calls
    # actually happen.
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=single, intra_dp_cp=single, cp=single, tp=single, is_pp_last_stage=True))


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


def test_exact_assert_checks_tokens_not_only_the_reduced_mean(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    rollout_data = _rollout_data(matching=True)
    # The two errors cancel in the scalar sample mean. A mean-only gate passes;
    # exact per-token parity must fail.
    rollout_data["rollout_log_probs"] = [torch.tensor([-0.09, -0.21, -0.3])]

    with pytest.raises(AssertionError, match="masked per-token"):
        log_utils.log_rollout_data(
            rollout_id=0,
            args=_args(true_on_policy_megatron_uses_sglang_backend=True),
            rollout_data=rollout_data,
        )


def test_exact_assert_ignores_positions_excluded_by_loss_mask(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    rollout_data = _rollout_data(matching=True)
    rollout_data["loss_masks"] = [torch.tensor([1.0, 0.0, 1.0])]
    rollout_data["rollout_log_probs"][0][1] = 99.0

    log_utils.log_rollout_data(
        rollout_id=0,
        args=_args(
            true_on_policy_megatron_uses_sglang_backend=True,
            # Isolate the exact masked gate from the legacy scalar checker,
            # which intentionally averages with the same mask but tolerates 0.03.
            ci_disable_logprobs_checker=False,
        ),
        rollout_data=rollout_data,
    )


def test_exact_assert_obeys_ci_disable_checker(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    log_utils.log_rollout_data(
        rollout_id=0,
        args=_args(
            true_on_policy_megatron_uses_sglang_backend=True,
            ci_disable_logprobs_checker=True,
        ),
        rollout_data=_rollout_data(matching=False),
    )


def test_direct_true_on_policy_mode_without_phase5_attribute_does_not_crash(monkeypatch):
    _single_process_state()
    monkeypatch.setattr(log_utils, "gather_log_data", _identity_gather_log_data)

    args = _args()
    del args.true_on_policy_megatron_uses_sglang_backend
    log_utils.log_rollout_data(rollout_id=0, args=args, rollout_data=_rollout_data(matching=False))


@pytest.mark.parametrize(
    ("cp_rank", "response_indices"),
    [
        (0, [0]),
        (1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_exact_gate_uses_dsv4_padded_thd_cp_local_response_mask(cp_rank, response_indices):
    single = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=cp_rank, size=2, group=None)
    set_parallel_state(ParallelState(intra_dp=single, intra_dp_cp=single, cp=cp, tp=single))

    global_values = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
    local_values = global_values[torch.tensor(response_indices)]
    rollout_data = {
        "log_probs": [local_values.clone()],
        "rollout_log_probs": [local_values.clone()],
    }
    args = _args(true_on_policy_megatron_uses_sglang_backend=True)

    log_utils._assert_true_on_policy_logprob_parity(
        args,
        rollout_data,
        total_lengths=[11],
        response_lengths=[7],
        loss_masks=[torch.ones(7)],
        max_seq_lens=[16],
    )

    rollout_data["rollout_log_probs"][0][0] += 0.01
    with pytest.raises(AssertionError, match="masked per-token"):
        log_utils._assert_true_on_policy_logprob_parity(
            args,
            rollout_data,
            total_lengths=[11],
            response_lengths=[7],
            loss_masks=[torch.ones(7)],
            max_seq_lens=[16],
        )


def _worker_rank_local_parity_failure(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        world = GroupInfo(
            rank=rank,
            size=world_size,
            group=dist.group.WORLD,
            gloo_group=dist.group.WORLD,
        )
        single = GroupInfo(rank=0, size=1, group=None)
        set_parallel_state(
            ParallelState(intra_dp=world, intra_dp_cp=world, cp=single, tp=single, is_pp_last_stage=True)
        )
        # Only rank 1 has a mismatch. Both ranks must leave the synchronized
        # check with the same failure instead of rank 0 entering gather_log_data
        # while rank 1 exits early.
        rollout_data = _rollout_data(matching=rank == 0)
        with pytest.raises(AssertionError, match="rank 1"):
            log_utils.log_rollout_data(
                rollout_id=0,
                args=_args(true_on_policy_megatron_uses_sglang_backend=True),
                rollout_data=rollout_data,
            )
    finally:
        dist.destroy_process_group()


def test_rank_local_exact_mismatch_fails_all_dp_ranks_without_hanging() -> None:
    run_multiprocess(_worker_rank_local_parity_failure, world_size=2)
