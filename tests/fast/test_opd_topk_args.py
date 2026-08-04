"""TDD for --loss-type opd_topk_loss's arguments, validation, and coupling
(spec Phase D "raw-mass v1", plan Task 4).
"""

from argparse import Namespace

import pytest

from orbit.utils.arguments import _validate_opd_args, validate_opd_topk_loss_args


def _valid_args(**overrides) -> Namespace:
    """A fully valid opd_topk_loss config: external single-teacher transport,
    only-teacher strategy, untempered rollout, CP=1, no tail-bucket."""
    defaults = dict(
        loss_type="opd_topk_loss",
        opd_type="sglang",
        opd_log_prob_top_k=8,
        opd_top_k_strategy="only-teacher",
        opd_teacher_url="http://host:1234/generate",
        opd_teacher_urls=None,
        opd_teacher=None,
        opd_teacher_load=None,
        rollout_temperature=1.0,
        context_parallel_size=1,
        opd_topk_tail_bucket=False,
        opd_kl_type="reverse",
        opd_mixed_kl_weight=0.5,
        opd_topk_zero_outside=None,
        compute_advantages_and_returns=True,
        advantage_estimator="grpo",
        use_opd=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# --- no-op when not opd_topk_loss ---


def test_noop_when_loss_type_is_not_opd_topk_loss():
    args = Namespace(loss_type="policy_loss")
    validate_opd_topk_loss_args(args)  # must not raise or touch anything
    assert not hasattr(args, "compute_advantages_and_returns")


# --- rejection cases ---


def test_requires_positive_top_k():
    args = _valid_args(opd_log_prob_top_k=0)
    with pytest.raises(ValueError, match="opd-log-prob-top-k"):
        validate_opd_topk_loss_args(args)


def test_requires_only_teacher_strategy():
    args = _valid_args(opd_top_k_strategy="only-student")
    with pytest.raises(ValueError, match="opd-top-k-strategy only-teacher"):
        validate_opd_topk_loss_args(args)


def test_rejects_teacher_ensembles():
    args = _valid_args(
        opd_teacher_url=None,
        opd_teacher_urls=["default=http://h1:1/generate,http://h2:1/generate"],
    )
    with pytest.raises(ValueError, match="ensemble"):
        validate_opd_topk_loss_args(args)


def test_allows_multi_named_single_member_routing():
    # Multiple NAMED teachers are fine as long as no single group has >1 member.
    args = _valid_args(
        opd_teacher_url=None,
        opd_teacher_urls=["default=http://h1:1/generate", "math=http://h2:1/generate"],
    )
    validate_opd_topk_loss_args(args)


def test_rejects_managed_same_engine_teacher_path():
    # opd_teacher="base" with no external URL selects
    # orbit.rollout.opd_scoring.local_scoring_enabled's path, which does not
    # retain teacher_topk_ids/teacher_topk_logprobs (Task 1 gap).
    args = _valid_args(opd_teacher_url=None, opd_teacher_urls=None, opd_teacher="base")
    with pytest.raises(ValueError, match="external teacher"):
        validate_opd_topk_loss_args(args)


def test_requires_unit_rollout_temperature():
    args = _valid_args(rollout_temperature=0.7)
    with pytest.raises(ValueError, match="rollout-temperature"):
        validate_opd_topk_loss_args(args)


def test_requires_cp_size_one():
    args = _valid_args(context_parallel_size=2)
    with pytest.raises(ValueError, match="context-parallel-size"):
        validate_opd_topk_loss_args(args)


def test_rejects_tail_bucket():
    args = _valid_args(opd_topk_tail_bucket=True)
    with pytest.raises(ValueError, match="opd-topk-tail-bucket"):
        validate_opd_topk_loss_args(args)


# --- --opd-topk-zero-outside default resolution ---


def test_zero_outside_default_true_for_reverse():
    args = _valid_args(opd_kl_type="reverse", opd_topk_zero_outside=None)
    validate_opd_topk_loss_args(args)
    assert args.opd_topk_zero_outside is True


def test_zero_outside_default_true_for_mixed():
    args = _valid_args(opd_kl_type="mixed", opd_topk_zero_outside=None)
    validate_opd_topk_loss_args(args)
    assert args.opd_topk_zero_outside is True


def test_zero_outside_default_false_and_warns_for_forward(caplog):
    args = _valid_args(opd_kl_type="forward", opd_topk_zero_outside=None)
    with caplog.at_level("WARNING"):
        validate_opd_topk_loss_args(args)
    assert args.opd_topk_zero_outside is False
    assert any("opd-topk-zero-outside" in r.message.lower() for r in caplog.records)


def test_zero_outside_explicit_false_is_not_overridden():
    args = _valid_args(opd_kl_type="reverse", opd_topk_zero_outside=False)
    validate_opd_topk_loss_args(args)
    assert args.opd_topk_zero_outside is False


def test_zero_outside_explicit_true_with_forward_is_kept_and_does_not_warn(caplog):
    # Inert-but-accepted: opd_topk_loss_function itself warns at loss-compute time
    # (see loss.py's _topk_kl_terms); validation does not duplicate that warning
    # for an explicit user choice, only for the unset-default resolution above.
    args = _valid_args(opd_kl_type="forward", opd_topk_zero_outside=True)
    with caplog.at_level("WARNING"):
        validate_opd_topk_loss_args(args)
    assert args.opd_topk_zero_outside is True
    assert not any("opd-topk-zero-outside" in r.message.lower() for r in caplog.records)


# --- compute_advantages_and_returns coupling ---


def test_sets_compute_advantages_and_returns_false():
    args = _valid_args(compute_advantages_and_returns=True)
    validate_opd_topk_loss_args(args)
    assert args.compute_advantages_and_returns is False


# --- fully valid config passes ---


def test_fully_valid_config_passes():
    args = _valid_args()
    validate_opd_topk_loss_args(args)
    assert args.compute_advantages_and_returns is False
    assert args.opd_topk_zero_outside is True


# --- wired into _validate_opd_args (the enclosing OPD validation entry) ---


def test_wired_into_validate_opd_args_rejects():
    args = _valid_args(opd_log_prob_top_k=0)
    with pytest.raises(ValueError, match="opd-log-prob-top-k"):
        _validate_opd_args(args)


def test_wired_into_validate_opd_args_passes():
    args = _valid_args()
    _validate_opd_args(args)
    assert args.compute_advantages_and_returns is False
    assert args.opd_topk_zero_outside is True
