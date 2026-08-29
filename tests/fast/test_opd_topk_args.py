"""TDD for --loss-type opd_topk_loss's arguments, validation, and coupling
(spec Phase D "raw-mass v1", plan Task 4).
"""

from argparse import Namespace

import pytest

from orbit.utils.arguments import (
    _common_orbit_validate_args,
    _validate_opd_args,
    validate_opd_topk_loss_args,
    validate_opd_topk_vocab_size,
)


def _valid_args(**overrides) -> Namespace:
    """A fully valid opd_topk_loss config: external single-teacher transport,
    only-teacher strategy, untempered rollout, CP=1, no tail-bucket, correct
    OPD custom-reward hooks wired."""
    defaults = dict(
        loss_type="opd_topk_loss",
        opd_type="sglang",
        opd_log_prob_top_k=8,
        vocab_size=128,
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
        use_kl_loss=False,
        kl_coef=0.0,
        custom_rm_path="orbit.peft.opd.opd_sglang.reward_func",
        custom_reward_post_process_path="orbit.peft.opd.opd_sglang.post_process",
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


def test_rejects_top_k_larger_than_real_student_vocab():
    args = _valid_args(opd_log_prob_top_k=129, vocab_size=128)
    with pytest.raises(ValueError, match="real vocabulary"):
        validate_opd_topk_loss_args(args)


def test_vocab_size_recheck_rejects_after_tokenizer_fills_deferred_value():
    args = _valid_args(opd_log_prob_top_k=129, vocab_size=None)
    validate_opd_topk_vocab_size(args)
    args.vocab_size = 128
    with pytest.raises(ValueError, match="real vocabulary"):
        validate_opd_topk_vocab_size(args)


@pytest.mark.parametrize("override", [{"use_kl_loss": True}, {"kl_coef": 0.1}])
def test_rejects_ignored_reference_policy_kl_settings(override):
    args = _valid_args(**override)
    with pytest.raises(ValueError, match="reference-policy KL"):
        validate_opd_topk_loss_args(args)


def test_common_validation_rejects_topk_ref_kl_before_touching_missing_ref_load():
    args = Namespace(
        rollout_temperature=1.0,
        loss_type="opd_topk_loss",
        use_kl_loss=True,
        kl_coef=0.0,
        ref_load=None,
    )
    with pytest.raises(ValueError, match="reference-policy KL"):
        _common_orbit_validate_args(args)


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


def test_rejects_no_teacher_configured():
    # Finding 3 (final-review): nothing external configured at all (no
    # --opd-teacher-url(s), no --opd-serve-teacher, and an unset --opd-teacher so
    # local_scoring_enabled is False too) must be rejected with a clear message,
    # not silently sail through to the hooks check and only fail deep into a
    # rollout on a missing-key error.
    args = _valid_args(opd_teacher_url=None, opd_teacher_urls=None, opd_teacher=None)
    with pytest.raises(ValueError, match="external teacher"):
        validate_opd_topk_loss_args(args)


def test_allows_opd_serve_teacher_as_teacher_presence():
    # --opd-serve-teacher is a valid remedy: it publishes its endpoint as
    # --opd-teacher-url once its engines are up, so it must satisfy the presence
    # check even though opd_teacher_url is still unset at validation time.
    args = _valid_args(opd_teacher_url=None, opd_teacher_urls=None, opd_teacher=None, opd_serve_teacher=True)
    validate_opd_topk_loss_args(args)


def test_rejects_managed_same_engine_teacher_path():
    # opd_teacher="base" with no external URL selects
    # orbit.peft.opd.opd_scoring.local_scoring_enabled's path, which does not
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


def test_rejects_allgather_cp():
    # Finding 6 (final-review): --allgather-cp is only enforced at loss-compute time
    # today (get_log_probs_and_entropy's NotImplementedError) -- validate it up front
    # too, so a misconfigured run fails fast instead of after a full rollout.
    args = _valid_args(allgather_cp=True)
    with pytest.raises(ValueError, match="allgather-cp"):
        validate_opd_topk_loss_args(args)


def test_rejects_tail_bucket():
    args = _valid_args(opd_topk_tail_bucket=True)
    with pytest.raises(ValueError, match="opd-topk-tail-bucket"):
        validate_opd_topk_loss_args(args)


def test_rejects_missing_custom_rm_path():
    # opd_topk_loss bypasses needs_opd_teacher() (default grpo estimator, no
    # --use-opd), so nothing else enforces the OPD custom-reward hooks; without
    # this check a missing --custom-rm-path silently falls through to the
    # default reward path and teacher_topk_ids/logprobs never get populated.
    args = _valid_args(custom_rm_path=None)
    with pytest.raises(ValueError, match="custom-rm-path"):
        validate_opd_topk_loss_args(args)


def test_rejects_wrong_custom_rm_path():
    args = _valid_args(custom_rm_path="some.other.reward_func")
    with pytest.raises(ValueError, match="custom-rm-path"):
        validate_opd_topk_loss_args(args)


def test_rejects_missing_custom_reward_post_process_path():
    args = _valid_args(custom_reward_post_process_path=None)
    with pytest.raises(ValueError, match="custom-reward-post-process-path"):
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
