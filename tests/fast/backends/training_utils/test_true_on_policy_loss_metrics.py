from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import losses as loss_utils
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages

from .loss.loss_test_utils import make_parallel_state


def _make_args(*, use_rollout_logprobs: bool) -> Namespace:
    return Namespace(
        use_rollout_logprobs=use_rollout_logprobs,
        use_opsm=False,
        advantage_estimator="ppo",
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=False,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        kl_loss_coef=0.0,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        allgather_cp=False,
    )


def _make_batch(*, old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> dict:
    return {
        "advantages": [torch.tensor([1.0, -0.5], dtype=torch.float32)],
        "log_probs": [old_log_probs],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.tensor([7, 8, 9], dtype=torch.long)],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 1.0], dtype=torch.float32)],
    }


def _patch_single_rank_loss_helpers(monkeypatch):
    monkeypatch.setattr(
        loss_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_ess_ratio_contribution",
        lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0),
    )


@pytest.mark.parametrize(
    (
        "use_rollout_logprobs",
        "train_log_probs",
        "old_log_probs",
        "rollout_log_probs",
        "expected_reference_abs_diff",
        "expected_current_abs_diff",
    ),
    [
        (
            False,
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.45,
            0.0,
        ),
        (
            True,
            torch.tensor([0.50, 1.00], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.0,
            0.15,
        ),
    ],
)
def test_train_rollout_logprob_abs_diff_uses_policy_loss_reference_logprobs(
    monkeypatch,
    use_rollout_logprobs: bool,
    train_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    expected_reference_abs_diff: float,
    expected_current_abs_diff: float,
):
    args = _make_args(use_rollout_logprobs=use_rollout_logprobs)
    batch = _make_batch(old_log_probs=old_log_probs, rollout_log_probs=rollout_log_probs)

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [train_log_probs.clone()],
            "entropy": [torch.zeros_like(train_log_probs)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(expected_reference_abs_diff))
    torch.testing.assert_close(metrics["current_rollout_logprob_abs_diff"], torch.tensor(expected_current_abs_diff))
    assert torch.isfinite(metrics["current_rollout_kl"])


def test_zero_weighted_entropy_nan_does_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        assert kwargs["with_entropy"] is False
        return {"log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)]}

    monkeypatch.setattr(loss_utils, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["entropy_loss"], torch.tensor(0.0))


def test_policy_loss_adds_explicit_dagger_term_and_keeps_metric_namespace(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.opd_dagger_coef = 2.0
    args.opd_dagger_loss = "explicit_cross_entropy"
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {"log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)]},
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    def fake_dagger_loss(args, batch, logits, reducer):
        explicit_ce = logits.sum() * 0 + 0.75
        return explicit_ce, {
            "explicit_ce": explicit_ce.detach(),
            "teacher_topk_mass": explicit_ce.new_tensor(0.9),
        }

    monkeypatch.setattr(loss_utils, "compute_explicit_dagger_loss", fake_dagger_loss)
    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    torch.testing.assert_close(loss, torch.tensor(1.5))
    torch.testing.assert_close(metrics["opd_dagger/explicit_ce"], torch.tensor(0.75))
    torch.testing.assert_close(metrics["opd_dagger/loss"], torch.tensor(1.5))
    torch.testing.assert_close(metrics["opd_dagger/teacher_topk_mass"], torch.tensor(0.9))


def test_policy_loss_dispatches_complete_topk_rest_dagger_term(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.opd_dagger_coef = 1.5
    args.opd_dagger_loss = "cross_entropy"
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {"log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)]},
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.full_like(ppo_kl, 0.4),
            torch.zeros_like(ppo_kl),
        ),
    )

    def fake_topk_rest_loss(args, batch, logits, reducer):
        cross_entropy = logits.sum() * 0 + 0.8
        return cross_entropy, {
            "cross_entropy": cross_entropy.detach(),
            "explicit_ce": cross_entropy.new_tensor(0.6),
            "rest_ce": cross_entropy.new_tensor(0.2),
            "teacher_entropy": cross_entropy.new_tensor(0.5),
            "coarse_kl": cross_entropy.new_tensor(0.3),
            "teacher_rest_mass": cross_entropy.new_tensor(0.1),
            "student_rest_mass": cross_entropy.new_tensor(0.15),
        }

    monkeypatch.setattr(loss_utils, "compute_topk_rest_dagger_loss", fake_topk_rest_loss)
    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    torch.testing.assert_close(loss, torch.tensor(1.6))
    torch.testing.assert_close(metrics["loss"], torch.tensor(1.6))
    torch.testing.assert_close(metrics["pg_loss"], torch.tensor(0.4))
    torch.testing.assert_close(metrics["opd_dagger/cross_entropy"], torch.tensor(0.8))
    torch.testing.assert_close(metrics["opd_dagger/explicit_ce"], torch.tensor(0.6))
    torch.testing.assert_close(metrics["opd_dagger/rest_ce"], torch.tensor(0.2))
    torch.testing.assert_close(metrics["opd_dagger/teacher_entropy"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["opd_dagger/coarse_kl"], torch.tensor(0.3))
    torch.testing.assert_close(metrics["opd_dagger/loss"], torch.tensor(1.2))


def test_sampled_rkld_pg_and_topk_rest_dagger_compose_in_value_gradient_and_metrics(monkeypatch):
    """Milestone 03: the two independently validated branches share one forward."""
    make_parallel_state()
    _patch_single_rank_loss_helpers(monkeypatch)

    args = _make_args(use_rollout_logprobs=False)
    # Keep this fast composition test independent of the optional Megatron
    # fused-CE import. The DAgger branch still exercises the production
    # Stable-TP operator; only the sampled-token log-prob lookup uses the
    # single-rank full-vocabulary reference path.
    args.true_on_policy_mode = True
    args.vocab_size = 4
    args.opd_dagger_top_k = 2
    args.opd_dagger_loss = "cross_entropy"
    args.opd_dagger_coef = 0.7

    base_logits = torch.tensor(
        [[[1.2, -0.4, 0.8, -1.0], [-0.3, 1.1, 0.2, 0.7], [9.0, 9.0, 9.0, 9.0]]],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([3, 2], dtype=torch.long)
    with torch.no_grad():
        old_student_log_probs = (
            torch.log_softmax(base_logits[0, :2], dim=-1)
            .gather(
                dim=-1,
                index=sampled_tokens.unsqueeze(-1),
            )
            .squeeze(-1)
        )

    # Build the RKLD-PG coefficient exactly as the training pre-pass does. Both
    # q_old and p_T intentionally carry autograd history here so this test also
    # proves the stop-gradient boundary is enforced rather than assumed.
    expected_reverse_kl = torch.tensor([0.25, -0.15], dtype=torch.float32)
    old_student_leaf = old_student_log_probs.clone().requires_grad_()
    teacher_sampled_leaf = (old_student_log_probs - expected_reverse_kl).clone().requires_grad_()
    rkld_advantages = [torch.zeros(2, dtype=torch.float32)]
    rollout_data = {"teacher_log_probs": [teacher_sampled_leaf]}
    apply_opd_kl_to_advantages(
        Namespace(opd_type="sglang", opd_kl_coef=0.4),
        rollout_data,
        rkld_advantages,
        [old_student_leaf],
    )
    torch.testing.assert_close(rkld_advantages[0], -0.4 * expected_reverse_kl)
    assert rkld_advantages[0].requires_grad is False
    assert rollout_data["opd_reverse_kl"][0].requires_grad is False

    teacher_topk_probs = torch.tensor([[0.6, 0.3], [0.5, 0.2]], dtype=torch.float32)
    teacher_topk_log_probs = teacher_topk_probs.log().requires_grad_()
    common_batch = {
        "log_probs": [old_student_log_probs.detach()],
        "rollout_log_probs": [old_student_log_probs.detach().clone()],
        "unconcat_tokens": [torch.tensor([0, 3, 2], dtype=torch.long)],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.ones(2, dtype=torch.float32)],
        "opd_reverse_kl": [rollout_data["opd_reverse_kl"][0]],
        "teacher_topk_token_ids": [torch.tensor([[1, 2], [0, 1]], dtype=torch.long)],
        "teacher_topk_log_probs": [teacher_topk_log_probs],
        "teacher_topk_valid_mask": [torch.ones((2, 2), dtype=torch.bool)],
    }

    def run_branch(*, advantages: torch.Tensor, dagger_coef: float):
        branch_args = Namespace(**vars(args))
        branch_args.opd_dagger_coef = dagger_coef
        branch_logits = base_logits.clone().requires_grad_()
        batch = {**common_batch, "advantages": [advantages.clone()]}
        loss, metrics = loss_utils.policy_loss_function(
            branch_args,
            batch,
            branch_logits,
            sum_of_sample_mean=lambda tensor: tensor.float().mean(),
        )
        loss.backward()
        return loss.detach(), metrics, branch_logits.grad.detach().clone()

    hybrid_loss, hybrid_metrics, hybrid_grad = run_branch(
        advantages=rkld_advantages[0],
        dagger_coef=args.opd_dagger_coef,
    )
    rkld_loss, _, rkld_grad = run_branch(advantages=rkld_advantages[0], dagger_coef=0.0)
    dagger_loss, _, dagger_grad = run_branch(
        advantages=torch.zeros_like(rkld_advantages[0]),
        dagger_coef=args.opd_dagger_coef,
    )

    torch.testing.assert_close(hybrid_loss, rkld_loss + dagger_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(hybrid_grad, rkld_grad + dagger_grad, atol=1e-6, rtol=1e-6)
    assert torch.linalg.vector_norm(rkld_grad) > 0
    assert torch.linalg.vector_norm(dagger_grad) > 0

    torch.testing.assert_close(hybrid_metrics["loss"], hybrid_loss)
    torch.testing.assert_close(hybrid_metrics["pg_loss"], rkld_loss)
    torch.testing.assert_close(hybrid_metrics["opd_dagger/loss"], dagger_loss)
    torch.testing.assert_close(hybrid_metrics["opd_reverse_kl"], expected_reverse_kl.mean())
    assert "opd_dagger/cross_entropy" in hybrid_metrics
    assert old_student_leaf.grad is None
    assert teacher_sampled_leaf.grad is None
    assert teacher_topk_log_probs.grad is None


def test_zero_weighted_kl_nan_does_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.0
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )
    batch["ref_log_probs"] = [torch.tensor([float("nan"), float("nan")], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])


def test_masked_nonfinite_ppo_terms_do_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
    )
    batch["loss_masks"] = [torch.tensor([1.0, 0.0], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, float("nan")], dtype=torch.float32)],
        },
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: (tensor * batch["loss_masks"][0]).sum(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["ppo_kl"])
