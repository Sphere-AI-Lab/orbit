from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import loss as training_loss
# upstream moved policy_loss_function out of training_utils/loss.py into
# training_utils/loss_hub/losses.py (loss.py re-exports it). Collaborators have to
# be patched on the defining module or the real ones run.
from miles.backends.training_utils.loss_hub import losses as losses_mod
from miles.backends.training_utils.loss_hub.math_utils import _safe_clamp_log_ratio, compute_approx_kl


def _unbiased_kl(log_probs: torch.Tensor, old_log_probs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Mirrors the loss.py use_unbiased_kl site post-fix.
    importance_ratio = _safe_clamp_log_ratio(log_probs - old_log_probs).exp()
    return compute_approx_kl(log_probs, ref, kl_loss_type="k3", importance_ratio=importance_ratio)


def test_extreme_off_policy_drift_stays_finite() -> None:
    # Extreme drift lives in the RATIO exponent (x - old) — the quantity the
    # loss.py fix clamps. The KL pair (x vs ref) stays moderate: k3's own
    # exp(ref - x) is NOT clamped for plain k3 and must not need to be here.
    log_probs = torch.tensor([120.0, -120.0, 0.3], requires_grad=True)
    old_log_probs = torch.zeros(3)
    ref = (log_probs - 0.25).detach()
    kl = _unbiased_kl(log_probs, old_log_probs, ref)
    kl.sum().backward()
    assert torch.isfinite(kl).all()
    assert torch.isfinite(log_probs.grad).all()
    # The raw (unclamped) weight overflows exactly where the clamp saves us:
    assert not torch.isfinite(torch.exp(log_probs.detach() - old_log_probs)).all()


def test_in_band_ratio_matches_raw_exp() -> None:
    log_probs = torch.tensor([0.5, -1.5, 3.0])
    old_log_probs = torch.tensor([0.1, 0.2, -0.3])
    clamped = _safe_clamp_log_ratio(log_probs - old_log_probs).exp()
    torch.testing.assert_close(clamped, torch.exp(log_probs - old_log_probs))


def test_unbiased_kl_gradient_includes_score_term() -> None:
    # NeMo 33bce20d7's regression, in orbit terms: the IS weight w = exp(x - old)
    # depends on x, so d(w * kl)/dx = w * kl + w * d(kl)/dx. For k3 with
    # r = ref - x: kl = exp(r) - 1 - r, d(kl)/dx = 1 - exp(r).
    x = torch.tensor([0.4, -0.7, 1.2], requires_grad=True)
    old = torch.tensor([0.1, -0.2, 0.9])
    ref = torch.tensor([0.0, 0.3, 1.0])
    kl = _unbiased_kl(x, old, ref)
    kl.sum().backward()

    with torch.no_grad():
        w = torch.exp(x - old)
        r = ref - x
        kl_tok = torch.exp(r) - 1 - r
        expected = w * (kl_tok + 1 - torch.exp(r))
    torch.testing.assert_close(x.grad, expected)


@pytest.fixture
def _trivial_parallel_state() -> None:
    """A real single-rank ParallelState in the module global.

    Stubbing ``losses_mod.get_parallel_state`` is no longer enough: upstream's
    policy_loss_function reaches cp_utils.get_local_response_loss_masks, which resolves
    ``get_parallel_state`` in its own module. Before the restructure this test passed
    only when an earlier test in the same process had left the global set; pin it so
    the test is order-independent. group=None short-circuits GroupInfo's post-init
    verification, so no torch.distributed init is needed.
    """
    from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


@pytest.mark.parametrize(
    ("use_rollout_logprobs", "use_tis", "expected_denominator"),
    [
        (False, False, "trainer"),
        (True, False, "rollout"),
        (False, True, "rollout"),
    ],
)
@pytest.mark.usefixtures("_trivial_parallel_state")
def test_policy_loss_uses_sampling_policy_for_unbiased_kl(
    monkeypatch: pytest.MonkeyPatch,
    use_rollout_logprobs: bool,
    use_tis: bool,
    expected_denominator: str,
) -> None:
    """Exercise the real policy-loss wiring, including the async TIS case."""
    current = torch.tensor([0.4, -0.7, 1.2], requires_grad=True)
    trainer_old = torch.tensor([0.1, -0.2, 0.9])
    rollout_behavior = torch.tensor([-0.8, 0.5, 0.0])
    ref = torch.tensor([0.0, 0.3, 1.0])
    captured: dict[str, torch.Tensor] = {}

    args = SimpleNamespace(
        use_rollout_logprobs=use_rollout_logprobs,
        use_opsm=False,
        advantage_estimator="ppo",
        force_on_policy_ratio=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=use_tis,
        custom_tis_function_path=None,
        calculate_per_token_loss=True,
        qkv_format="thd",
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=True,
        use_unbiased_kl=True,
        kl_loss_type="k3",
        kl_loss_coef=1.0,
        # upstream's policy_loss_function reads these two before the KL block.
        skip_actor_forward_only=False,
        observe_training_entropy=False,
    )
    batch = {
        "advantages": [torch.zeros(3)],
        "log_probs": [trainer_old],
        "rollout_log_probs": [rollout_behavior],
        "ref_log_probs": [ref],
        "response_lengths": [3],
        "total_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "unconcat_tokens": [torch.arange(3)],
    }

    monkeypatch.setattr(losses_mod, "get_parallel_state", lambda: object())
    monkeypatch.setattr(
        losses_mod,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [current],
            "entropy": [torch.zeros_like(current)],
        },
    )
    monkeypatch.setattr(
        losses_mod,
        "compute_policy_loss",
        lambda ppo_kl, *args, **kwargs: (torch.zeros_like(ppo_kl), torch.zeros_like(ppo_kl)),
    )

    real_compute_approx_kl = losses_mod.compute_approx_kl

    def capture_kl(
        log_probs: torch.Tensor,
        log_probs_base: torch.Tensor,
        kl_loss_type: str,
        importance_ratio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if importance_ratio is None:
            # upstream's train-vs-rollout mismatch diagnostic calls compute_approx_kl
            # too (kl_loss_type="low_var_kl", no importance ratio). That is not the call
            # under test; let the real implementation serve it.
            return real_compute_approx_kl(log_probs, log_probs_base, kl_loss_type=kl_loss_type)
        captured["importance_ratio"] = importance_ratio
        return importance_ratio * (log_probs - log_probs_base).square()

    monkeypatch.setattr(losses_mod, "compute_approx_kl", capture_kl)
    monkeypatch.setattr(
        losses_mod,
        "vanilla_tis_function",
        lambda **kwargs: (kwargs["pg_loss"], kwargs["loss_masks"], {}),
    )

    def reducer(values: torch.Tensor) -> torch.Tensor:
        return values.sum()

    monkeypatch.setattr(losses_mod, "get_sum_of_sample_mean", lambda *args, **kwargs: reducer)
    monkeypatch.setattr(losses_mod, "_response_masked_max", lambda values, **kwargs: values.max())

    loss, _ = training_loss.policy_loss_function(
        args,
        batch,
        torch.zeros(1, 3, 2, requires_grad=True),
        reducer,
    )
    loss.backward()

    denominator = trainer_old if expected_denominator == "trainer" else rollout_behavior
    expected = _safe_clamp_log_ratio(current.detach() - denominator).exp()
    torch.testing.assert_close(captured["importance_ratio"].detach(), expected)
    delta = current.detach() - ref
    # d[exp(x-denominator) * (x-ref)^2]/dx includes both the ordinary KL
    # derivative and the score-function derivative through the IS weight.
    expected_grad = expected * (2 * delta + delta.square())
    torch.testing.assert_close(current.grad, expected_grad)
