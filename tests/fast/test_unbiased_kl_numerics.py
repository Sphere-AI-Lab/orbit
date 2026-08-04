import torch

from orbit.utils.ppo_utils import _safe_clamp_log_ratio, compute_approx_kl


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
