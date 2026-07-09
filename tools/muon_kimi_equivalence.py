#!/usr/bin/env python
"""Numerical receipt: emerging_optimizers Muon (with the Kimi preset flags)
reproduces slm-research's vendored Kimi-Muon on the 2D orthogonalized update.

Runs both optimizers from an identical weight through an identical grad
sequence and reports, per step, the update's directional agreement (cosine)
and relative L2 gap. Kimi's Newton-Schulz runs in bf16, so the raw gap sits
at bf16 precision (~1e-2) while direction should be ~1.0 — that split is the
whole point: same algorithm, bf16-limited precision, not a different update.

Kimi preset  ==  --optimizer muon --muon-scale-mode spectral
                 --muon-extra-scale-factor 0.2 --muon-nesterov
                 --muon-coefficient-type simple --muon-num-ns-steps 5

NOTE coefficient_type="simple" (Keller-Jordan's (3.4445,-4.7750,2.0315)) is
Kimi's actual set — NOT emerging's default "quintic" (a newer better-
converging set). With "simple" the per-step update matches Kimi to bf16
precision (cos>0.999); the residual gap is Kimi's own bf16 Newton-Schulz.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/lustre/fast/fast/lechen/clthegoat/slm-research")
from src.optim._kimi_muon import Muon as KimiMuon  # noqa: E402
from emerging_optimizers.orthogonalized_optimizers.muon import Muon as EmergingMuon  # noqa: E402


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def run(nesterov: bool, wd: float, steps: int = 5, shape=(256, 512), lr=1e-2, momentum=0.95):
    torch.manual_seed(0)
    w0 = torch.randn(*shape, dtype=torch.float32)
    grads = [torch.randn(*shape, dtype=torch.float32) for _ in range(steps)]

    wk = w0.clone().requires_grad_(True)
    ke = KimiMuon(lr=lr, wd=wd, muon_params=[wk], momentum=momentum, nesterov=nesterov, ns_steps=5)

    we = w0.clone().requires_grad_(True)
    em = EmergingMuon(
        [we], lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov,
        weight_decay_method="decoupled", scale_mode="spectral", extra_scale_factor=0.2,
        coefficient_type="simple", num_ns_steps=5, fp32_matmul_prec="high",
    )

    worst_cos, worst_rel = 1.0, 0.0
    for g in grads:
        pk, pe = wk.detach().clone(), we.detach().clone()
        wk.grad = g.clone()
        we.grad = g.clone()
        ke.step()
        em.step()
        dk, de = wk.detach() - pk, we.detach() - pe   # the two updates
        worst_cos = min(worst_cos, _cos(dk, de))
        worst_rel = max(worst_rel, _rel(dk, de))
    final_rel = _rel(wk.detach(), we.detach())
    return worst_cos, worst_rel, final_rel


def main() -> int:
    print(f"{'config':28s} {'min cos(update)':>16s} {'max rel-gap':>14s} {'final W rel-gap':>16s}")
    ok = True
    for nesterov in (True, False):
        for wd in (0.0, 0.1):
            c, r, fr = run(nesterov=nesterov, wd=wd)
            label = f"nesterov={nesterov}, wd={wd}"
            print(f"{label:28s} {c:16.6f} {r:14.4f} {fr:16.4f}")
            # same algorithm + coeffs ⇒ cos matches to Kimi's bf16-NS precision
            ok = ok and c > 0.999 and fr < 0.02
    print(f"\n### MUON_KIMI_EQUIV {'PASS' if ok else 'FAIL'}  "
          f"(cos>0.999 = same update direction; gap is bf16-NS precision)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
