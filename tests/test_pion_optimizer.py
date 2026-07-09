"""Unit tests for the Pion optimizer port (pion + pion_msign).

Covers the orbit-side wiring that's testable without a GPU model: the
ZeRO-disable predicate, the OptimizerConfig field surface, and that the
Megatron getters import. The real construct/step path is verified by the
GPU smoke (logs/local_pion_smoke.sh), mirroring the Muon verification.
"""

from __future__ import annotations

import dataclasses

from orbit.backends.megatron_utils.arguments import _is_muon_optimizer, _is_pion_optimizer


def test_is_pion_predicate():
    for name in ("pion", "pion_msign", "Pion", "PION_MSIGN"):
        assert _is_pion_optimizer(name), name
    for name in ("adam", "sgd", "muon", "dist_muon", None, ""):
        assert not _is_pion_optimizer(name), name


def test_pion_and_muon_predicates_disjoint():
    # pion must not be classified as muon (they take different getters)
    assert not _is_muon_optimizer("pion")
    assert not _is_muon_optimizer("pion_msign")
    assert not _is_pion_optimizer("muon")


def test_zero_disabled_for_pion():
    from types import SimpleNamespace

    from orbit.backends.megatron_utils import arguments as A

    # exercise just the ZeRO-disable line the shim runs
    for opt, expect_dist in (("adam", True), ("muon", False), ("pion", False), ("pion_msign", False)):
        use_dist = not (A._is_muon_optimizer(opt) or A._is_pion_optimizer(opt))
        assert use_dist is expect_dist, opt


def test_optimizer_config_has_pion_fields():
    from megatron.core.optimizer import OptimizerConfig

    fields = {f.name for f in dataclasses.fields(OptimizerConfig)}
    # the fields the Pion getters read must exist so config forwarding works
    required = {
        "pion_momentum", "pion_update_side", "pion_scaling", "pion_rms",
        "pion_beta1", "pion_beta2", "pion_degree", "pion_exp_map",
        "pion_first_momentum", "pion_second_momentum", "pion_12_momentum",
        "pion_use_second_momentum", "pion_qkv_split_granularity",
        "pion_msign_lambda", "pion_spectrum_reset_interval",
    }
    missing = required - fields
    assert not missing, f"missing pion OptimizerConfig fields: {sorted(missing)}"


def test_pion_getters_import():
    from megatron.core.optimizer.pion import get_megatron_pion_optimizer
    from megatron.core.optimizer.pion_msign import get_megatron_pion_ortho_exp_optimizer

    assert callable(get_megatron_pion_optimizer)
    assert callable(get_megatron_pion_ortho_exp_optimizer)


def test_build_optimizer_routes_pion():
    # the dispatch branch in _build_optimizer_and_scheduler must select the
    # pion getters for pion/pion_msign and fall through otherwise
    import inspect

    from orbit.backends.megatron_utils import model

    src = inspect.getsource(model._build_optimizer_and_scheduler)
    assert 'if "pion" in optimizer_type' in src
    assert "get_megatron_pion_ortho_exp_optimizer" in src
    assert "get_megatron_pion_optimizer" in src
