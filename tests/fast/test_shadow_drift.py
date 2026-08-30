"""Upstream may not silently change a method orbit overrides via a mixin.

A mixin listed first shadows the vendored method, which makes that method dead
code — and dead code never conflicts, so an upstream rewrite of it lands
silently while orbit's override goes on winning. This pins a hash of every
shadowed body so that change becomes a reviewable failure instead.
See tools/check_shadow_drift.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "check_shadow_drift.py"

SPEC = importlib.util.spec_from_file_location("orbit_check_shadow_drift", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_no_upstream_drift_under_an_orbit_override():
    errors = CHECKER.collect_errors()
    assert not errors, "shadowed-method drift:\n  " + "\n  ".join(errors)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo sources use PEP 695 syntax")
def test_the_guard_actually_finds_the_mixins():
    """A discovery bug would make this guard vacuously green, which is worse than
    not having it: assert it still sees all three orbit mixin sites.

    Note this asserts on DISCOVERY, not on pins. Only a mixin that overrides a
    method the base class also defines produces something to pin, and today only
    OrbitTrainActorExtensions does: OrbitEngineExtensions' methods have no
    RayActor counterpart, and UpdateWeightFromTensor's only base IS its mixin.
    Asserting all three produce pins would fail for a correct guard.
    """
    sites = {mixin for _rel, _cls, _tree, _src, mixin, _methods in CHECKER._mixin_sites()}
    for expected in (
        "orbit.megatron.actor_ext.OrbitTrainActorExtensions",
        "orbit.sglang.engine_ext.OrbitEngineExtensions",
        "orbit.transport.update_weight_ext.OrbitUpdateWeightExtensions",
    ):
        assert expected in sites, f"{expected} no longer discovered; guard is blind"
