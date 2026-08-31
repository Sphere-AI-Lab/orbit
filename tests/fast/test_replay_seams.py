"""Orbit's MoE routing-replay changes still happen with both vendored files pristine.

Two vendored files carried them:

* ``miles/utils/replay_base.py`` -- a whole ``_sanitize_replay_top_indices``
  function plus six swapped lines inside ``BaseReplayManager.get_topk_fn``. Both
  are in orbit/utils/replay_ext.py now, the method installed on the vendored
  class by an import-time seam.
* ``miles/backends/megatron_utils/replay_utils.py`` -- the whole R3 wiring pass
  and its adapter, now orbit/megatron/routing_replay.py.

The bug the sanitiser exists for is silent: on this miles base upstream repairs
only rows that are ENTIRELY padded, so a partially padded row keeps a -1 expert
id; Megatron's dispatch map takes it and the model routes somewhere it never
chose. So every assertion below is on the ids that come out,
and the upstream half is exercised through the saved
``_orbit_unpatched_get_topk_fn`` to prove it is still upstream's body.
"""

import ast
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401,E402  -- importing orbit arms the seam

from miles.utils.replay_base import (  # noqa: E402
    BaseReplayManager,
    RoutingReplayManager,
    routing_replay_manager,
)
from orbit.utils import replay_ext  # noqa: E402
from orbit.utils.replay_ext import (  # noqa: E402
    GET_TOPK_FN_UPSTREAM_SHA,
    _sanitize_replay_top_indices,
    install_replay_sanitizer,
    method_sha,
)

REPO = Path(__file__).resolve().parents[2]
SEAM_MODULE = "orbit.utils.replay_ext"


class _StubReplay:
    """A Replay whose pops do not need a CUDA device.

    ``Replay.pop_forward`` ends in ``.to(torch.cuda.current_device())``, so the
    real class cannot be exercised on the CPU gate. Cloning on every pop also
    keeps upstream's IN-PLACE repair from corrupting the fixture between the two
    halves of a comparison.
    """

    def __init__(self, indices):
        self._indices = indices

    def pop_forward(self):
        return self._indices.clone()

    pop_backward = pop_forward


def _manager(indices):
    manager = RoutingReplayManager()
    manager.enabled = True
    manager.stage = "replay_forward"
    manager.set_current(_StubReplay(indices))
    return manager


def _unreachable(*args, **kwargs):  # the replay stages never call old_topk_fn
    raise AssertionError("old_topk_fn must not be called on the replay path")


# A replayed route with one padded slot per token -- and neither row is entirely
# padded, which is exactly the case upstream's repair skips on this base.
PADDED = torch.tensor([[0, -1], [1, -1]])
SCORES = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])


# --------------------------------------------------------------------------
# miles/utils/replay_base.py: the lifted function and the class seam
# --------------------------------------------------------------------------


def test_the_vendored_module_no_longer_carries_the_orbit_code():
    """The point of the move. If this fails the file went dirty again."""
    src = (REPO / "miles" / "utils" / "replay_base.py").read_text()
    assert "_sanitize_replay_top_indices" not in src
    assert "ORBIT" not in src


def test_the_seam_is_installed_on_the_vendored_class():
    assert BaseReplayManager.get_topk_fn.__module__ == SEAM_MODULE
    assert hasattr(BaseReplayManager, "_orbit_unpatched_get_topk_fn"), (
        "the pristine upstream method must be kept so drift stays observable"
    )


def test_installing_twice_is_a_no_op():
    """It runs once per process today; a second run must not stack or re-save."""
    assert install_replay_sanitizer() is False
    assert BaseReplayManager._orbit_unpatched_get_topk_fn.__module__ == (
        "miles.utils.replay_base"
    )


def test_the_pin_matches_the_vendored_source_statically():
    """The CPU-gate half of the pin: tools/check_patch_pins.py only reads
    ``@patch_function`` declarations, and this seam is on a class method."""
    src = (REPO / "miles" / "utils" / "replay_base.py").read_text()
    tree = ast.parse(src)
    segment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "BaseReplayManager":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "get_topk_fn":
                    segment = ast.get_source_segment(src, member)
    assert segment is not None, "upstream no longer defines BaseReplayManager.get_topk_fn"
    actual = method_sha(segment)
    assert actual == GET_TOPK_FN_UPSTREAM_SHA, (
        f"upstream's body changed; review orbit's copy in {SEAM_MODULE} and then "
        f"set GET_TOPK_FN_UPSTREAM_SHA = {actual!r}"
    )


def test_the_static_and_runtime_hashes_are_the_same_hash():
    """Two readers of one body: ``ast.get_source_segment`` dedents only the def
    line, ``inspect.getsource`` dedents nothing. If these ever disagree the
    static gate above is checking something the runtime never verifies."""
    runtime = method_sha(inspect.getsource(BaseReplayManager._orbit_unpatched_get_topk_fn))
    assert runtime == GET_TOPK_FN_UPSTREAM_SHA


def test_orbit_repairs_padding_without_repeating_an_expert():
    """The behaviour the seam exists for."""
    out = _manager(PADDED).get_topk_fn(_unreachable, return_probs=False)(SCORES, 2)
    for row in out.tolist():
        assert len(set(row)) == len(row), f"token routed to the same expert twice: {row}"
        assert all(0 <= expert < SCORES.shape[1] for expert in row)
    # The ids orbit did not have to repair are untouched.
    assert [row[0] for row in out.tolist()] == [0, 1]


def test_upstream_alone_leaves_the_padding_this_seam_repairs():
    """Proves the previous test measured orbit and not a coincidence, and that
    the saved original is still UPSTREAM's body rather than a copy of orbit's.

    What upstream does wrong here is base-dependent, and this base is the worse
    of the two. On orbit-main-isolated's older miles, the inline repair rewrote
    EVERY -1 with ``arange % num_experts`` and could hand a token the same expert
    twice ([[0, 0], [1, 1]] for this input). miles @ dbbab1566 repairs only rows
    that are entirely -1, so a partially padded row keeps its -1 and a NEGATIVE
    expert id reaches Megatron's sparse dispatch map. Orbit's sanitizer repairs
    both.
    """
    upstream = BaseReplayManager._orbit_unpatched_get_topk_fn
    out = upstream(_manager(PADDED), _unreachable, False)(SCORES, 2)
    assert out.tolist() == [[0, -1], [1, -1]]


def test_probabilities_are_gathered_against_the_repaired_ids():
    """Why the change cannot be made outside ``get_topk_fn``: the gather happens
    in the same closure as the repair, so sanitising the returned ids afterwards
    would leave the probabilities matched to upstream's."""
    probs, indices = _manager(PADDED).get_topk_fn(_unreachable, return_probs=True)(SCORES, 2)
    expected = SCORES.gather(1, indices)
    assert torch.equal(probs, expected)


def test_the_untouched_stages_still_run_upstreams_logic():
    """The seam replaces the whole method, so its non-replay stages are orbit's
    copy of upstream's. Keep them observably identical."""
    manager = _manager(PADDED)
    calls = []

    def old_topk_fn(scores, topk, *args, **kwargs):
        calls.append(topk)
        return torch.tensor([[3, 2], [2, 3]])

    manager.stage = "fallthrough"
    assert manager.get_topk_fn(old_topk_fn, False)(SCORES, 2).tolist() == [[3, 2], [2, 3]]

    manager.enabled = False
    manager.stage = "replay_forward"
    assert manager.get_topk_fn(old_topk_fn, False)(SCORES, 2).tolist() == [[3, 2], [2, 3]]
    assert calls == [2, 2]


def test_sanitize_rejects_routes_it_cannot_repair():
    with pytest.raises(ValueError, match="out of range"):
        _sanitize_replay_top_indices(torch.tensor([[0, 9]]), 4)
    with pytest.raises(ValueError, match="duplicate expert ids"):
        _sanitize_replay_top_indices(torch.tensor([[2, 2]]), 4)
    with pytest.raises(ValueError, match="cannot be represented"):
        _sanitize_replay_top_indices(torch.tensor([[0, 1, 2]]), 2)


def test_sanitize_leaves_a_clean_route_alone():
    clean = torch.tensor([[0, 3], [1, 2]])
    assert torch.equal(_sanitize_replay_top_indices(clean, 4), clean)
    assert _sanitize_replay_top_indices(torch.empty(0, 2, dtype=torch.long), 4).numel() == 0


# --------------------------------------------------------------------------
# megatron_utils/replay_utils.py: the lifted R3 wiring
# --------------------------------------------------------------------------


def test_the_vendored_replay_utils_no_longer_carries_the_wiring():
    src = (REPO / "miles" / "backends" / "megatron_utils" / "replay_utils.py").read_text()
    assert "wire_routing_replay_to_models" not in src
    assert "ORBIT" not in src


def test_the_actor_calls_the_wiring_from_its_new_home():
    tree = ast.parse((REPO / "miles" / "backends" / "megatron_utils" / "actor.py").read_text())
    sources = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(a.name == "wire_routing_replay_to_models" for a in node.names)
    }
    assert sources == {"orbit.megatron.routing_replay"}


def test_the_adapter_routes_megatrons_hook_through_orbits_get_topk_fn(monkeypatch):
    """The adapter's whole job: Megatron's per-layer replay hook must reach the
    manager (and therefore the sanitiser), not Megatron's inference RouterReplay.
    """
    from orbit.megatron.routing_replay import _R3_ADAPTER

    seen = {}

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        seen.update(scores=scores, topk=topk, num_groups=num_groups, group_topk=group_topk)
        return SCORES, torch.tensor([[0, 1], [1, 0]])

    monkeypatch.setattr(routing_replay_manager, "enabled", False)
    out = _R3_ADAPTER.get_replay_topk(SCORES, 2, 1, 1, compute_topk)
    assert out[1].tolist() == [[0, 1], [1, 0]]
    assert seen == {"scores": SCORES, "topk": 2, "num_groups": 1, "group_topk": 1}


def _fake_megatron_classes(monkeypatch):
    router_mod = pytest.importorskip("megatron.core.transformer.moe.router")
    moe_layer_mod = pytest.importorskip("megatron.core.transformer.moe.moe_layer")

    class FakeRouter:
        def __init__(self):
            self.router_replay = None
            self.hooks = []

        def register_forward_pre_hook(self, fn):
            self.hooks.append(fn)

    class FakeMoELayer:
        def __init__(self, is_mtp_layer):
            self.is_mtp_layer = is_mtp_layer

    monkeypatch.setattr(router_mod, "TopKRouter", FakeRouter)
    monkeypatch.setattr(moe_layer_mod, "MoELayer", FakeMoELayer)
    return FakeRouter, FakeMoELayer


class _FakeModel:
    def __init__(self, named):
        self._named = named

    def named_modules(self):
        return list(self._named)


def test_wiring_registers_every_moe_router_and_skips_mtp(monkeypatch):
    from orbit.megatron.routing_replay import _R3_ADAPTER, wire_routing_replay_to_models

    FakeRouter, FakeMoELayer = _fake_megatron_classes(monkeypatch)
    monkeypatch.setattr(routing_replay_manager, "enabled", True)
    monkeypatch.setattr(routing_replay_manager, "replays", [])

    normal, mtp = FakeRouter(), FakeRouter()
    model = _FakeModel(
        [
            ("decoder.layers.0.mlp", FakeMoELayer(is_mtp_layer=False)),
            ("decoder.layers.0.mlp.router", normal),
            ("decoder.layers.1.mlp", FakeMoELayer(is_mtp_layer=True)),
            ("decoder.layers.1.mlp.router", mtp),
        ]
    )
    wire_routing_replay_to_models(model)

    assert normal.router_replay is _R3_ADAPTER
    assert normal.hooks, "no forward pre-hook, so the per-layer replay is never selected"
    assert hasattr(normal, "routing_replay")
    # MTP routes fresh every step; replaying into it is the bug the skip prevents.
    assert mtp.router_replay is None
    assert not hasattr(mtp, "routing_replay")
    assert len(routing_replay_manager.replays) == 1


def test_wiring_is_a_no_op_when_replay_is_off(monkeypatch):
    from orbit.megatron.routing_replay import wire_routing_replay_to_models

    FakeRouter, _ = _fake_megatron_classes(monkeypatch)
    monkeypatch.setattr(routing_replay_manager, "enabled", False)
    router = FakeRouter()
    wire_routing_replay_to_models(_FakeModel([("decoder.layers.0.mlp.router", router)]))
    assert router.router_replay is None


def test_the_seam_module_is_registered_so_import_orbit_arms_it():
    from orbit.patch.on_import import check_seam_targets, fired, registry

    assert replay_ext._REPLAY_BASE in {module for module, _ in registry()}
    assert replay_ext._REPLAY_BASE in {module for module, _ in fired()}
    assert check_seam_targets() == []


def test_the_seam_survives_replay_base_being_imported_before_orbit():
    """Import ORDER must not decide whether the seam takes effect: a module
    already in sys.modules missed the meta_path hook, and on_import fires the
    callback immediately instead. Run it for real in a subprocess."""
    import os
    import subprocess
    import sys

    probe = (
        "import miles.utils.replay_base as rb\n"  # BEFORE orbit
        "import orbit\n"
        "print(rb.BaseReplayManager.get_topk_fn.__module__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith(SEAM_MODULE), out.stdout
