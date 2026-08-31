"""Qwen3.6 MTP-expert packing after the move into orbit's home mixin.

miles_plugins/mbridge/qwen3_5.py bridges Qwen3.5. Orbit registers the same
bridge for Qwen3.6, which shares the ``qwen3_5_moe`` config schema but packs the
MTP layer's experts as fused 3-D tensors instead of per-expert ``.weight``
files. orbit/megatron/qwen3_5_ext.py adds the fused mapping, the detector, and
the ``_orbit_mtp_mlp_mapping`` property that picks between them.

The shape of the seam is the thing to protect:

* Upstream's ``_MTP_MLP_MAPPING`` dict stays byte-pristine in the vendored class
  -- it IS the unfused variant, so orbit consumes it rather than forking it.
* The property is therefore called ``_orbit_mtp_mlp_mapping`` and NOT
  ``_MTP_MLP_MAPPING``. A mixin cannot shadow an attribute the vendored class
  defines in its own body, so a same-named property would lose to upstream's
  dict, silently, and every Qwen3.6 MTP expert would be looked up under Qwen3.5
  names -- a load failure far from its cause.
* ``_weight_name_mapping_mtp_mlp`` is 6% orbit, so it stayed upstream's; its
  only orbit-owned line is the ``self._orbit_mtp_mlp_mapping`` read.

``mbridge`` is not installed in the CPU gate, so the vendored module cannot be
imported here. The class-body facts are read out of the source with ``ast``
(which is also the only way to prove a name is ABSENT from the class body), and
the behaviour is driven through a host class that mixes in the same mixin over
the vendored file's own literal mapping.
"""

import ast
from pathlib import Path

import pytest

from orbit.megatron.qwen3_5_ext import OrbitQwen35BridgeExtensions

VENDORED = Path(__file__).resolve().parents[2] / "miles_plugins/mbridge/qwen3_5.py"
MTP_NAME = "mtp.layers.0.mlp.experts.linear_fc1.weight3"


def _class_node() -> ast.ClassDef:
    tree = ast.parse(VENDORED.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5Bridge")


def _class_body_names() -> set[str]:
    names = set()
    for node in _class_node().body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _vendored_unfused_mapping() -> dict:
    node = next(
        n
        for n in _class_node().body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_MTP_MLP_MAPPING" for t in n.targets)
    )
    return ast.literal_eval(node.value)


class _Host(OrbitQwen35BridgeExtensions):
    """Stands in for Qwen3_5Bridge: same mixin, same pristine class dict."""

    _MTP_MLP_MAPPING = _vendored_unfused_mapping()
    _MLP_MAPPING = {"mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"]}

    # Upstream's method body, read from the vendored file so this cannot drift.
    exec(  # noqa: S102 - the source is this repo's own vendored file
        ast.unparse(
            next(
                n
                for n in _class_node().body
                if isinstance(n, ast.FunctionDef) and n.name == "_weight_name_mapping_mtp_mlp"
            )
        )
    )


@pytest.mark.parametrize("name", ("_mtp_experts_fused", "_orbit_mtp_mlp_mapping", "_MTP_MLP_MAPPING_FUSED"))
def test_orbit_names_are_absent_from_the_vendored_class_body(name):
    assert name not in _class_body_names(), (
        f"{name} is defined in Qwen3_5Bridge's own body; the class __dict__ beats "
        f"every base, so that copy would shadow the mixin"
    )
    assert hasattr(OrbitQwen35BridgeExtensions, name), f"{name} must live on the mixin"


def test_upstreams_mapping_dict_is_still_in_the_vendored_class_and_unfused():
    """Orbit consumes upstream's dict instead of forking it; if it ever moved
    into the mixin, the property below would recurse or read a stale copy."""
    assert "_MTP_MLP_MAPPING" in _class_body_names()
    assert "_MTP_MLP_MAPPING" not in vars(OrbitQwen35BridgeExtensions)
    assert all(
        "{expert_id}" in target
        for targets in _vendored_unfused_mapping().values()
        for target in targets
    ), "upstream's dict must still be the per-expert (unfused) variant"


def test_the_property_is_not_named_after_the_vendored_dict():
    """A mixin property named _MTP_MLP_MAPPING would be shadowed by the vendored
    class attribute of that name -- silently."""
    assert isinstance(vars(OrbitQwen35BridgeExtensions)["_orbit_mtp_mlp_mapping"], property)
    assert "_MTP_MLP_MAPPING" not in vars(OrbitQwen35BridgeExtensions)


def test_the_vendored_class_lists_the_mixin_first_and_registers_qwen3_6():
    node = _class_node()
    assert [b.id for b in node.bases if isinstance(b, ast.Name)][0] == "OrbitQwen35BridgeExtensions"
    registered = {
        s.value
        for d in node.decorator_list
        if isinstance(d, ast.Call)
        for arg in d.args
        if isinstance(arg, ast.List)
        for s in arg.elts
        if isinstance(s, ast.Constant)
    }
    assert {"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe"} <= registered


def test_the_vendored_method_reads_the_orbit_property_not_the_raw_dict():
    body = ast.unparse(
        next(
            n
            for n in _class_node().body
            if isinstance(n, ast.FunctionDef) and n.name == "_weight_name_mapping_mtp_mlp"
        )
    )
    assert "self._orbit_mtp_mlp_mapping" in body
    assert "self._MTP_MLP_MAPPING " not in body, "reading the raw dict would pin Qwen3.5's packing"


def test_no_safetensor_index_means_unfused_and_nothing_is_cached():
    """A pre-init probe must not freeze the answer for the real load."""
    host = _Host()
    assert host._mtp_experts_fused() is False
    assert not hasattr(host, "_mtp_experts_fused_cached")
    assert host._orbit_mtp_mlp_mapping is _Host._MTP_MLP_MAPPING


def test_qwen3_5_index_resolves_to_upstreams_per_expert_names():
    host = _Host()
    host.safetensor_io = type("IO", (), {"index": ["mtp.layers.0.mlp.experts.0.gate_proj.weight"]})()
    assert host._mtp_experts_fused() is False
    assert host._weight_name_mapping_mtp_mlp(MTP_NAME) == [
        "mtp.layers.0.mlp.experts.3.gate_proj.weight",
        "mtp.layers.0.mlp.experts.3.up_proj.weight",
    ]


def test_qwen3_6_index_resolves_to_the_fused_names_and_caches():
    host = _Host()
    host.safetensor_io = type(
        "IO",
        (),
        {"index": ["mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"]},
    )()
    assert host._mtp_experts_fused() is True
    assert host._mtp_experts_fused_cached is True
    assert host._weight_name_mapping_mtp_mlp(MTP_NAME) == ["mtp.layers.0.mlp.experts.gate_up_proj"]


def test_a_fused_key_outside_the_mtp_block_does_not_trigger_detection():
    """Regular layers are ALWAYS fused; only mtp.layers keys may decide this."""
    host = _Host()
    host.safetensor_io = type(
        "IO", (), {"index": ["model.language_model.layers.0.mlp.experts.gate_up_proj"]}
    )()
    assert host._mtp_experts_fused() is False


def test_non_expert_mtp_names_still_route_through_the_plain_mlp_mapping():
    host = _Host()
    assert host._weight_name_mapping_mtp_mlp("mtp.layers.0.mlp.linear_fc2.weight") == [
        "model.layers.0.mlp.down_proj.weight"
    ]


@pytest.mark.parametrize("name", ("_mtp_experts_fused", "_orbit_mtp_mlp_mapping"))
def test_real_bridge_resolves_to_the_mixin_when_mbridge_is_available(name):
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge

    assert name not in Qwen3_5Bridge.__dict__
    mro = Qwen3_5Bridge.__mro__
    assert mro.index(OrbitQwen35BridgeExtensions) == 1
