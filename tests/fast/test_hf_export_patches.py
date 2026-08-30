"""Orbit's Megatron->HF converter patches add names without shadowing upstream.

These mappings used to be `elif` branches edited into seven vendored converters.
They are now delegating patches, so the vendored files are byte-pristine. The
properties that matter are that orbit's names work, that upstream's names still
go through UPSTREAM's body (not a copy), and that a genuinely unknown name still
raises -- a patch that swallowed the error would turn a loud export failure into
a silently missing tensor.
"""

import argparse

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.backends.megatron_utils.megatron_to_hf import qwen2, qwen3moe


def _args():
    return argparse.Namespace(
        kv_channels=None, hidden_size=8, num_attention_heads=4, num_query_groups=2
    )


def test_the_patch_is_actually_installed():
    assert qwen2.convert_qwen2_to_hf.__module__ == "orbit.megatron.hf_export_patches"
    assert hasattr(qwen2, "_orbit_unpatched_convert_qwen2_to_hf"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_orbit_maps_the_bare_layernorm_names_upstream_rejects():
    name = "module.module.decoder.layers.3.input_layernorm.weight"
    out = qwen2.convert_qwen2_to_hf(_args(), name, torch.ones(8))
    assert out[0][0] == "model.layers.3.input_layernorm.weight"

    # ...and prove the patch is what did it: upstream alone cannot.
    with pytest.raises(ValueError, match="Unknown parameter name"):
        qwen2._orbit_unpatched_convert_qwen2_to_hf(_args(), name, torch.ones(8))


def test_upstream_names_still_run_upstreams_body():
    """The delegation property. If this ever fails because orbit copied the body
    instead, upstream's fixes to that converter stop reaching us."""
    name = "module.module.decoder.layers.3.self_attention.linear_proj.weight"
    param = torch.ones(8)
    patched = qwen2.convert_qwen2_to_hf(_args(), name, param)
    upstream = qwen2._orbit_unpatched_convert_qwen2_to_hf(_args(), name, param)
    assert patched == upstream


def test_qwen3moe_maps_grouped_expert_fc1_oft_adapters():
    name = "module.module.decoder.layers.2.mlp.experts.linear_fc1.adapter_gate.5.oft_r"
    out = qwen3moe.convert_qwen3moe_to_hf(_args(), name, torch.ones(8))
    assert out[0][0] == "model.layers.2.mlp.experts.5.gate_proj.oft_R"


def test_an_unknown_name_still_raises():
    with pytest.raises(ValueError, match="Unknown parameter name"):
        qwen2.convert_qwen2_to_hf(
            _args(), "module.module.decoder.layers.3.nonsense.weight", torch.ones(8)
        )


def test_patched_modules_are_leaves_so_import_time_apply_cannot_cycle():
    """orbit/__init__.py applies patches on import, which is only safe while no
    patched module reaches back into orbit. Enforce that invariant rather than
    trusting it: a patch on a module that imports orbit needs a deferred apply.
    """
    import ast
    from pathlib import Path

    from orbit.patch import registry

    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for patch in registry():
        path = repo / (patch.module.replace(".", "/") + ".py")
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(errors="surrogateescape"))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = node.names[0].name
            if mod and mod.split(".")[0] == "orbit":
                offenders.append(f"{patch.module} imports {mod}")
    assert not offenders, "patched module is not a leaf:\n  " + "\n  ".join(offenders)
