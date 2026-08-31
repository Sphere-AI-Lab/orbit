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


def test_every_entrypoint_that_reaches_a_patched_module_arms_the_patches():
    """The failure this catches actually happened.

    Patches install through a hook that ``import orbit`` arms, so a standalone
    script that imports a patched module WITHOUT importing orbit silently runs
    upstream's unpatched function -- no error, just orbit's mappings missing.
    Both checkpoint->HF converter CLIs did exactly that, and no test noticed
    because every test imports orbit itself.

    Rule: a file with a ``__main__`` block that imports a patched module (or the
    package that pulls it in) must also import orbit.
    """
    import ast
    import subprocess
    from pathlib import Path

    from orbit.patch import registry

    repo = Path(__file__).resolve().parents[2]
    # A module counts as "reaching" a patch target if it imports the target or
    # the package whose __init__ imports it.
    reaching = set()
    for patch in registry():
        reaching.add(patch.module)
        reaching.add(patch.module.rsplit(".", 1)[0])

    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "*.py"], capture_output=True, text=True
    ).stdout.split()

    offenders = []
    for rel in tracked:
        if rel.startswith(("miles/", "miles_plugins/", "orbit/", "tests/")):
            continue  # library code and tests run inside processes that import orbit
        path = repo / rel
        try:
            tree = ast.parse(path.read_text(errors="surrogateescape"))
        except (OSError, SyntaxError):
            continue
        src = path.read_text(errors="surrogateescape")
        if '__name__ == "__main__"' not in src and "__name__ == '__main__'" not in src:
            continue
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                # `from miles.backends.megatron_utils import megatron_to_hf` has
                # node.module == the PARENT of what it actually binds. Without
                # this the guard never even considered the Ray converter CLI --
                # it was structurally blind to the exact file it was written for.
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        if not (imported & reaching):
            continue
        if not any(m == "orbit" or m.startswith("orbit.") for m in imported):
            offenders.append(rel)

    assert not offenders, (
        "entrypoint reaches a patched module without arming the patches "
        "(add `import orbit`):\n  " + "\n  ".join(offenders)
    )


def test_patch_survives_the_converter_package_being_imported_before_orbit():
    """Import ORDER must not decide whether the patch takes effect.

    Patching `qwen2.convert_qwen2_to_hf` is not enough: the package
    `megatron_to_hf/__init__.py` does `from .qwen2 import convert_qwen2_to_hf`
    at import time, and `_convert_to_hf_core` dispatches through THAT binding.
    If the package was already imported when the hook is armed, the re-export is
    stale and callers get upstream's unpatched function while the patch looks
    installed.

    Not hypothetical: a Ray actor imports the converter package while unpickling
    its worker class, before the actor body runs `import orbit`. Run in a
    subprocess so the import order is real rather than simulated.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    probe = (
        "import miles.backends.megatron_utils.megatron_to_hf as pkg\n"   # BEFORE orbit
        "import orbit\n"
        "print(pkg.convert_qwen2_to_hf.__module__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo,
        capture_output=True,
        text=True,
        # Inherit the environment: torch/sglang need the CUDA and loader paths the
        # activated env sets. Only the two knobs this probe cares about are forced.
        env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("orbit.megatron.hf_export_patches"), (
        f"package-level re-export was not re-pointed: {out.stdout.strip()!r}"
    )
