"""Orbit's two patches on the ``megatron_to_hf`` PACKAGE itself.

Both adapt an argument on the way IN, which is the reason they compose with
orbit/megatron/hf_export_patches.py instead of shadowing it: upstream's own
dispatch still runs, so it still reaches the package bindings that
orbit/patch/runtime.py::_repoint_reexports keeps pointed at the patched
converters. The last test pins exactly that, because the way this would break is
silent -- a copied dispatch chain would keep working while quietly calling the
UNPATCHED converters.
"""

import argparse

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.backends.megatron_utils import megatron_to_hf as m2h


def _args():
    return argparse.Namespace(
        vocab_size=4,
        kv_channels=None,
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        q_lora_rank=None,
    )


_PROJ = "module.module.decoder.layers.3.self_attention.linear_proj.weight"


def test_the_patches_are_actually_installed():
    assert m2h.postprocess_hf_param.__module__ == "orbit.megatron.megatron_to_hf_patches"
    assert m2h._convert_to_hf_core.__module__ == "orbit.megatron.megatron_to_hf_patches"
    assert hasattr(m2h, "_orbit_unpatched_postprocess_hf_param")
    assert hasattr(m2h, "_orbit_unpatched__convert_to_hf_core")


@pytest.mark.parametrize("missing", ["", None])
def test_padding_is_stripped_by_hf_name_when_the_megatron_name_is_missing(missing):
    """What megatron-bridge's HFWeightTuple leaves orbit with."""
    param = torch.arange(8)
    out = m2h.postprocess_hf_param(_args(), missing, "model.embed_tokens.weight", param)
    assert len(out) == 4

    # ...and prove the patch is what did it: upstream keeps the padding.
    upstream = m2h._orbit_unpatched_postprocess_hf_param(
        _args(), missing, "model.embed_tokens.weight", param
    )
    assert len(upstream) == 8


def test_a_present_megatron_name_still_runs_upstreams_body():
    """The delegation property: orbit substitutes a name, nothing more."""
    param = torch.arange(8)
    name = "embedding.word_embeddings.weight"
    patched = m2h.postprocess_hf_param(_args(), name, "model.embed_tokens.weight", param)
    upstream = m2h._orbit_unpatched_postprocess_hf_param(
        _args(), name, "model.embed_tokens.weight", param
    )
    assert torch.equal(patched, upstream)
    assert len(patched) == 4


def test_a_non_vocab_param_is_untouched_either_way():
    param = torch.arange(8)
    assert torch.equal(
        m2h.postprocess_hf_param(_args(), "", "model.layers.0.mlp.down_proj.weight", param),
        param,
    )


def test_qwen3_6_dispatches_to_the_qwen3_5_converter():
    param = torch.ones(8)
    out = m2h._convert_to_hf_core(_args(), "qwen3_6", _PROJ, param)
    assert out[0][0] == "model.language_model.layers.3.self_attn.o_proj.weight"

    # ...and prove the alias is orbit's: upstream falls through to the LATER
    # `"qwen3" in model_name` branch and converts it as Qwen2 -- silently wrong
    # rather than an error, which is why the alias exists.
    upstream = m2h._orbit_unpatched__convert_to_hf_core(_args(), "qwen3_6", _PROJ, param)
    assert upstream[0][0] == "model.layers.3.self_attn.o_proj.weight"


def test_a_known_family_still_runs_upstreams_dispatch():
    """The delegation property for the dispatcher."""
    param = torch.ones(8)
    patched = m2h._convert_to_hf_core(_args(), "qwen2", _PROJ, param)
    upstream = m2h._orbit_unpatched__convert_to_hf_core(_args(), "qwen2", _PROJ, param)
    assert [n for n, _ in patched] == [n for n, _ in upstream]
    assert patched[0][0] == "model.layers.3.self_attn.o_proj.weight"


def test_an_unsupported_model_still_raises():
    with pytest.raises(ValueError, match="Unsupported model"):
        m2h._convert_to_hf_core(_args(), "not-a-model", _PROJ, torch.ones(8))


def test_the_alias_still_reaches_the_PATCHED_qwen3_5_converter():
    """The composition property, and the one that would fail silently.

    orbit/megatron/hf_export_patches.py adds the bare-layernorm names upstream's
    converters reject. Aliasing the model name (rather than calling a converter
    directly) keeps upstream's dispatch in charge, so those names must still
    resolve through a `qwen3_6` model name.
    """
    name = "module.module.decoder.layers.3.input_layernorm.weight"
    out = m2h._convert_to_hf_core(_args(), "qwen3_6", name, torch.ones(8))
    assert out[0][0] == "model.language_model.layers.3.input_layernorm.weight"


def test_patching_the_package_does_not_unseat_the_converter_re_exports():
    """Import ORDER, for the module that re-exports the patched converters.

    The package's own two functions are patched directly, so they are safe. What
    is NOT automatic is that a package imported BEFORE the hook is armed still
    hands out the patched CONVERTERS: those are re-exports, and
    orbit/patch/runtime.py::_repoint_reexports is what fixes them. Patching the
    package itself must not disturb that. Run in a subprocess so the import
    order is real rather than simulated.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    probe = (
        "import miles.backends.megatron_utils.megatron_to_hf as pkg\n"  # BEFORE orbit
        "import orbit\n"
        "print(pkg.postprocess_hf_param.__module__)\n"
        "print(pkg._convert_to_hf_core.__module__)\n"
        "print(pkg.convert_qwen3_5_to_hf.__module__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    package, dispatch, converter = out.stdout.strip().splitlines()[-3:]
    assert package == "orbit.megatron.megatron_to_hf_patches"
    assert dispatch == "orbit.megatron.megatron_to_hf_patches"
    assert converter == "orbit.megatron.hf_export_patches"
