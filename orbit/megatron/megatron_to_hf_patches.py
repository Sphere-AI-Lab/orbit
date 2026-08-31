"""Orbit's two additions to the ``megatron_to_hf`` PACKAGE itself.

The per-model converters are patched in orbit/megatron/hf_export_patches.py.
These two live in ``megatron_to_hf/__init__.py``, the module that re-exports
those converters, and both are DELEGATING patches that adapt an ARGUMENT on the
way in -- which is exactly why neither of them disturbs the converter patches:
upstream's own dispatch still runs, so it still reaches the package bindings that
orbit/patch/runtime.py::_repoint_reexports keeps pointed at the patched
converters. Copying the dispatch chain into orbit is what would have broken them.

* ``postprocess_hf_param`` -- newer megatron-bridge hands back an
  ``HFWeightTuple`` with no megatron-side name, so the megatron name arrives
  empty and upstream's ``remove_padding`` cannot recognize an embedding or
  output layer to strip vocab padding from. The HF name identifies those same
  layers, so orbit substitutes it when the megatron name is missing. Upstream
  still owns everything ``postprocess_hf_param`` does.

* ``_convert_to_hf_core`` -- Qwen3.6 shares Qwen3.5's parameter layout and
  miles_plugins/mbridge/qwen3_5.py registers both families on one bridge, but
  upstream's dispatch has no ``qwen3_6`` case. It cannot simply fall through
  either: ``"qwen3" in model_name`` is a LATER branch, so an unaliased
  ``qwen3_6`` would silently take the Qwen2 converter instead of raising.
  Orbit therefore renames the family at the boundary. ``model_name`` is read
  nowhere else in upstream's body except the "unsupported model" message, which
  an aliased name never reaches.

Nothing here imports torch or miles at module scope: ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_MEGATRON_TO_HF = "miles.backends.megatron_utils.megatron_to_hf"

_POSTPROCESS_REASON = (
    "megatron-bridge's HFWeightTuple no longer carries the megatron parameter "
    "name, and remove_padding needs a name to recognize the embedding/output "
    "layers it strips vocab padding from; the HF name identifies the same "
    "layers, and upstream has no fallback"
)

_DISPATCH_REASON = (
    "Qwen3.6 shares Qwen3.5's parameter layout (miles_plugins/mbridge/qwen3_5.py "
    "registers both), but upstream's dispatch has no qwen3_6 case and would fall "
    "through to the later 'qwen3' branch -- silently converting it as Qwen2 "
    "rather than raising"
)

_QWEN3_6 = "qwen3_6"
_QWEN3_5 = "qwen3_5"


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf",
    "postprocess_hf_param",
    upstream_sha="87d97432355ee8f4ac07a96fea0416807349e1081945eb3473aa8b542960e8fd",
    reason=_POSTPROCESS_REASON,
)
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    """Upstream's post-processing, with the HF name standing in for a missing
    megatron name."""
    return original(_MEGATRON_TO_HF, "postprocess_hf_param")(
        args, megatron_param_name or hf_param_name, hf_param_name, param
    )


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf",
    "_convert_to_hf_core",
    upstream_sha="d362b5ee77b6b89ddd0f718d955dbac7b6365d73fdb83585c3248047490d5bc4",
    reason=_DISPATCH_REASON,
)
def _convert_to_hf_core(args, model_name, name, param):
    """Upstream's dispatch, with qwen3_6 aliased onto the qwen3_5 converter.

    Aliasing the name rather than calling the converter directly keeps orbit out
    of the dispatch business entirely: upstream picks the converter, through the
    package binding, so the patched Qwen3.5 converter is what actually runs.
    """
    if _QWEN3_6 in model_name and _QWEN3_5 not in model_name:
        model_name = model_name.replace(_QWEN3_6, _QWEN3_5)
    return original(_MEGATRON_TO_HF, "_convert_to_hf_core")(args, model_name, name, param)
