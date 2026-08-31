"""Orbit's additions to miles' Megatron->HF parameter-name converters.

Orbit needs two things upstream's converters do not emit:

* the BARE ``input_layernorm.weight`` / ``pre_mlp_layernorm.weight`` names that
  non-fused-QKV layouts produce (upstream only maps the fused
  ``self_attention.linear_qkv.layer_norm_weight`` / ``mlp.linear_fc1.
  layer_norm_weight`` spellings), and
* for Qwen3-MoE, the grouped-expert FC1 OFT adapters, which have no upstream
  spelling at all.

These used to be extra ``elif`` branches edited into seven vendored files. They
are now DELEGATING patches, which is why the vendored converters are pristine
again: each replacement calls upstream's function first and only handles the
names upstream rejects.

There are SIX patches here, not seven, and the missing one is the point of the
mechanism rather than an oversight: miles now maps the bare
``input_layernorm.weight`` and ``pre_mlp_layernorm.weight`` names in
``qwen2.py`` itself, so orbit's branch there would never be reached. Upstream
absorbed the change; orbit drops it and reuses upstream. That is the "no
conflict, no stale duplicate" outcome promised below, observed. Two consequences worth stating, because they are the
whole reason to prefer this over copying the bodies:

* orbit carries ~10 lines per converter instead of the 52-165 line upstream
  body, so upstream's own fixes to those bodies keep running;
* if upstream ever adds these mappings itself, orbit's fallback simply stops
  being reached -- no conflict, no stale duplicate.

The upstream fallthrough is ``raise ValueError(f"Unknown parameter name: ...")``,
so "upstream rejected it" is detectable precisely. The message prefix is matched
rather than the bare exception type, so a genuine ValueError from inside a
tensor reshape still propagates instead of being mistaken for a miss.
"""

from __future__ import annotations

import re

from orbit.patch import original, patch_function

_MEGATRON_TO_HF = "miles.backends.megatron_utils.megatron_to_hf"
_LAYER_RE = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")
_UNKNOWN = "Unknown parameter name:"

# rest-name -> HF suffix, for the layouts upstream's fused spellings miss.
_BARE_NORMS = {
    "input_layernorm.weight": "input_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
}

_EXPERT_FC1_OFT = re.compile(r"mlp\.experts\.linear_fc1\.adapter_(gate|up)\.(\d+)\.oft_r")

_REASON = (
    "non-fused-QKV layouts emit bare input_layernorm/pre_mlp_layernorm names "
    "that upstream's converter does not map"
)


def _upstream_missed(exc: ValueError) -> bool:
    return str(exc).startswith(_UNKNOWN)


def _bare_norm(name, param, layer_prefix: str):
    """Map the bare layernorm spellings, or None when this is not one."""
    match = _LAYER_RE.match(name)
    if not match:
        return None
    layer_idx, rest = match.groups()
    suffix = _BARE_NORMS.get(rest)
    if suffix is None:
        return None
    return [(f"{layer_prefix.format(layer_idx=layer_idx)}.{suffix}", param)]


def _delegate(module: str, attr: str, args, name, param, layer_prefix: str, extra=None):
    """Upstream first; orbit handles only what upstream rejects."""
    try:
        return original(module, attr)(args, name, param)
    except ValueError as exc:
        if not _upstream_missed(exc):
            raise
    if extra is not None:
        hit = extra(name, param)
        if hit is not None:
            return hit
    hit = _bare_norm(name, param, layer_prefix)
    if hit is not None:
        return hit
    raise ValueError(f"{_UNKNOWN} {name}")


def _expert_fc1_oft(name, param):
    """Qwen3-MoE grouped-expert FC1 OFT adapters -> per-expert HF ``oft_R``."""
    match = _LAYER_RE.match(name)
    if not match:
        return None
    layer_idx, rest = match.groups()
    hit = _EXPERT_FC1_OFT.match(rest)
    if hit is None:
        return None
    proj, expert_idx = hit.groups()
    hf_proj = "gate_proj" if proj == "gate" else "up_proj"
    return [
        (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{hf_proj}.oft_R", param)
    ]


_STD = "model.layers.{layer_idx}"


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf.glm4", "convert_glm4_to_hf", upstream_sha="0563ed7909be4bbba7d0ce136024db3e459f56c80c3b0b5685ecece724e4561d", reason=_REASON)
def convert_glm4_to_hf(args, name, param):
    return _delegate(
        "miles.backends.megatron_utils.megatron_to_hf.glm4", "convert_glm4_to_hf", args, name, param, _STD)


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf.glm4moe", "convert_glm4moe_to_hf", upstream_sha="71b13bdd16cec55d2d3c63893569ba4568d772b4d6ad9e8c530a9ff88ed5b3ec", reason=_REASON)
def convert_glm4moe_to_hf(args, name, param):
    return _delegate(
        "miles.backends.megatron_utils.megatron_to_hf.glm4moe", "convert_glm4moe_to_hf", args, name, param, _STD)


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf.llama", "convert_llama_to_hf", upstream_sha="ce3166fc6dfc1d1786d8ae81d6d5573f0d51800213fda194190b381b7409916e", reason=_REASON)
def convert_llama_to_hf(args, name, param):
    return _delegate(
        "miles.backends.megatron_utils.megatron_to_hf.llama", "convert_llama_to_hf", args, name, param, _STD)


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf.qwen3_next", "convert_qwen3_next_to_hf", upstream_sha="8a88f826fa9415ab4e5a1ea420729146ca9c2a997d675e304bbdc7bd85b85126", reason=_REASON)
def convert_qwen3_next_to_hf(args, name, param):
    return _delegate(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3_next", "convert_qwen3_next_to_hf", args, name, param, _STD
    )


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf.qwen3_5", "convert_qwen3_5_to_hf", upstream_sha="7cb22ed152d0246cc154a723e14bcc8243a2b93d689dd59c4ec3576b89484569", reason=_REASON)
def convert_qwen3_5_to_hf(args, name, param):
    # Qwen3.5 nests the decoder under `model.language_model`.
    return _delegate(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3_5",
        "convert_qwen3_5_to_hf",
        args,
        name,
        param,
        "model.language_model.layers.{layer_idx}",
    )


@patch_function(
    "miles.backends.megatron_utils.megatron_to_hf.qwen3moe",
    "convert_qwen3moe_to_hf",
    upstream_sha="822050b8d464a892b26df9defd53ac491b7ab1e305d2d647e8685a799a7af44c",
    reason=_REASON + "; plus grouped-expert FC1 OFT adapters, which upstream has no spelling for",
)
def convert_qwen3moe_to_hf(args, name, param):
    return _delegate(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3moe",
        "convert_qwen3moe_to_hf",
        args,
        name,
        param,
        _STD,
        extra=_expert_fc1_oft,
    )
