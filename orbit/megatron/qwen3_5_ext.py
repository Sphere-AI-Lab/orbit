"""Orbit's Qwen3.6 support on top of miles' Qwen3.5 mbridge plugin.

Home mixin for the methods lifted out of miles_plugins/mbridge/qwen3_5.py.
``Qwen3_5Bridge`` in the miles file lists this mixin as its first base:

    class Qwen3_5Bridge(OrbitQwen35BridgeExtensions, Qwen2MoEBridge):

Qwen3.6-35B-A3B reuses the ``qwen3_5_moe`` HF config schema, so the vendored
bridge covers it once it is registered for those model types (an ORBIT-SEAM on
the ``@register_model`` decorator, which is the one thing that cannot be
expressed from out here -- the decorator runs at class creation). One
STRUCTURAL difference remains:

* Qwen3.5-35B-A3B ships the MTP layer's experts as individual per-expert
  tensors -- ``mtp.layers.{n}.mlp.experts.{id}.gate_proj.weight`` and friends.
* Qwen3.6-35B-A3B packs them the way the regular layers already are: fused 3-D
  ``gate_up_proj`` / ``down_proj`` tensors.

Upstream hard-codes the unfused shape in ``Qwen3_5Bridge._MTP_MLP_MAPPING``.
Rather than fork that dict, orbit keeps it byte-pristine (it IS the unfused
variant) and adds the fused counterpart plus a detector here;
``_orbit_mtp_mlp_mapping`` returns whichever matches the checkpoint actually
being loaded. The vendored ``_weight_name_mapping_mtp_mlp`` -- 6% orbit, so
deliberately NOT moved -- reads that property instead of the raw dict, which is
its only orbit-owned line besides a corrected docstring. Upstream's mapping
logic keeps running, and an upstream fix to it keeps flowing.

Why the property is named ``_orbit_mtp_mlp_mapping`` and not
``_MTP_MLP_MAPPING``: a mixin CANNOT shadow an attribute the vendored class
defines in its own body. Python resolves the class's ``__dict__`` before any
base, so a property named ``_MTP_MLP_MAPPING`` here would lose to upstream's
dict of the same name, silently, and every Qwen3.6 MTP expert would be looked up
under Qwen3.5's per-expert names -- a load failure far from its cause. The
distinct name makes the vendored dict a plain input to this property instead of
a competitor.

Detection reads ``safetensor_io.index``, which is only populated once the bridge
has been constructed against a real checkpoint. The result is cached ONLY once
``safetensor_io`` exists, so a pre-init probe (tests build the bridge with
``__new__``) returns False without freezing that answer for the real load.

Plain mixin: no ``__init__``, no ``super()`` call (from here ``super()`` is
``Qwen2MoEBridge``, mbridge's own base -- deliberately not used: nothing here
overrides an mbridge method).
"""

from __future__ import annotations


class OrbitQwen35BridgeExtensions:
    # Fused counterpart of the vendored (pristine) Qwen3_5Bridge._MTP_MLP_MAPPING,
    # which is the unfused per-expert variant. Qwen3.6 packs MTP experts fused.
    _MTP_MLP_MAPPING_FUSED = {
        "mlp.experts.linear_fc1": ["mtp.layers.{layer_number}.mlp.experts.gate_up_proj"],
        "mlp.experts.linear_fc2": ["mtp.layers.{layer_number}.mlp.experts.down_proj"],
    }

    def _mtp_experts_fused(self) -> bool:
        """Detect whether MTP expert weights are stored in fused 3-D tensors.

        Qwen3.5 MoE-A3B: unfused per-expert tensors (keys end in ``.weight``).
        Qwen3.6 MoE-A3B: fused ``gate_up_proj`` / ``down_proj`` tensors.
        Resolved from ``safetensor_io.index`` on first call; result is only
        cached once ``safetensor_io`` is available, so early pre-init access
        (e.g. from tests that instantiate via ``__new__``) does not lock in
        a wrong answer.
        """
        cached = getattr(self, "_mtp_experts_fused_cached", None)
        if cached is not None:
            return cached
        io = getattr(self, "safetensor_io", None)
        index = getattr(io, "index", None) if io is not None else None
        if not index:
            return False
        fused = any(
            "mtp.layers." in k and "mlp.experts." in k and (k.endswith("gate_up_proj") or k.endswith("down_proj"))
            for k in index
        )
        self._mtp_experts_fused_cached = fused
        return fused

    @property
    def _orbit_mtp_mlp_mapping(self):
        """The MTP expert mapping matching this checkpoint's packing.

        ``self._MTP_MLP_MAPPING`` is the vendored class's pristine dict, i.e.
        Qwen3.5's unfused per-expert names.
        """
        return self._MTP_MLP_MAPPING_FUSED if self._mtp_experts_fused() else self._MTP_MLP_MAPPING


__all__ = ["OrbitQwen35BridgeExtensions"]
