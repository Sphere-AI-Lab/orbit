"""Orbit's ``HfWeightIteratorBase.__init__``.

Home mixin for the constructor rewritten out of
miles/backends/megatron_utils/update_weight/hf_weight_iterator_base.py.
``HfWeightIteratorBase`` in the miles file lists this mixin as its first base:

    class HfWeightIteratorBase(OrbitHfWeightIteratorExtensions, ABC):

What changed and why: base stores a boolean ``self.is_lora``, which cannot
express OFT. Orbit's export paths need three states (``"none" | "lora" |
"oft"``), so the constructor stores ``self.peft_method`` and keeps
``self.is_lora`` as a derived compat alias. ``hf_weight_iterator_bridge.py``
reads ``self.peft_method`` to pick the LoRA / OFT / full-weight export branch.

``is_lora`` survives as an accepted KEYWORD too, so a caller written against
base's API keeps working: ``is_lora=True`` resolves to ``peft_method="lora"``
and ``is_lora=False`` to ``"none"``, but ONLY when ``peft_method`` was left at
its default. An explicit ``peft_method`` always wins, which is what makes the
vendored ``HfWeightIteratorBase.create`` byte-pristine: base's ``create`` has no
``peft_method`` parameter, so a ``peft_method=`` argument rides through its
``**kwargs`` and lands here alongside base's unconditional ``is_lora=False`` --
and the "explicit peft_method wins" rule is exactly what makes that pair resolve
correctly. Both are covered by tests/fast/test_hf_weight_iterator_peft_method.py.

The default is ``is_lora=None``, not base's ``False``, so "caller said nothing"
is distinguishable from "caller said not-LoRA"; the two happen to resolve to the
same ``peft_method``, but only the None default lets a future third state be
added without silently overriding it.

MRO note: this is the ONLY ``__init__`` in the chain --
``HfWeightIteratorBase`` deliberately keeps none of its own, or its own
``__dict__`` would shadow this one and ``self.peft_method`` would never be set
(``hf_weight_iterator_bridge.py`` would then raise AttributeError on the first
export). ``HfWeightIteratorDirect`` and ``HfWeightIteratorBridge`` both forward
``*args, **kwargs`` through ``super().__init__``, so they reach this unchanged.

Plain mixin: no state of its own beyond what it sets on ``self``, and no
``super().__init__`` call -- from here ``super()`` is ``ABC``/``object``, which
is what base's constructor also implicitly ended at.
"""

from __future__ import annotations


class OrbitHfWeightIteratorExtensions:
    def __init__(self, args, model, model_name, quantization_config, *, peft_method="none", is_lora=None):
        if is_lora is not None and peft_method == "none":
            peft_method = "lora" if is_lora else "none"

        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.peft_method = peft_method
        self.is_lora = peft_method == "lora"


__all__ = ["OrbitHfWeightIteratorExtensions"]
