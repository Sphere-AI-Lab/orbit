"""``HfWeightIteratorBase.__init__`` after the move into orbit's home mixin.

Upstream stores a boolean ``self.is_lora``, which cannot express OFT. Orbit's
constructor (orbit/megatron/hf_weight_iterator_ext.py) stores a three-state
``self.peft_method`` and keeps ``is_lora`` as a derived alias;
hf_weight_iterator_bridge.py reads ``self.peft_method`` to choose the LoRA / OFT
/ full-weight export branch.

Two failure modes are pinned here.

1. **Shadowing.** If ``HfWeightIteratorBase`` re-grew an ``__init__`` of its own
   it would beat the mixin in attribute lookup, ``self.peft_method`` would never
   be set, and the first OFT export would die on AttributeError -- at weight-sync
   time, on a GPU, far from the cause.

2. **The pristine ``create``.** ``create`` is 40% orbit, so it was NOT moved --
   and it did not even need a seam: base's signature is
   ``create(args, model, *, is_lora=False, **kwargs)``, so a ``peft_method=``
   argument rides through ``**kwargs`` and arrives alongside base's
   unconditional ``is_lora=False``. That resolves correctly ONLY because the
   constructor lets an explicit ``peft_method`` win over ``is_lora``. Change
   that precedence and every OFT run silently exports as full weights, so the
   whole ``create``-shaped argument matrix is asserted below.
"""

import pytest

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase
from orbit.megatron.hf_weight_iterator_ext import OrbitHfWeightIteratorExtensions


class _Iterator(HfWeightIteratorBase):
    """Concrete subclass; forwards like the real Direct/Bridge iterators do."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_hf_weight_chunks(self, megatron_local_weights):  # pragma: no cover - unused
        return []


def _build(**kwargs):
    """What base's pristine ``create`` passes on: is_lora is ALWAYS forwarded."""
    return _Iterator("ARGS", "MODEL", model_name="m", quantization_config=None, is_lora=False, **kwargs)


def test_init_resolves_to_the_mixin_and_is_not_shadowed():
    assert "__init__" not in HfWeightIteratorBase.__dict__, (
        "HfWeightIteratorBase defines __init__ in its own body; that copy shadows "
        "the mixin and self.peft_method is never set"
    )
    assert HfWeightIteratorBase.__init__.__qualname__ == "OrbitHfWeightIteratorExtensions.__init__"
    assert HfWeightIteratorBase.__mro__[:2] == (HfWeightIteratorBase, OrbitHfWeightIteratorExtensions)


def test_create_stayed_in_the_vendored_class():
    """40% orbit ownership: left upstream's, and pristine."""
    assert "create" in HfWeightIteratorBase.__dict__
    assert "peft_method" not in HfWeightIteratorBase.create.__code__.co_varnames


@pytest.mark.parametrize(
    ("kwargs", "peft_method", "is_lora"),
    (
        ({}, "none", False),
        ({"peft_method": "none"}, "none", False),
        ({"peft_method": "lora"}, "lora", True),
        ({"peft_method": "oft"}, "oft", False),
    ),
)
def test_explicit_peft_method_wins_over_creates_is_lora_default(kwargs, peft_method, is_lora):
    iterator = _build(**kwargs)
    assert iterator.peft_method == peft_method
    assert iterator.is_lora is is_lora


@pytest.mark.parametrize(
    ("is_lora", "peft_method"), ((True, "lora"), (False, "none"), (None, "none"))
)
def test_legacy_is_lora_still_resolves_when_no_peft_method_is_given(is_lora, peft_method):
    iterator = _Iterator("ARGS", "MODEL", model_name="m", quantization_config=None, is_lora=is_lora)
    assert iterator.peft_method == peft_method
    assert iterator.is_lora is (peft_method == "lora")


def test_is_lora_is_derived_not_stored_independently():
    """A caller must never be able to produce peft_method="oft" with is_lora
    True: the bridge exporter would then take the LoRA branch for OFT weights."""
    iterator = _Iterator(
        "ARGS", "MODEL", model_name="m", quantization_config=None, peft_method="oft", is_lora=True
    )
    assert (iterator.peft_method, iterator.is_lora) == ("oft", False)


def test_upstream_attributes_are_still_set():
    iterator = _build()
    assert (iterator.args, iterator.model, iterator.model_name, iterator.quantization_config) == (
        "ARGS",
        "MODEL",
        "m",
        None,
    )
