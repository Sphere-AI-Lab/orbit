"""``expected_ids_for_finish_reason`` after the move into orbit's home mixins.

The hook is orbit-added (miles has no counterpart) and lives in
orbit/utils/chat_template_utils/tito_ext.py. Two things can silently break it:

1. The vendored class re-grows a body of its own. Python resolves a class's
   ``__dict__`` before any base, so a method left in ``TITOTokenizer`` /
   ``Qwen3TITOTokenizer`` shadows the mixin with NO error -- the override just
   stops happening.
2. The mixin bases get listed in the wrong order, which would give the Qwen3
   subclass the identity default and leave the trailing ``<|im_end|>`` in the
   canonical ids of every truncated generation.

Both are asserted here, alongside the actual trimming behaviour.
"""

import pytest

from miles.utils.chat_template_utils.tito_tokenizer import (
    GLM47TITOTokenizer,
    Qwen3TITOTokenizer,
    TITOTokenizer,
)
from orbit.utils.chat_template_utils.tito_ext import (
    OrbitQwen3TITOTokenizerExtensions,
    OrbitTITOTokenizerExtensions,
)

HOOK = "expected_ids_for_finish_reason"


@pytest.mark.parametrize(
    ("cls", "owner"),
    (
        (TITOTokenizer, OrbitTITOTokenizerExtensions),
        (Qwen3TITOTokenizer, OrbitQwen3TITOTokenizerExtensions),
        (GLM47TITOTokenizer, OrbitTITOTokenizerExtensions),
    ),
)
def test_hook_resolves_to_the_mixin_and_is_not_shadowed(cls, owner):
    assert HOOK not in cls.__dict__, (
        f"{cls.__name__} defines {HOOK} in its own body; a class's __dict__ beats "
        f"every base, so that copy silently shadows the mixin and orbit's override "
        f"stops running"
    )
    assert getattr(cls, HOOK).__qualname__ == f"{owner.__name__}.{HOOK}"


def test_mixins_precede_the_vendored_classes_in_the_mro():
    mro = Qwen3TITOTokenizer.__mro__
    assert mro.index(OrbitQwen3TITOTokenizerExtensions) < mro.index(TITOTokenizer), (
        "the Qwen3 mixin must precede TITOTokenizer, or Qwen3 gets the identity default"
    )
    assert mro.index(TITOTokenizer) < mro.index(OrbitTITOTokenizerExtensions)


def _qwen3(newline_id: int = 198, im_end_id: int = 151645) -> Qwen3TITOTokenizer:
    """A Qwen3 tokenizer without ``__init__`` -- the hook reads only these two ids."""
    tokenizer = Qwen3TITOTokenizer.__new__(Qwen3TITOTokenizer)
    tokenizer._newline_id = newline_id
    tokenizer._im_end_id = im_end_id
    return tokenizer


def test_base_hook_is_identity_but_returns_a_copy():
    ids = [1, 2, 3]
    out = TITOTokenizer.__new__(TITOTokenizer).expected_ids_for_finish_reason(ids, "length")
    assert out == ids
    assert out is not ids, "callers must not be handed the caller's own list back"


@pytest.mark.parametrize("finish_reason", (None, "stop", "abort", "tool_calls"))
def test_qwen3_keeps_the_canonical_ids_unless_truncated(finish_reason):
    """Any finish reason but "length" means the model did emit its stop token."""
    ids = [5, 6, 151645, 198]
    assert _qwen3().expected_ids_for_finish_reason(ids, finish_reason) == ids


def test_qwen3_drops_im_end_and_its_trailing_newline_on_length():
    assert _qwen3().expected_ids_for_finish_reason([5, 6, 151645, 198], "length") == [5, 6]


def test_qwen3_drops_a_bare_im_end_with_no_trailing_newline():
    assert _qwen3().expected_ids_for_finish_reason([5, 6, 151645], "length") == [5, 6]


def test_qwen3_leaves_a_trailing_newline_that_is_not_preceded_by_im_end():
    """Only the closing control token is removed; a newline the model really
    could have emitted must survive."""
    assert _qwen3().expected_ids_for_finish_reason([5, 6, 198], "length") == [5, 6, 198]


@pytest.mark.parametrize("ids", ([], [151645], [198]))
def test_qwen3_handles_degenerate_sequences(ids):
    out = _qwen3().expected_ids_for_finish_reason(ids, "length")
    assert out == ([] if ids in ([], [151645]) else [198])
