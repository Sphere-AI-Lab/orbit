"""Orbit's added ``TITOTokenizer`` finish-reason hook.

Home mixins for the methods lifted out of
miles/utils/chat_template_utils/tito_tokenizer.py: the
``expected_ids_for_finish_reason`` hook (identity default on the base class,
im_end/newline-trimming override on the Qwen3 subclass).

Why the hook exists: a chat template renders a COMPLETED assistant message,
closing control tokens included. A generation cut short by ``max_new_tokens``
never emitted those tokens, so comparing the model's ids against the canonical
render reports a spurious mismatch on the tail. The hook lets a tokenizer family
drop exactly the suffix the model could not have produced, and nothing else.

Both methods are orbit-ADDED -- there is no upstream counterpart, so nothing was
deleted from the vendored class bodies. The vendored classes list these mixins
as their FIRST base:

    class TITOTokenizer(OrbitTITOTokenizerExtensions):
    class Qwen3TITOTokenizer(OrbitQwen3TITOTokenizerExtensions, TITOTokenizer):

which linearizes to Qwen3TITOTokenizer -> OrbitQwen3TITOTokenizerExtensions ->
TITOTokenizer -> OrbitTITOTokenizerExtensions -> object. The Qwen3 override
therefore wins for Qwen3 and ``GLM47TITOTokenizer`` (a plain ``TITOTokenizer``
subclass) still gets the identity default.

Plain mixins: no ``__init__``, no state of their own. The Qwen3 method reads
``self._newline_id`` / ``self._im_end_id``, which the vendored
``Qwen3TITOTokenizer.__init__`` sets, the normal attribute-lookup way.

No ``super()`` call: from the base mixin ``super()`` is ``object``, and the
Qwen3 method is a total replacement of the default rather than an extension of
it, so neither needs one.
"""

from __future__ import annotations


class OrbitTITOTokenizerExtensions:
    def expected_ids_for_finish_reason(
        self,
        expected_ids: list[int],
        finish_reason: str | None,
    ) -> list[int]:
        """Adjust canonical IDs for a model-specific incomplete finish.

        A chat template renders a completed assistant message, including its
        closing control tokens.  A length-truncated generation does not emit
        those tokens.  Most tokenizer families need no adjustment; subclasses
        may remove only the canonical suffix that the model could not have
        produced for the supplied finish reason.
        """
        return list(expected_ids)


class OrbitQwen3TITOTokenizerExtensions:
    def expected_ids_for_finish_reason(
        self,
        expected_ids: list[int],
        finish_reason: str | None,
    ) -> list[int]:
        """Drop the ``<|im_end|>``(``\\n``) suffix a length-truncated Qwen3
        generation could not have emitted.

        Only for ``finish_reason == "length"``: every other finish reason means
        the model did reach its stop token, so the canonical ids stand.
        """
        ids = list(expected_ids)
        if finish_reason != "length":
            return ids

        boundary = len(ids)
        if boundary and ids[boundary - 1] == self._newline_id:
            boundary -= 1
        if boundary and ids[boundary - 1] == self._im_end_id:
            return ids[: boundary - 1]
        return ids


__all__ = ["OrbitQwen3TITOTokenizerExtensions", "OrbitTITOTokenizerExtensions"]
