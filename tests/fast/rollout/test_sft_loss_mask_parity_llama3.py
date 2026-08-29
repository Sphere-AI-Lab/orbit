"""Gate G3-llama: Orbit's Llama-3 loss mask must equal the HF oracle's label mask.

Scoring a different token set than an HF SFT recipe shifts every NLL in the study
by a constant -- invisible in the shape of a loss curve, fatal to any comparison.

The oracle below is deliberately ALGORITHM-INDEPENDENT from the implementation.

    implementation (``gen_multi_turn_loss_mask_llama3``): tokenizes the whole
        conversation, then finds each assistant turn by scanning the TOKEN id
        stream for the ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``
        id subsequence and walking forward to the ``<|eot_id|>`` id.

    oracle (``_hf_label_mask`` below): renders the conversation to TEXT, finds
        each assistant turn by ``str.index`` over that text, giving CHARACTER
        offsets, and only then maps characters to tokens via the fast
        tokenizer's ``return_offsets_mapping``. It never looks at a token id.

A bug in the token SCAN -- an off-by-one on the header length, an off-by-one on the
``<|eot_id|>`` boundary, a missed or spurious occurrence -- cannot hide in the char
scan, because the char scan has no token scan to be wrong in the same way. Both
mutation proofs in the task report exercise exactly this class and both are caught.
The Qwen3 sibling gate (``test_sft_loss_mask_parity.py``) shipped an oracle that
reused the implementation's own ``<|im_end|>`` token scan; that was recorded as a
known weakness and is deliberately not repeated here.

Exactly where that independence STOPS
-------------------------------------
The independence is in the ALGORITHM (character offsets vs token scan), not in the
inputs. The two sides share three things:

- the tokenizer and the chat template -- these are the *spec*, and a gate that used
  a different one would be answering a different question;
- the header LITERAL ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``;
- the terminator LITERAL ``<|eot_id|>``.

So a wrong *literal* is a shared-input bug, and the parity mechanism is blind to it
by construction: both sides would locate the same wrong span by different means and
agree. This was demonstrated, not merely reasoned about. Dropping the trailing
``\\n\\n`` from the header literal on BOTH sides leaves ``test_token_ids_match``,
``test_scored_token_count_matches``, ``test_scored_token_positions_match`` and
``test_oracle_scores_a_nonempty_span_per_assistant_turn`` all passing.

The only thing that catches it is
``test_scored_text_is_exactly_the_assistant_turns``, which compares the DECODED
scored text against ``message["content"].strip() + EOT`` built from the fixture's own
``content`` -- independent of the HEADER literal only. It still uses the same
module-level ``EOT = "<|eot_id|>"`` constant the oracle's span search uses (see
``_hf_label_mask`` below), so it is NOT independent of the terminator literal: a wrong
``EOT`` would still make ``expected`` and the oracle's span agree on the same wrong
text, which is exactly the corresponding hole described next.
**That test carries the shared-HEADER-literal guarantee. Do not delete it as
redundant with the parity tests; it is the only assertion here that is not
downstream of the header literal above.**

The corresponding hole that nothing in this gate closes: if a turn ended with a
terminator other than ``<|eot_id|>``, implementation and oracle would produce
BYTE-IDENTICAL wrong masks, both over-running into the following turn. The template
does have such a path -- it emits ``<|eom_id|>`` instead of ``<|eot_id|>`` (see
``llama3_chat_template.py``) -- but only for a ``tool_calls`` message AND only when
``builtin_tools`` is defined in the Jinja context. Nothing in Orbit passes
``builtin_tools``; ``tools=`` alone still yields ``<|eot_id|>``. So this is
unreachable today rather than a live hole -- but the gate is structurally incapable
of catching it, so if ``builtin_tools`` ever becomes reachable this file must grow a
terminator-independent check, not just another fixture row.

On ``offset_mapping`` and special tokens
----------------------------------------
The brief's sketch guarded against ``offset_mapping`` reporting an empty ``(0, 0)``
span for added special tokens on some backends. That was MEASURED against this
tokenizer (``PreTrainedTokenizerFast``, tokenizers 0.22.2, transformers 4.57.1) over
all 6278 tokens of all 12 fixture rows: there are ZERO degenerate offsets. Every
added token -- ``<|begin_of_text|>``, ``<|start_header_id|>``, ``<|end_header_id|>``,
``<|eot_id|>`` -- reports its true character span, and ``text[a:b]`` round-trips to
``decode([id])`` for every token. So the guard is not shipped as dead code; instead
``_assert_offsets_are_usable`` ASSERTS the property, so a future backend that starts
returning degenerate offsets fails this gate loudly rather than silently producing a
mask that is wrong at every special token. Falling back to a token scan is not an
option: it would collapse the oracle onto the implementation and destroy the only
property this gate has.

One real subtlety the measurement did surface: offsets can OVERLAP. Llama-3 is a
byte-level BPE, so a multi-byte character (CJK, fixture row 8) is split across
several tokens that each report the *whole character's* char span. Two readings of
"how many", neither more "the" count than the other: **62** tokens in row 8 share an
identical ``(a, b)`` offset pair with at least one other token (the strict reading --
two or more tokens claiming the same span); **76** tokens in row 8 have a *solo*
``tokenizer.decode([id])`` that contains the U+FFFD replacement character, i.e. half of
a multi-byte sequence (the broader reading). That overlap is harmless for
containment-based masking (both halves of the character sit inside the same span) and
is why the mapping below uses an explicit inside/disjoint classification rather than
assuming a clean partition.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests/fast/fixtures/lora_regret/llama3_sample.jsonl"
LLAMA31_8B = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B")

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>"

pytestmark = pytest.mark.skipif(
    not (LLAMA31_8B / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from orbit.peft.utils.llama3_chat_template import ensure_llama3_chat_template

    # Llama-3.1-8B *base* ships no chat_template, so apply_chat_template would raise
    # and MultiTurnLossMaskGenerator could not even be constructed. Must happen before
    # the generator is built, not after.
    tok = AutoTokenizer.from_pretrained(str(LLAMA31_8B))
    ensure_llama3_chat_template(tok)
    return tok


@pytest.fixture(scope="module")
def conversations():
    return [json.loads(line)["prompt"] for line in FIXTURE.read_text().splitlines()]


@pytest.fixture(scope="module")
def generator(tokenizer):
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    return MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="llama3")


def _assert_offsets_are_usable(offsets, ids, text, tokenizer):
    """The oracle's preconditions on the fast backend's offset_mapping.

    Measured true for this tokenizer; asserted rather than assumed so that a backend
    change degrades into a loud failure instead of a silently wrong mask.
    """
    covered_to = 0
    for i, (a, b) in enumerate(offsets):
        assert a < b, (
            f"offset_mapping returned a degenerate span {(a, b)} for token {i} "
            f"({tokenizer.convert_ids_to_tokens(ids[i])!r}). The char-offset oracle "
            "cannot place a zero-width token; resolve this within the char-offset "
            "approach -- do NOT fall back to a token scan, which would make this "
            "gate a copy of the implementation it tests."
        )
        # No gap: every character belongs to at least one token. Overlap IS allowed --
        # a multi-byte character split across several byte-level BPE tokens has each
        # of them report the whole character's span: 62 tokens in fixture row 8 share an
        # identical (a, b) offset pair with another token (the strict "overlapping"
        # reading), or 76 by the broader reading -- any token whose solo decode contains
        # U+FFFD. A per-token `text[a:b] == decode([id])` round-trip is therefore NOT a
        # valid precondition here: half a character decodes to U+FFFD.
        assert a <= covered_to, (
            f"offset_mapping leaves characters [{covered_to}, {a}) = "
            f"{text[covered_to:a]!r} unassigned to any token, before token {i}; "
            "a mask built from these offsets would silently omit them."
        )
        covered_to = max(covered_to, b)
    assert covered_to == len(text), (
        f"offset_mapping covers only {covered_to} of {len(text)} characters"
    )
    assert tokenizer.decode(ids) == text, (
        "the token stream does not decode back to the rendered text the oracle "
        "measured its character offsets against"
    )


def _hf_label_mask(tokenizer, messages):
    """Reference mask by CHARACTER offsets. Shares no algorithm with the implementation.

    1. Render the conversation to text with the chat template.
    2. Locate each assistant turn's scored span as a pair of CHARACTER indices, by
       string search: the span runs from the first character after the assistant
       header through the last character of the ``<|eot_id|>`` that closes the turn.
       ``<|eot_id|>`` is scored -- the model must learn to stop.
    3. Tokenize that same text and use ``offset_mapping`` to decide, per token,
       whether its character span lies inside a scored span.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    _assert_offsets_are_usable(offsets, ids, text, tokenizer)

    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert text.count(ASSISTANT_HEADER) == len(assistant_messages), (
        f"rendered text contains {text.count(ASSISTANT_HEADER)} assistant header(s) "
        f"but the conversation has {len(assistant_messages)} assistant message(s); "
        "a message's content probably contains the header literal, which would make "
        "the oracle's own char search unsound."
    )

    spans, cursor = [], 0
    for _ in assistant_messages:
        header_at = text.index(ASSISTANT_HEADER, cursor)
        start = header_at + len(ASSISTANT_HEADER)
        end = text.index(EOT, start) + len(EOT)
        spans.append((start, end))
        cursor = end

    # Every span boundary must fall exactly between two tokens. If it does not, the
    # containment test below would silently drop a straddling token, so refuse.
    token_starts = {a for a, _ in offsets}
    token_ends = {b for _, b in offsets}
    for start, end in spans:
        assert start in token_starts, f"span start {start} is not a token boundary"
        assert end in token_ends, f"span end {end} is not a token boundary"

    mask = [0] * len(ids)
    for i, (a, b) in enumerate(offsets):
        for start, end in spans:
            inside = a >= start and b <= end
            disjoint = b <= start or a >= end
            assert inside or disjoint, (
                f"token {i} at offsets {(a, b)} partially overlaps scored span "
                f"{(start, end)}; the oracle cannot decide whether to score it."
            )
            if inside:
                mask[i] = 1

    return ids, mask


def _scored_runs(mask):
    """[start, end) index ranges of contiguous 1-runs."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j < len(mask) and mask[j] == 1:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


@pytest.fixture(scope="module")
def oracle(tokenizer, conversations):
    """Oracle output for every fixture row, computed once."""
    return [_hf_label_mask(tokenizer, messages) for messages in conversations]


def test_fixture_actually_exercises_multi_turn(conversations):
    """Non-vacuity: the rows this gate iterates must exist and must be multi-turn.

    Every assertion below is inside a `for row in conversations` loop, so an empty or
    truncated fixture would make the whole gate pass without testing anything. The
    Qwen3 equivalent of this gate passed every single-turn case and failed only on
    multi-turn, a bug that reached 13% of the study's held-out set.
    """
    assert len(conversations) == 12, f"expected 12 fixture rows, got {len(conversations)}"
    multi = [c for c in conversations if sum(m["role"] == "assistant" for m in c) >= 2]
    assert len(multi) >= 6, f"expected >=6 multi-turn rows, got {len(multi)}"
    assert any(m["role"] == "system" for c in conversations for m in c), "no system-message row"
    assert max(len(c) for c in conversations) >= 9, "no long (>=9 message) row"


def test_oracle_scores_a_nonempty_span_per_assistant_turn(conversations, oracle):
    """Non-vacuity: an oracle that scored nothing would make parity trivially true."""
    for row, (messages, (ids, mask)) in enumerate(zip(conversations, oracle, strict=True)):
        n_assistant = sum(m["role"] == "assistant" for m in messages)
        runs = _scored_runs(mask)
        assert len(runs) == n_assistant, (
            f"row {row}: oracle found {len(runs)} scored span(s) for {n_assistant} "
            f"assistant turn(s)"
        )
        assert 0 < sum(mask) < len(ids), (
            f"row {row}: oracle scored {sum(mask)}/{len(ids)} tokens; a mask that is "
            "all-zero or all-one cannot discriminate anything"
        )


def test_token_ids_match(conversations, generator, oracle):
    """Orbit's tokenization must equal the oracle's, or mask comparison is meaningless.

    A COHERENCE GUARD, not an independence check. In transformers 4.57
    ``apply_chat_template(tokenize=True)`` is literally
    ``self(rendered_chat, ..., add_special_tokens=False, ...)`` -- the same call the
    oracle makes on the same rendered string -- so this cannot catch a tokenizer bug
    and is expected to hold trivially. Its job is to confirm the two sides really are
    comparing masks over the SAME token sequence, so that a length or alignment
    mismatch surfaces here with a clear message instead of as a confusing
    position-mismatch failure downstream.
    """
    for row, (messages, (their_ids, _)) in enumerate(zip(conversations, oracle, strict=True)):
        our_ids, _ = generator.get_loss_mask(messages)
        assert our_ids == their_ids, (
            f"row {row}: token ids differ (orbit={len(our_ids)} tokens, "
            f"oracle={len(their_ids)} tokens)"
        )


def test_scored_token_count_matches(conversations, generator, oracle):
    """How MANY tokens are scored. A count bug rescales every NLL in the study.

    Kept separate from the position assertion so that a count mismatch cannot shadow
    a position mismatch, and so the failure names which of the two bugs occurred.
    """
    for row, (messages, (_, their_mask)) in enumerate(zip(conversations, oracle, strict=True)):
        _, our_mask = generator.get_loss_mask(messages)
        assert sum(our_mask) == sum(their_mask), (
            f"row {row}: scored-token COUNT differs: orbit={sum(our_mask)} "
            f"oracle={sum(their_mask)} (delta={sum(our_mask) - sum(their_mask)})"
        )


def test_scored_token_positions_match(tokenizer, conversations, generator, oracle):
    """WHICH tokens are scored. A pure shift keeps the count and still ruins the loss."""
    for row, (messages, (ids, their_mask)) in enumerate(zip(conversations, oracle, strict=True)):
        _, our_mask = generator.get_loss_mask(messages)
        if our_mask == their_mask:
            continue
        first = next(i for i in range(len(their_mask)) if our_mask[i] != their_mask[i])
        pytest.fail(
            f"row {row}: scored-token POSITIONS differ. First divergence at token "
            f"{first} (id {ids[first]}, "
            f"{tokenizer.convert_ids_to_tokens(ids[first])!r}): "
            f"orbit={our_mask[first]} oracle={their_mask[first]}. "
            f"{sum(a != b for a, b in zip(our_mask, their_mask, strict=True))} "
            f"token(s) disagree in total."
        )


def test_scored_text_is_exactly_the_assistant_turns(tokenizer, conversations, generator):
    """Semantic anchor, in pure string space: what does the mask actually select?

    Independent of both the token scan and the char-offset oracle. The scored text of
    each turn must be exactly that assistant message's (template-trimmed) content plus
    the ``<|eot_id|>`` that closes it -- no header, no leading newlines, no bleed into
    the next user turn.

    DO NOT DELETE THIS AS REDUNDANT WITH THE PARITY TESTS. It looks redundant on a
    green run and is not: the parity tests compare Orbit against an oracle that shares
    the header and terminator LITERALS with it, so a wrong literal makes both sides
    agree on the same wrong span. Verified by injecting exactly that -- dropping the
    header literal's trailing ``\\n\\n`` on BOTH sides leaves the other six tests in
    this file passing and is caught here alone, because ``expected`` is rebuilt from
    the fixture's own ``content`` using the same module-level ``EOT`` constant the
    oracle's span search uses (see ``_hf_label_mask``). It is independent of the HEADER
    literal only -- NOT of the terminator literal, which it still consults via ``EOT``.
    See the module docstring's "Exactly where that independence STOPS".
    """
    for row, messages in enumerate(conversations):
        ids, mask = generator.get_loss_mask(messages)
        expected = [m["content"].strip() + EOT for m in messages if m["role"] == "assistant"]
        actual = [tokenizer.decode(ids[a:b]) for a, b in _scored_runs(mask)]
        assert actual == expected, (
            f"row {row}: scored text is not exactly the assistant turns.\n"
            f"  expected {len(expected)} span(s), got {len(actual)}\n"
            + "\n".join(
                f"  span {i}: got {g!r:.160} != want {w!r:.160}"
                for i, (g, w) in enumerate(zip(actual, expected, strict=False))
                if g != w
            )
        )


def test_system_and_user_turns_are_never_scored(tokenizer, conversations, generator):
    """The complement of the above: nothing outside an assistant turn may be scored."""
    for row, messages in enumerate(conversations):
        ids, mask = generator.get_loss_mask(messages)
        unscored = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if not m])
        for message in messages:
            if message["role"] in ("system", "user"):
                content = message["content"].strip()
                assert content in unscored, (
                    f"row {row}: a {message['role']} turn is not fully unscored"
                )
