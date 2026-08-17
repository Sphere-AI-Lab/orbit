"""Gate G3: Orbit's SFT loss mask must equal the HF oracle's label mask.

A constant offset between the two masks shifts every NLL in the reproduction
by a constant, which is invisible in the shape of a loss curve but makes the
numbers uncomparable to michaelbzhu's published table.

Skipped unless the Qwen3-4B tokenizer is present locally.

FIXED (fix round 1, see task-8-report.md): this gate originally FAILED on the
multi-turn conversation below. Root cause: Orbit's
``MultiTurnLossMaskGenerator.gen_multi_turn_loss_mask_qwen3``
(`orbit/utils/mask_utils.py`) used to render each message in isolation,
paired with a synthetic single-user "prefix" message, to work out that
message's token span. For Qwen3's *base* chat template this was unsafe: the
template decides whether to wrap an assistant turn in an empty
``<think>\\n\\n</think>\\n\\n`` block based on whether that turn is the LAST
assistant response following the LAST user turn in the WHOLE conversation
(Qwen3 deliberately strips reasoning wrappers from assistant turns earlier in
the history). Rendered in isolation, every assistant message trivially
looked like "the last message following the last user turn," so Orbit
inserted that empty think-block before *every* assistant turn, not just the
true final one. ``no_robots_train.jsonl`` for this reproduction is 8.4%
multi-turn (535/6400) and ``no_robots_test.jsonl`` -- the held-out NLL split
-- is 13.0% multi-turn (13/100), so this was a live risk to the study's
headline metric, not a fixture curiosity.

``gen_multi_turn_loss_mask_qwen3`` was rewritten to tokenize the whole
conversation once and locate assistant-turn spans within that single
tokenization (the same approach validated below in ``_hf_label_mask``),
which cannot be fooled by Qwen3's context-sensitive think-tag insertion.
Verified against the corrected reference on all 100 rows of
``no_robots_test.jsonl`` and an 802-row sample of ``no_robots_train.jsonl``
(a random 300 plus every one of the 535 multi-turn rows in the whole file) --
zero disagreements. See task-8-report.md for the full remediation record.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests/fast/fixtures/lora_regret/no_robots_sample.jsonl"
QWEN3_4B = Path("/lustre/fast/fast/zqiu/hf_models/Qwen3-4B")

pytestmark = pytest.mark.skipif(
    not (QWEN3_4B / "tokenizer_config.json").exists(),
    reason="Qwen3-4B tokenizer not downloaded (Task 6)",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(QWEN3_4B), trust_remote_code=True)


@pytest.fixture(scope="module")
def conversations():
    return [json.loads(line)["prompt"] for line in FIXTURE.read_text().splitlines()]


def _hf_label_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Reference mask: score assistant turns only, the way HF SFT recipes do.

    Tokenizes the FULL conversation exactly once (``add_generation_prompt=False``)
    and then locates each assistant turn's scored span *within that single
    tokenization* by scanning for the literal ``<|im_start|>assistant\\n``
    header token sequence and the ``<|im_end|>`` (plus a trailing newline,
    if present) that closes it.

    This deliberately does NOT re-tokenize prefixes or per-turn sub-lists in
    isolation (an earlier version of this helper did, via
    ``apply_chat_template(messages[:i], ..., add_generation_prompt=True)``
    diffed against ``apply_chat_template(messages[:i+1], ...)``). That
    approach is unsound for Qwen3's base chat template: whether an assistant
    turn gets wrapped in an empty ``<think>\\n\\n</think>\\n\\n`` block depends
    on whether it is the last assistant turn following the last user turn in
    the WHOLE conversation. Truncating to ``messages[:i+1]`` makes turn ``i``
    trivially "last" regardless of what actually follows it later in the real
    conversation, silently corrupting the span computed for any assistant
    turn that isn't truly final. Verified against this fixture: the
    single-tokenization/boundary-scan approach here reproduces the naive
    diffing approach exactly for every single-assistant-turn conversation,
    and only diverges (correctly) on the multi-turn one.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_newline = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]
    header = [im_start, *assistant_newline]

    # return_dict=False: transformers 5 flipped the default to True, which turns
    # this into a BatchEncoding; the reference mask below indexes a flat id list.
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=False
    )
    mask = [0] * len(full)

    i = 0
    while i <= len(full) - len(header):
        if full[i : i + len(header)] != header:
            i += 1
            continue
        start = i + len(header)
        j = start
        while j < len(full) and full[j] != im_end:
            j += 1
        end = j + 1  # include <|im_end|>
        if end < len(full) and full[end] == assistant_newline[-1]:
            end += 1  # include the trailing "\n" after <|im_end|>
        for k in range(start, min(end, len(full))):
            mask[k] = 1
        i = end
    return full, mask


def test_orbit_and_hf_tokenize_to_the_same_ids(tokenizer, conversations):
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    for messages in conversations:
        orbit_ids, _ = gen.get_loss_mask(messages)
        hf_ids, _ = _hf_label_mask(tokenizer, messages)
        assert orbit_ids == hf_ids, f"token ids differ for {messages}"


def test_orbit_and_hf_score_the_same_tokens(tokenizer, conversations):
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    for messages in conversations:
        _, orbit_mask = gen.get_loss_mask(messages)
        _, hf_mask = _hf_label_mask(tokenizer, messages)
        assert sum(orbit_mask) == sum(hf_mask), (
            f"scored-token COUNT differs for {messages}: "
            f"orbit={sum(orbit_mask)} hf={sum(hf_mask)}"
        )
        assert orbit_mask == hf_mask, f"scored-token POSITIONS differ for {messages}"


def test_system_prompt_is_not_scored(tokenizer, conversations):
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = conversations[2]  # the one with a system turn
    ids, mask = gen.get_loss_mask(messages)
    scored = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m])
    assert "terse" not in scored


def _scored_runs(mask: list[int]) -> list[tuple[int, int]]:
    """Return the [start, end) index ranges of contiguous 1-runs in a mask."""
    runs = []
    i = 0
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


def test_non_final_assistant_turn_is_not_wrapped_in_think_tags(tokenizer, conversations):
    """Regression guard for the isolated-per-message rendering bug (Task 8 fix round 1).

    Qwen3's chat template wraps ONLY the final assistant turn (the one following the
    last real user turn in the whole conversation) in an empty
    ``<think>\\n\\n</think>\\n\\n`` block. A non-final assistant turn must never be
    scored with that wrapper -- if it is, `mask_utils.py` has regressed to rendering
    turns in isolation again.
    """
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = conversations[1]  # multi-turn: user, assistant, user, assistant
    ids, mask = gen.get_loss_mask(messages)
    runs = _scored_runs(mask)
    assert len(runs) == 2, f"expected 2 scored (assistant-turn) spans, got {len(runs)}: {runs}"

    first_start, first_end = runs[0]
    non_final_scored = tokenizer.decode(ids[first_start:first_end])
    assert "<think>" not in non_final_scored, (
        f"non-final assistant turn was scored with a <think> wrapper: {non_final_scored!r}"
    )

    last_start, last_end = runs[-1]
    final_scored = tokenizer.decode(ids[last_start:last_end])
    assert "<think>" in final_scored, (
        f"the true final assistant turn should still get the think wrapper: {final_scored!r}"
    )


def test_step_loss_mask_zero_zeroes_the_turn(tokenizer, conversations):
    """Regression guard: `step_loss_mask=0` on an assistant message must still zero
    that turn's loss mask entirely, unaffected by the qwen3 rewrite to single-shot
    tokenization + boundary scan.
    """
    import copy

    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    baseline_messages = conversations[1]  # user, assistant, user, assistant
    baseline_ids, baseline_mask = gen.get_loss_mask(baseline_messages)
    baseline_runs = _scored_runs(baseline_mask)
    assert len(baseline_runs) == 2

    # Zero out only the FIRST assistant turn; the second must be untouched.
    messages = copy.deepcopy(baseline_messages)
    assistant_seen = 0
    for message in messages:
        if message["role"] == "assistant":
            assistant_seen += 1
            if assistant_seen == 1:
                message["step_loss_mask"] = 0
    ids, mask = gen.get_loss_mask(messages)
    assert ids == baseline_ids, "step_loss_mask must not change the token sequence, only the mask"
    runs = _scored_runs(mask)
    assert len(runs) == 1, f"expected exactly 1 scored span (the untouched turn), got {runs}"
    assert runs[0] == baseline_runs[1], "the surviving span should be exactly the second turn's span"

    # Zero out BOTH assistant turns; nothing should be scored.
    messages_all_off = copy.deepcopy(baseline_messages)
    for message in messages_all_off:
        if message["role"] == "assistant":
            message["step_loss_mask"] = 0
    _, mask_all_off = gen.get_loss_mask(messages_all_off)
    assert sum(mask_all_off) == 0, f"expected 0 scored tokens, got {sum(mask_all_off)}"
