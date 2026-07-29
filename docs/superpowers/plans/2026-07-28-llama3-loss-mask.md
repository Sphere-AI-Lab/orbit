# Llama-3 Loss Mask + Parity Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Orbit a Llama-3 multi-turn loss-mask generator, gated token-for-token against
an independent HuggingFace oracle, so SFT on Llama-3.1-8B scores exactly the tokens an HF SFT
recipe would.

**Architecture:** Mirror the *post-fix* Qwen3 generator: tokenize the whole conversation
exactly once, then locate each assistant turn's scored span inside that single tokenization by
scanning for the literal assistant-header token sequence and its `<|eot_id|>` terminator. The
parity test deliberately uses a **different** algorithm from the implementation — character
offsets from the fast tokenizer's `offset_mapping` — so a shared bug cannot hide from it.

**Tech stack:** `transformers` fast tokenizers, pytest, no GPU. Every test in this plan runs
on CPU in seconds.

This unblocks **P2** of
`docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md`.

## Two facts established before writing this plan

Both were checked against the downloaded checkpoint, not assumed.

1. **Llama-3.1-8B base has no chat template at all.** `tokenizer_config.json` has no
   `chat_template` key and its `eos_token` is `<|end_of_text|>`, not `<|eot_id|>`. Any call to
   `apply_chat_template` raises — which means `MultiTurnLossMaskGenerator.__init__` raises
   too, because `get_system_message_length()` calls it. A template must be supplied before the
   generator can even be constructed. **This is Task 1 and it is the real blocker; the mask
   algorithm is the easy part.**
2. **The chat special tokens exist in the base vocabulary**: `<|start_header_id|>` = 128006,
   `<|end_header_id|>` = 128007, `<|eot_id|>` = 128009, `<|begin_of_text|>` = 128000. So the
   Llama-3.1 Instruct template can be applied to the base tokenizer with no vocab surgery and
   no embedding resize.

**Decision — which template to pin.** Use the **Llama-3.1-Instruct template verbatim**, not
Tulu3's native `<|user|>` / `<|assistant|>` format. Three reasons: the tokens are already in
the base vocab (Tulu's are not, and adding them means resizing embeddings on a base model we
are about to fine-tune, introducing untrained rows); `<|eot_id|>` gives an unambiguous
turn terminator that mirrors Qwen's `<|im_end|>`, so the implementation stays structurally
identical to the one already validated; and the blog's claims are all internal comparisons
across arms, so any *consistently applied* template is admissible provided it is pinned and
recorded. The alternative is worth revisiting only if a Tulu3-native template later proves to
change the measured optima.

**Checked and benign:** the Instruct template injects an unconditional system block
(`Cutting Knowledge Date: December 2023` / `Today Date: …`) even when no system message is
supplied. Its date is a **literal default `"26 Jul 2024"`, not `strftime_now`** — verified by
reading the template source — so tokenization is deterministic across days. Had it used
`strftime_now`, every run would tokenize differently by date and the whole campaign would be
irreproducible. Task 1 pins the template text so a future upstream change cannot silently
introduce that.

## Global Constraints

- Working interpreter: `/lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python`
- Tests need no `PYTHONPATH` shim: `orbit/rollout/rm_hub/math_alignment.py` calls
  `_ensure_vendored_math_eval_on_path()` before importing `latex2sympy`.
- Canonical test command:
  ```
  cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
  /lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python -m pytest <paths> -q -p no:cacheprovider
  ```
- **Always pass explicit test paths.** `pytest tests/fast/` silently skips
  `tests/fast/scripts/` and `tests/fast/tools/` — `norecursedirs` matches those basenames at
  any depth.
- Pre-existing failures, not caused by this plan: `test_quantizer_ci.py` (collection error),
  `test_tensor_backper.py::test_tensor_backuper_allows_filtered_placeholder_source`,
  `test_orbit_launcher_contract.py::test_active_launchers_are_thin_orbit_entrypoints`
  (**15** legacy launchers — the count must stay 15),
  `test_shared_launcher_knobs.py::test_train_lib_exposes_dump_details_knob`,
  `test_megatron_cli_flags.py::test_post_layernorm_flags_propagate_to_megatron`.
- Model path: `/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B` (base, tokenizer files present).
- Do not touch `gen_multi_turn_loss_mask_qwen`, `_qwen3`, or `_distill_qwen`. Their byte-level
  stability is asserted by the existing G3 gate.

## File Structure

- `orbit/utils/llama3_chat_template.py` *(new)* — the pinned template string and a
  `ensure_llama3_chat_template(tokenizer)` helper. One responsibility: make a template-less
  Llama-3 tokenizer usable, idempotently.
- `orbit/utils/mask_utils.py` *(modify)* — add `gen_multi_turn_loss_mask_llama3` and one
  dispatch branch. Nothing else in the file changes.
- `orbit/utils/arguments.py` *(modify)* — add `llama3` to `--loss-mask-type` choices.
- `tests/fast/utils/test_llama3_chat_template.py` *(new)*
- `tests/fast/utils/test_llama3_loss_mask.py` *(new)* — unit tests, hand-built conversations.
- `tests/fast/rollout/test_sft_loss_mask_parity_llama3.py` *(new)* — the gate, against an
  independent oracle, over real fixture rows.
- `tests/fast/fixtures/lora_regret/llama3_sample.jsonl` *(new)*

---

### Task 1: Pin the Llama-3 chat template and make the base tokenizer usable

**Files:**
- Create: `orbit/utils/llama3_chat_template.py`
- Test: `tests/fast/utils/test_llama3_chat_template.py`

**Interfaces:**
- Produces: `LLAMA3_CHAT_TEMPLATE: str` and
  `ensure_llama3_chat_template(tokenizer) -> None` — sets `tokenizer.chat_template` only when
  it is unset; never overwrites a template the tokenizer already carries.

- [ ] **Step 1: Write the failing test**

```python
# tests/fast/utils/test_llama3_chat_template.py
"""The Llama-3.1 base tokenizer ships no chat template; we pin one."""

from pathlib import Path

import pytest

LLAMA31_8B = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B")

pytestmark = pytest.mark.skipif(
    not (LLAMA31_8B / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def base_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(LLAMA31_8B))


def test_base_tokenizer_really_has_no_template(base_tokenizer):
    """If this ever fails, upstream added a template and Task 1's premise changed."""
    assert base_tokenizer.chat_template is None


def test_ensure_sets_the_template(base_tokenizer):
    from orbit.utils.llama3_chat_template import (
        LLAMA3_CHAT_TEMPLATE,
        ensure_llama3_chat_template,
    )

    ensure_llama3_chat_template(base_tokenizer)
    assert base_tokenizer.chat_template == LLAMA3_CHAT_TEMPLATE


def test_ensure_is_idempotent_and_never_overwrites(base_tokenizer):
    from orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    base_tokenizer.chat_template = "SENTINEL"
    ensure_llama3_chat_template(base_tokenizer)
    assert base_tokenizer.chat_template == "SENTINEL"


def test_template_date_is_a_literal_not_a_clock():
    """A strftime_now date would make every run tokenize differently by day."""
    from orbit.utils.llama3_chat_template import LLAMA3_CHAT_TEMPLATE

    assert "strftime_now" not in LLAMA3_CHAT_TEMPLATE
    assert '"26 Jul 2024"' in LLAMA3_CHAT_TEMPLATE


def test_rendered_conversation_has_expected_markers(base_tokenizer):
    from orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    ensure_llama3_chat_template(base_tokenizer)
    text = base_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False,
    )
    assert "<|start_header_id|>assistant<|end_header_id|>\n\n" in text
    assert text.rstrip().endswith("<|eot_id|>")


def test_exactly_one_bos_and_it_is_first(base_tokenizer):
    """apply_chat_template must not double-add bos on top of the template's own."""
    from orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    ensure_llama3_chat_template(base_tokenizer)
    ids = base_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=True,
    )
    bos = base_tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    assert ids[0] == bos
    assert ids.count(bos) == 1
```

- [ ] **Step 2: Run it and watch it fail**

```
/lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python -m pytest \
    tests/fast/utils/test_llama3_chat_template.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: No module named 'orbit.utils.llama3_chat_template'`.

- [ ] **Step 3: Implement**

Copy the template **verbatim** from
`/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B-Instruct/tokenizer_config.json` (key
`chat_template`, 4614 characters) into the module as a string literal. Do not retype it, and
do not reformat it — a single changed whitespace changes tokenization.

```python
# orbit/utils/llama3_chat_template.py
"""The Llama-3.1 chat template, pinned.

Llama-3.1-8B *base* ships no `chat_template`, so `apply_chat_template` raises and
`MultiTurnLossMaskGenerator` cannot even be constructed against it. The base vocab
does contain the chat control tokens (<|start_header_id|> 128006, <|end_header_id|>
128007, <|eot_id|> 128009), so the Instruct template applies cleanly with no vocab
resize.

Pinned rather than read from the Instruct checkpoint at runtime for two reasons: the
Instruct model is not a dependency of an experiment that fine-tunes base, and an
upstream template revision must never silently change what tokens we score.

The template's `date_string` default is the LITERAL "26 Jul 2024", not strftime_now,
so rendering is deterministic across days. Verified when pinning; the accompanying
test asserts it stays that way.
"""

LLAMA3_CHAT_TEMPLATE = r"""<PASTE THE 4614-CHARACTER TEMPLATE HERE VERBATIM>"""


def ensure_llama3_chat_template(tokenizer) -> None:
    """Give a template-less Llama-3 tokenizer the pinned template.

    Idempotent, and never overwrites a template the tokenizer already has -- an
    Instruct checkpoint carries its own and must keep it.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
```

- [ ] **Step 4: Run the tests — all 6 pass**

- [ ] **Step 5: Commit**

```bash
git add orbit/utils/llama3_chat_template.py tests/fast/utils/test_llama3_chat_template.py
git commit -m "feat(mask): pin the llama-3.1 chat template for the template-less base tokenizer"
```

---

### Task 2: Build the parity fixture from real multi-turn conversations

**Files:**
- Create: `tests/fast/fixtures/lora_regret/llama3_sample.jsonl`
- Modify: `tools/lora_regret/prepare_data.py` (add the extraction helper used to build it)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: a JSONL fixture, one `{"prompt": [messages]}` per line, that Task 5's gate reads.

**Why real rows and not hand-built ones.** The Qwen3 gate failed on exactly the case a
hand-built fixture would have gotten right — single-turn conversations passed, multi-turn did
not — and the failure was worth 13% of the held-out set. Tulu3 is far more multi-turn than No
Robots was, so the fixture must be multi-turn-heavy by construction.

- [ ] **Step 1: Extract 12 conversations from Tulu3**

At least 6 with ≥2 assistant turns, at least 1 with a system message, at least 1 with 4+
turns. If Tulu3 is not yet prepared (experiment-plan P4), pull directly:

```python
from datasets import load_dataset
ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
```

Select deterministically (sort by index, take the first N matching each predicate) so the
fixture is reproducible. Write `{"prompt": [{"role": ..., "content": ...}, ...]}` per line.

- [ ] **Step 2: Assert the fixture's own shape in a test**

```python
def test_fixture_is_multi_turn_heavy():
    rows = [json.loads(l)["prompt"] for l in FIXTURE.read_text().splitlines()]
    assert len(rows) == 12
    multi = [r for r in rows if sum(m["role"] == "assistant" for m in r) >= 2]
    assert len(multi) >= 6, "fixture must exercise the multi-turn path that broke Qwen3"
    assert any(m["role"] == "system" for r in rows for m in r)
    assert any(len(r) >= 8 for r in rows)
```

- [ ] **Step 3: Run it, then commit the fixture**

```bash
git add tests/fast/fixtures/lora_regret/llama3_sample.jsonl tools/lora_regret/prepare_data.py
git commit -m "test(mask): multi-turn tulu3 fixture for the llama-3 loss-mask gate"
```

---

### Task 3: `gen_multi_turn_loss_mask_llama3`

**Files:**
- Modify: `orbit/utils/mask_utils.py` (add one method; touch nothing else)
- Test: `tests/fast/utils/test_llama3_loss_mask.py`

**Interfaces:**
- Consumes: `ensure_llama3_chat_template` from Task 1.
- Produces: `MultiTurnLossMaskGenerator.gen_multi_turn_loss_mask_llama3(messages, tools=None)
  -> tuple[list[int], list[int]]` — `(all_token_ids, all_loss_masks)`, same contract as the
  qwen3 method.

- [ ] **Step 1: Write the failing tests**

```python
# tests/fast/utils/test_llama3_loss_mask.py
from pathlib import Path

import pytest

LLAMA31_8B = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B")

pytestmark = pytest.mark.skipif(
    not (LLAMA31_8B / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def gen():
    from transformers import AutoTokenizer

    from orbit.utils.llama3_chat_template import ensure_llama3_chat_template
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    tok = AutoTokenizer.from_pretrained(str(LLAMA31_8B))
    ensure_llama3_chat_template(tok)
    return MultiTurnLossMaskGenerator(tok, tokenizer_type="llama3")


SINGLE = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
MULTI = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "Goodbye!"},
]


def test_lengths_match(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(SINGLE)
    assert len(ids) == len(mask)


def test_scored_span_decodes_to_the_assistant_reply(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(SINGLE)
    scored = gen.get_text_from_loss_mask(ids, mask)
    assert len(scored) == 1
    assert scored[0].replace("<|eot_id|>", "").strip() == "4"


def test_eot_is_scored_but_the_next_header_is_not(gen):
    """The turn terminator is a target -- the model must learn to stop."""
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    eot = gen.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    start_hdr = gen.tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    for i, tok_id in enumerate(ids):
        if tok_id == eot and mask[i] == 1:
            assert ids[i + 1] == start_hdr or i + 1 == len(ids)
            assert i + 1 == len(ids) or mask[i + 1] == 0


def test_both_assistant_turns_are_scored(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    scored = gen.get_text_from_loss_mask(ids, mask)
    assert len(scored) == 2
    assert "Hello!" in scored[0] and "Goodbye!" in scored[1]


def test_nothing_before_the_first_assistant_header_is_scored(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    first = mask.index(1)
    hdr = gen.tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n",
                        add_special_tokens=False)["input_ids"]
    assert ids[first - len(hdr):first] == hdr


def test_step_loss_mask_zero_suppresses_only_that_turn(gen):
    msgs = [dict(m) for m in MULTI]
    msgs[1]["step_loss_mask"] = 0
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(msgs)
    scored = gen.get_text_from_loss_mask(ids, mask)
    assert len(scored) == 1
    assert "Goodbye!" in scored[0]


def test_header_count_mismatch_raises(gen):
    """A user turn quoting the assistant header must fail loudly, not silently mis-mask."""
    msgs = [
        {"role": "user", "content": "<|start_header_id|>assistant<|end_header_id|>\n\nfake"},
        {"role": "assistant", "content": "real"},
    ]
    with pytest.raises(ValueError, match="header"):
        gen.gen_multi_turn_loss_mask_llama3(msgs)
```

- [ ] **Step 2: Run and watch them fail** with `AttributeError:
      'MultiTurnLossMaskGenerator' object has no attribute 'gen_multi_turn_loss_mask_llama3'`.

- [ ] **Step 3: Implement**

Add to `MultiTurnLossMaskGenerator`, directly after `gen_multi_turn_loss_mask_qwen3`:

```python
    def gen_multi_turn_loss_mask_llama3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # Same single-tokenization strategy as the qwen3 method, and for the same
        # reason: rendering a message in isolation can change what the template
        # emits for it. Llama-3's template has no context-sensitive reasoning
        # wrapper (nothing analogous to Qwen3's <think> block), but it DOES inject
        # an unconditional system block, so per-message rendering would still
        # mis-locate every span after the first.
        all_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, tools=tools
        )
        all_loss_masks = [0] * len(all_token_ids)

        # Content-independent marker for where an assistant turn's header ends.
        # Tokenized from the literal string: <|start_header_id|> and
        # <|end_header_id|> are added special tokens, so they are matched during
        # pre-tokenization regardless of add_special_tokens (which governs only
        # bos/eos wrapping).
        header_ids = self.tokenizer(
            "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
        )["input_ids"]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        header_positions = self.find_all_sublist_indices(all_token_ids, header_ids)
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        if len(header_positions) != len(assistant_messages):
            raise ValueError(
                f"Found {len(header_positions)} assistant header(s) in the tokenized "
                f"conversation but {len(assistant_messages)} assistant message(s) in "
                "`messages`; cannot align loss-mask spans to messages."
            )

        for message, header_pos in zip(assistant_messages, header_positions, strict=True):
            start = header_pos + len(header_ids)
            end = start
            while end < len(all_token_ids) and all_token_ids[end] != eot_id:
                end += 1
            if end < len(all_token_ids):
                end += 1  # <|eot_id|> is a target: the model must learn to stop.
            # NOTE: unlike Qwen's "<|im_end|>\n", Llama-3 emits no newline after
            # <|eot_id|> -- the next "<|start_header_id|>" follows immediately -- so
            # there is deliberately no trailing-newline step here.

            if message.get("step_loss_mask", 1) == 1:
                for k in range(start, min(end, len(all_token_ids))):
                    all_loss_masks[k] = 1

        return all_token_ids, all_loss_masks
```

- [ ] **Step 4: Run the tests — all 7 pass**

- [ ] **Step 5: Prove the tests are not tautological**

Temporarily change `end += 1` (the `<|eot_id|>` inclusion) to `pass` and re-run.
`test_eot_is_scored_but_the_next_header_is_not` and
`test_scored_span_decodes_to_the_assistant_reply` must fail. Revert.

- [ ] **Step 6: Commit**

```bash
git add orbit/utils/mask_utils.py tests/fast/utils/test_llama3_loss_mask.py
git commit -m "feat(mask): llama-3 multi-turn loss mask via single-pass span location"
```

---

### Task 4: Dispatch and CLI

**Files:**
- Modify: `orbit/utils/mask_utils.py` (the `get_loss_mask` branch)
- Modify: `orbit/utils/arguments.py:1713-1717`
- Test: `tests/fast/utils/test_llama3_loss_mask.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def test_dispatch_routes_llama3(gen):
    a = gen.get_loss_mask(MULTI)
    b = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    assert a == b


def test_unknown_type_still_raises():
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    obj = MultiTurnLossMaskGenerator.__new__(MultiTurnLossMaskGenerator)
    obj.tokenizer_type = "not_a_real_type"
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        MultiTurnLossMaskGenerator.get_loss_mask(obj, MULTI)


def test_argparse_accepts_llama3_and_rejects_junk():
    """The launcher can only pass what argparse allows."""
    import subprocess, sys
    # exercised through the real parser; see tests/fast/utils/test_peft_arguments.py
    # for the established pattern of driving orbit's parser in-process.
```

Follow the in-process parser pattern already used by
`tests/fast/utils/test_peft_arguments.py::test_lora_a_init_method_real_parser_rejects_uniform`
rather than shelling out — assert `llama3` parses and that an invalid value raises
`SystemExit(2)`.

- [ ] **Step 2: Run, watch fail** (`Unsupported tokenizer type: llama3`).

- [ ] **Step 3: Implement — two edits**

```python
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "llama3":
            return self.gen_multi_turn_loss_mask_llama3(messages, tools)
```

```python
                choices=["qwen", "qwen3", "distill_qwen", "llama3"],
```

- [ ] **Step 4: Run tests — pass.** Then run the full CPU suite and confirm the pre-existing
      failure list is unchanged, in particular that the launcher-contract failure still reads
      **15**, not 16.

- [ ] **Step 5: Commit**

```bash
git add orbit/utils/mask_utils.py orbit/utils/arguments.py tests/fast/utils/test_llama3_loss_mask.py
git commit -m "feat(mask): expose llama3 as a --loss-mask-type choice"
```

---

### Task 5: The gate — parity against an independent HF oracle

**Files:**
- Create: `tests/fast/rollout/test_sft_loss_mask_parity_llama3.py`

**Interfaces:**
- Consumes: Tasks 1-4 and the Task 2 fixture.
- Produces: the Llama-3 equivalent of gate G3. **The experiment campaign does not start until
  this passes.**

**The oracle must not share the implementation's algorithm.** The Qwen3 gate's committed
oracle used the same `<|im_end|>`-scan as the code it tested, which was recorded as a deferred
weakness — a bug shared by both would have been invisible. Here the oracle works from
**character offsets**: render the conversation to text, locate each assistant turn's content
span by string search over the rendered text, then map characters to tokens with
`return_offsets_mapping`. It cannot fail the same way a token-scan fails.

- [ ] **Step 1: Write the gate**

```python
"""Gate G3-llama: Orbit's Llama-3 loss mask must equal the HF oracle's label mask.

Scoring a different token set than an HF SFT recipe shifts every NLL in the study
by a constant -- invisible in the shape of a loss curve, fatal to any comparison.

The oracle below is deliberately algorithm-independent from the implementation:
it locates assistant content by CHARACTER offsets in the rendered conversation and
maps them to tokens via offset_mapping, where the implementation scans for header
and <|eot_id|> TOKEN ids. A bug in the token scan cannot hide in the char scan.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests/fast/fixtures/lora_regret/llama3_sample.jsonl"
LLAMA31_8B = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B")

pytestmark = pytest.mark.skipif(
    not (LLAMA31_8B / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    tok = AutoTokenizer.from_pretrained(str(LLAMA31_8B))
    ensure_llama3_chat_template(tok)
    return tok


@pytest.fixture(scope="module")
def conversations():
    return [json.loads(line)["prompt"] for line in FIXTURE.read_text().splitlines()]


def _hf_label_mask(tokenizer, messages):
    """Reference mask by character offsets. Independent of the implementation."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    spans, cursor = [], 0
    for m in messages:
        if m["role"] != "assistant":
            continue
        header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        h = text.index(header, cursor)
        start = h + len(header)
        end = text.index("<|eot_id|>", start) + len("<|eot_id|>")
        spans.append((start, end))
        cursor = end

    mask = [0] * len(ids)
    for i, (a, b) in enumerate(offsets):
        if a == b:  # special tokens can report empty offsets on some backends
            a = b = _char_of_token(text, ids, i, tokenizer)
        for s, e in spans:
            if a >= s and b <= e:
                mask[i] = 1
    return ids, mask
```

If `offset_mapping` reports `(0, 0)` for added special tokens with this tokenizer backend,
resolve it by re-encoding with `add_special_tokens=False` on the decoded text and asserting
round-trip equality; do **not** silently fall back to a token scan, which would destroy the
oracle's independence. Verify the backend's behaviour first and delete `_char_of_token` if it
is unnecessary.

- [ ] **Step 2: Assert exact parity on every fixture row**

```python
def test_token_ids_match(tokenizer, conversations):
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="llama3")
    for i, msgs in enumerate(conversations):
        ours, _ = gen.gen_multi_turn_loss_mask_llama3(msgs)
        theirs, _ = _hf_label_mask(tokenizer, msgs)
        assert ours == theirs, f"row {i}: token ids differ"


def test_masks_match_exactly(tokenizer, conversations):
    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator

    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="llama3")
    for i, msgs in enumerate(conversations):
        _, ours = gen.gen_multi_turn_loss_mask_llama3(msgs)
        _, theirs = _hf_label_mask(tokenizer, msgs)
        assert sum(ours) == sum(theirs), f"row {i}: scored-token COUNT differs"
        assert ours == theirs, f"row {i}: mask positions differ"


def test_multi_turn_rows_are_actually_exercised(conversations):
    multi = [c for c in conversations if sum(m["role"] == "assistant" for m in c) >= 2]
    assert len(multi) >= 6
```

- [ ] **Step 3: Run the gate**

```
/lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python -m pytest \
    tests/fast/rollout/test_sft_loss_mask_parity_llama3.py -q -p no:cacheprovider
```

**If it fails, that is the gate working — do not weaken it.** Diagnose before changing either
side. For Qwen3 the equivalent failure was a real production bug affecting 13% of the held-out
set, and the fix was worth more than the gate.

- [ ] **Step 4: Prove non-vacuity by mutation**

Temporarily change the implementation's `start = header_pos + len(header_ids)` to
`header_pos + len(header_ids) - 1` and re-run: `test_masks_match_exactly` must fail with a
position mismatch. Then change `end += 1` to `pass`: the count assertion must fail. Revert
both. Record both observed failures in the commit message — a gate that has never been seen
to fail is not evidence of anything.

- [ ] **Step 5: Commit**

```bash
git add tests/fast/rollout/test_sft_loss_mask_parity_llama3.py
git commit -m "test(mask): gate llama-3 loss mask against a char-offset HF oracle"
```

---

### Task 6: Record the gate and unblock the campaign

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md`
- Modify: `docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md` (tick P2)

- [ ] **Step 1: Add a "G3-llama" section to the gate log** with the command, the result, the
      number of fixture rows and multi-turn rows exercised, and the two mutations that were
      observed to fail. If a real bug was found, write down what it was and its exposure as a
      fraction of the training and held-out sets — that is what the Qwen3 entry does, and it
      is the part that turns out to matter later.
- [ ] **Step 2: Tick P2 in the experiment plan.**
- [ ] **Step 3: Commit.**

```bash
git add docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md \
        docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md
git commit -m "docs(repro): record the llama-3 loss-mask gate result"
```

---

## Self-Review Notes

**Spec coverage.** P2 asks for a Llama-3 loss-mask generator plus a parity gate; Tasks 3-5
deliver both, Task 1 covers the template blocker that P2 did not anticipate, and Tasks 2 and 6
cover the fixture and the record.

**Deliberately out of scope.** Wiring `LOSS_MASK_TYPE=llama3` into a Llama SFT launcher — that
belongs with the launcher work, which does not exist yet and is a separate task. Nothing here
changes any existing mask generator, so the current G3 must stay green throughout; if it goes
red, the change is wrong.

**Known weakness, inherited.** A user turn whose content literally contains
`<|start_header_id|>assistant<|end_header_id|>` raises rather than silently mis-masking
(Task 3's last test pins this). Fail-loud is the right trade, but it is a new crash path on
untrusted input, exactly as the Qwen3 version has. Check the fixture and the real Tulu3 split
for such rows before the sweep; if any exist, they must be filtered rather than crash a 40-run
sweep at hour six.
