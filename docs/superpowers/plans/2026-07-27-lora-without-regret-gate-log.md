# LoRA Without Regret — Gate Log

Gates defined in `docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md` §7.2.
Fill in as each is run. Do not launch the sweep until G2 is marked PASS.

Run order is cheapest-first: G3 → G4 → seed noise → G1 → G2.

## G3 — loss-mask parity (CPU)

- Command:
  ```
  cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
  /lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/fast/rollout/test_sft_loss_mask_parity.py
  ```
- Result: **PASSED** (re-verified 2026-07-28 as part of the full CPU sweep: 296 passed,
  1 pre-existing failure — `test_active_launchers_are_thin_orbit_entrypoints`, 15 legacy
  launchers, unchanged from `orbit-v0`).
- Notes: this gate FAILED on first run and found a real bug in Orbit production code.
  `gen_multi_turn_loss_mask_qwen3` called `apply_chat_template` per message, so Qwen3's
  template emitted `<think>\n\n</think>\n\n` for every assistant turn instead of only the
  last, injecting 4 phantom scored tokens before each non-final reply. Exposure: 8.4% of
  no_robots_train, 13.0% of the 100-row held-out test set. Fixed in `be42437`; verified by
  an independent char-offset oracle (100/100 test rows, 902/902 train rows) and confirmed
  non-vacuous (13/13 flagged against the pre-fix code).

## G3-llama — loss-mask parity, Llama-3.1 (CPU)

- Command:
  ```
  cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
  /lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/fast/rollout/test_sft_loss_mask_parity_llama3.py \
      tests/fast/rollout/test_sft_loss_mask_parity.py
  ```
- Result: **PASSED** — `12 passed, 2 warnings` (7 llama-3 + 5 Qwen3, the sibling gate stays
  green throughout). Re-verified 2026-07-28 immediately before writing this entry:
  `12 passed, 2 warnings in 0.72s`. Exercises all **12** fixture rows (4 single-turn, 8
  multi-turn — 6 two-turn, 1 three-turn+system, 1 four-turn+system with 9 messages), and on
  every row Orbit's token ids and Orbit's mask are **exactly equal** to the oracle's — 0
  mismatches, not a tolerance.
- **Scope of this gate, stated precisely:** everything above runs the tokenizer with
  `LLAMA3_CHAT_TEMPLATE` (`orbit/utils/llama3_chat_template.py`) set via
  `ensure_llama3_chat_template`, i.e. it covers the **Python constant**. Production does not
  read that constant at all — `load_tokenizer` (`orbit/utils/processing_utils.py`) only ever
  sets `tokenizer.chat_template` from a **file** named by `--chat-template-path`. The file a
  real Llama-3 SFT run must pass is
  `orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja`, added alongside this fix
  wave and pinned byte-identical to the constant by
  `test_bundled_jinja_matches_the_pinned_python_constant`
  (`tests/fast/utils/test_llama3_chat_template.py`). So G3-llama's parity guarantee reaches
  production only through that second, separate test — not directly.
- Unlike G3, **no real bug was found in Orbit production code this time.** The gate is green
  on a correct implementation (`gen_multi_turn_loss_mask_llama3`, added by this same plan) and
  so, unlike the Qwen3 entry above, there is no exposure percentage to report — its value is
  **prospective**: it is a detector installed and proven to fire, not (yet) a detector that
  has fired on something real.
- What makes the oracle trustworthy: it locates every span by **character offsets** via
  `tokenizer(..., return_offsets_mapping=True)`, walking the rendered text with `str.index`
  for the header and terminator literals, and only maps back to token indices as the last
  step. The implementation instead scans the **token id** stream for the header subsequence
  and for the `<|eot_id|>` id. The two sides share no algorithm — this is a deliberate
  improvement over this same gate's Qwen3 sibling, whose oracle reused the implementation's
  own token scan and therefore could not have caught the Qwen3 bug above by construction; this
  one does not repeat that weakness.
- Non-vacuity, both directions observed on `orbit/utils/mask_utils.py`, reverted after each
  (md5 confirmed unchanged: `64fe0553bdd6758fa1e49cbd4af83fde`):
  - mutation (a) `start = header_pos + len(header_ids)` → `... - 1`: predicted the
    mask-POSITION assertion fails; **observed** — `test_scored_token_positions_match` failed
    (row 0, first divergence at token 64), plus the count and semantic-text tests (3 failed, 4
    passed). Note: the brief's own draft put the count and position assertions in one test
    function with count checked first, under which this mutation's position failure would
    never have been *evaluated* (the count assert fires first and aborts the function) — the
    brief's own mutation proof would not have bound as written. Splitting them into two
    separate test functions is what made both predictions independently observable. This is a
    brief-level defect, not a code defect; no ordinal count of how many times this plan has
    hit this class of defect is given here (an earlier draft of this section said "third,"
    Task 5's own report said "second" of its own instance — the two do not agree and neither
    was independently re-derived, so the count is dropped rather than guessed).
  - mutation (b) `end += 1` → `pass`: predicted the scored-COUNT assertion fails; **observed**
    — `test_scored_token_count_matches` failed (`orbit=561 oracle=562`), i.e. exactly one
    `<|eot_id|>` per turn goes unmasked (3 failed, 4 passed).
- **The gate's limits, stated as plainly as its successes:**
  - Independence is in the **algorithm**, not the inputs. Both sides read the same header
    (`<|start_header_id|>assistant<|end_header_id|>\n\n`) and terminator (`<|eot_id|>`)
    literals. A review-time injection confirmed the asymmetry: a wrong **header** literal (its
    trailing `\n\n` dropped) is caught only by `test_scored_text_is_exactly_the_assistant_turns`
    (a pure string/semantic anchor rebuilt from the fixture's own message content — 4 of the
    other tests pass blind); a wrong **terminator** literal is caught by **nothing** —
    implementation and oracle agree on the same wrong span by construction. The module
    docstring now states this and marks the semantic-anchor test do-not-delete.
  - `step_loss_mask=0` is outside the oracle's reach entirely (it has no notion of the flag);
    that path is covered only by Task 3's unit tests, not by this gate.
  - The fixture has **no** row with `tools=`/`tool_calls`, `ipython`/`tool` role, empty
    assistant content, or an assistant turn whose content literally contains the header or
    `<|eot_id|>` — all real branches of the pinned template, all untested in both directions.
  - The gate is currently green on a correct implementation — see the "no real bug" note
    above; a green run here proves the implementation matches the oracle on these 12 rows
    today, nothing about a bug it has caught.
- **Two findings worth carrying to any future token-to-character work on this tokenizer:**
  1. Llama-3's byte-level BPE produces **overlapping** `offset_mapping` spans on multi-byte
     characters — a CJK character split across tokens has each token report the *whole
     character's* span. Re-derived directly against the shipped fixture and tokenizer (all
     787 tokens of row 8, 0 in the other 11, both reproduce): **62** tokens share an identical
     `(a, b)` offset pair with at least one other token in the row (the strict reading of
     "overlapping" — two or more tokens claiming the same span); a broader reading — any token
     whose *solo* `tokenizer.decode([id])` contains the U+FFFD replacement character, i.e. half
     of a multi-byte sequence — gives **76**. Report the definition with the number; the two
     readings disagree and neither is "the" count. A per-token round-trip precondition
     (`text[a:b] == decode([id])`) is false under either reading; the oracle asserts no-gap
     coverage plus a whole-stream round-trip instead.
  2. `<|eom_id|>` is **unreachable today**: the template only emits it for a `tool_calls`
     message when `builtin_tools` is present in the Jinja context
     ([llama3_chat_template.py:108](/lustre/fast/fast/zqiu/orbit-iclr/orbit/orbit/utils/llama3_chat_template.py#L108)),
     and nothing in `orbit/` passes `builtin_tools` (`tools=` alone still yields `<|eot_id|>`,
     verified against the real implementation). If it ever becomes reachable, this gate is
     structurally incapable of catching a mis-handled `<|eom_id|>` boundary, because the
     implementation's terminator scan and the oracle's terminator search share the same
     `<|eot_id|>`-only literal.

## G4 — step-0 NLL agreement

Pass condition (design §7.2): (1) scored-token count and sample count must match Orbit-vs-HF
**exactly**; (2) the NLL delta must fall within the **measured** HF bf16-vs-fp32 spread on
this reference set, not a fixed tolerance.

| | token-weighted NLL | scored tokens | samples |
|---|---|---|---|
| HF bf16 | 3.592773 | 18472 | 100 |
| Orbit bf16 | 3.589597 | 18472 | 100 |
| HF fp32 | 3.585589 | 18472 | 100 |

- Counts match exactly: yes (18472 tokens, 100 samples, all three rows)
- HF bf16-vs-fp32 spread (measured): 0.007184 nats
- Orbit-vs-HF-bf16 delta: 0.003176 nats (inside the measured spread)
- Result: **G4 PASSED** (measured 2026-07-28)
- Logs: `logs/lora_regret/g4_hf.log`, `logs/lora_regret/g4_hf_fp32.log`;
  script `tools/lora_regret/g4_hf_nll.py --dtype {bfloat16,float32}`
- Consequence: the 0.0057-nat attention-vs-MLP gap is SMALLER than this cross-implementation
  spread. Within Orbit the offset is constant across arms and cancels, so internal claims
  hold, but absolute comparisons to michaelbzhu carry ~±0.005 nats of slack.

## Seed noise (prerequisite for G2's tolerance)

LoRA r256 all-modules at lr=2.5e-4, `LORA_A_INIT_METHOD=kaiming`, three seeds, run 2026-07-28.
Each run: 200 rollouts (6400 rows / batch 32 / 1 epoch), constant LR, held-out NLL on the
same 100 rows (18472 scored tokens, 100 samples — every run, no truncation).

| seed | test NLL (token-weighted) | sample-mean NLL |
|---|---|---|
| 0 | 1.900615 | 1.799106 |
| 1 | 1.900560 | 1.798425 |
| 2 | 1.898870 | 1.794420 |

- mean: 1.900015
- **sigma (sample std): 0.000992**
- max − min: 0.001745
- **G2 tolerance = 2·sigma = 0.00198 nats**
- Is sigma small enough to resolve the 0.0057 attn-vs-mlp gap? **Yes** — the gap is 5.7·sigma,
  and the tighter MLP-vs-all-modules separation (0.0034) is 3.4·sigma.
- Caveat: n=3 gives the variance estimate 2 degrees of freedom, so the 95% CI on sigma itself
  is roughly [0.0005, 0.006]. The point estimate clears the bar with 2x margin and the raw
  spread is well inside the effect size, but this is a weak estimate in the formal sense.
  If a layer-ablation result later lands within ~2x of sigma, add seeds on those arms before
  claiming it.
- Logs: `logs/seednoise_s0_20260728_173157.log`, `_s1_20260728_173207`, `_s2_20260728_173216`
- wandb: project `lora-without-regret`, runs `my5dv4us` (s0), `b4ktf2k8` (s2); s1's final
  sync threw a `BrokenPipeError` in wandb's atexit teardown — benign, after
  `progress rollout=199/199 completed=200/200 remaining=0`, and the log is authoritative.

**Run provenance (deviations from the plan's serial command, none affecting the numbers):**
the three seeds ran CONCURRENTLY on one H100. Independent processes, so seeds stay
statistically independent and sigma is unaffected; only wall clock and per-step timers are.
Two consequences to avoid repeating: all three resolved `SAVE_DIR` to the same
`orbit_ckpts/qwen3-4b-norobots-sft` (last writer wins — two adapter sets lost, final save
293s vs 97s in the smoke), and co-scheduled wandb teardown broke seed 1's socket. Task 12's
driver sets a per-arm `SAVE_DIR` (`tools/lora_regret/sweep.py:131`), so the sweep is exempt.

## OPEN ISSUE — reduction mismatch, resolve BEFORE G1/G2

Orbit reports a **global token-weighted** NLL. The oracle reports
`val_loss = total_loss / len(val_dataloader)` (`third_party/lora-without-regret/sft_lora.py:326`)
— an unweighted mean **over batches** of HF's per-batch token-mean, at batch size 2. That is a
third convention, and G2 compares the two numbers directly.

Evidence the mismatch is material: our two reductions BRACKET the published value.

```
sample-mean       1.799   <   published 1.8457   <   token-weighted 1.900
```

A batch-of-2 mean of token-means lands between those by construction, so the 0.055-nat gap
between our 1.900 and the published 1.8457 is plausibly bookkeeping, not training. The
training recipes otherwise match closely (verified by reading the oracle): `train[:6400]` /
`test[:100]`, 1 epoch, effective batch 2x16 = 32, constant-LR `torch.optim.AdamW`, bf16.
One genuine difference: torch `AdamW` defaults to `weight_decay=0.01`, our launcher sets
`WEIGHT_DECAY=0.0`.

Recommended fix before G1: accumulate `loss * n_tokens` alongside the existing sum in the
oracle's `eval()` and print BOTH numbers. Training is untouched, so G1 still produces the
published-style number, and G2 gains an apples-to-apples comparison.

## G1 — oracle reproduces the published number

| LR | test NLL (oracle reduction) | test NLL (token-weighted) |
|---|---|---|
| 1.2e-4 | | |
| 2.5e-4 | | |
| 5e-4 | | |

- Minimum: (expected ~1.8457)
- Result:

## G2 — Orbit parity with the oracle

| LR | oracle NLL | orbit NLL | delta |
|---|---|---|---|
| 1.2e-4 | | | |
| 2.5e-4 | | | |
| 5e-4 | | | |

- sigma (measured above): 0.000992
- Band: max |delta| must be <= **0.00198**
- argmin agreement: (oracle argmin LR vs orbit argmin LR — must match)
- Result:
- Note: both sides are bf16-against-bf16 (`sft_lora.py` hardcodes `torch_dtype=bfloat16`;
  the Orbit launcher sets `PRECISION_PROFILE="bf16"`). The G4 precision offset is present on
  both the before and after measurements and largely cancels, but state it in the write-up.

## G5 — `latex2sympy` import on the launcher chain (CPU, port gate)

Added by the port (`docs/superpowers/plans/2026-07-29-lora-without-regret-gap.md`, G5).
Measured before deciding, because the plan's own instruction was "measure before fixing".

- Command:
  ```
  cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
  /lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python \
      -c "import orbit.rollout.rm_hub.math_alignment"
  ```
- Result: **IMPORT OK**, with no `PYTHONPATH` set.
- Mechanism: `orbit/rollout/rm_hub/math_alignment.py` still imports `latex2sympy` at module
  scope, but calls `_ensure_vendored_math_eval_on_path()` first, which
  `sys.path.insert(0, ...)`s the in-tree vendored copy. The import resolves to
  `examples/peft_arena/backend/third_party/math_eval/latex2sympy/latex2sympy2.py`.
- **Decision: the old repo's `PYTHONPATH` shim must NOT be ported.** Two independent
  reasons. (a) It is unnecessary — the in-tree shim is `__file__`-relative and inserted at
  position 0, so it works regardless of cwd and wins over any installed package. (b) The
  directory it names, `examples/peft-arena/third_party/math_eval`, **does not exist in this
  repo** (here it is `peft_arena`, with a `backend/` level); exporting it would prepend a
  non-existent path and mislead the next reader.
- Secondary measurement, since the gap plan flagged it as open: `latex2sympy2-extended==1.11.0`
  (`pyproject.toml`) does **not** provide this import path. It installs as
  `latex2sympy2_extended`; a bare `import latex2sympy` still raises `ModuleNotFoundError`.
  So the vendored copy is load-bearing, not redundant with the declared dependency.
- Caveat: measured in the pinned-version proxy venv above, not the project's own `.venv`,
  which was still building. The conclusion is env-independent — the resolution is by
  `sys.path` order, and the only way an installed package could interfere is by claiming the
  top-level name `latex2sympy`, which `latex2sympy2-extended` does not.
