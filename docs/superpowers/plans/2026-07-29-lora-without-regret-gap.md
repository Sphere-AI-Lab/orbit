# LoRA-Without-Regret — the gap on Sphere-AI-Lab/orbit

What the campaign needs, minus what already ported cleanly. Derived from the original
implementation plan (`orbit-infra/orbit`, `docs/superpowers/plans/2026-07-27-lora-without-regret-repro.md`,
Tasks 1-14) re-scored against this repo at `feat/dev` (`8476b7f`).

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Steps use checkbox (`- [ ]`) syntax.

## Status: G1-G6 closed 2026-07-29

All six gates are applied on `feat/lora-without-regret`. CPU verification ran under
`/lustre/fast/fast/zqiu/clthegoat-cu13/.venv` (Python 3.12.13, transformers 4.57.1,
pytest 9.0.3 — the versions `pyproject.toml` pins); the project's own `.venv` was still
building. **373 passed, 0 failed**, against 5 collection errors that are pre-existing and
CUDA-related (verified identical with the branch stashed). No GPU ran.

Both knowingly-red `TestLogFormatPins` tests are now green **without being edited** —
`git diff` on `test_lora_regret_sweep.py` is empty.

Evidence recorded while applying, not asserted afterwards:

- **G1** — the Qwen3 gate was ported first and *watched fail*: token ids diverged at index
  12 (id 151667 = `<think>`, "Left contains 4 more items") and the scored-token count was
  `orbit=16 hf=12`. Both mutation proofs were then re-run **on this repo**:
  `start = header_pos + len(header_ids) - 1` failed the POSITION assertion at token 64
  (`'ĊĊ'`), and `end += 1` → `pass` failed it at token 626 (`<|eot_id|>`).
- **G2** — non-tautology of the LoRA-init gate proven by reverting the `getattr` key to the
  capital-A spelling, yielding exactly `assert 'xavier' == 'kaiming'`.
- **G3** — all 11 wiring tests passed on the first attempt; the stop rule (two fix rounds)
  was never approached.
- **G5** — measured, and the answer is *do not port*: see the entry below.

Three places where this repo's base forced a decision the plan could not have anticipated:

1. **G4's shared-lib knobs have no shared lib to live in.** This repo's launchers are
   standalone by contract (`test_sft_launchers_are_standalone`), so `LOSS_TYPE`,
   `APPLY_CHAT_TEMPLATE`, `LOSS_MASK_TYPE`, `LABEL_KEY`, `SEED` and `ROLLOUT_SEED` are
   declared in the launcher itself. The "every existing launcher stays byte-identical"
   requirement is then satisfied trivially rather than by careful restatement: no other
   launcher is touched. The two knobs that carry real meaning are still pinned by tests —
   the no-colon `${LABEL_KEY-}` form and the `ROLLOUT_SEED`→`SEED` tie.
2. **The repro launcher forgoes `--cuda-graph-scope full_iteration`**, unlike every sibling
   in `examples/sft/`. Megatron asserts `not check_for_nan_in_loss_and_grad` whenever
   `cuda_graph_impl=local` and `full_iteration` are combined, and offers only the negative
   `--no-check-for-nan-in-loss-and-grad` — there is no flag to turn the check back on. For
   an LR sweep that trade is backwards: a silently-NaN arm reads as a bad learning rate.
3. **`--training-mode sft` replaces the old `ORBIT_DEBUG_MODE=train`/`--debug-train-only`
   route**, which this repo has as a first-class mode. The plan's warning still applies
   verbatim, and was re-confirmed: `train.py` calls `create_rollout_manager()`
   unconditionally, so the launcher still passes `--prompt-data`.

## What is already here

Ported in `bea2ec8` and `161730c` — every file whose base was byte-identical upstream, or
which upstream did not have at all:

| Landed | Status |
|---|---|
| `orbit/utils/peft_param_match.py` + test | green, 26 tests with the template tests |
| `orbit/utils/llama3_chat_template.py`, `templates/llama3.1_pinned.jinja` + test | green |
| `orbit/backends/megatron_utils/lora_utils.py` (capital-A `getattr` fix) | applied; **inert until G2** gives it a CLI arg |
| `orbit/utils/eval_nll.py` | module only; **inert until G3** wires it |
| `tools/lora_regret/{arms,sweep,prepare_data,g4_hf_nll}.py` + tests | 82-arm matrix verified, 42/5/35 by selector |
| `third_party/lora-without-regret/**` (vendored oracle, incl. the token-weighted-NLL patch) | as-is |
| `tests/fast/fixtures/lora_regret/{no_robots,llama3}_sample.jsonl` | as-is |
| Gate log, experiments plan, llama3-loss-mask plan | paths still say `orbit-infra` — **G8** |

**Two tests are knowingly red** and stay red until G3:
`test_lora_regret_sweep.py::TestLogFormatPins::{test_template_matches_train_py_source,
test_phase_labels_match_train_py_source}`. They assert the sweep's NLL parser matches the
format string in `train.py`; that string arrives with the wiring. Their redness is the gap
being visible, not a defect — do not "fix" them by relaxing the assertion.

**Deliberately held back**, because their assertions are about code that does not exist yet:
`tests/fast/utils/test_eval_nll.py` (11 of its tests assert wiring), the llama3 loss-mask
tests, both parity gates, the SFT launcher and its tests. Each is named in the task that
should bring it in.

## The three upstream bugs, still open here

All verified present on `feat/dev`. Two need a gap task; one is half-landed.

1. **Qwen3 loss mask** — `gen_multi_turn_loss_mask_qwen3` still renders per message, injecting
   4 phantom `<think>` tokens before every non-final assistant turn. **G1.**
2. **`lora_A_init_method` unreachable** — the `lora_utils.py` half is ported; the CLI arg it
   reads does not exist. **G2.**
3. **`latex2sympy` module-level import** on every launcher's chain. Upstream now declares
   `latex2sympy2-extended`, which may or may not provide the `latex2sympy.latex2sympy2` path
   that import needs. **G5 measures before fixing.**

---

### G1: `mask_utils.py` — qwen3 fix, llama3 method, dispatch

Base diverged: upstream added a `response_only` type and gates `get_system_message_length()`
on it in `__init__`. Both changes must preserve that.

- [x] Port `tests/fast/rollout/test_sft_loss_mask_parity.py` **first** and watch it fail —
      that failure is upstream bug 1 reproducing here. Record the output.
- [x] Replace `gen_multi_turn_loss_mask_qwen3` with the single-tokenization version. Re-run:
      the gate passes. Verify the other three generators are byte-identical to upstream's.
- [x] Add `gen_multi_turn_loss_mask_llama3` after it, plus the dispatch branch — placed so it
      cannot shadow the `qwen` branch's nested `distill_qwen` check or the `response_only`
      branch. Bring in `tests/fast/utils/test_llama3_loss_mask.py` (10 tests).
- [x] Bring in `tests/fast/rollout/test_sft_loss_mask_parity_llama3.py` and **re-run both
      mutation proofs here** — `start = header_pos + len(header_ids) - 1` must fail the
      POSITION assertion, `end += 1` → `pass` must fail the COUNT assertion. A gate carried
      across repos without being seen to fail is not evidence.

### G2: `arguments.py` — three flag groups

Base diverged by 272 lines; re-derive every insertion point, do not port by line number.

- [x] `--loss-mask-type`: add `"llama3"` to choices. **Default stays `"qwen"`.** Upstream's
      dispatch already handles `response_only` but does not list it in choices — leave that
      alone; it is not ours.
- [x] `--lora-a-init-method`, `choices=[xavier, normal, kaiming, zero]`. **`uniform` is not
      legal** — Bridge routes Megatron parallel linears to `ParallelLinearAdapter`, whose
      `_get_init_fn` raises `NotImplementedError` otherwise; PEFT's `kaiming_uniform_(a=√5)`
      is spelled `kaiming` there and its bound is exactly `1/√d_in`, the blog's convention.
      Bring in `test_lora_a_init_method_reaches_adapter.py`; prove non-tautology by reverting
      the getattr key and watching `assert 'xavier' == 'kaiming'`.
- [x] The eval-NLL flags (`--eval-nll-data`, `--eval-nll-interval`, `--eval-nll-micro-batch-size`).
- [x] `tests/fast/utils/{test_peft_arguments,test_lora_arguments}.py` do not exist upstream —
      create them carrying only our assertions, not the old files wholesale.

### G3: eval-NLL wiring — the risky one

`actor.py` moved 125 lines. The integration points all still exist and are the same shape
(`train.py`: `async def train(args)`, `_timed_phase`, `should_run_periodic_action`, local
`offload_train()`; `actor_group.py`: `_broadcast`, `async def train`; `actor.py`: `train`,
`compute_log_prob`), but every line number differs.

Four properties, each learned the hard way, that must survive:

- The eval sits **between** `actor_model.train(...)` and `offload_train()`. The existing eval
  block runs after offload because it goes through SGLang, not the training model.
- Actors return `(sum_neg_logprob, n_tokens)`, reduced over the **DP group only** — TP/PP
  replicas hold identical samples, DP shards hold different token counts.
- Both DP all-reduces sit **inside** the wake/sleep window; `sleep()` destroys process groups
  and the monkeypatched `dist.all_reduce` then silently substitutes WORLD.
- Every held-out row is scored, asserted by count — `get_data_iterator` floor-divides, so 100
  rows at global batch 32 silently becomes 96 and the metric starts depending on batch size.

- [x] Re-derive and apply the five edits (`actor.py`, `actor_group.py`, `train_actor.py`,
      `train.py`, `train_async.py`). Keep `train_async.py`'s **refusal** — the async loop
      overlaps generation with training, so "weights at the moment of measurement" is undefined.
- [x] Restore `tests/fast/utils/test_eval_nll.py`; all 11 wiring tests must go green.
- [x] The two `TestLogFormatPins` tests must go green **without touching them**.
- [x] **Stop rule:** if this needs more than two fix rounds, reconsider whether the eval
      belongs as a standalone module invoked from the launcher rather than threaded through
      the Ray dispatch chain.

### G4: the launcher layer — written fresh, not ported

Upstream deleted `scripts/lib/{peft,rollout,train}.sh`. Nothing ports. Requirements, as
requirements:

- [x] Knobs, each defaulting to an exact restatement of the current argparse default so every
      existing launcher's command line stays byte-identical: `LOSS_TYPE`,
      `APPLY_CHAT_TEMPLATE`, `LOSS_MASK_TYPE`, `LABEL_KEY`, `SEED`, `ROLLOUT_SEED`.
- [x] `LABEL_KEY` uses the **no-colon** form `${LABEL_KEY-label}`. SFT rows are
      `{"prompt": [...]}` with no label field; the colon form re-defaults an intentionally
      empty value back to `"label"` and crashes the loader.
- [x] `ROLLOUT_SEED` defaults to 42 in the shared lib — a true restatement, since it also
      seeds SGLang generation and changing it would move other people's RL runs — and is tied
      to `SEED` **in the repro launcher only**, so a seed sweep varies data order as well as init.
- [x] The launcher is **Llama-3.1-8B / Tulu3**, not Qwen3/No-Robots: the campaign re-anchored.
      It needs `--chat-template-path orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja`,
      since Llama-3.1 base ships no chat template.
- [x] `--debug-train-only` gives the rollout placement group 0 GPUs, but `train.py` still
      constructs `RolloutManager` unconditionally and its `__init__` loads the dataset — a
      pure-SFT launcher is not exempt from the loader's contract.
- [x] `tests/fast/scripts/test_orbit_launcher_contract.py` does not exist upstream; if an
      equivalent does, extend its glob to see `examples/sft/`.

### G5: `latex2sympy` — measure before fixing

- [x] Against the built env: `python -c "import orbit.rollout.rm_hub.math_alignment"`. If it
      imports, the old repo's `PYTHONPATH` shim is **unnecessary and must not be ported** — it
      would shadow a properly installed package. If it fails, fix it the way upstream's
      dependency set implies. Either way, record the measurement.

### G6: docs

- [x] Port the old implementation plan and design spec (they read as modifications because
      they were committed to `orbit-v0` before the branch existed).
- [x] Correct every path in the ported docs — they name `orbit-infra/orbit` and the old venvs.
- [x] Add a provenance note to the experiments plan: which repo this came from, that the
      histories are unrelated, and which upstream bugs the port fixed on the way in.

---

## What does not transfer, and must not be treated as if it did

**Gate G4 (step-0 NLL vs HuggingFace) and the seed-noise σ = 0.000992 were measurements on
Qwen3-4B / No Robots.** The campaign has re-anchored to Llama-3.1-8B / Tulu3, and both numbers
are model- and dataset-specific. They stay in the gate log as history. E1-0 of the experiments
plan re-measures σ; G4's equivalent must be re-run against the Llama checkpoint before any
absolute claim rests on it.

**The campaign's own prerequisites are unchanged by this port** and remain open: P0 (FullFT on
8B needs ≥4 GPUs; this box has one), P3 (the DP>1 NLL reduction has never executed), P4
(Tulu3 / OpenThoughts3 / MATH / GSM8K prep), P5 (RL launcher). P1 is half-satisfied — the
Llama-3.1-8B base weights and Megatron checkpoint exist, but under the old repo's path.
