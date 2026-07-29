# Port the LoRA-Without-Regret work onto Sphere-AI-Lab/orbit

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Move the model-agnostic half of a 44-commit body of work from a stale repo onto
`Sphere-AI-Lab/orbit`, adapting it where upstream has diverged, and rebuild the launcher layer
that upstream deleted rather than porting it.

**Architecture:** The two repos have **unrelated histories** (different root commits, no merge
base), so nothing here is a rebase, a merge, or a cherry-pick — every task re-applies a change
by hand against upstream's current shape and re-runs the tests that pinned it. The old branch
is available locally as the git remote `old` for reading diffs.

**Tech stack:** Orbit on the CUDA-13.2 / torch-2.11 stack, Python 3.12, env at
`/fast/zqiu/orbit-iclr/orbit_env` built by `source env.sh && uv sync --extra allinone`.

## Where things are

| | |
|---|---|
| New repo | `/fast/zqiu/orbit-iclr/orbit`, branch `feat/lora-without-regret` off `feat/dev` (`8476b7f`) |
| Old repo | `/lustre/fast/fast/zqiu/orbit-infra/orbit`, branch `feat/lora-without-regret` (`6ad07e5`), 44 commits off `orbit-v0` (`dc1f554`) |
| Reading the old work | already added as remote `old`; use `git diff old/orbit-v0 old/feat/lora-without-regret -- <path>` |
| Env | `/fast/zqiu/orbit-iclr/orbit_env` |
| The campaign this serves | `docs/superpowers/plans/2026-07-28-lora-without-regret-experiments.md` (ported in Task 10) |

## Global Constraints

- **Interpreter:** `/fast/zqiu/orbit-iclr/orbit_env/bin/python` once the build finishes. Until
  then, CPU-only tests may be run with `/lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python`
  — but every task must be **re-verified against the new env** before it is called done.
- **Test command:** `<python> -m pytest <explicit paths> -q -p no:cacheprovider`. Always pass
  explicit paths: `norecursedirs` matches `tools` and `scripts` at any depth, so a bare
  `pytest tests/fast` silently skips whole directories. Never place a new test under
  `tests/fast/tools/`.
- **Do not port anything that touches `scripts/lib/peft.sh`, `rollout.sh`, or `train.sh`** —
  upstream deleted all three. The launcher layer is Task 9, written fresh.
- Upstream added a `response_only` mask type and gates `get_system_message_length()` on it in
  `MultiTurnLossMaskGenerator.__init__`. Every mask change must preserve that.
- Record the pre-existing failure list on `feat/dev` **before** Task 1 and hold it constant;
  any new failure is ours.
- Never `--no-verify`. Do not push. One commit per task unless a task says otherwise.

## Three upstream bugs this port fixes

Verified present on `feat/dev` at `8476b7f`. Each is a real defect in the new repo, not merely
a difference from ours, and each already has a test on the old branch.

1. **Qwen3 multi-turn loss mask** (`orbit/utils/mask_utils.py`, `gen_multi_turn_loss_mask_qwen3`)
   still renders each message in isolation against a synthetic prefix. Qwen3's template emits
   `<think>\n\n</think>\n\n` only for the final assistant turn, so every non-final turn gets
   4 phantom scored tokens. Measured exposure on the old branch's data: 8.4% of training rows,
   13.0% of the held-out set. **Task 1.**
2. **`lora_A_init_method` is unreachable** (`orbit/backends/megatron_utils/lora_utils.py:59`):
   `getattr(args, "lora_A_init_method", "xavier")` reads a capital-A attribute that no CLI
   argument ever sets, so the adapter always initializes `xavier` — ~2.4x the std of PEFT's
   convention, which shifts every measured optimal learning rate. **Task 6.**
3. **`latex2sympy` import** (`orbit/rollout/rm_hub/math_alignment.py:43`) is module-level and
   sits on the import chain of every launcher. Upstream now declares `latex2sympy2-extended`
   as a dependency, which may or may not provide the `latex2sympy.latex2sympy2` module path
   that line needs. **Task 9 verifies this against the new env before porting any fix** — the
   old repo's `PYTHONPATH` shim may be unnecessary here, and shipping it blindly would shadow
   a real installed package.

---

### Task 0: Baseline

- [ ] **Step 1: Record the pre-existing failure list.** With the new env if it is ready, else
      the cu13 venv, run the CPU suites this port will touch and write the failures to
      `.superpowers/sdd/<workspace>/baseline.md`. Include the exact command. Every later task
      compares against this list; a task that changes it has broken something.
- [ ] **Step 2: Confirm the env.** `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"` and record it in the baseline file. If the build has failed, stop and report — nothing below is trustworthy against a half-built env.

---

### Task 1: Qwen3 loss-mask fix + its parity gate

**Files:** modify `orbit/utils/mask_utils.py`; create
`tests/fast/rollout/test_sft_loss_mask_parity.py`, `tests/fast/fixtures/lora_regret/no_robots_sample.jsonl`.

The old implementation and its gate are at
`git show old/feat/lora-without-regret:orbit/utils/mask_utils.py` and
`...:tests/fast/rollout/test_sft_loss_mask_parity.py`.

- [ ] **Step 1: Port the gate first and watch it FAIL** against upstream's current qwen3
      method. That failure is the bug reproducing; record its exact output — it is the
      evidence that this task fixes something real rather than churning working code.
- [ ] **Step 2: Replace `gen_multi_turn_loss_mask_qwen3`** with the single-tokenization
      version, preserving upstream's `response_only` handling elsewhere in the file.
- [ ] **Step 3: Re-run — the gate passes.** Then verify the other three generators are
      byte-identical to upstream's (`git diff old/... -- ...` on each), so this task changed
      exactly one method.
- [ ] **Step 4: Commit.** `fix(mask): tokenize qwen3 conversations once to stop phantom think-block tokens`

---

### Task 2: The pinned Llama-3 chat template

**Files:** create `orbit/utils/llama3_chat_template.py`,
`orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja`,
`tests/fast/utils/test_llama3_chat_template.py`.

Straight port; upstream's `chat_template_utils/templates/` already exists and holds three
qwen3 `.jinja` files, so the bundled file slots in beside them.

- [ ] **Step 1: Port all three files** from `old/feat/lora-without-regret`.
- [ ] **Step 2: Verify byte-identity three ways** — the Python constant against
      `/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B-Instruct/tokenizer_config.json`, the
      `.jinja` against the constant, and both tests present and passing (8 tests).
- [ ] **Step 3: Confirm `autofix.py`'s rules still contain only qwen3 patterns** and that
      nothing globs the template directory — the bundled file must never be auto-applied over
      a checkpoint's own template.
- [ ] **Step 4: Commit.** `feat(mask): pin the llama-3.1 chat template and bundle it as a .jinja`

---

### Task 3: Llama-3 loss mask, dispatch, CLI

**Files:** modify `orbit/utils/mask_utils.py`, `orbit/utils/arguments.py`; create
`tests/fast/utils/test_llama3_loss_mask.py`.

- [ ] **Step 1: Port the 10 tests first**, run them, confirm they fail for the right reason.
- [ ] **Step 2: Port `gen_multi_turn_loss_mask_llama3`** directly after the qwen3 method.
- [ ] **Step 3: Add the dispatch branch**, placed so it cannot shadow upstream's `qwen`
      branch (whose nested `distill_qwen` check must still fire) or its `response_only`
      branch, and add `"llama3"` to `--loss-mask-type` choices. **The default stays `"qwen"`.**
- [ ] **Step 4: Re-run.** The strategy guard (exactly one `apply_chat_template` call carrying
      the whole conversation) and the `tools` guard must both pass — they are the two that
      took extra rounds to get right the first time.
- [ ] **Step 5: Commit.** `feat(mask): llama-3 multi-turn loss mask via single-pass span location`

---

### Task 4: Llama-3 parity gate + Tulu3 fixture

**Files:** create `tests/fast/rollout/test_sft_loss_mask_parity_llama3.py`,
`tests/fast/fixtures/lora_regret/llama3_sample.jsonl`; modify `tools/lora_regret/prepare_data.py`
(created in Task 8 — **run Task 8 first if you are executing out of order**).

- [ ] **Step 1: Port the fixture and the gate.**
- [ ] **Step 2: Re-run both mutation proofs** on the new repo's code, since the file they
      mutate is not byte-identical to the old one: `start = header_pos + len(header_ids) - 1`
      must fail the POSITION assertion, and `end += 1` → `pass` must fail the COUNT assertion.
      Revert after each and confirm a clean tree. A gate carried across repos without
      re-proving it is a gate nobody has seen fail here.
- [ ] **Step 3: Commit.** `test(mask): gate llama-3 loss mask against a char-offset HF oracle`

---

### Task 5: Held-out NLL eval

**Files:** create `orbit/utils/eval_nll.py`, `tests/fast/utils/test_eval_nll.py`; modify
`orbit/backends/megatron_utils/actor.py`, `orbit/ray/actor_group.py`,
`orbit/ray/train_actor.py`, `train.py`, `train_async.py`, `orbit/utils/arguments.py`.

**This is the riskiest task.** `actor.py` diverged by 125 lines between the repos. The
integration points all still exist upstream and are recognizably the same shape — `train.py`
has `async def train(args)`, `_timed_phase`, `should_run_periodic_action` and a local
`offload_train()`; `actor_group.py` has `_broadcast` and `async def train`; `actor.py` has
`train` and `compute_log_prob` — but every line number differs. Re-derive each insertion
point; do not port by line number.

Four properties the old implementation established the hard way, all of which must survive:

- The NLL eval must sit **between** `actor_model.train(...)` and `offload_train()`. The
  existing eval block runs after offload because it evaluates through SGLang, not the
  training model.
- Actors return `(sum_neg_logprob, n_tokens)` and reduce over the **DP group only**. TP/PP
  replicas hold identical samples (double-count) and DP shards hold different token counts, so
  a token-weighted mean is not the mean of per-rank means.
- Both DP all-reduces must sit **inside** the wake/sleep window: `sleep()` destroys process
  groups, and the monkeypatched `dist.all_reduce` then substitutes the WORLD group.
- Every held-out row must be scored, asserted by count. `get_data_iterator` floor-divides, so
  a 100-row set at global batch 32 silently becomes 96 and the metric starts depending on
  batch size.

- [ ] **Step 1: Port `eval_nll.py` and its tests** (the module is deliberately CPU-testable;
      `megatron_utils` cannot be imported without CUDA).
- [ ] **Step 2: Re-derive and apply the five wiring edits.**
- [ ] **Step 3: Keep `train_async.py`'s refusal** — the async loop overlaps next-rollout
      generation with current-rollout training, so "weights at the moment of measurement" has
      no meaning there. It must refuse the flag, not silently produce nothing.
- [ ] **Step 4: Run the tests; confirm the baseline failure list is unchanged.**
- [ ] **Step 5: Commit.** `feat(eval): forward-only held-out NLL eval for SFT runs`

---

### Task 6: PEFT init plumbing (upstream bug 2)

**Files:** modify `orbit/backends/megatron_utils/lora_utils.py`, `orbit/utils/arguments.py`;
create `tests/fast/utils/test_lora_a_init_method_reaches_adapter.py`; extend
`tests/fast/utils/test_peft_arguments.py` and `test_lora_arguments.py` if they exist upstream,
else create them.

- [ ] **Step 1: Expose `--lora-a-init-method`** with Megatron-Bridge's real vocabulary,
      `choices=[xavier, normal, kaiming, zero]`. **`uniform` is not a legal value** — Bridge
      routes Megatron parallel linears to `ParallelLinearAdapter`, whose `_get_init_fn`
      raises `NotImplementedError` on anything else. PEFT's `kaiming_uniform_(a=√5)` is
      spelled `kaiming` there, and it is provably the blog's convention: its bound is
      `√(6/((1+5)·d_in))` = `1/√d_in`.
- [ ] **Step 2: Fix the capital-A `getattr`** at `lora_utils.py:59` so the flag reaches
      `create_lora_instance`.
- [ ] **Step 3: Prove the test is not tautological** by reverting the getattr key and
      watching it fail with `assert 'xavier' == 'kaiming'`.
- [ ] **Step 4: Commit.** `fix(peft): make --lora-a-init-method reach the adapter`

---

### Task 7: Matched-parameter OFT solver

**Files:** create `orbit/utils/peft_param_match.py`, `tests/fast/utils/test_peft_param_match.py`.

Self-contained, no upstream conflict. Note `_find_nearest_divisor` lives on
`OFTRotationModule`, not `OFTLinear`. Verified block sizes at Qwen3-4B dims: r1→b=5 (ratio
exactly 1.0), r16→b=64, r256→b=1280 (loose, 1.249 — must be flagged as loosely matched in any
table).

- [ ] **Step 1: Port both files, run the tests, commit.**
      `feat(peft): matched-parameter OFT block-size solver`

---

### Task 8: Repro tooling and the vendored oracle

**Files:** create `tools/lora_regret/{__init__,arms,prepare_data,sweep,g4_hf_nll}.py`,
`scripts/lora_regret/fetch_models.sh`, `third_party/lora-without-regret/**`; create tests
under `tests/fast/utils/` (**not** `tests/fast/tools/` — see Global Constraints); modify
`.gitignore` for the vendored `.venv`.

- [ ] **Step 1: Port the tooling and the vendored oracle**, including the local patch that
      makes both oracle scripts report a token-weighted NLL beside their own batch-mean
      `val_loss`, and `README.orbit.md` recording that patch.
- [ ] **Step 2: Re-run the sweep dry-run** and confirm it still yields 82 arms at 2560/9728,
      partitioned 42 / 5 / 35 by `'^(full|lora)-'`, `'^oftscout-'`, `'^oft-'`. `--only` is a
      **regex, not a glob, and is not repeatable.**
- [ ] **Step 3: Commit.** `feat(repro): lora-without-regret tooling, matrix and vendored oracle`

---

### Task 9: The launcher layer, written fresh

**Files:** create a Llama-3 SFT launcher under `examples/sft/`; modify whichever of
`scripts/lib/{common,driver,launcher,paths,preflight,tool_env}.sh` the new structure requires.

Upstream deleted `peft.sh`, `rollout.sh` and `train.sh`, so **nothing is ported here** — read
the new library's conventions and write against them. What the old launcher needed, as
requirements rather than as code:

- Knobs, each defaulting to an exact restatement of the current argparse default so every
  existing launcher's command line stays byte-identical: `LOSS_TYPE`, `APPLY_CHAT_TEMPLATE`,
  `LOSS_MASK_TYPE`, `LABEL_KEY`, `SEED`, `ROLLOUT_SEED`.
- `LABEL_KEY` must use the **no-colon** form `${LABEL_KEY-label}`: SFT rows are
  `{"prompt": [...]}` with no label field, and the colon form would re-default an
  intentionally empty value back to `"label"` and crash the data loader.
- `ROLLOUT_SEED` defaults to 42 in the shared library (a true restatement — it also seeds
  SGLang generation, so changing it would silently move other people's RL runs) and is tied to
  `SEED` **in the repro launcher only**, so a seed sweep varies data order as well as init.
- `--debug-train-only` gives the rollout placement group 0 GPUs, but `train.py` still
  constructs `RolloutManager` unconditionally and its `__init__` loads the dataset — a pure
  SFT launcher is not exempt from the loader's contract.
- A Llama-3 run additionally needs `--chat-template-path` pointing at the `.jinja` from Task 2.
- [ ] **Step 1: Verify the `latex2sympy` import** against the new env before porting any fix
      (upstream bug 3): `python -c "import orbit.rollout.rm_hub.math_alignment"`. If it
      succeeds, the old repo's `PYTHONPATH` shim is unnecessary and must **not** be ported —
      it would shadow a properly installed package. If it fails, fix it the way upstream's
      dependency set implies, not by re-adding a vendored path.
- [ ] **Step 2: Write the launcher, smoke it on CPU as far as it goes, and commit.**

---

### Task 10: Documentation

**Files:** create `docs/superpowers/plans/2026-07-27-lora-without-regret-repro.md`,
`2026-07-28-lora-without-regret-experiments.md`, `2026-07-27-lora-without-regret-gate-log.md`,
`2026-07-28-llama3-loss-mask.md`, `docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md`.

- [ ] **Step 1: Port all five documents.**
- [ ] **Step 2: Correct every path and repo reference** — they name
      `/lustre/fast/fast/zqiu/orbit-infra/orbit` and the old venvs throughout, and the
      experiments plan's P-list must now reflect this repo's state.
- [ ] **Step 3: Add a provenance note** at the head of the experiments plan: which repo this
      came from, that the histories are unrelated, and which three upstream bugs the port
      fixed on the way in.
- [ ] **Step 4: Commit.** `docs(repro): port the lora-without-regret plans and gate log`

---

## Self-Review Notes

**Deliberately not ported:** the Qwen3/No-Robots SFT launcher and its README, the five knobs as
*code*, `tests/fast/scripts/test_sft_launcher_args.py`, `test_tool_env_latex2sympy_path.py`, and
the `test_orbit_launcher_contract.py` glob extension — all of them bind to the three deleted
`scripts/lib` files. Their requirements survive as Task 9's specification.

**Gates that must be re-earned, not assumed.** G3 and G3-llama both get re-run here (Tasks 1
and 4) because the code they gate is not byte-identical to what they gated before. G4 (step-0
NLL against HuggingFace) and the seed-noise σ were **measurements on Qwen3-4B/No Robots**, and
the campaign has since re-anchored to Llama-3.1-8B/Tulu3 — they do not transfer and must not be
copied forward as if they did. The gate log records them as history, not as current evidence.

**The riskiest assumption in this plan** is that Task 5's five wiring edits land correctly
against an `actor.py` that moved 125 lines. If that task needs more than two fix rounds, stop
and reconsider whether the eval belongs as a separate module invoked from the launcher rather
than threaded through the Ray dispatch chain.
