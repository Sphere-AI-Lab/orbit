# LoRA Without Regret Reproduction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the SFT and RL results of the "LoRA Without Regret" blog post inside Orbit, with matched-parameter OFT arms, producing an LR-tuned LoRA/FullFT reference line for future PEFT comparisons.

**Architecture:** Three separable layers. A throwaway *oracle* (michaelbzhu's HF/PEFT scripts, vendored) supplies known-good reference numbers. A *training* layer adds an SFT launcher that drives Orbit's existing-but-untested `sft_loss` path with `--debug-train-only` (no SGLang engine), plus a held-out-NLL eval hook built on the actor's existing `compute_log_prob` forward-only primitive. A *sweep* layer drives launcher invocations by env prefix and appends one JSONL record per arm, so plots regenerate without re-running anything.

**Tech Stack:** Bash launchers (`scripts/lib/` contract), Python 3.12, PyTorch, Megatron-Core + Megatron-Bridge (PEFT), Ray, SGLang (RL half only), pytest, uv, Weights & Biases.

**Spec:** `docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md`

## Global Constraints

- **Repo:** `orbit`, branch off `orbit-v0`. All paths below are relative to the orbit repo root unless absolute.
- **LoRA alpha is 32** in every SFT and RL arm (blog + michaelbzhu fixed value). Orbit's default is 16 — always set it explicitly.
- **LoRA A init must be `kaiming`** (= `kaiming_uniform_(a=sqrt(5))`, Bridge's `ParallelLinearAdapter._get_init_fn` vocabulary — `{xavier, normal, kaiming, zero}`), never the Orbit default `xavier` (`xavier_normal_` on this path). Measured std ratio xavier/kaiming = 2.4293 at d_in=2560, r=16; leaving the default biases every measured optimal LR by ~2.4x.
- **LR schedule is constant** — `LR_DECAY_STYLE=constant`, no warmup, no cooldown.
- **NaN checks stay on.** Never set `SKIP_NAN_CHECK_IN_LOSS_AND_GRAD=1` in a repro launcher.
- **Never push.** Commit locally only.
- **GPU work is operator-run.** Tasks tagged `[GPU — operator-run]` end by handing the user an exact command wrapped in `codexlog <name> ...`; the implementing agent does not launch them. Tasks tagged `[CPU]` are run by the implementing agent and must actually pass before the task is marked done.
- **Megatron imports need the CUDA env.** Any pytest that imports `megatron.core` requires `source load_cuda12_9_nccl_env.sh` (or the CUDA 13.2 equivalent) first; a bare venv raises a TE/cuBLAS symbol `OSError`.
- **Interpreter and PYTHONPATH (verified at setup — do not use `python` directly).** Orbit's own `.venv/` is empty: it has neither `ray` nor `torch`, so `python -m pytest` fails at conftest import. Use this exact invocation for every test command in this plan:

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
/lustre/fast/fast/zqiu/clthegoat-cu13/.venv/bin/python -m pytest <paths> -q -p no:cacheprovider
```

  No `PYTHONPATH` entry is needed on this repo. `orbit/rollout/rm_hub/math_alignment.py` still imports
  `latex2sympy` at module scope, but it calls `_ensure_vendored_math_eval_on_path()` first, which
  `sys.path.insert(0, ...)`s the vendored copy under `examples/peft_arena/backend/third_party/math_eval/`.
  Measured: `import orbit.rollout.rm_hub.math_alignment` succeeds with no shim. Do NOT re-add the old
  repo's `export PYTHONPATH=.../examples/peft-arena/third_party/math_eval:$PWD` — that directory does not
  exist here (note `peft_arena`/`backend`, not `peft-arena`), and the export would shadow the in-tree copy.
  Note also that `latex2sympy2-extended` does NOT satisfy this import: it installs as `latex2sympy2_extended`.
  Where a task step below writes `python -m pytest`, read it as the invocation above.
- **Pre-existing test failures — NOT caused by this plan; do not attempt to fix them.** Measured on `feat/lora-without-regret` at `dc1f554` with zero plan changes applied:
  - `tests/fast/utils/test_quantizer_ci.py` — collection error (`AssertionError` at import). Pass `--ignore=tests/fast/utils/test_quantizer_ci.py`.
  - `tests/fast/utils/test_tensor_backper.py::test_tensor_backuper_allows_filtered_placeholder_source` — FAILED.
  - `tests/fast/scripts/test_orbit_launcher_contract.py::test_active_launchers_are_thin_orbit_entrypoints` — FAILED: 16 existing launchers lack `ORBIT_ROOT` or do not source `scripts/lib/`. **New launchers added by Tasks 9 and 13 must still satisfy this test's per-launcher checks** (env-bash shebang, `set -euo pipefail`, `ORBIT_ROOT`, sources `scripts/lib/`, no `MILES_ROOT`/`miles_ckpts`/`miles-` strings) so they do not add to the failure list — the plan's launcher templates already do.
  - `tests/fast/scripts/test_shared_launcher_knobs.py::test_train_lib_exposes_dump_details_knob` — FAILED: `build_debug_args` reads `TRAIN_ENV_VARS` unbound under `set -u` when `apply_train_defaults` was not called first.

  Where a task step says "Expected: all pass", it means **all pass except the four above**. A task is done when its own new tests pass and this failure list has not grown. Verify with `git stash`/compare if unsure — never by "fixing" an unrelated pre-existing failure.
- **Bash snippet tests must call `apply_*_defaults` before `build_*_args`.** Under `set -euo pipefail` the `build_*` functions reference variables that only the matching `apply_*_defaults` sets; skipping it produces `unbound variable`, which is precisely how `test_train_lib_exposes_dump_details_knob` fails today. Follow the pattern in `tests/fast/scripts/test_shared_launcher_knobs.py`.
- **Launcher contract:** every knob is `${VAR:-default}`; leaf launchers set vars then `source "${ORBIT_ROOT}/scripts/lib/launcher.sh"` as the last line.
- **Reference numbers to beat** (michaelbzhu, Qwen3-4B, No Robots 6400, 200 steps): FullFT 1.8457 @ 2.5e-5; r256-all 1.8457 @ 2.5e-4; r256-attn 1.8548 @ 3.5e-4; r256-mlp 1.8491 @ 3.0e-4; r16-all 1.8473 @ 2.2e-4; r1-all 1.8489 @ 1.2e-4.

---

### Task 1: Vendor the oracle and prove it runs

**Files:**
- Create: `third_party/lora-without-regret/` (git clone, pinned)
- Create: `third_party/lora-without-regret/README.orbit.md`
- Modify: `.gitignore` (add `third_party/lora-without-regret/.venv/`)

**Interfaces:**
- Consumes: nothing.
- Produces: a runnable `uv run --directory third_party/lora-without-regret sft_lora.py` entrypoint, used by Tasks 6, 11, and 12.

- [ ] **Step 1: Clone the oracle at a pinned commit**

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p third_party
git clone https://github.com/michaelbzhu/lora-without-regret third_party/lora-without-regret
cd third_party/lora-without-regret
git rev-parse HEAD   # record this SHA — it goes in README.orbit.md
rm -rf .git          # vendor it; do not nest a git repo
```

- [ ] **Step 2: Create the provenance note**

Create `third_party/lora-without-regret/README.orbit.md`:

```markdown
# lora-without-regret (vendored oracle)

Snapshot of https://github.com/michaelbzhu/lora-without-regret at commit `<SHA from Step 1>`.

## Why this is here

This is a **throwaway validation oracle**, not part of Orbit's supported surface.
It exists to answer one question: does Orbit's `sft_loss` path produce the same
test NLL as a known-good HF/PEFT trainer? See
`docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md` §7 (gates G1/G2).

Once G2 passes, nothing in the reproduction depends on this directory.

## Running it

It has its own uv environment, deliberately isolated from Orbit's:

    uv sync --directory third_party/lora-without-regret
    CUDA_VISIBLE_DEVICES=0 uv run --directory third_party/lora-without-regret \
        sft_lora.py --lr 2.5e-4 --lora-rank 256 --lora-type all --no-wandb

## Do not

- Import from this directory in Orbit code.
- "Fix" it to match Orbit. If they disagree, that is the signal we are looking for.
```

- [ ] **Step 3: Ignore its virtualenv**

Append to `.gitignore`:

```
third_party/lora-without-regret/.venv/
```

- [ ] **Step 4: Verify the environment resolves [CPU]**

Run: `uv sync --directory third_party/lora-without-regret`
Expected: dependency resolution succeeds and `.venv/` is created. If resolution fails on a torch/CUDA pin, record the exact error in `README.orbit.md` under a "Known install deviations" heading and pin the working versions rather than silently editing `pyproject.toml`.

- [ ] **Step 5: Verify the entrypoint parses [CPU]**

Run: `uv run --directory third_party/lora-without-regret sft_lora.py --help`
Expected: usage text listing `--lr`, `--lora-rank`, `--lora-type`, `--no-wandb`. This confirms the oracle is invocable without a GPU.

- [ ] **Step 6: Commit**

```bash
git add third_party/lora-without-regret .gitignore
git commit -m "chore(repro): vendor lora-without-regret oracle"
```

---

### Task 2: Convert No Robots and competition_math to Orbit JSONL

**Files:**
- Create: `tools/lora_regret/__init__.py`
- Create: `tools/lora_regret/prepare_data.py`
- Test: `tests/fast/tools/test_lora_regret_prepare_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `prepare_no_robots(out_dir: Path, n_train: int = 6400, n_test: int = 100) -> tuple[Path, Path]` returning `(train_jsonl, test_jsonl)`.
  - `prepare_competition_math(out_dir: Path, n_train: int = 7500, val_start: int = 7500, val_end: int = 8500) -> tuple[Path, Path]`.
  - Output schema for No Robots, one JSON object per line:
    `{"prompt": [{"role": ..., "content": ...}, ...]}` — a **messages list**, because
    `sft_rollout.generate_rollout` passes `sample.prompt` straight into
    `MASK_GENERATOR.get_loss_mask(messages)` (`orbit/rollout/sft_rollout.py:47-50`).
  - Output schema for competition_math: `{"prompt": "<problem text>", "label": "<boxed answer>"}`
    matching `build_rollout_args`' hardcoded `--input-key prompt --label-key label`
    (`scripts/lib/rollout.sh:52-55`).

- [ ] **Step 1: Write the failing test**

Create `tests/fast/tools/test_lora_regret_prepare_data.py`:

```python
"""Schema tests for the LoRA-without-regret data preparation.

These run without network access by monkeypatching the dataset loader, so the
JSONL contract is pinned independently of HuggingFace availability.
"""

import json
from pathlib import Path

import pytest

from tools.lora_regret.prepare_data import (
    prepare_competition_math,
    prepare_no_robots,
    _write_jsonl,
)


def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [{"prompt": [{"role": "user", "content": "hi"}]}, {"prompt": "x", "label": "1"}]
    out = tmp_path / "out.jsonl"
    _write_jsonl(out, rows)
    read_back = [json.loads(line) for line in out.read_text().splitlines()]
    assert read_back == rows


def test_no_robots_emits_messages_list(tmp_path: Path, monkeypatch):
    fake = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
    monkeypatch.setattr(
        "tools.lora_regret.prepare_data._load_split",
        lambda name, split: fake,
    )
    train, test = prepare_no_robots(tmp_path, n_train=6, n_test=2)

    train_rows = [json.loads(l) for l in train.read_text().splitlines()]
    test_rows = [json.loads(l) for l in test.read_text().splitlines()]

    assert len(train_rows) == 6
    assert len(test_rows) == 2
    # The contract sft_rollout depends on: prompt is a list of message dicts.
    assert isinstance(train_rows[0]["prompt"], list)
    assert train_rows[0]["prompt"][0]["role"] == "user"
    assert set(train_rows[0].keys()) == {"prompt"}


def test_no_robots_train_test_are_disjoint_prefixes(tmp_path: Path, monkeypatch):
    fake = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
    calls = []

    def _fake_load(name, split):
        calls.append(split)
        return fake

    monkeypatch.setattr("tools.lora_regret.prepare_data._load_split", _fake_load)
    prepare_no_robots(tmp_path, n_train=6, n_test=2)
    # train comes from the train split, test from the test split — never a slice of one.
    assert calls == ["train", "test"]


def test_competition_math_emits_prompt_label(tmp_path: Path, monkeypatch):
    fake = [{"problem": f"p{i}", "solution": rf"x \boxed{{{i}}} y"} for i in range(20)]
    monkeypatch.setattr(
        "tools.lora_regret.prepare_data._load_split",
        lambda name, split: fake,
    )
    train, val = prepare_competition_math(tmp_path, n_train=5, val_start=5, val_end=8)

    train_rows = [json.loads(l) for l in train.read_text().splitlines()]
    val_rows = [json.loads(l) for l in val.read_text().splitlines()]

    assert len(train_rows) == 5
    assert len(val_rows) == 3
    assert train_rows[0] == {"prompt": "p0", "label": "0"}
    assert val_rows[0] == {"prompt": "p5", "label": "5"}


def test_competition_math_skips_rows_without_boxed_answer(tmp_path: Path, monkeypatch):
    fake = [
        {"problem": "good", "solution": r"\boxed{42}"},
        {"problem": "bad", "solution": "no answer here"},
        {"problem": "good2", "solution": r"\boxed{7}"},
    ]
    monkeypatch.setattr("tools.lora_regret.prepare_data._load_split", lambda name, split: fake)
    train, _ = prepare_competition_math(tmp_path, n_train=3, val_start=3, val_end=3)
    rows = [json.loads(l) for l in train.read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["good", "good2"]
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_prepare_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.lora_regret'`

- [ ] **Step 3: Write the implementation**

Create `tools/lora_regret/__init__.py` (empty file).

Create `tools/lora_regret/prepare_data.py`:

```python
"""Convert the LoRA-without-regret datasets into Orbit's JSONL format.

Two different schemas, because two different consumers:

* No Robots feeds the SFT path. `sft_rollout.generate_rollout` reads
  `sample.prompt` and hands it straight to `MultiTurnLossMaskGenerator.get_loss_mask`,
  which expects a list of `{"role", "content"}` dicts. So `prompt` stays a list and
  the launcher must NOT pass `--apply-chat-template`.
* competition_math feeds the RL path, which uses the standard
  `--input-key prompt --label-key label` contract with a string prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

NO_ROBOTS_REPO = "HuggingFaceH4/no_robots"
COMPETITION_MATH_REPO = "qwedsacf/competition_math"


def _load_split(name: str, split: str) -> list[dict[str, Any]]:
    """Load one split as a list of dicts. Split out so tests can monkeypatch it."""
    from datasets import load_dataset

    return list(load_dataset(name, split=split))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def extract_boxed(solution: str) -> str | None:
    """Return the contents of the last \\boxed{...} in a solution, or None.

    Uses arbitrary-depth brace counting, not a fixed-depth regex: competition_math
    solutions routinely nest braces two or more levels deep (e.g.
    ``\\boxed{\\frac{2\\sqrt{35}}{35}}``), which a single-level regex silently
    mis-drops as "no boxed answer".

    This mirrors ``last_boxed_only_string`` / ``remove_boxed`` in
    `orbit/rollout/rm_hub/math_utils.py` (the same extraction the RL reward path
    uses to grade rollouts, so ground-truth extraction here stays consistent with
    how answers are later graded) rather than importing them: importing that
    module pulls in the whole `orbit.rollout.rm_hub` package, which transitively
    imports torch and ray (~2400 extra modules, ~9s just to import) — heavy
    runtime deps this CPU-only, dependency-light dataset-prep script has no
    other reason to need.
    """
    idx = solution.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    depth = 0
    right_brace_idx = None
    while i < len(solution):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed = solution[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if boxed[: len(left)] != left or boxed[-1] != "}":
        return None
    return boxed[len(left) : -1].strip()


def prepare_no_robots(out_dir: Path, n_train: int = 6400, n_test: int = 100) -> tuple[Path, Path]:
    """Write No Robots train/test JSONL in Orbit chat format.

    Returns (train_path, test_path).
    """
    out_dir = Path(out_dir)
    train_raw = _load_split(NO_ROBOTS_REPO, "train")[:n_train]
    test_raw = _load_split(NO_ROBOTS_REPO, "test")[:n_test]

    def _convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"prompt": row["messages"]} for row in rows]

    train_path = _write_jsonl(out_dir / "no_robots_train.jsonl", _convert(train_raw))
    test_path = _write_jsonl(out_dir / "no_robots_test.jsonl", _convert(test_raw))
    return train_path, test_path


def prepare_competition_math(
    out_dir: Path,
    n_train: int = 7500,
    val_start: int = 7500,
    val_end: int = 8500,
) -> tuple[Path, Path]:
    """Write competition_math train/val JSONL in Orbit prompt/label format.

    Rows whose solution has no \\boxed{...} answer are dropped, since the math
    reward function cannot grade them.

    Returns (train_path, val_path).
    """
    out_dir = Path(out_dir)
    raw = _load_split(COMPETITION_MATH_REPO, "train")

    def _convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            answer = extract_boxed(row["solution"])
            if answer is None:
                continue
            out.append({"prompt": row["problem"], "label": answer})
        return out

    train_path = _write_jsonl(out_dir / "competition_math_train.jsonl", _convert(raw[:n_train]))
    val_path = _write_jsonl(out_dir / "competition_math_val.jsonl", _convert(raw[val_start:val_end]))
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/lustre/fast/fast/groups/ei-slm/data/lora_regret"),
        help="Directory to write the JSONL files into.",
    )
    parser.add_argument(
        "--dataset",
        choices=["no_robots", "competition_math", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.dataset in ("no_robots", "both"):
        train, test = prepare_no_robots(args.out_dir)
        print(f"no_robots: {train} {test}")
    if args.dataset in ("competition_math", "both"):
        train, val = prepare_competition_math(args.out_dir)
        print(f"competition_math: {train} {val}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_prepare_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Materialize the real datasets [CPU, needs network]**

Run:

```bash
codexlog lora_regret_prepare_data \
    python -m tools.lora_regret.prepare_data \
        --out-dir /lustre/fast/fast/groups/ei-slm/data/lora_regret
```

Expected: four files written. Verify counts:

```bash
wc -l /lustre/fast/fast/groups/ei-slm/data/lora_regret/*.jsonl
```

Expected: `no_robots_train.jsonl` = 6400, `no_robots_test.jsonl` = 100, `competition_math_train.jsonl` <= 7500, `competition_math_val.jsonl` <= 1000. The math counts are `<=` because rows without a boxed answer are dropped — record the actual numbers in the commit message. With the corrected (arbitrary-depth) `extract_boxed`, the observed counts on `qwedsacf/competition_math` are `competition_math_train.jsonl` = 7498, `competition_math_val.jsonl` = 1000 (only 2 of the first 7500 rows and 0 of rows `[7500:8500)` genuinely lack a `\boxed{...}` answer).

If the machine has no network access, hand the user this command and stop; the rest of the plan can proceed except Tasks 11-17.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret tests/fast/tools/test_lora_regret_prepare_data.py
git commit -m "feat(repro): dataset preparation for lora-without-regret"
```

---

### Task 3: Expose `--lora-a-init-method`

**Files:**
- Modify: `orbit/utils/arguments.py:21-27` (`_PEFT_LORA_DEFAULTS`) and `orbit/utils/arguments.py:1177+` (`add_lora_arguments`)
- Modify: `orbit/backends/megatron_utils/lora_utils.py:59`
- Modify: `scripts/lib/peft.sh` (`apply_peft_defaults`, `build_peft_args`)
- Test: `tests/fast/utils/test_lora_arguments.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - CLI flag `--lora-a-init-method` with `choices=["xavier", "normal", "kaiming", "zero"]`, `default="xavier"`, landing on `args.lora_a_init_method`.
  - Env knob `LORA_A_INIT_METHOD` (default `xavier`), emitted by `build_peft_args` only when `PEFT_METHOD=lora`.

**Why:** `lora_utils.py:59` reads `getattr(args, "lora_A_init_method", "xavier")` — an attribute no CLI argument ever sets, so Orbit is permanently on `xavier`. Orbit's Megatron linears (`linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`) never satisfy Bridge's `nn.Linear`/`te.Linear` fast-path check in `lora.py`, so they always land on `ParallelLinearAdapter`, whose `_get_init_fn` (`megatron/bridge/peft/utils.py:651-672`) accepts only `{xavier, normal, kaiming, zero}` — `"uniform"` is not a legal value here and raises `NotImplementedError`. The blog and HF PEFT use `kaiming_uniform_(a=sqrt(5))`, which this path calls `"kaiming"`; `"xavier"` on this path is `xavier_normal_`, not `xavier_uniform_`. Measured std ratio is 2.4293, and `init_A` enters both of the blog's two effective hyperparameter dimensions linearly, so the default biases every measured optimal LR by ~2.4x.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fast/utils/test_lora_arguments.py`:

```python
class TestLoraAInitMethod:
    def test_default_is_xavier(self):
        args = _make_args(lora_rank=16, target_modules="q_proj", lora_a_init_method="xavier")
        result = _apply_peft_validation(args)
        assert result.lora_a_init_method == "xavier"

    def test_kaiming_survives_normalization(self):
        args = _make_args(lora_rank=16, target_modules="q_proj", lora_a_init_method="kaiming")
        result = _apply_peft_validation(args)
        assert result.lora_a_init_method == "kaiming"

    def test_non_default_init_rejected_when_peft_method_is_oft(self):
        args = _make_args(
            peft_method="oft",
            lora_rank=0,
            oft_block_size=64,
            target_modules="q_proj",
            lora_a_init_method="kaiming",
        )
        with pytest.raises(AssertionError, match="LoRA flags require --peft-method lora"):
            _apply_peft_validation(args)

    def test_default_init_accepted_when_peft_method_is_oft(self):
        args = _make_args(
            peft_method="oft",
            lora_rank=0,
            oft_block_size=64,
            target_modules="q_proj",
            lora_a_init_method="xavier",
        )
        result = _apply_peft_validation(args)
        assert result.peft_method == "oft"
```

Also add `"lora_a_init_method": "xavier"` to the `args` dict inside `_make_args` at the top of that file, so every existing test keeps passing.

- [ ] **Step 2: Run the tests to verify they fail [CPU]**

Run: `python -m pytest tests/fast/utils/test_lora_arguments.py -v -k LoraAInitMethod`
Expected: `test_non_default_init_rejected_when_peft_method_is_oft` FAILS (no assertion raised) because `lora_a_init_method` is not yet in `_PEFT_LORA_DEFAULTS`.

- [ ] **Step 3: Add the default to the LoRA guard set**

In `orbit/utils/arguments.py`, extend `_PEFT_LORA_DEFAULTS`:

```python
_PEFT_LORA_DEFAULTS = {
    "lora_rank": 0,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_type": "lora",
    "lora_adapter_path": None,
    "lora_sync_from_tensor": False,
    "lora_a_init_method": "xavier",
}
```

This makes the existing guard at `arguments.py:140-142` reject `--lora-a-init-method kaiming` under `--peft-method oft`, matching how every other LoRA flag behaves.

- [ ] **Step 4: Add the CLI argument**

In `add_lora_arguments` in `orbit/utils/arguments.py`, immediately after the `--lora-dropout` block:

```python
            parser.add_argument(
                "--lora-a-init-method",
                type=str,
                default="xavier",
                choices=["xavier", "normal", "kaiming", "zero"],
                help="Initialization for LoRA matrix A, forwarded to Megatron-Bridge's "
                "ParallelLinearAdapter (the path Orbit's Megatron linears actually take). "
                "'kaiming' is kaiming_uniform_(a=sqrt(5)), matching HF PEFT and the "
                "LoRA-without-regret paper; 'xavier' is xavier_normal_ and is Bridge's "
                "default. These differ by ~2.4x in std, which shifts the optimal learning "
                "rate (default: xavier)",
            )
```

- [ ] **Step 5: Thread it into the adapter**

In `orbit/backends/megatron_utils/lora_utils.py`, change line 59 from
`lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),` to:

```python
        lora_A_init_method=getattr(args, "lora_a_init_method", "xavier"),
```

and extend the log line just below so the effective init is visible in every run log:

```python
    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, a_init={getattr(args, 'lora_a_init_method', 'xavier')}, "
        f"target_modules={target_modules}, exclude_modules={exclude_modules}"
    )
```

- [ ] **Step 6: Add the launcher knob**

In `scripts/lib/peft.sh`, add to `apply_peft_defaults` after the `LORA_DROPOUT` line:

```bash
    LORA_A_INIT_METHOD=${LORA_A_INIT_METHOD:-xavier}
```

and in `build_peft_args`, inside the `lora)` case, append to the `PEFT_ARGS` array:

```bash
                --lora-a-init-method "${LORA_A_INIT_METHOD}"
```

- [ ] **Step 7: Run the tests to verify they pass [CPU]**

Run: `python -m pytest tests/fast/utils/test_lora_arguments.py tests/fast/utils/test_peft_arguments.py -v`
Expected: all pass, including the pre-existing tests.

- [ ] **Step 8: Verify bash syntax [CPU]**

Run: `bash -n scripts/lib/peft.sh`
Expected: no output, exit 0.

- [ ] **Step 9: Commit**

```bash
git add orbit/utils/arguments.py orbit/backends/megatron_utils/lora_utils.py \
        scripts/lib/peft.sh tests/fast/utils/test_lora_arguments.py
git commit -m "feat(peft): expose --lora-a-init-method for PEFT-compatible LoRA init"
```

---

### Task 4: Add SFT launcher knobs (`LOSS_TYPE`, `APPLY_CHAT_TEMPLATE`, `LOSS_MASK_TYPE`)

**Files:**
- Modify: `scripts/lib/train.sh` (`apply_train_defaults`, `build_loss_args`)
- Modify: `scripts/lib/rollout.sh` (`apply_rollout_defaults`, `build_rollout_args`)
- Test: `tests/fast/scripts/test_sft_launcher_args.py`

**Interfaces:**
- Consumes: nothing.
- Produces three env knobs consumed by Task 7's launcher:
  - `LOSS_TYPE` (default `policy_loss`) → `--loss-type <value>`
  - `APPLY_CHAT_TEMPLATE` (default `1`) → emits `--apply-chat-template` only when truthy
  - `LOSS_MASK_TYPE` (default empty) → `--loss-mask-type <value>` only when non-empty

**Why:** `build_rollout_args` (`scripts/lib/rollout.sh:52-56`) hardcodes `--apply-chat-template`, which makes `data.py:215-221` run the prompt through `tokenizer.apply_chat_template` and return a **string**. `sft_rollout.generate_rollout` needs the raw **messages list** to pass to `get_loss_mask` (`orbit/rollout/sft_rollout.py:47-50`), so SFT must be able to turn that flag off. And there is currently no `LOSS_TYPE` knob anywhere in `scripts/lib/`, so `--loss-type sft_loss` is unreachable from a launcher. `LOSS_MASK_TYPE` must be `qwen3` for Qwen3 models — `MultiTurnLossMaskGenerator` accepts only `{"qwen", "qwen3", "distill_qwen"}` (`orbit/utils/mask_utils.py:134-144`).

- [ ] **Step 1: Write the failing test**

Create `tests/fast/scripts/test_sft_launcher_args.py`:

```python
"""Assert the SFT-related launcher knobs assemble the right CLI flags.

These source the shared lib functions in a subshell and print the resulting
arrays, so the bash contract is pinned without needing a GPU or Ray.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "scripts" / "lib"


def _run_snippet(env_assignments: str, snippet: str) -> str:
    script = f"""
set -euo pipefail
source "{LIB}/common.sh"
source "{LIB}/rollout.sh"
source "{LIB}/train.sh"
{env_assignments}
NUM_ROLLOUT=1
TRAIN_JSONL=/dev/null
{snippet}
"""
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=REPO
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_loss_type_defaults_to_policy_loss():
    out = _run_snippet("", 'apply_train_defaults; build_loss_args; printf "%s\\n" "${LOSS_ARGS[@]}"')
    assert "--loss-type" in out
    assert "policy_loss" in out


def test_loss_type_override_reaches_loss_args():
    out = _run_snippet(
        "LOSS_TYPE=sft_loss",
        'apply_train_defaults; build_loss_args; printf "%s\\n" "${LOSS_ARGS[@]}"',
    )
    assert "sft_loss" in out


def test_apply_chat_template_on_by_default():
    out = _run_snippet(
        "", 'apply_rollout_defaults; build_rollout_args; printf "%s\\n" "${ROLLOUT_ARGS[@]}"'
    )
    assert "--apply-chat-template" in out.splitlines()


def test_apply_chat_template_can_be_disabled():
    out = _run_snippet(
        "APPLY_CHAT_TEMPLATE=0",
        'apply_rollout_defaults; build_rollout_args; printf "%s\\n" "${ROLLOUT_ARGS[@]}"',
    )
    assert "--apply-chat-template" not in out.splitlines()


def test_loss_mask_type_omitted_when_unset():
    out = _run_snippet(
        "", 'apply_rollout_defaults; build_rollout_args; printf "%s\\n" "${ROLLOUT_ARGS[@]}"'
    )
    assert "--loss-mask-type" not in out.splitlines()


def test_loss_mask_type_emitted_when_set():
    out = _run_snippet(
        "LOSS_MASK_TYPE=qwen3",
        'apply_rollout_defaults; build_rollout_args; printf "%s\\n" "${ROLLOUT_ARGS[@]}"',
    )
    lines = out.splitlines()
    assert "--loss-mask-type" in lines
    assert lines[lines.index("--loss-mask-type") + 1] == "qwen3"
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/scripts/test_sft_launcher_args.py -v`
Expected: `test_loss_type_defaults_to_policy_loss`, `test_apply_chat_template_can_be_disabled`, and `test_loss_mask_type_emitted_when_set` FAIL.

- [ ] **Step 3: Add `LOSS_TYPE` to train.sh**

In `scripts/lib/train.sh`, add to `apply_train_defaults`:

```bash
    LOSS_TYPE=${LOSS_TYPE:-policy_loss}
```

and change `build_loss_args` so the array starts with the loss type:

```bash
build_loss_args() {
    LOSS_ARGS=(
       --loss-type "${LOSS_TYPE}"
       --calculate-per-token-loss
    )
    if [[ -n "${LOG_PROBS_CHUNK_SIZE}" ]]; then
        LOSS_ARGS+=(--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}")
    fi
}
```

- [ ] **Step 4: Make `--apply-chat-template` conditional and add `LOSS_MASK_TYPE`**

In `scripts/lib/rollout.sh`, add to `apply_rollout_defaults`:

```bash
    APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-1}
    LOSS_MASK_TYPE=${LOSS_MASK_TYPE:-}
```

In `build_rollout_args`, remove the bare `--apply-chat-template` line from the array literal so it reads:

```bash
    ROLLOUT_ARGS=(
       --prompt-data "${TRAIN_JSONL}"
       --input-key prompt
       --label-key label
       --rollout-shuffle
       --rm-type "${RM_TYPE}"
       --num-rollout "${NUM_ROLLOUT}"
       --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
       --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
       --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
       --rollout-temperature "${ROLLOUT_TEMPERATURE}"
       --global-batch-size "${GLOBAL_BATCH_SIZE}"
    )
    if is_true "${APPLY_CHAT_TEMPLATE}"; then
        ROLLOUT_ARGS+=(--apply-chat-template)
    fi
    if [[ -n "${LOSS_MASK_TYPE}" ]]; then
        ROLLOUT_ARGS+=(--loss-mask-type "${LOSS_MASK_TYPE}")
    fi
```

Leave the rest of the function unchanged.

- [ ] **Step 5: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/scripts/test_sft_launcher_args.py -v`
Expected: 6 passed

- [ ] **Step 6: Verify no existing launcher regressed [CPU]**

Run: `python -m pytest tests/fast/scripts/ tests/fast/test_megatron_cli_flags.py -v`
Expected: all pass. `--apply-chat-template` still defaults on, so every RL launcher behaves identically.

Also confirm `--loss-mask-type` is a real CLI flag before relying on it:

Run: `grep -n "loss-mask-type" orbit/utils/arguments.py`
Expected: a `parser.add_argument("--loss-mask-type", ...)` line. If it is absent, add it to the same argument group as `--rollout-function-path` with `type=str, default="qwen"`, and note the addition in the commit message.

- [ ] **Step 7: Commit**

```bash
git add scripts/lib/train.sh scripts/lib/rollout.sh tests/fast/scripts/test_sft_launcher_args.py
git commit -m "feat(launcher): add LOSS_TYPE, APPLY_CHAT_TEMPLATE, LOSS_MASK_TYPE knobs"
```

---

### Task 5: Matched-parameter OFT block-size solver

**Files:**
- Create: `orbit/utils/peft_param_match.py`
- Test: `tests/fast/utils/test_peft_param_match.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `lora_param_count(rank: int, d_in: int, d_out: int) -> int`
  - `oft_param_count(block_size: int, d_in: int, block_share: bool = False) -> int`
  - `nearest_divisor(n: int, target: int) -> int` — mirrors `OFTLinear._find_nearest_divisor`
  - `matched_oft_block_size(rank: int, d_in: int, d_out: int) -> int`
  - `match_report(rank: int, d_in: int, d_out: int) -> dict` with keys
    `{"rank", "d_in", "d_out", "ideal_block_size", "block_size", "lora_params", "oft_params", "ratio"}`

**Why:** OFT stores `oft_r` of shape `(d_in // b, b(b-1)/2)` (`megatron/bridge/peft/oft_layers.py:282-290`), so params = `d_in * (b - 1) / 2` against LoRA's `rank * (d_in + d_out)`. Equating gives `b = 1 + 2*rank*(d_in + d_out)/d_in`, which for a square matrix is `1 + 4*rank`. Bridge snaps a non-dividing `b` to the nearest divisor of `d_in` (`oft_layers.py:265-267`), so the realized parameter count differs from the ideal — and at rank 256 it differs a lot. `match_report` exists so every OFT arm can log its *realized* parameter count rather than being described as "matched" in prose.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_peft_param_match.py`:

```python
"""Parameter-count matching between LoRA rank and OFT block size.

The formulas here must track megatron/bridge/peft/oft_layers.py: oft_r has
shape (d_in // block_size, block_size * (block_size - 1) // 2), and a block
size that does not divide d_in is snapped to the nearest divisor.
"""

import pytest

from orbit.utils.peft_param_match import (
    lora_param_count,
    match_report,
    matched_oft_block_size,
    nearest_divisor,
    oft_param_count,
)


class TestParamCounts:
    def test_lora_param_count_square(self):
        assert lora_param_count(rank=16, d_in=2560, d_out=2560) == 16 * 5120

    def test_lora_param_count_rectangular(self):
        assert lora_param_count(rank=8, d_in=2560, d_out=9728) == 8 * (2560 + 9728)

    def test_oft_param_count_matches_bridge_shape(self):
        # (d_in // b) blocks, each b(b-1)/2 elements.
        d_in, b = 2560, 64
        assert oft_param_count(b, d_in) == (d_in // b) * (b * (b - 1) // 2)
        assert oft_param_count(b, d_in) == d_in * (b - 1) // 2

    def test_oft_param_count_block_share_ties_all_blocks(self):
        assert oft_param_count(64, 2560, block_share=True) == 64 * 63 // 2


class TestNearestDivisor:
    def test_exact_divisor_is_unchanged(self):
        assert nearest_divisor(2560, 64) == 64

    def test_snaps_below_when_closer(self):
        # divisors of 2560 around 70: 64 and 80 -> 64 is nearer
        assert nearest_divisor(2560, 70) == 64

    def test_snaps_above_when_closer(self):
        assert nearest_divisor(2560, 78) == 80

    def test_never_returns_zero(self):
        assert nearest_divisor(2560, 1) == 1


class TestMatchedBlockSize:
    def test_rank_1_square_is_exact(self):
        # b = 1 + 4*1 = 5, and 5 divides 2560
        b = matched_oft_block_size(rank=1, d_in=2560, d_out=2560)
        assert b == 5
        assert oft_param_count(b, 2560) == lora_param_count(1, 2560, 2560)

    def test_rank_16_square_snaps_to_64(self):
        # ideal b = 65, nearest divisor of 2560 is 64
        assert matched_oft_block_size(rank=16, d_in=2560, d_out=2560) == 64

    def test_rank_16_match_is_within_two_percent(self):
        rep = match_report(rank=16, d_in=2560, d_out=2560)
        assert 0.98 <= rep["ratio"] <= 1.02

    def test_rank_256_match_is_loose_and_reported_as_such(self):
        rep = match_report(rank=256, d_in=2560, d_out=2560)
        assert rep["ideal_block_size"] == 1025
        # The snap is far away, so the ratio must NOT be near 1 — and the
        # report must expose that rather than hide it.
        assert not (0.9 <= rep["ratio"] <= 1.1)

    def test_report_exposes_all_keys(self):
        rep = match_report(rank=16, d_in=2560, d_out=9728)
        assert set(rep) == {
            "rank", "d_in", "d_out", "ideal_block_size",
            "block_size", "lora_params", "oft_params", "ratio",
        }

    def test_block_size_cannot_exceed_d_in(self):
        b = matched_oft_block_size(rank=4096, d_in=2560, d_out=2560)
        assert b <= 2560
        assert 2560 % b == 0

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            matched_oft_block_size(rank=0, d_in=2560, d_out=2560)
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/utils/test_peft_param_match.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orbit.utils.peft_param_match'`

- [ ] **Step 3: Write the implementation**

Create `orbit/utils/peft_param_match.py`:

```python
"""Match OFT block size to LoRA rank by trainable-parameter count.

OFT stores one skew-symmetric vector per block:

    oft_r shape = (d_in // b, b * (b - 1) // 2)
    params      = d_in * (b - 1) / 2

LoRA stores two factors:

    params      = rank * (d_in + d_out)

Equating them gives the ideal block size

    b = 1 + 2 * rank * (d_in + d_out) / d_in

which reduces to b = 1 + 4 * rank when d_in == d_out. Megatron-Bridge snaps a
block size that does not divide d_in to the nearest divisor
(oft_layers.py:265-267), so the realized count can differ from the ideal --
`match_report` reports the realized ratio so no arm is described as "matched"
when it is not.
"""

from __future__ import annotations

import math


def lora_param_count(rank: int, d_in: int, d_out: int) -> int:
    """Trainable parameters in a LoRA adapter on a (d_out, d_in) linear."""
    return rank * (d_in + d_out)


def oft_param_count(block_size: int, d_in: int, block_share: bool = False) -> int:
    """Trainable parameters in an OFT adapter on a linear with `d_in` inputs."""
    per_block = block_size * (block_size - 1) // 2
    num_blocks = 1 if block_share else d_in // block_size
    return num_blocks * per_block


def nearest_divisor(n: int, target: int) -> int:
    """Nearest divisor of `n` to `target`. Mirrors OFTLinear._find_nearest_divisor."""
    best = 1
    best_dist = abs(target - 1)
    for i in range(1, math.isqrt(n) + 1):
        if n % i:
            continue
        for cand in (i, n // i):
            dist = abs(target - cand)
            if dist < best_dist:
                best, best_dist = cand, dist
    return best


def matched_oft_block_size(rank: int, d_in: int, d_out: int) -> int:
    """Block size whose OFT parameter count is closest to LoRA at `rank`."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    ideal = 1 + 2 * rank * (d_in + d_out) // d_in
    ideal = min(max(ideal, 1), d_in)
    if d_in % ideal == 0:
        return ideal
    return nearest_divisor(d_in, ideal)


def match_report(rank: int, d_in: int, d_out: int) -> dict:
    """Full accounting of a LoRA-to-OFT parameter match, for logging."""
    ideal = 1 + 2 * rank * (d_in + d_out) // d_in
    block_size = matched_oft_block_size(rank, d_in, d_out)
    lora_params = lora_param_count(rank, d_in, d_out)
    oft_params = oft_param_count(block_size, d_in)
    return {
        "rank": rank,
        "d_in": d_in,
        "d_out": d_out,
        "ideal_block_size": ideal,
        "block_size": block_size,
        "lora_params": lora_params,
        "oft_params": oft_params,
        "ratio": oft_params / lora_params,
    }
```

- [ ] **Step 4: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/utils/test_peft_param_match.py -v`
Expected: 14 passed

- [ ] **Step 5: Print the real match table for the plan record [CPU]**

Once Qwen3-4B is downloaded (Task 6), run:

```bash
python - <<'PY'
import json, pathlib
from orbit.utils.peft_param_match import match_report
cfg = json.loads(pathlib.Path(
    "/lustre/fast/fast/zqiu/hf_models/Qwen3-4B/config.json").read_text())
h, ffn = cfg["hidden_size"], cfg["intermediate_size"]
for rank in (1, 16, 256):
    print("attn(square) ", match_report(rank, h, h))
    print("mlp  (fc1)   ", match_report(rank, h, ffn))
PY
```

Record the output in the commit message. If `hidden_size` is not divisible by 5, the rank-1 arm will snap and its ratio will not be 1.0 — that is information, not a failure.

- [ ] **Step 6: Commit**

```bash
git add orbit/utils/peft_param_match.py tests/fast/utils/test_peft_param_match.py
git commit -m "feat(peft): matched-parameter OFT block-size solver"
```

---

### Task 6: Download models and convert to Megatron checkpoints [GPU — operator-run]

**Files:**
- Create: `scripts/lora_regret/fetch_models.sh`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `/lustre/fast/fast/zqiu/hf_models/Qwen3-4B` (base, not Instruct-2507)
  - `/lustre/fast/fast/zqiu/hf_models/Qwen3-1.7B`
  - `${ORBIT_ROOT}/checkpoints/Qwen3-4B_torch_dist`
  - `${ORBIT_ROOT}/checkpoints/Qwen3-1.7B_torch_dist`

  Tasks 7, 11-17 consume these paths as `HF_CKPT` and `MEGATRON_LOAD`.

- [ ] **Step 1: Write the fetch script**

Create `scripts/lora_regret/fetch_models.sh`:

```bash
#!/usr/bin/env bash
# Download the base models the LoRA-without-regret reproduction needs.
#
# NOTE: Qwen3-4B (base) is NOT the same as Qwen3-4B-Instruct-2507, which is
# already present locally. The reproduction anchors to the base model because
# michaelbzhu's published table does.
set -euo pipefail

HF_MODELS_DIR=${HF_MODELS_DIR:-/lustre/fast/fast/zqiu/hf_models}
mkdir -p "${HF_MODELS_DIR}"

for repo in Qwen/Qwen3-4B Qwen/Qwen3-1.7B; do
    name="${repo#*/}"
    dest="${HF_MODELS_DIR}/${name}"
    if [[ -f "${dest}/config.json" ]]; then
        echo "skip ${name}: already at ${dest}"
        continue
    fi
    echo "downloading ${repo} -> ${dest}"
    huggingface-cli download "${repo}" --local-dir "${dest}"
done

echo "done. hidden sizes:"
for name in Qwen3-4B Qwen3-1.7B; do
    python -c "import json,sys;c=json.load(open('${HF_MODELS_DIR}/${name}/config.json'));print('${name}', c['hidden_size'], c['intermediate_size'])"
done
```

- [ ] **Step 2: Verify bash syntax [CPU]**

Run: `bash -n scripts/lora_regret/fetch_models.sh && chmod +x scripts/lora_regret/fetch_models.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Read the conversion contract [CPU]**

Run: `cat scripts/conversion/README.md && grep -n "usage\|Usage\|HF_CKPT\|OUTPUT" scripts/conversion/convert_checkpoint.sh | head -30`
Expected: the exact env-var contract for `convert_checkpoint.sh`. Use whatever it documents in Step 4 rather than the placeholder form below — if the two disagree, the README wins and the plan is wrong.

- [ ] **Step 4: Hand the operator the download + conversion commands [GPU — operator-run]**

Give the user, and stop:

```bash
codexlog lora_regret_fetch_models \
    bash scripts/lora_regret/fetch_models.sh

codexlog lora_regret_convert_qwen3_4b \
    HF_CKPT=/lustre/fast/fast/zqiu/hf_models/Qwen3-4B \
    OUTPUT_DIR=$PWD/checkpoints/Qwen3-4B_torch_dist \
    bash scripts/conversion/convert_checkpoint.sh

codexlog lora_regret_convert_qwen3_1p7b \
    HF_CKPT=/lustre/fast/fast/zqiu/hf_models/Qwen3-1.7B \
    OUTPUT_DIR=$PWD/checkpoints/Qwen3-1.7B_torch_dist \
    bash scripts/conversion/convert_checkpoint.sh
```

Acceptance: each output directory contains `latest_checkpointed_iteration.txt` and an `iter_*` subdirectory.

- [ ] **Step 5: Commit**

```bash
git add scripts/lora_regret/fetch_models.sh
git commit -m "chore(repro): model fetch script for lora-without-regret"
```

---

### Task 7: Held-out NLL eval hook

**Files:**
- Create: `orbit/utils/eval_nll.py`
- Modify: `orbit/backends/megatron_utils/actor.py` (add `compute_eval_nll` method)
- Modify: `orbit/utils/arguments.py` (add `--eval-nll-data`, `--eval-nll-interval`)
- Modify: `scripts/lib/rollout.sh` (`EVAL_NLL_DATA`, `EVAL_NLL_INTERVAL` knobs)
- Test: `tests/fast/utils/test_eval_nll.py`

> Paths corrected during Task 7. The module is deliberately NOT under
> `orbit/backends/megatron_utils/`: that package's `__init__.py` does `import deep_ep`,
> which raises a bare `AssertionError` when `CUDA_HOME` is unset, so nothing under it can
> be imported or unit-tested on CPU. Do not move it back.

**Interfaces:**
- Consumes: `get_data_iterator(args, model, rollout_data) -> tuple[list[DataIterator], list[int]]` from `orbit/backends/training_utils/data.py:330`; `MegatronActor.compute_log_prob(data_iterator, num_microbatches, store_prefix="") -> dict[str, list[torch.Tensor]]` from `orbit/backends/megatron_utils/actor.py:394`.
- Produces:
  - `reduce_nll(log_probs: list[torch.Tensor], response_lengths: list[int]) -> float` — token-weighted mean negative log-likelihood.
  - `MegatronActor.compute_eval_nll(self, eval_batch) -> float`, logged to W&B as `eval/test_nll`.
  - CLI: `--eval-nll-data PATH` (default `None`), `--eval-nll-interval N` (default `0` = off).
  - Env: `EVAL_NLL_DATA`, `EVAL_NLL_INTERVAL`.

**Why:** every SFT figure in the blog is held-out NLL, and Orbit's only eval path is generation-based `math_eval` pass@k. The actor already has a forward-only primitive used for reference log-probs (`compute_ref_log_probs` → `compute_log_prob` → `forward_only`), so this reuses it rather than adding a second forward path. Token-weighted (not sample-weighted) reduction is required because the oracle's HF `Trainer` reports token-mean NLL; a sample-mean would disagree by a factor that varies with length distribution and would fail gate G2 for the wrong reason.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_eval_nll.py`:

```python
"""Reduction semantics for held-out NLL.

Kept free of megatron imports so it runs in a bare venv: only the pure
reduction function is exercised here. The actor wiring is covered by gate G4
(a real forward pass), not by a unit test.
"""

import math

import pytest
import torch

from orbit.utils.eval_nll import reduce_nll


def test_single_sample_mean_of_negatives():
    lp = [torch.tensor([-1.0, -2.0, -3.0])]
    assert reduce_nll(lp, [3]) == pytest.approx(2.0)


def test_token_weighted_not_sample_weighted():
    # Sample A: 1 token at -10. Sample B: 9 tokens at 0.
    # Token-weighted -> 10/10 = 1.0. Sample-weighted would be (10 + 0)/2 = 5.0.
    lp = [torch.tensor([-10.0]), torch.zeros(9)]
    assert reduce_nll(lp, [1, 9]) == pytest.approx(1.0)


def test_matches_naive_concatenation():
    lp = [torch.tensor([-0.5, -1.5]), torch.tensor([-2.5])]
    expected = -(torch.cat(lp).sum().item()) / 3
    assert reduce_nll(lp, [2, 1]) == pytest.approx(expected)


def test_empty_input_returns_nan():
    assert math.isnan(reduce_nll([], []))


def test_zero_total_length_returns_nan():
    assert math.isnan(reduce_nll([torch.tensor([])], [0]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        reduce_nll([torch.tensor([-1.0, -2.0])], [3])
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/utils/test_eval_nll.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orbit.utils.eval_nll'`

- [ ] **Step 3: Write the reduction module**

Create `orbit/utils/eval_nll.py`:

```python
"""Held-out negative log-likelihood evaluation for SFT runs.

Orbit's built-in eval generates completions and grades them. The
LoRA-without-regret SFT figures are all test NLL, so this adds a forward-only
eval that reuses the actor's existing `compute_log_prob` primitive.

The reduction is **token-weighted**: sum of negative log-probs over all
response tokens, divided by the total number of response tokens. This matches
what HF's Trainer reports, which matters because gate G2 compares the two
numbers directly.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def reduce_nll(log_probs: list[torch.Tensor], response_lengths: list[int]) -> float:
    """Token-weighted mean negative log-likelihood.

    Args:
        log_probs: one 1-D tensor of per-token log-probabilities per sample.
        response_lengths: number of scored tokens per sample.

    Returns:
        Mean NLL in nats, or NaN when there is nothing to score.

    Raises:
        ValueError: if a tensor's length disagrees with its declared response length.
    """
    if len(log_probs) != len(response_lengths):
        raise ValueError(
            f"length mismatch: {len(log_probs)} tensors vs {len(response_lengths)} lengths"
        )
    total_tokens = 0
    total_logprob = 0.0
    for tensor, length in zip(log_probs, response_lengths, strict=True):
        if tensor.numel() != length:
            raise ValueError(
                f"length mismatch: tensor has {tensor.numel()} elements, expected {length}"
            )
        total_tokens += length
        total_logprob += float(tensor.sum())
    if total_tokens == 0:
        return math.nan
    return -total_logprob / total_tokens
```

- [ ] **Step 4: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/utils/test_eval_nll.py -v`
Expected: 6 passed

- [ ] **Step 5: Add the actor method**

In `orbit/backends/megatron_utils/actor.py`, add this method immediately after `compute_log_prob` (which ends at line 409):

```python
    def compute_eval_nll(self, eval_batch: RolloutBatch) -> float:
        """Token-weighted held-out NLL under the current (adapted) model.

        Forward-only; no optimizer state is touched. Reuses compute_log_prob so
        there is exactly one forward path in the actor.
        """
        from orbit.utils.eval_nll import reduce_nll

        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, eval_batch)
        with timer("eval_nll"):
            out = self.compute_log_prob(data_iterator, num_microbatches, store_prefix="eval_")
        nll = reduce_nll(out["eval_log_probs"], eval_batch["response_lengths"])
        logger.info(f"eval_nll rollout_id={self._last_rollout_id} nll={nll:.6f}")
        return nll
```

If `self._last_rollout_id` does not exist on the actor, drop that field from the log line rather than inventing state — the sweep driver keys results by run, not by the log line.

Confirm the key name: `compute_log_prob` returns whatever `get_log_probs_and_entropy` stores under `store_prefix`. Run `grep -n "store_prefix" orbit/backends/training_utils/*.py | head` and use the actual key (it will be `f"{store_prefix}log_probs"`, i.e. `"eval_log_probs"`); if it differs, fix the line above to match.

- [ ] **Step 6: Add the CLI arguments**

In `orbit/utils/arguments.py`, in the same argument group as `--eval-interval`:

```python
            parser.add_argument(
                "--eval-nll-data",
                type=str,
                default=None,
                help="JSONL of held-out examples for forward-only NLL eval. Uses the same "
                "chat schema as --prompt-data under the SFT rollout function. "
                "Unset disables NLL eval (default: None)",
            )
            parser.add_argument(
                "--eval-nll-interval",
                type=int,
                default=0,
                help="Run held-out NLL eval every N rollout steps. 0 disables (default: 0)",
            )
```

- [ ] **Step 7: Add the launcher knobs**

In `scripts/lib/rollout.sh`, add to `apply_rollout_defaults`:

```bash
    EVAL_NLL_DATA=${EVAL_NLL_DATA:-}
    EVAL_NLL_INTERVAL=${EVAL_NLL_INTERVAL:-0}
```

and to `build_eval_args`, **outside** the `if ! is_true "${DISABLE_EVAL}"` block so NLL eval works with generation eval switched off:

```bash
    if [[ -n "${EVAL_NLL_DATA}" ]]; then
        EVAL_ARGS+=(--eval-nll-data "${EVAL_NLL_DATA}" --eval-nll-interval "${EVAL_NLL_INTERVAL}")
    fi
```

Note `EVAL_ARGS` is initialized to `()` at the top of `build_eval_args`, so appending after the `if` block is safe.

- [ ] **Step 8: Wire the call site**

In the training loop that calls `actor.train(...)` (find it with `grep -n "\.train(" train.py`), add this import at the top of `train.py`:

```python
from orbit.utils.eval_nll import build_eval_nll_batch
```

and add after the train call:

```python
    if args.eval_nll_interval and rollout_id % args.eval_nll_interval == 0:
        eval_batch = build_eval_nll_batch(args)
        nll = ray.get([actor.compute_eval_nll.remote(eval_batch) for actor in actors])[0]
        log_dict = {"eval/test_nll": nll, "rollout/step": rollout_id}
        if args.use_wandb:
            wandb.log(log_dict)
        print(f"eval/test_nll step={rollout_id} nll={nll:.6f}")
```

`build_eval_nll_batch(args)` loads `args.eval_nll_data` through the same tokenize-and-mask path `sft_rollout.generate_rollout` uses. Add it to `orbit/utils/eval_nll.py`:

```python
def build_eval_nll_batch(args):
    """Tokenize the held-out NLL set into a RolloutBatch-shaped dict.

    Mirrors sft_rollout.generate_rollout's masking so train and eval agree
    token-for-token; gate G3 asserts that agreement against the HF oracle.
    """
    import json
    from pathlib import Path

    from orbit.utils.mask_utils import MultiTurnLossMaskGenerator
    from orbit.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(
        args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
    )
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)

    tokens, loss_masks, response_lengths, total_lengths = [], [], [], []
    for line in Path(args.eval_nll_data).read_text(encoding="utf-8").splitlines():
        messages = json.loads(line)["prompt"]
        token_ids, loss_mask = mask_generator.get_loss_mask(messages)
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        tokens.append(token_ids)
        loss_masks.append(loss_mask[-response_length:])
        response_lengths.append(response_length)
        total_lengths.append(len(token_ids))

    return {
        "unconcat_tokens": tokens,
        "loss_masks": loss_masks,
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
    }
```

- [ ] **Step 9: Verify imports and syntax [CPU]**

Run:

```bash
python -m py_compile orbit/utils/eval_nll.py \
    orbit/backends/megatron_utils/actor.py orbit/utils/arguments.py train.py
bash -n scripts/lib/rollout.sh
python -m pytest tests/fast/utils/test_eval_nll.py tests/fast/scripts/test_sft_launcher_args.py -v
```

Expected: compile clean, bash clean, all tests pass.

- [ ] **Step 10: Verify argument parsing end-to-end [CPU, needs CUDA env]**

Run:

```bash
source load_cuda12_9_nccl_env.sh
python -c "
from orbit.utils.arguments import parse_args
import sys
sys.argv = ['x', '--eval-nll-data', '/tmp/x.jsonl', '--eval-nll-interval', '5']
" 2>&1 | tail -5
```

If `parse_args` requires many other flags, instead run `grep -n "eval-nll" orbit/utils/arguments.py` and confirm both flags are registered in a group reachable from the main parser. State which check you ran.

- [ ] **Step 11: Commit**

```bash
git add orbit/utils/eval_nll.py orbit/backends/megatron_utils/actor.py \
        orbit/utils/arguments.py scripts/lib/rollout.sh train.py \
        tests/fast/utils/test_eval_nll.py
git commit -m "feat(eval): forward-only held-out NLL eval for SFT runs"
```

---

### Task 8: Gate G3 — loss-mask parity against the oracle

**Files:**
- Create: `tests/fast/rollout/test_sft_loss_mask_parity.py`
- Create: `tests/fast/fixtures/lora_regret/no_robots_sample.jsonl`

**Interfaces:**
- Consumes: `MultiTurnLossMaskGenerator.get_loss_mask(messages, tools=None) -> tuple[list[int], list[int]]` (`orbit/utils/mask_utils.py:133`); the vendored oracle from Task 1.
- Produces: a permanent regression test that Orbit's supervised token mask equals the oracle's label mask.

**Why:** if Orbit masks a different token set than the oracle — one extra `<|im_end|>`, a system prompt scored or not — every NLL in the study shifts by a roughly constant amount. That is invisible in the *shape* of the loss curves, so the LR sweep would look perfectly healthy while being uncomparable to the published table. This is the cheapest gate and it runs on CPU.

- [ ] **Step 1: Create the fixture**

Create `tests/fast/fixtures/lora_regret/no_robots_sample.jsonl` with three lines covering single-turn, multi-turn, and a system prompt:

```jsonl
{"prompt": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
{"prompt": [{"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}, {"role": "user", "content": "Another?"}, {"role": "assistant", "content": "Green."}]}
{"prompt": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]}
```

- [ ] **Step 2: Write the parity test**

Create `tests/fast/rollout/test_sft_loss_mask_parity.py`:

```python
"""Gate G3: Orbit's SFT loss mask must equal the HF oracle's label mask.

A constant offset between the two masks shifts every NLL in the reproduction
by a constant, which is invisible in the shape of a loss curve but makes the
numbers uncomparable to michaelbzhu's published table.

Skipped unless the Qwen3-4B tokenizer is present locally.
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

    Built by tokenizing the conversation prefix-by-prefix, so the scored span
    for each assistant turn is exactly the tokens that turn contributes.
    """
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    mask = [0] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True
        )
        upto = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )
        for j in range(len(prefix), len(upto)):
            mask[j] = 1
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
    scored = tokenizer.decode([t for t, m in zip(ids, mask) if m])
    assert "terse" not in scored
```

- [ ] **Step 3: Run the test [CPU]**

Run: `python -m pytest tests/fast/rollout/test_sft_loss_mask_parity.py -v`

Expected, in order of preference:
1. **PASS** — G3 clears; proceed.
2. **SKIPPED** — Qwen3-4B not yet downloaded. Acceptable only before Task 6; re-run after and record the result.
3. **FAIL on positions but not count** — an off-by-one in where the assistant span starts. Fix `_hf_label_mask`'s `add_generation_prompt` handling first, since the reference is the more likely suspect; only then suspect `mask_utils`.
4. **FAIL on count** — a genuine masking disagreement. **Stop the plan and investigate.** Do not proceed to the sweep. Record the per-conversation counts in the task notes.

- [ ] **Step 4: Commit**

```bash
git add tests/fast/rollout/test_sft_loss_mask_parity.py tests/fast/fixtures/lora_regret
git commit -m "test(sft): gate G3 loss-mask parity against HF oracle"
```

**Remediation (fix round 1):** G3 initially ran (not skipped) and returned **FAIL on
count** for the multi-turn fixture: Orbit scored 16 tokens, the oracle 12. Root cause:
`gen_multi_turn_loss_mask_qwen3` (`orbit/utils/mask_utils.py`) rendered each message in
isolation via a synthetic single-user prefix; Qwen3's chat template only wraps the
*final* assistant turn (the one after the last real user turn in the whole conversation)
in an empty `<think>\n\n</think>\n\n` block, and in isolation every message trivially
looks final, so Orbit scored that spurious 4-token block on every non-final assistant
turn too. `_hf_label_mask` (the task brief's own reference) had the identical
truncation flaw and was fixed first, to a single-tokenization + boundary-scan
construction, before concluding Orbit was actually wrong. Fix: `mask_utils.py::gen_multi_turn_loss_mask_qwen3`
was rewritten to tokenize the whole conversation once and locate assistant-turn spans
within that single tokenization (same approach as the corrected reference), preserving
`gen_token_length`, `tools` pass-through, system-message handling, and the
`step_loss_mask` override. Verified against the corrected reference on all 100 rows of
`no_robots_test.jsonl` and an 802-row sample of `no_robots_train.jsonl` (random 300 +
all 535 multi-turn rows in the file) — 0 disagreements. G3 now PASSES, plus two added
regression tests (no `<think>` wrapper on non-final turns; `step_loss_mask=0` still
zeroes a turn). Full record: `.superpowers/sdd/2026-07-27-lora-without-regret-repro/task-8-report.md`.

---

### Task 9: SFT launcher

**Files:**
- Create: `examples/sft/run-qwen3-4b-norobots-sft.sh`
- Create: `examples/sft/README.md`

**Interfaces:**
- Consumes: `LOSS_TYPE`, `APPLY_CHAT_TEMPLATE`, `LOSS_MASK_TYPE` (Task 4); `LORA_A_INIT_METHOD` (Task 3); `EVAL_NLL_DATA`, `EVAL_NLL_INTERVAL` (Task 7); the JSONL from Task 2; the checkpoints from Task 6.
- Produces: a launcher the sweep driver (Task 10) invokes with an env prefix. Overridable knobs the driver relies on: `PEFT_METHOD`, `LORA_RANK`, `TARGET_MODULES`, `OFT_BLOCK_SIZE`, `LR`, `SEED`, `WANDB_GROUP`, `SAVE_DIR`, `RUN_LOG`.

- [ ] **Step 1: Write the launcher**

Create `examples/sft/run-qwen3-4b-norobots-sft.sh`:

```bash
#!/usr/bin/env bash
# Qwen3-4B base + SFT on No Robots (6400 examples), for the LoRA-without-regret
# reproduction. See docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md
#
# This launcher runs the TRAIN phase only: ORBIT_DEBUG_MODE=train skips SGLang
# engine allocation entirely, and the "rollout" function just tokenizes and
# builds loss masks. No generation happens.
#
# Full fine-tuning:   PEFT_METHOD=none  LR=2.5e-5
# LoRA r256 all:      PEFT_METHOD=lora  LORA_RANK=256 LR=2.5e-4
# LoRA r256 attn:     PEFT_METHOD=lora  LORA_RANK=256 TARGET_MODULES=linear_qkv,linear_proj
# LoRA r256 mlp:      PEFT_METHOD=lora  LORA_RANK=256 TARGET_MODULES=linear_fc1,linear_fc2
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-"run_qwen3_4b_norobots_sft"}
PRECISION_PROFILE="bf16"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}

# === Model spec ===
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-4B.sh"}

# === SFT wiring: train-only, no rollout engine ===
ORBIT_DEBUG_MODE=${ORBIT_DEBUG_MODE:-train}
ROLLOUT_FUNCTION_PATH=${ROLLOUT_FUNCTION_PATH:-orbit.rollout.sft_rollout.generate_rollout}
LOSS_TYPE=${LOSS_TYPE:-sft_loss}
# sft_rollout needs the raw messages list, not a rendered chat string.
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}
LOSS_MASK_TYPE=${LOSS_MASK_TYPE:-qwen3}
# Generation-based eval is meaningless here; NLL eval replaces it.
DISABLE_EVAL=${DISABLE_EVAL:-1}

# === Data ===
DATA_DIR=${DATA_DIR:-/lustre/fast/fast/groups/ei-slm/data/lora_regret}
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_DIR}/no_robots_train.jsonl}
TEST_JSONL=${TEST_JSONL:-${DATA_DIR}/no_robots_test.jsonl}
EVAL_NLL_DATA=${EVAL_NLL_DATA:-${TEST_JSONL}}
EVAL_NLL_INTERVAL=${EVAL_NLL_INTERVAL:-10}

# === Checkpoints ===
HF_CKPT=${HF_CKPT:-/lustre/fast/fast/zqiu/hf_models/Qwen3-4B}
MEGATRON_LOAD=${MEGATRON_LOAD:-"${ORBIT_ROOT}/checkpoints/Qwen3-4B_torch_dist"}
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/qwen3-4b-norobots-sft}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}   # effectively "never" for a 200-step run
RUN_LOG=${RUN_LOG:-"${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"}

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

# === Training schedule ===
# 6400 examples / batch 32 = 200 optimizer steps over one epoch.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# === Optimizer ===
# Constant LR, no warmup, no cooldown -- the blog's protocol.
LR=${LR:-2.5e-4}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}

# === PEFT ===
PEFT_METHOD=${PEFT_METHOD:-lora}
TARGET_MODULES=${TARGET_MODULES:-linear_qkv,linear_proj,linear_fc1,linear_fc2}
LORA_RANK=${LORA_RANK:-256}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
# PEFT-compatible init. Do NOT leave this at Orbit's xavier default: the two
# differ by ~2.4x in std, which shifts the measured optimal LR.
LORA_A_INIT_METHOD=${LORA_A_INIT_METHOD:-kaiming}

# === RL knobs that must stay off ===
USE_KL_LOSS=${USE_KL_LOSS:-0}

# === Safety ===
# Keep NaN checks on -- a silently-NaN arm would look like a bad LR.
SKIP_NAN_CHECK_IN_LOSS_AND_GRAD=${SKIP_NAN_CHECK_IN_LOSS_AND_GRAD:-0}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-lora-without-regret}
WANDB_GROUP=${WANDB_GROUP:-sft-qwen3-4b-norobots}

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
```

- [ ] **Step 2: Handle `PEFT_METHOD=none`**

`build_peft_args` (`scripts/lib/peft.sh:24-58`) exits 2 on any `PEFT_METHOD` other than `oft` or `lora`, so full fine-tuning cannot be launched. Add a `none` case:

```bash
        none)
            PEFT_ARGS=(--peft-method none)
            ;;
```

immediately before the `*)` catch-all, and update the error message to `expected 'none', 'oft' or 'lora'`.

Then add a test to `tests/fast/scripts/test_sft_launcher_args.py`:

```python
def test_peft_method_none_emits_only_the_method_flag():
    out = _run_snippet(
        "PEFT_METHOD=none",
        'source "%s/peft.sh"; apply_peft_defaults; PEFT_METHOD=none; '
        'build_peft_args; printf "%%s\\n" "${PEFT_ARGS[@]}"' % LIB,
    )
    lines = out.splitlines()
    assert lines == ["--peft-method", "none"]
```

- [ ] **Step 3: Write the README**

Create `examples/sft/README.md`:

```markdown
# SFT launchers

Supervised fine-tuning recipes. Unlike the RL launchers, these run the train
phase only — `ORBIT_DEBUG_MODE=train` skips SGLang engine allocation, and the
rollout function (`orbit.rollout.sft_rollout.generate_rollout`) only tokenizes
and builds loss masks.

## run-qwen3-4b-norobots-sft.sh

Qwen3-4B base on No Robots (6400 examples, 200 steps at batch 32), for the
LoRA-without-regret reproduction.

Arms:

    # full fine-tuning
    PEFT_METHOD=none LR=2.5e-5 bash examples/sft/run-qwen3-4b-norobots-sft.sh

    # LoRA rank 256, all modules
    PEFT_METHOD=lora LORA_RANK=256 LR=2.5e-4 \
        bash examples/sft/run-qwen3-4b-norobots-sft.sh

    # LoRA rank 256, attention only
    PEFT_METHOD=lora LORA_RANK=256 LR=3.5e-4 \
        TARGET_MODULES=linear_qkv,linear_proj \
        bash examples/sft/run-qwen3-4b-norobots-sft.sh

    # matched OFT (block size from orbit.utils.peft_param_match)
    PEFT_METHOD=oft OFT_BLOCK_SIZE=64 LR=1e-4 \
        bash examples/sft/run-qwen3-4b-norobots-sft.sh

Smoke test (one step, no W&B):

    TOTAL_EPOCHS=1 ROLLOUT_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=2 NUM_ROLLOUT=1 \
    EVAL_NLL_INTERVAL=1 ENABLE_WANDB=0 \
        bash examples/sft/run-qwen3-4b-norobots-sft.sh

`EVAL_NLL_INTERVAL=0` is only valid when `EVAL_NLL_DATA` is genuinely unset;
this launcher defaults `EVAL_NLL_DATA` to `TEST_JSONL`, and `${VAR:-default}`
treats an empty override (`EVAL_NLL_DATA=`) as unset too, so there is no way
to turn NLL eval off here short of pointing `TEST_JSONL` elsewhere. The smoke
above uses `EVAL_NLL_INTERVAL=1` instead, which also exercises the NLL path.

`LORA_A_INIT_METHOD` defaults to `kaiming` here, not Orbit's global `xavier`
default. The two differ by ~2.4x in std, which shifts the optimal learning
rate — see the design doc §3.2.
```

- [ ] **Step 4: Verify syntax and tests [CPU]**

Run:

```bash
bash -n examples/sft/run-qwen3-4b-norobots-sft.sh scripts/lib/peft.sh
chmod +x examples/sft/run-qwen3-4b-norobots-sft.sh
python -m pytest tests/fast/scripts/ -v
```

Expected: clean syntax, all tests pass.

Also check whether `orbit_plugins/model_args/qwen3-4B.sh` exists:

Run: `ls orbit_plugins/model_args/ | grep -i qwen3`

**Verified during execution — this branch will not fire.** `orbit_plugins/model_args/qwen3-4B.sh` and `qwen3-1.7B.sh` both already exist, and were cross-checked field-by-field against the downloaded configs (layers, hidden, FFN, heads, kv-groups, rotary base, vocab, kv-channels). No edits needed.

**Rotary-base correction.** An earlier draft of this plan claimed base Qwen3 uses `rope_theta=1e4`. That is wrong. Measured from the downloaded configs: **Qwen3-4B base and Qwen3-1.7B are both `rope_theta = 1e6`**; only `Qwen3-4B-Instruct-2507` uses `5e6`. The real trap runs the other way — several existing launchers default `MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}` because they target Instruct-2507, and two of them source the *base* `qwen3-4B.sh` and override its correct 1e6 default. **A base-model launcher copied from those templates must NOT carry `MODEL_ARGS_ROTARY_BASE=5000000` — leave the knob unset so `qwen3-4B.sh`'s own 1e6 applies.**

- [ ] **Step 5: Hand the operator the smoke test [GPU — operator-run]**

```bash
codexlog lora_regret_sft_smoke \
    TOTAL_EPOCHS=1 ROLLOUT_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=2 NUM_ROLLOUT=1 \
    EVAL_NLL_INTERVAL=1 ENABLE_WANDB=0 \
    bash examples/sft/run-qwen3-4b-norobots-sft.sh
```

Acceptance: the run completes one optimizer step with a finite loss, and the log contains `Created LoRA: rank=256, alpha=32, ... a_init=kaiming`. No rollout engine receives any GPUs in `--debug-train-only` mode (`orbit/ray/placement_group.py:85-88`), so no SGLang engine process should appear in the log. The `RolloutManager` actor itself is still constructed unconditionally (`train.py:97`) and its `__init__` still loads `TRAIN_JSONL` through the normal dataset-loader contract, so this is a GPU-free / engine-free path, not a rollout-manager-free one.

- [ ] **Step 6: Commit**

```bash
git add examples/sft scripts/lib/peft.sh tests/fast/scripts/test_sft_launcher_args.py
git commit -m "feat(sft): Qwen3-4B No Robots SFT launcher and PEFT_METHOD=none support"
```

---

### Task 10: Sweep driver and results ledger

**Files:**
- Create: `tools/lora_regret/sweep.py`
- Create: `tools/lora_regret/arms.py`
- Test: `tests/fast/tools/test_lora_regret_sweep.py`

**Interfaces:**
- Consumes: `matched_oft_block_size` and `match_report` from `orbit.utils.peft_param_match` (Task 5); the launcher from Task 9.
- Produces:
  - `Arm` dataclass with fields `name: str`, `method: str` (`"full"|"lora"|"oft"`), `rank: int | None`, `oft_block_size: int | None`, `target_modules: str`, `lr: float`, `seed: int`.
  - `sft_arms(hidden_size: int, ffn_size: int) -> list[Arm]` — the full 82-arm SFT matrix (42 LoRA/FullFT + 40 OFT).
  - `arm_env(arm: Arm) -> dict[str, str]` — env overrides for one launcher invocation.
  - `load_ledger(path: Path) -> set[str]` / `append_result(path: Path, record: dict) -> None`.
  - Results schema, one JSON object per line in `results/lora_regret_sft.jsonl`:
    `{"arm", "method", "rank", "oft_block_size", "target_modules", "lr", "seed", "test_nll", "adapter_params", "wandb_run_id", "steps", "status"}`
    where `status` is `"ok"` or `"failed"`.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/tools/test_lora_regret_sweep.py`:

```python
"""Arm enumeration and the resume ledger for the LoRA-without-regret sweep."""

import json
from pathlib import Path

import pytest

from tools.lora_regret.arms import Arm, LORA_LR_GRID, FULL_LR_GRID, arm_env, sft_arms
from tools.lora_regret.sweep import append_result, load_ledger

H, FFN = 2560, 9728


class TestLrGrids:
    def test_lora_grid_brackets_every_published_optimum(self):
        # published LoRA optima span 1.2e-4 .. 3.5e-4
        assert min(LORA_LR_GRID) < 1.2e-4
        assert max(LORA_LR_GRID) > 3.5e-4
        assert len(LORA_LR_GRID) == 7

    def test_full_grid_brackets_the_fullft_optimum(self):
        assert min(FULL_LR_GRID) < 2.5e-5 < max(FULL_LR_GRID)
        assert len(FULL_LR_GRID) == 7

    def test_grids_are_monotonic(self):
        assert LORA_LR_GRID == sorted(LORA_LR_GRID)
        assert FULL_LR_GRID == sorted(FULL_LR_GRID)


class TestSftArms:
    def test_lora_and_full_arm_count_is_42(self):
        arms = [a for a in sft_arms(H, FFN) if a.method in ("lora", "full")]
        assert len(arms) == 42

    def test_one_full_finetune_config(self):
        full = [a for a in sft_arms(H, FFN) if a.method == "full"]
        assert len(full) == 7
        assert all(a.rank is None for a in full)

    def test_layer_ablation_target_modules(self):
        arms = sft_arms(H, FFN)
        targets = {a.target_modules for a in arms if a.method == "lora" and a.rank == 256}
        assert targets == {
            "linear_qkv,linear_proj,linear_fc1,linear_fc2",
            "linear_qkv,linear_proj",
            "linear_fc1,linear_fc2",
        }

    def test_ranks_present(self):
        ranks = {a.rank for a in sft_arms(H, FFN) if a.method == "lora"}
        assert ranks == {1, 16, 256}

    def test_oft_block_sizes_come_from_the_solver(self):
        from orbit.utils.peft_param_match import matched_oft_block_size

        oft = [a for a in sft_arms(H, FFN) if a.method == "oft"]
        blocks = {a.oft_block_size for a in oft}
        assert matched_oft_block_size(1, H, H) in blocks
        assert matched_oft_block_size(16, H, H) in blocks

    def test_arm_names_are_unique(self):
        names = [a.name for a in sft_arms(H, FFN)]
        assert len(names) == len(set(names))


class TestArmEnv:
    def test_full_finetune_env(self):
        env = arm_env(Arm("x", "full", None, None, "", 2.5e-5, 0))
        assert env["PEFT_METHOD"] == "none"
        assert env["LR"] == "2.5e-05"
        assert "LORA_RANK" not in env

    def test_lora_env_sets_alpha_and_init(self):
        env = arm_env(Arm("x", "lora", 16, None, "linear_fc1", 2e-4, 3))
        assert env["PEFT_METHOD"] == "lora"
        assert env["LORA_RANK"] == "16"
        assert env["LORA_ALPHA"] == "32"
        assert env["LORA_A_INIT_METHOD"] == "kaiming"
        assert env["TARGET_MODULES"] == "linear_fc1"
        assert env["SEED"] == "3"

    def test_oft_env_sets_block_size(self):
        env = arm_env(Arm("x", "oft", None, 64, "linear_fc1", 1e-4, 0))
        assert env["PEFT_METHOD"] == "oft"
        assert env["OFT_BLOCK_SIZE"] == "64"
        assert "LORA_RANK" not in env


class TestLedger:
    def test_load_ledger_of_missing_file_is_empty(self, tmp_path: Path):
        assert load_ledger(tmp_path / "nope.jsonl") == set()

    def test_append_then_load_round_trip(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        append_result(path, {"arm": "a1", "status": "ok", "test_nll": 1.84})
        append_result(path, {"arm": "a2", "status": "ok", "test_nll": 1.85})
        assert load_ledger(path) == {"a1", "a2"}

    def test_failed_arms_are_not_treated_as_done(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        append_result(path, {"arm": "a1", "status": "failed", "test_nll": None})
        assert load_ledger(path) == set()

    def test_ledger_survives_a_truncated_final_line(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        append_result(path, {"arm": "a1", "status": "ok"})
        with path.open("a") as fh:
            fh.write('{"arm": "a2", "sta')
        assert load_ledger(path) == {"a1"}
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.lora_regret.arms'`

- [ ] **Step 3: Write the arm enumeration**

Create `tools/lora_regret/arms.py`:

```python
"""The LoRA-without-regret SFT experiment matrix.

Two LR grids, because the LoRA and FullFT optima sit a decade apart and one
shared grid would spend most of its points where nothing happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from orbit.utils.peft_param_match import matched_oft_block_size

ALL_MODULES = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
ATTN_MODULES = "linear_qkv,linear_proj"
MLP_MODULES = "linear_fc1,linear_fc2"

# Brackets every published LoRA optimum (1.2e-4 .. 3.5e-4) with >=2 points a side.
LORA_LR_GRID = [5e-5, 8e-5, 1.2e-4, 2e-4, 3e-4, 5e-4, 8e-4]
# Same shape, one decade down; brackets the FullFT optimum 2.5e-5.
FULL_LR_GRID = [5e-6, 8e-6, 1.2e-5, 2e-5, 3e-5, 5e-5, 8e-5]
# OFT's natural LR scale is unknown a priori: it parameterizes a rotation, not
# an additive update. Scout wide, then refine around the argmin.
OFT_SCOUT_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

LORA_ALPHA = 32


@dataclass(frozen=True)
class Arm:
    name: str
    method: str  # "full" | "lora" | "oft"
    rank: int | None
    oft_block_size: int | None
    target_modules: str
    lr: float
    seed: int


def _name(method: str, tag: str, modules: str, lr: float, seed: int) -> str:
    short = {ALL_MODULES: "all", ATTN_MODULES: "attn", MLP_MODULES: "mlp"}.get(modules, "na")
    return f"{method}-{tag}-{short}-lr{lr:g}-s{seed}"


def sft_arms(hidden_size: int, ffn_size: int, seed: int = 0) -> list[Arm]:
    """The 82-arm SFT matrix: 42 LoRA/FullFT plus 40 OFT (5 scout + 5x7)."""
    arms: list[Arm] = []

    for lr in FULL_LR_GRID:
        arms.append(Arm(_name("full", "na", "", lr, seed), "full", None, None, "", lr, seed))

    lora_configs = [
        (256, ALL_MODULES),
        (256, ATTN_MODULES),
        (256, MLP_MODULES),
        (16, ALL_MODULES),
        (1, ALL_MODULES),
    ]
    for rank, modules in lora_configs:
        for lr in LORA_LR_GRID:
            arms.append(
                Arm(_name("lora", f"r{rank}", modules, lr, seed), "lora", rank, None, modules, lr, seed)
            )

    # Matched OFT. Block size is solved against the square (attention) shape so
    # all arms share one OFT_BLOCK_SIZE; per-layer snapping handles the rest.
    oft_configs = [
        (1, ALL_MODULES),
        (16, ALL_MODULES),
        (256, ALL_MODULES),
        (16, ATTN_MODULES),
        (16, MLP_MODULES),
    ]
    scout_block = matched_oft_block_size(16, hidden_size, hidden_size)
    for lr in OFT_SCOUT_GRID:
        arms.append(
            Arm(_name("oftscout", f"b{scout_block}", ALL_MODULES, lr, seed),
                "oft", None, scout_block, ALL_MODULES, lr, seed)
        )
    for rank, modules in oft_configs:
        block = matched_oft_block_size(rank, hidden_size, hidden_size)
        for lr in LORA_LR_GRID:
            arms.append(
                Arm(_name("oft", f"b{block}", modules, lr, seed), "oft", None, block, modules, lr, seed)
            )

    return arms


def arm_env(arm: Arm) -> dict[str, str]:
    """Environment overrides for one launcher invocation."""
    env = {"LR": f"{arm.lr:g}", "SEED": str(arm.seed)}
    if arm.method == "full":
        env["PEFT_METHOD"] = "none"
        return env
    env["TARGET_MODULES"] = arm.target_modules
    if arm.method == "lora":
        env["PEFT_METHOD"] = "lora"
        env["LORA_RANK"] = str(arm.rank)
        env["LORA_ALPHA"] = str(LORA_ALPHA)
        env["LORA_A_INIT_METHOD"] = "kaiming"
    elif arm.method == "oft":
        env["PEFT_METHOD"] = "oft"
        env["OFT_BLOCK_SIZE"] = str(arm.oft_block_size)
    else:
        raise ValueError(f"unknown method {arm.method!r}")
    return env
```

Note the `test_full_finetune_env` test expects `env["LR"] == "2.5e-05"`; `f"{2.5e-5:g}"` renders as `2.5e-05`. If your Python renders it differently, fix the test to match the implementation, not the other way round.

- [ ] **Step 4: Write the sweep driver**

Create `tools/lora_regret/sweep.py`:

```python
"""Drive the LoRA-without-regret sweep, one launcher invocation per arm.

Resumable: every completed arm appends a record to the results JSONL, and a
restart skips arms already recorded as "ok". A failed arm is retried on the
next run rather than silently skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from tools.lora_regret.arms import Arm, arm_env, sft_arms

LAUNCHER = "examples/sft/run-qwen3-4b-norobots-sft.sh"
_NLL_LINE = re.compile(r"eval/test_nll step=(\d+) nll=([0-9.]+)")


def load_ledger(path: Path) -> set[str]:
    """Arm names already completed successfully. Tolerates a truncated tail."""
    if not Path(path).exists():
        return set()
    done = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from an interrupted write
        if record.get("status") == "ok":
            done.add(record["arm"])
    return done


def append_result(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def parse_final_nll(log_text: str) -> tuple[float | None, int | None]:
    """Last `eval/test_nll step=N nll=X` line in a run log."""
    matches = _NLL_LINE.findall(log_text)
    if not matches:
        return None, None
    step, nll = matches[-1]
    return float(nll), int(step)


def run_arm(arm: Arm, repo_root: Path, results_path: Path, dry_run: bool) -> None:
    log_path = repo_root / "logs" / "lora_regret" / f"{arm.name}.log"
    env = dict(os.environ)
    env.update(arm_env(arm))
    env.update(
        {
            "LAUNCHER_NAME": arm.name,
            "RUN_LOG": str(log_path),
            "WANDB_GROUP": "lora-regret-sft",
            "SAVE_DIR": str(repo_root / "orbit_ckpts" / "lora_regret" / arm.name),
        }
    )
    cmd = ["bash", str(repo_root / LAUNCHER)]
    if dry_run:
        overrides = " ".join(f"{k}={v}" for k, v in sorted(arm_env(arm).items()))
        print(f"{overrides} bash {LAUNCHER}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, env=env, cwd=repo_root)
    nll, steps = (None, None)
    if log_path.exists():
        nll, steps = parse_final_nll(log_path.read_text(encoding="utf-8", errors="replace"))

    append_result(
        results_path,
        {
            "arm": arm.name,
            "method": arm.method,
            "rank": arm.rank,
            "oft_block_size": arm.oft_block_size,
            "target_modules": arm.target_modules,
            "lr": arm.lr,
            "seed": arm.seed,
            "test_nll": nll,
            "adapter_params": None,
            "wandb_run_id": None,
            "steps": steps,
            "status": "ok" if (proc.returncode == 0 and nll is not None) else "failed",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--ffn-size", type=int, required=True)
    parser.add_argument("--results", type=Path, default=Path("results/lora_regret_sft.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        default=None,
        help="Regex; run only arms whose name matches (e.g. '^lora-r256' or '^oftscout').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    arms = sft_arms(args.hidden_size, args.ffn_size, seed=args.seed)
    if args.only:
        pattern = re.compile(args.only)
        arms = [a for a in arms if pattern.search(a.name)]

    done = load_ledger(args.results)
    todo = [a for a in arms if a.name not in done]
    print(f"{len(arms)} arms selected, {len(done)} already done, {len(todo)} to run")

    for i, arm in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {arm.name}")
        run_arm(arm, repo_root, args.results, args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_sweep.py -v`
Expected: 17 passed

- [ ] **Step 6: Verify the dry run enumerates the right matrix [CPU]**

Run:

```bash
python -m tools.lora_regret.sweep --hidden-size 2560 --ffn-size 9728 --dry-run | head -5
python -m tools.lora_regret.sweep --hidden-size 2560 --ffn-size 9728 --dry-run | wc -l
```

Expected: 82 lines (42 LoRA/FullFT + 40 OFT), each a runnable env-prefixed command. Spot-check that one FullFT line has `PEFT_METHOD=none` and no `LORA_RANK`.

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py \
        tests/fast/tools/test_lora_regret_sweep.py
git commit -m "feat(repro): sweep driver and resumable results ledger"
```

---

### Task 11: Gates G4, G1, G2 [GPU — operator-run]

**Files:**
- Create: `docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md`

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: a gate log recording each gate's measured numbers, which Task 12 onward depends on. **The sweep does not launch until G2 passes.**

- [ ] **Step 1: Create the gate log skeleton**

Create `docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md`:

```markdown
# LoRA Without Regret — Gate Log

Gates defined in `docs/superpowers/specs/2026-07-27-lora-without-regret-repro-design.md` §7.2.
Fill in as each is run. Do not launch the sweep until G2 is marked PASS.

## G3 — loss-mask parity (CPU)

- Command: `python -m pytest tests/fast/rollout/test_sft_loss_mask_parity.py -v`
- Result:
- Notes:

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
- HF bf16-vs-fp32 spread (measured): 0.0072 nats
- Orbit-vs-HF-bf16 delta: 0.003176 nats (inside the measured spread)
- Result: **G4 PASSED** (measured 2026-07-28)

## G1 — oracle reproduces the published number

| LR | test NLL |
|---|---|
| 1.2e-4 | |
| 2.5e-4 | |
| 5e-4 | |

- Minimum: (expected ~1.8457)
- Result:

## G2 — Orbit parity with the oracle

| LR | oracle NLL | orbit NLL | delta |
|---|---|---|---|
| 1.2e-4 | | | |
| 2.5e-4 | | | |
| 5e-4 | | | |

- sigma (from §7.1 seed measurement):
- Max |delta|: (must be <= 2*sigma)
- argmin agreement: (oracle argmin LR vs orbit argmin LR — must match)
- Result:

## Seed noise (prerequisite for G2's tolerance)

LoRA r256 all-modules at lr=2.5e-4, three seeds:

| seed | test NLL |
|---|---|
| 0 | |
| 1 | |
| 2 | |

- sigma (sample std):
- Is sigma small enough to resolve the 0.0057 attn-vs-mlp gap? (needs roughly sigma < 0.002)
- Decision if not:
```

- [x] **Step 2: Hand the operator G4 (step-0 NLL) [GPU — operator-run]** — done 2026-07-28, G4 PASSED (numbers below)

```bash
codexlog lora_regret_g4_orbit_step0 \
    PEFT_METHOD=none NUM_ROLLOUT=1 TOTAL_EPOCHS=1 \
    EVAL_NLL_INTERVAL=1 ENABLE_WANDB=0 LR=0.0 \
    bash examples/sft/run-qwen3-4b-norobots-sft.sh
```

The HF side of this gate does **not** go through the vendored oracle. `sft_full.py`'s real
flag surface (verified by reading the script) is exactly `--model-id --lr --wandb-project
--wandb-run-name --no-wandb --batch-size --gradient-accumulation-steps --num-epochs
--output-dir --seed` — there is no `--max-steps` and no `--dtype`. Nothing in that list
stops the run after step 0: `--lr 0.0` alone still runs a full epoch of forward+backward
over all 6400 training rows as a training-shaped no-op, purely to reach the same step-0
print this gate needs. "Evaluate the base model and stop" is not expressible with the
oracle's flags, so G4's HF side was instead scored with a purpose-written script, checked in
at `tools/lora_regret/g4_hf_nll.py`. It reuses Orbit's own `MultiTurnLossMaskGenerator`
(already verified token-for-token against HF by G3) and HF's own `AutoModelForCausalLM`
forward pass — run once per precision:

```bash
codexlog lora_regret_g4_hf_bf16 python tools/lora_regret/g4_hf_nll.py --dtype bfloat16
codexlog lora_regret_g4_hf_fp32 python tools/lora_regret/g4_hf_nll.py --dtype float32
```

Record both numbers in the gate log, along with each side's scored-token and sample count. Pass condition (design §7.2, corrected from the earlier ~1e-3-nat fixed bar, which is not achievable): (1) scored-token count and sample count must match Orbit-vs-HF **exactly** — any mismatch is structural and blocks the gate; (2) the NLL delta must fall within the **measured** HF bf16-vs-fp32 spread on this reference set (score it at both precisions on the HF side to establish that spread — do not hardcode a tolerance). A gap larger than that spread, or any count mismatch, means the checkpoint conversion or the tokenization differs — stop and investigate.

**Measured 2026-07-28:** HF bf16 3.592773, Orbit bf16 3.589597, HF fp32 3.585589, all at 18472 scored tokens / 100 samples (`logs/lora_regret/g4_hf.log`, `logs/lora_regret/g4_hf_fp32.log`). Counts match exactly; HF's own bf16-vs-fp32 spread is 0.0072 nats, and the Orbit-vs-HF-bf16 delta (0.003176 nats) falls inside it, closer to HF's own bf16 than to HF fp32. **G4 PASSED.**

- [ ] **Step 3: Hand the operator the seed-noise measurement [GPU — operator-run]**

```bash
for s in 0 1 2; do
  codexlog lora_regret_seed_$s \
      PEFT_METHOD=lora LORA_RANK=256 LORA_ALPHA=32 LORA_A_INIT_METHOD=kaiming \
      LR=2.5e-4 SEED=$s LAUNCHER_NAME=seednoise_s$s \
      bash examples/sft/run-qwen3-4b-norobots-sft.sh
done
```

Record the three final NLLs and their sample standard deviation in the gate log. **This number sets G2's tolerance and decides whether the layer-ablation claim is resolvable at all** — the attention-vs-MLP gap is 0.0057, so roughly sigma < 0.002 is needed.

**Precision note for G1 and G2 below.** Both `sft_full.py` and `sft_lora.py` hardcode
`torch_dtype=torch.bfloat16` in their `model_kwargs` (no `--dtype` flag on either script —
verified by reading both), and this recipe's Orbit launcher sets `PRECISION_PROFILE="bf16"`.
So G1's oracle run and G2's Orbit run are bf16-against-bf16 — the correct regime, not a
precision mismatch polluting the 2·sigma comparison.

- [ ] **Step 4: Hand the operator G1 (oracle) [GPU — operator-run]**

```bash
for lr in 1.2e-4 2.5e-4 5e-4; do
  codexlog lora_regret_g1_hf_lr$lr \
      env CUDA_VISIBLE_DEVICES=0 uv run --directory third_party/lora-without-regret \
          sft_lora.py --lr $lr --lora-rank 256 --lora-type all --no-wandb
done
```

Expected minimum ~1.8457. If the oracle does not reproduce its own published number, the problem is environmental (dataset version, tokenizer, torch version) and must be resolved before G2 means anything.

- [ ] **Step 5: Hand the operator G2 (Orbit parity) [GPU — operator-run]**

```bash
for lr in 1.2e-4 2.5e-4 5e-4; do
  codexlog lora_regret_g2_orbit_lr$lr \
      PEFT_METHOD=lora LORA_RANK=256 LORA_ALPHA=32 LORA_A_INIT_METHOD=kaiming \
      LR=$lr LAUNCHER_NAME=g2_orbit_lr$lr \
      bash examples/sft/run-qwen3-4b-norobots-sft.sh
done
```

Pass condition: every |orbit − oracle| <= 2*sigma, **and** both sides pick the same argmin LR.

If G2 fails, the ranked suspects are: (1) loss-mask disagreement — re-run G3; (2) `LORA_A_INIT_METHOD` not actually reaching the adapter — grep the run log for `a_init=kaiming`; (3) token-weighted vs sample-weighted NLL reduction — compare against `reduce_nll`'s docstring; (4) LR schedule not actually constant — grep the log for the scheduler line. **Do not proceed to Task 12 on a failed G2.**

- [ ] **Step 6: Commit the completed gate log**

```bash
git add docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md
git commit -m "docs(repro): gate log for lora-without-regret validation"
```

---

### Task 12: Run the SFT sweeps [GPU — operator-run]

**Files:**
- Create: `results/lora_regret_sft.jsonl` (generated)
- Modify: `.gitignore` (do NOT ignore `results/` — the JSONL is the deliverable)

**Interfaces:**
- Consumes: the sweep driver (Task 10), a passed G2 (Task 11).
- Produces: `results/lora_regret_sft.jsonl` with 82 records.

- [ ] **Step 1: Confirm G2 passed**

Read `docs/superpowers/plans/2026-07-27-lora-without-regret-gate-log.md`. If G2 is not marked PASS, stop and report that the sweep is blocked.

- [ ] **Step 2: Hand the operator the LoRA/FullFT sweep [GPU — operator-run]**

```bash
codexlog lora_regret_sweep_lora_full \
    python -m tools.lora_regret.sweep \
        --hidden-size <H> --ffn-size <FFN> \
        --results results/lora_regret_sft.jsonl \
        --only '^(lora|full)-'
```

Substitute `<H>` and `<FFN>` with the values printed by Task 6 Step 1. 42 arms; resumable — a rerun skips completed arms.

- [ ] **Step 3: Hand the operator the OFT scout, then the OFT sweep [GPU — operator-run]**

```bash
codexlog lora_regret_sweep_oft_scout \
    python -m tools.lora_regret.sweep \
        --hidden-size <H> --ffn-size <FFN> \
        --results results/lora_regret_sft.jsonl \
        --only '^oftscout-'
```

Read the scout's argmin out of the results file:

```bash
python -c "
import json
rows = [json.loads(l) for l in open('results/lora_regret_sft.jsonl')]
scout = [r for r in rows if r['arm'].startswith('oftscout') and r['test_nll']]
print(min(scout, key=lambda r: r['test_nll']))
"
```

If the scout's argmin is at an endpoint of `OFT_SCOUT_GRID` (1e-5 or 1e-3), the optimum is outside the scouted range — extend `OFT_SCOUT_GRID` in `tools/lora_regret/arms.py`, commit that change, and re-scout before continuing. Otherwise recenter `LORA_LR_GRID` for the OFT arms on the argmin and run:

```bash
codexlog lora_regret_sweep_oft \
    python -m tools.lora_regret.sweep \
        --hidden-size <H> --ffn-size <FFN> \
        --results results/lora_regret_sft.jsonl \
        --only '^oft-'
```

- [ ] **Step 4: Verify completeness [CPU]**

```bash
python -c "
import json, collections
rows = [json.loads(l) for l in open('results/lora_regret_sft.jsonl')]
status = collections.Counter(r['status'] for r in rows)
print(status)
failed = [r['arm'] for r in rows if r['status'] == 'failed']
print('failed:', failed)
"
```

Expected: 82 `ok`, 0 `failed`. Re-run the driver to retry any failures. If an arm fails repeatedly, record why in the gate log rather than dropping it silently — a silently missing arm reads as "covered everything".

- [ ] **Step 5: Commit the results**

```bash
git add results/lora_regret_sft.jsonl
git commit -m "results(repro): SFT sweep for lora-without-regret"
```

---

### Task 13: RL launchers and sweep [GPU — operator-run]

**Files:**
- Create: `examples/high_precision/run-qwen3-1.7b-math-grpo.sh`
- Create: `tools/lora_regret/rl_arms.py`
- Test: `tests/fast/tools/test_lora_regret_rl_arms.py`

**Interfaces:**
- Consumes: `matched_oft_block_size` (Task 5); the competition_math JSONL (Task 2); the Qwen3-1.7B checkpoint (Task 6); `Arm` and `arm_env` (Task 10).
- Produces: `rl_arms(hidden_size: int, ffn_size: int) -> list[Arm]` — 28 arms (7 configs x 4 LRs); `results/lora_regret_rl.jsonl`.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/tools/test_lora_regret_rl_arms.py`:

```python
"""RL arm enumeration for the LoRA-without-regret reproduction."""

from tools.lora_regret.rl_arms import RL_LR_GRID, rl_arms

H, FFN = 2048, 6144


def test_four_lrs_per_config():
    assert len(RL_LR_GRID) == 4


def test_twenty_eight_arms():
    assert len(rl_arms(H, FFN)) == 28


def test_seven_distinct_configs():
    configs = {(a.method, a.rank, a.oft_block_size) for a in rl_arms(H, FFN)}
    assert len(configs) == 7


def test_includes_full_and_rank_one():
    arms = rl_arms(H, FFN)
    assert any(a.method == "full" for a in arms)
    assert any(a.method == "lora" and a.rank == 1 for a in arms)


def test_names_unique():
    names = [a.name for a in rl_arms(H, FFN)]
    assert len(names) == len(set(names))
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_rl_arms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.lora_regret.rl_arms'`

- [ ] **Step 3: Write the RL arm enumeration**

Create `tools/lora_regret/rl_arms.py`:

```python
"""RL (GRPO) arm matrix for the LoRA-without-regret reproduction.

Four LRs per config rather than the SFT half's seven: the RL claim is
qualitative parity between LoRA and FullFT, not a precise optimum.
"""

from __future__ import annotations

from orbit.utils.peft_param_match import matched_oft_block_size
from tools.lora_regret.arms import ALL_MODULES, Arm

RL_LR_GRID = [1e-6, 3e-6, 1e-5, 3e-5]
RL_LORA_LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4]


def rl_arms(hidden_size: int, ffn_size: int, seed: int = 0) -> list[Arm]:
    arms: list[Arm] = []
    for lr in RL_LR_GRID:
        arms.append(Arm(f"rl-full-lr{lr:g}-s{seed}", "full", None, None, "", lr, seed))
    for rank in (256, 16, 1):
        for lr in RL_LORA_LR_GRID:
            arms.append(
                Arm(f"rl-lora-r{rank}-lr{lr:g}-s{seed}", "lora", rank, None, ALL_MODULES, lr, seed)
            )
    for rank in (256, 16, 1):
        block = matched_oft_block_size(rank, hidden_size, hidden_size)
        for lr in RL_LORA_LR_GRID:
            arms.append(
                Arm(f"rl-oft-b{block}-lr{lr:g}-s{seed}", "oft", None, block, ALL_MODULES, lr, seed)
            )
    return arms
```

- [ ] **Step 4: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_rl_arms.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the RL launcher**

Create `examples/high_precision/run-qwen3-1.7b-math-grpo.sh`, modelled on `run-qwen2.5-0.5b-bf16-math-lora.sh`:

```bash
#!/usr/bin/env bash
# Qwen3-1.7B + GRPO on competition_math, for the LoRA-without-regret RL half.
# 50 GRPO steps, 32 prompts/step, 8 rollouts/prompt, max 1024 new tokens.
#
#   PEFT_METHOD=none  LR=3e-6  bash examples/high_precision/run-qwen3-1.7b-math-grpo.sh
#   PEFT_METHOD=lora  LORA_RANK=1 LR=1e-4 bash ...
#   PEFT_METHOD=oft   OFT_BLOCK_SIZE=5 LR=1e-4 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

LAUNCHER_NAME=${LAUNCHER_NAME:-"run_qwen3_1p7b_math_grpo"}
PRECISION_PROFILE="bf16"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}

MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-1.7B.sh"}

# === Data ===
DATA_DIR=${DATA_DIR:-/lustre/fast/fast/groups/ei-slm/data/lora_regret}
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_DIR}/competition_math_train.jsonl}
TEST_JSONL=${TEST_JSONL:-${DATA_DIR}/competition_math_val.jsonl}
DATASET=${DATASET:-competition_math}

# === Checkpoints ===
HF_CKPT=${HF_CKPT:-/lustre/fast/fast/zqiu/hf_models/Qwen3-1.7B}
MEGATRON_LOAD=${MEGATRON_LOAD:-"${ORBIT_ROOT}/checkpoints/Qwen3-1.7B_torch_dist"}
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/qwen3-1.7b-math-grpo}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
RUN_LOG=${RUN_LOG:-"${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"}

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}

# === Training schedule: 50 GRPO steps, 32 prompts x 8 rollouts ===
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}   # 32 x 8: one update per GRPO step
NUM_ROLLOUT=${NUM_ROLLOUT:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}

# === RL ===
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo}
USE_KL_LOSS=${USE_KL_LOSS:-0}
RM_TYPE=${RM_TYPE:-math}

# === Optimizer ===
LR=${LR:-1e-4}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

# === PEFT ===
PEFT_METHOD=${PEFT_METHOD:-lora}
TARGET_MODULES=${TARGET_MODULES:-linear_qkv,linear_proj,linear_fc1,linear_fc2}
LORA_RANK=${LORA_RANK:-256}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_A_INIT_METHOD=${LORA_A_INIT_METHOD:-kaiming}

# === Eval ===
DISABLE_EVAL=${DISABLE_EVAL:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-4}

# === Rollout backend ===
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.60}

# === Safety ===
SKIP_NAN_CHECK_IN_LOSS_AND_GRAD=${SKIP_NAN_CHECK_IN_LOSS_AND_GRAD:-0}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-lora-without-regret}
WANDB_GROUP=${WANDB_GROUP:-rl-qwen3-1.7b-math}

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
```

Confirm `orbit_plugins/model_args/qwen3-1.7B.sh` exists; if not, create it from the nearest Qwen3 sibling with `hidden_size`, `num_layers`, `num_attention_heads`, `num_query_groups`, `ffn_hidden_size`, and `rope_theta` read from Qwen3-1.7B's `config.json`.

- [ ] **Step 6: Verify syntax [CPU]**

Run: `bash -n examples/high_precision/run-qwen3-1.7b-math-grpo.sh && chmod +x examples/high_precision/run-qwen3-1.7b-math-grpo.sh`
Expected: exit 0.

- [ ] **Step 7: Hand the operator the smoke test, then the sweep [GPU — operator-run]**

```bash
codexlog lora_regret_rl_smoke \
    NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=4 \
    DISABLE_EVAL=1 ENABLE_WANDB=0 \
    bash examples/high_precision/run-qwen3-1.7b-math-grpo.sh
```

Acceptance: one GRPO step completes, SGLang serves rollouts, reward is finite and non-constant.

Then extend `tools/lora_regret/sweep.py` with an `--rl` flag that swaps `sft_arms` for `rl_arms` and `LAUNCHER` for the RL launcher, and run the 28-arm sweep into `results/lora_regret_rl.jsonl`.

- [ ] **Step 8: Commit**

```bash
git add examples/high_precision/run-qwen3-1.7b-math-grpo.sh tools/lora_regret/rl_arms.py \
        tests/fast/tools/test_lora_regret_rl_arms.py
git commit -m "feat(rl): Qwen3-1.7B GRPO launcher and RL arm matrix"
```

---

### Task 14: Analysis, figures, and the write-up

**Files:**
- Create: `tools/lora_regret/analyze.py`
- Create: `docs/experiments/lora_without_regret.md`
- Test: `tests/fast/tools/test_lora_regret_analyze.py`

**Interfaces:**
- Consumes: `results/lora_regret_sft.jsonl`, `results/lora_regret_rl.jsonl`.
- Produces:
  - `argmin_by_config(rows: list[dict]) -> dict[tuple, dict]` — best-LR record per (method, rank, oft_block_size, target_modules).
  - `lr_ratio(rows: list[dict]) -> float` — LoRA-r256-all optimal LR divided by FullFT optimal LR.
  - `plot_lr_curves(rows, out_path)` and `plot_rank_curves(rows, out_path)`.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/tools/test_lora_regret_analyze.py`:

```python
"""Analysis of the sweep results. Pure functions over the JSONL records."""

import pytest

from tools.lora_regret.analyze import argmin_by_config, lr_ratio

ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"


def _row(method, rank, lr, nll, modules=ALL, block=None, status="ok"):
    return {
        "method": method, "rank": rank, "oft_block_size": block,
        "target_modules": modules, "lr": lr, "test_nll": nll, "status": status,
    }


def test_argmin_picks_lowest_nll_per_config():
    rows = [
        _row("lora", 256, 1e-4, 1.90),
        _row("lora", 256, 2.5e-4, 1.8457),
        _row("lora", 256, 5e-4, 1.87),
    ]
    best = argmin_by_config(rows)
    assert len(best) == 1
    assert next(iter(best.values()))["lr"] == 2.5e-4


def test_argmin_separates_configs():
    rows = [
        _row("lora", 256, 2.5e-4, 1.8457),
        _row("lora", 1, 1.2e-4, 1.8489),
        _row("full", None, 2.5e-5, 1.8457),
    ]
    assert len(argmin_by_config(rows)) == 3


def test_argmin_ignores_failed_and_null_rows():
    rows = [
        _row("lora", 256, 1e-4, None, status="failed"),
        _row("lora", 256, 2.5e-4, 1.8457),
    ]
    best = argmin_by_config(rows)
    assert next(iter(best.values()))["lr"] == 2.5e-4


def test_lr_ratio_reproduces_ten_x():
    rows = [
        _row("lora", 256, 2.5e-4, 1.8457),
        _row("lora", 256, 5e-4, 1.87),
        _row("full", None, 2.5e-5, 1.8457),
        _row("full", None, 5e-5, 1.87),
    ]
    assert lr_ratio(rows) == pytest.approx(10.0)


def test_lr_ratio_raises_when_an_arm_is_missing():
    rows = [_row("lora", 256, 2.5e-4, 1.8457)]
    with pytest.raises(ValueError, match="no full fine-tuning"):
        lr_ratio(rows)
```

- [ ] **Step 2: Run the test to verify it fails [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.lora_regret.analyze'`

- [ ] **Step 3: Write the analysis module**

Create `tools/lora_regret/analyze.py`:

```python
"""Reduce the sweep JSONL to the blog's claims.

Every figure regenerates from the JSONL alone, so plots never require re-running
training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ALL_MODULES = "linear_qkv,linear_proj,linear_fc1,linear_fc2"


def _config_key(row: dict) -> tuple:
    return (row["method"], row["rank"], row["oft_block_size"], row["target_modules"])


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def argmin_by_config(rows: list[dict]) -> dict[tuple, dict]:
    """Best (lowest test NLL) record per configuration, ignoring failed arms."""
    best: dict[tuple, dict] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("test_nll") is None:
            continue
        key = _config_key(row)
        if key not in best or row["test_nll"] < best[key]["test_nll"]:
            best[key] = row
    return best


def lr_ratio(rows: list[dict]) -> float:
    """Optimal LoRA(r256, all-modules) LR divided by optimal FullFT LR."""
    best = argmin_by_config(rows)
    lora = next(
        (v for k, v in best.items() if k[0] == "lora" and k[1] == 256 and k[3] == ALL_MODULES),
        None,
    )
    full = next((v for k, v in best.items() if k[0] == "full"), None)
    if lora is None:
        raise ValueError("no LoRA r256 all-modules arm in results")
    if full is None:
        raise ValueError("no full fine-tuning arm in results")
    return lora["lr"] / full["lr"]


def plot_lr_curves(rows: list[dict], out_path: Path) -> Path:
    """Test NLL vs LR, one line per configuration (the blog's Figure 2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_config: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("test_nll") is None:
            continue
        by_config.setdefault(_config_key(row), []).append(row)

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, group in sorted(by_config.items(), key=lambda kv: str(kv[0])):
        group = sorted(group, key=lambda r: r["lr"])
        label = f"{key[0]} r={key[1]} b={key[2]} {key[3].count(',') + 1 if key[3] else 0} mods"
        ax.plot([r["lr"] for r in group], [r["test_nll"] for r in group], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("test NLL (nats)")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return Path(out_path)


def plot_rank_curves(rows: list[dict], out_path: Path) -> Path:
    """Best test NLL vs rank, for the capacity claim (the blog's Figure 1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = argmin_by_config(rows)
    lora = sorted(
        [(k[1], v["test_nll"]) for k, v in best.items() if k[0] == "lora" and k[3] == ALL_MODULES]
    )
    full = next((v["test_nll"] for k, v in best.items() if k[0] == "full"), None)

    fig, ax = plt.subplots(figsize=(6, 4))
    if lora:
        ax.plot([r for r, _ in lora], [n for _, n in lora], marker="o", label="LoRA (all modules)")
    if full is not None:
        ax.axhline(full, linestyle="--", label="full fine-tuning")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("best test NLL (nats)")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return Path(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/lora_regret_sft.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/lora_regret"))
    args = parser.parse_args()

    rows = load_rows(args.results)
    best = argmin_by_config(rows)
    print(f"{'config':60s} {'lr':>10s} {'test_nll':>10s}")
    for key, row in sorted(best.items(), key=lambda kv: str(kv[0])):
        print(f"{str(key):60s} {row['lr']:>10.3g} {row['test_nll']:>10.4f}")
    try:
        print(f"\nLoRA/FullFT optimal-LR ratio: {lr_ratio(rows):.2f}x (published: 10.0x)")
    except ValueError as exc:
        print(f"\nLR ratio unavailable: {exc}")

    print(plot_lr_curves(rows, args.out_dir / "lr_curves.png"))
    print(plot_rank_curves(rows, args.out_dir / "rank_curves.png"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes [CPU]**

Run: `python -m pytest tests/fast/tools/test_lora_regret_analyze.py -v`
Expected: 5 passed

- [ ] **Step 5: Generate the figures and the write-up [CPU, after Task 12]**

Run:

```bash
python -m tools.lora_regret.analyze --results results/lora_regret_sft.jsonl
```

Then write `docs/experiments/lora_without_regret.md` containing, at minimum:

- A table of measured optimal LR and test NLL per arm, **side by side with michaelbzhu's published values** from the Global Constraints section.
- The measured LoRA/FullFT LR ratio against the published 10.0.
- The measured seed sigma, and an explicit statement of which comparisons it does and does not resolve. If the attention-vs-MLP gap is below sigma, say so plainly and report it as a failure to reproduce that specific claim.
- For every OFT arm: its block size, realized parameter count, and the ratio against its LoRA counterpart — with the rank-256 arm explicitly flagged as loosely matched.
- Any arm that failed or was dropped, and why. Silent truncation reads as "covered everything".

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/analyze.py tests/fast/tools/test_lora_regret_analyze.py \
        docs/experiments/lora_without_regret.md plots/lora_regret
git commit -m "feat(repro): analysis, figures, and write-up for lora-without-regret"
```

---

## Self-Review Notes

**Spec coverage.** Every spec section maps to a task: §3.2 init bug → Task 3; §5.1 arg exposure → Task 3; §5.2 checkpoints → Task 6; §5.3 data → Task 2; §5.4 NLL eval → Task 7; §6.1 SFT matrix → Tasks 9-10, 12; §6.2 matched OFT → Tasks 5, 10, 12; §6.3 RL → Task 13; §7.1 seed noise → Task 11 Step 3; §7.2 gates G1-G4 → Tasks 8 and 11; §7.3 unit tests → Tasks 3, 5, 7, 8, 10; §7.4 failure isolation → Task 9 (NaN checks on, `REQUIRE_MEGATRON_LOAD=1`) and Task 10 (ledger); §8 success criteria → Task 14.

**Verified during self-review** (checks run, not assumed):

- `--loss-mask-type` **is** already registered at `orbit/utils/arguments.py:1668`. Task 4 Step 6's fallback ("add it if absent") will not fire.
- `orbit_plugins/model_args/qwen3-4B.sh` and `qwen3-1.7B.sh` **both exist**. Tasks 9 and 13 can use them directly; their fallback branches are dead paths kept only as a guard.
- `compute_log_prob`'s output key **is** `f"{store_prefix}{key}"` (`orbit/backends/megatron_utils/model.py:321`), and `compute_ref_log_probs` uses `store_prefix="ref_"` to get `"ref_log_probs"`. So `store_prefix="eval_"` yields `"eval_log_probs"` exactly as Task 7 Step 5 assumes.

**Remaining soft spots, flagged rather than hidden:**

1. Task 7 Step 8 wires a call site in `train.py` that this plan has not read line-by-line. The implementer must locate the actual rollout loop and adapt; the surrounding interface (`compute_eval_nll`, `build_eval_nll_batch`) is fully specified, but the insertion point is not.
2. Task 6 Step 3 reads `scripts/conversion/README.md` because the conversion env-var contract was not verified; the README is declared authoritative over the plan's placeholder command.
3. Whether Qwen3-4B's `hidden_size` is divisible by 5 (needed for the rank-1 OFT match to be exact) is unknown until Task 6 downloads the model. Task 5 Step 5 prints the real match table; a non-exact match there is information to report, not a failure.
