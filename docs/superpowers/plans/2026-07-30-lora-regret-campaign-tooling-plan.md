# LoRA-Without-Regret campaign tooling — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six operator-side gaps so the campaign can be launched on a
reservation, run to completion, and read into claims C1–C6 without any
hand-written heredoc.

**Architecture:** Four new modules under `tools/lora_regret/` — `trace.py`
(log → NLL curve), `analyze.py` (ledgers → claim readings), `p3_check.py`
(DP=1 vs DP=4 equality), `preflight.py` (fail-fast reservation audit) — plus
enrichment of `sweep.py`'s ledger records and a new `e1long` matrix in
`arms.py`. Every unit is a pure function over text or JSON, so the whole plan is
CPU-verifiable with no GPU.

**Tech Stack:** Python 3.12, pytest 9.0.3, stdlib only (`re`, `json`,
`statistics`, `glob`, `argparse`). Reuses `orbit.utils.peft_param_match` and
`orbit.utils.eval_nll`.

**Spec:** `docs/superpowers/specs/2026-07-30-lora-regret-campaign-tooling-design.md`

## Global Constraints

- **Environment for every command in this plan** — activate before sourcing
  `env.sh`, so `$VIRTUAL_ENV` resolves site-packages correctly:
  ```bash
  source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
  cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
  export CUDA_HOME=/is/software/nvidia/cuda-13.2
  source env.sh
  ```
  `env.sh` is required even for CPU-only work: `megatron.core` imports
  `deep_ep`, which asserts on an unset `CUDA_HOME`.
- **Always pass explicit test paths.** `norecursedirs` matches the basenames
  `tools` and `scripts` at any depth, so `pytest tests/fast/` silently skips
  whole directories. Use `pytest tests` or a named file.
- **Baseline is 502 passed, 0 failed** (measured 2026-07-30, 162 s). Every task
  ends green at 502 + that task's new tests.
- **No GPU is needed for any task in this plan.** If a step seems to need one,
  the step is wrong.
- **`logs/` is gitignored** (`.gitignore:201`). No test may read from it; use
  the committed fixture created in Task 1.
- Commit messages: `feat(lora_regret):` / `fix(lora_regret):` / `docs(...)`,
  one short line, no attribution trailer.
- Llama-3.1-8B geometry, used throughout: `hidden_size=4096`, `ffn_size=14336`,
  `num_layers=32`, `qkv_output_size=6144` (`arms.LLAMA31_8B_QKV_OUTPUT`).

---

### Task 1: `trace.py` — the NLL curve, and one regex definition

**Files:**
- Create: `tools/lora_regret/trace.py`
- Create: `tests/fast/fixtures/lora_regret/smoke_lora_r256_eval_lines.log`
- Create: `tests/fast/utils/test_lora_regret_trace.py`
- Modify: `tools/lora_regret/sweep.py:74-81` (replace the regex definition with an import)

**Interfaces:**
- Consumes: `orbit.utils.eval_nll.EVAL_NLL_METRIC_KEY`
- Produces: `NllPoint` (NamedTuple), `parse_trace(log_text) -> list[NllPoint]`,
  `parse_trace_file(path) -> list[NllPoint]`,
  `trace_is_consistent(points) -> tuple[bool, str]`, and the module-level
  constants `NLL_LINE`, `PHASE_BEFORE_TRAIN`, `PHASE_AFTER_TRAIN`

`sweep.py` currently owns `_NLL_LINE`, built from `EVAL_NLL_METRIC_KEY` so a
rename of that constant cannot desync the parser from the metric. A second copy
in `trace.py` would reintroduce exactly the failure the first copy prevents, so
the definition **moves** here and `sweep.py` imports it under its existing
private names — which keeps `TestLogFormatPins` and every other current test
binding valid.

- [ ] **Step 1: Create the fixture from the real smoke log**

These are the three lines the 2026-07-30 GPU smoke actually emitted, prefix
included. Note `rollout_id=0 step=0` appears **twice**, once per phase — that
is the case `parse_final_nll`'s docstring exists for, so the fixture carries it
rather than a synthesized approximation.

Create `tests/fast/fixtures/lora_regret/smoke_lora_r256_eval_lines.log`:

```
[2026-07-30 15:16:12] train.py:39 - eval/test_nll rollout_id=0 step=0 phase=before_train nll=1.209810 sample_mean=1.478078 tokens=308760 samples=1000
[2026-07-30 15:18:08] train.py:39 - eval/test_nll rollout_id=0 step=0 phase=after_train nll=1.199709 sample_mean=1.455645 tokens=308760 samples=1000
[2026-07-30 15:19:26] train.py:39 - eval/test_nll rollout_id=1 step=1 phase=after_train nll=1.194836 sample_mean=1.421378 tokens=308760 samples=1000
```

- [ ] **Step 2: Write the failing tests**

Create `tests/fast/utils/test_lora_regret_trace.py`:

```python
"""The NLL curve behind C1's departure step.

parse_final_nll answers "what did this arm score"; parse_trace answers "how did
it get there". The fixture is the real 2026-07-30 smoke's three eval lines, not
synthesized text, so a parser that only satisfies its own format string fails
here.
"""

from pathlib import Path

import pytest

from tools.lora_regret.trace import (
    PHASE_AFTER_TRAIN,
    PHASE_BEFORE_TRAIN,
    NllPoint,
    parse_trace,
    parse_trace_file,
    trace_is_consistent,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "lora_regret"
    / "smoke_lora_r256_eval_lines.log"
)


class TestParseTrace:
    def test_parses_the_real_smoke_log(self):
        points = parse_trace_file(FIXTURE)
        assert [p.nll for p in points] == [1.209810, 1.199709, 1.194836]
        assert [p.phase for p in points] == [
            PHASE_BEFORE_TRAIN,
            PHASE_AFTER_TRAIN,
            PHASE_AFTER_TRAIN,
        ]
        assert [p.step for p in points] == [0, 0, 1]

    def test_carries_every_field(self):
        first = parse_trace_file(FIXTURE)[0]
        assert first == NllPoint(
            rollout_id=0,
            step=0,
            phase=PHASE_BEFORE_TRAIN,
            nll=1.209810,
            sample_mean=1.478078,
            tokens=308760,
            samples=1000,
        )

    def test_before_train_sorts_ahead_of_after_train_at_the_same_step(self):
        """The base-model measurement precedes the post-step one at step 0.

        Multi-rank log buffering can place them in either physical order, so the
        ordering must come from (step, phase), not from file position.
        """
        text = "\n".join(reversed(FIXTURE.read_text().splitlines()))
        points = parse_trace(text)
        assert [p.phase for p in points[:2]] == [PHASE_BEFORE_TRAIN, PHASE_AFTER_TRAIN]

    def test_a_log_with_no_eval_lines_is_an_empty_trace(self):
        assert parse_trace("Traceback (most recent call last):\n  boom\n") == []


class TestTraceIsConsistent:
    def _point(self, step, nll, samples=1000, tokens=308760):
        return NllPoint(step, step, PHASE_AFTER_TRAIN, nll, nll, tokens, samples)

    def test_accepts_a_constant_held_out_set(self):
        ok, why = trace_is_consistent([self._point(0, 1.2), self._point(1, 1.1)])
        assert ok, why

    def test_rejects_a_shrinking_sample_count(self):
        """1000 rows at global batch 32 silently becoming 992 is floor division.

        That makes the metric depend on batch size, which is exactly what E2
        varies -- so it must be caught, not averaged over.
        """
        ok, why = trace_is_consistent(
            [self._point(0, 1.2, samples=1000), self._point(1, 1.1, samples=992)]
        )
        assert not ok
        assert "992" in why and "1000" in why

    def test_rejects_a_changing_token_count(self):
        ok, why = trace_is_consistent(
            [self._point(0, 1.2, tokens=308760), self._point(1, 1.1, tokens=306000)]
        )
        assert not ok

    def test_rejects_an_empty_trace(self):
        ok, why = trace_is_consistent([])
        assert not ok
        assert "empty" in why


class TestSweepSharesOneRegex:
    def test_sweep_reuses_the_trace_regex_object(self):
        """One definition, pinned to EVAL_NLL_METRIC_KEY. Not two copies."""
        from tools.lora_regret import sweep, trace

        assert sweep._NLL_LINE is trace.NLL_LINE
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
python -m pytest tests/fast/utils/test_lora_regret_trace.py -q -p no:cacheprovider
```
Expected: collection error — `ModuleNotFoundError: No module named 'tools.lora_regret.trace'`

- [ ] **Step 4: Write `tools/lora_regret/trace.py`**

```python
"""The held-out NLL curve for one arm, extracted from its launcher log.

`sweep.parse_final_nll` answers "what did this arm score". This module answers
"how did it get there", which is what C1's departure step is measured from and
what no ledger field previously carried.

The line regex lives here and `sweep.py` imports it. It is built from
`EVAL_NLL_METRIC_KEY` rather than a re-spelled "eval/test_nll" literal so a
rename of that constant cannot silently desync the parser from the metric it
tracks -- and a second copy of the regex would reintroduce precisely that risk,
which is why this is a move rather than an addition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from orbit.utils.eval_nll import EVAL_NLL_METRIC_KEY

# train.py:_log_eval_nll emits one line per held-out NLL measurement, e.g.:
#
#   eval/test_nll rollout_id=12 step=12 phase=after_train nll=1.845700 \
#       sample_mean=1.801234 tokens=4096 samples=32
NLL_LINE = re.compile(
    re.escape(EVAL_NLL_METRIC_KEY)
    + r" rollout_id=(?P<rollout_id>\d+) step=(?P<step>\d+) phase=(?P<phase>\S+)"
    r" nll=(?P<nll>[0-9.]+) sample_mean=(?P<sample_mean>[0-9.]+)"
    r" tokens=(?P<tokens>\d+) samples=(?P<samples>\d+)"
)
# "before_train" is the untouched base model, logged once at rollout/step 0
# before any optimizer step -- gate G4's number. "after_train" is a
# post-optimizer-step measurement from the periodic hook.
PHASE_BEFORE_TRAIN = "before_train"
PHASE_AFTER_TRAIN = "after_train"


class NllPoint(NamedTuple):
    rollout_id: int
    step: int
    phase: str
    nll: float
    sample_mean: float
    tokens: int
    samples: int


def parse_trace(log_text: str) -> list[NllPoint]:
    """Every held-out measurement in the log, in measurement order.

    Both phases are retained: `before_train` is a meaningful number (the
    pristine base model), it simply must never be picked as an arm's *result* --
    that exclusion belongs in `parse_final_nll`, not here.

    Sorted by `(step, phase != before_train)` rather than by file position.
    Multi-rank log buffering can place the two step-0 rows in either physical
    order, and at equal step the base-model measurement is by construction the
    earlier one.
    """
    points = [
        NllPoint(
            rollout_id=int(m["rollout_id"]),
            step=int(m["step"]),
            phase=m["phase"],
            nll=float(m["nll"]),
            sample_mean=float(m["sample_mean"]),
            tokens=int(m["tokens"]),
            samples=int(m["samples"]),
        )
        for m in NLL_LINE.finditer(log_text)
    ]
    return sorted(points, key=lambda p: (p.step, p.phase != PHASE_BEFORE_TRAIN))


def parse_trace_file(path: str | Path) -> list[NllPoint]:
    return parse_trace(Path(path).read_text(encoding="utf-8", errors="replace"))


def trace_is_consistent(points: list[NllPoint]) -> tuple[bool, str]:
    """Whether every measurement scored the same held-out set.

    `get_data_iterator` floor-divides, so 1,000 rows at global batch 32 would
    silently become 992 and the metric would start depending on batch size --
    which is the axis E2 varies, so the gap E2 measures would be partly an
    artifact of its own instrument. Returns the reason as text so the caller can
    put it in a ledger rather than only in a traceback.
    """
    if not points:
        return False, "empty trace: no eval/test_nll lines in the log"
    tokens = sorted({p.tokens for p in points})
    samples = sorted({p.samples for p in points})
    if len(tokens) > 1 or len(samples) > 1:
        return False, (
            f"held-out set changed size mid-run: tokens={tokens} samples={samples}; "
            "get_data_iterator floor-divides, so this metric depends on batch size"
        )
    return True, ""
```

- [ ] **Step 5: Point `sweep.py` at the shared definition**

In `tools/lora_regret/sweep.py`, delete the `_NLL_LINE` / `_PHASE_BEFORE_TRAIN`
/ `_PHASE_AFTER_TRAIN` block (currently lines 62–81, comment included) and
replace it with an import that keeps the existing private names bound, so no
current test or call site changes:

```python
# The eval-line regex and phase labels live in trace.py -- one definition,
# built from EVAL_NLL_METRIC_KEY. Imported under the existing private names so
# every call site and the TestLogFormatPins pins keep working unchanged.
from tools.lora_regret.trace import (
    NLL_LINE as _NLL_LINE,
    PHASE_AFTER_TRAIN as _PHASE_AFTER_TRAIN,
    PHASE_BEFORE_TRAIN as _PHASE_BEFORE_TRAIN,
    parse_trace,
    trace_is_consistent,
)
```

Place it with the other imports at the top of the file, next to the existing
`from tools.lora_regret.arms import ...`. Remove the now-unused
`from orbit.utils.eval_nll import EVAL_NLL_METRIC_KEY` import and the `re`
import only if `re` is unused elsewhere — it is still used by
`_EVAL_LINE`/`_EVAL_SCORE` and by `main`'s `--only`, so **keep `import re`**.

- [ ] **Step 6: Run the new and existing tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_trace.py tests/fast/utils/test_lora_regret_sweep.py -q -p no:cacheprovider
```
Expected: all pass, including the pre-existing `TestLogFormatPins` and
`TestParseFinalNll` classes.

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/trace.py tools/lora_regret/sweep.py \
        tests/fast/utils/test_lora_regret_trace.py \
        tests/fast/fixtures/lora_regret/smoke_lora_r256_eval_lines.log
git commit -m "feat(lora_regret): extract the held-out NLL trace from arm logs"
```

---

### Task 2: record the trace and its consistency in the ledger

**Files:**
- Modify: `tools/lora_regret/sweep.py:230-286` (`run_arm`)
- Modify: `tests/fast/utils/test_lora_regret_sweep.py` (add a class)

**Interfaces:**
- Consumes: `trace.parse_trace`, `trace.trace_is_consistent` (Task 1)
- Produces: ledger records gain `nll_trace: list[dict] | None`,
  `trace_consistent: bool | None`, `trace_warning: str | None`,
  **`global_batch_size: int | None`** and **`dataset: str | None`**

The last two are not cosmetic. `Arm` carries both and E2 sets them
(`global_batch_size=batch, dataset="openthoughts3"`), but `run_arm`'s record
drops them — so **C3 is currently unreadable from the ledger**: its whole claim
is `best_LoRA(batch) − best_FullFT(batch)` at each batch size, and the batch size
each arm ran at survives only inside the arm's *name* (`...-b512-...`). Recovering
it by parsing names is string archaeology that breaks the first time a name
format changes.

An inconsistent trace does **not** mark the arm failed. The compute is already
spent and the record is evidence; discarding it loses the evidence. It is
recorded and `analyze.py` (Task 6) refuses to quote it — fail closed at the
layer that makes the claim, not at the layer that collects the data.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_sweep.py`:

```python
class TestRunArmRecordsTheTrace:
    """The ledger carries the whole curve, not only its last point.

    C1's departure step cannot be recovered from a single final NLL, and the
    logs it would otherwise have to be re-parsed from are gitignored and
    routinely cleaned.
    """

    def _arm(self):
        return Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)

    def _run(self, tmp_path, monkeypatch, log_body):
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(log_body)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        run_arm(self._arm(), tmp_path, results, dry_run=False)
        return json.loads(results.read_text().splitlines()[0])

    def test_the_trace_lands_in_the_record(self, tmp_path, monkeypatch):
        body = _build_log([
            _render(0, 0, _PHASE_BEFORE_TRAIN, 1.209810, tokens=308760, samples=1000),
            _render(0, 0, _PHASE_AFTER_TRAIN, 1.199709, tokens=308760, samples=1000),
            _render(1, 1, _PHASE_AFTER_TRAIN, 1.194836, tokens=308760, samples=1000),
        ])
        record = self._run(tmp_path, monkeypatch, body)
        assert [p["nll"] for p in record["nll_trace"]] == [1.209810, 1.199709, 1.194836]
        assert record["trace_consistent"] is True
        assert record["trace_warning"] is None
        assert record["test_nll"] == 1.194836

    def test_a_floor_divided_held_out_set_is_flagged_but_still_recorded(
        self, tmp_path, monkeypatch
    ):
        body = _build_log([
            _render(0, 0, _PHASE_AFTER_TRAIN, 1.2, tokens=308760, samples=1000),
            _render(1, 1, _PHASE_AFTER_TRAIN, 1.1, tokens=306000, samples=992),
        ])
        record = self._run(tmp_path, monkeypatch, body)
        assert record["trace_consistent"] is False
        assert "992" in record["trace_warning"]
        # The arm still succeeded; it is analyze.py that refuses to quote it.
        assert record["status"] == "ok"


class TestRunArmRecordsTheArmsIdentity:
    """C3 groups by batch size, so the batch size has to be in the record.

    Arm carries global_batch_size and dataset and e2_arms sets both, but the
    ledger dropped them -- leaving the batch an E2 arm ran at recoverable only
    by parsing its name.
    """

    def test_batch_size_and_dataset_reach_the_ledger(self, tmp_path, monkeypatch):
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(
                _build_log([_render(0, 0, _PHASE_AFTER_TRAIN, 1.5)])
            )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        arm = e2_arms()[0]
        assert arm.global_batch_size is not None, "fixture assumes e2 sets a batch"
        run_arm(arm, tmp_path, results, dry_run=False)
        record = json.loads(results.read_text().splitlines()[0])
        assert record["global_batch_size"] == arm.global_batch_size
        assert record["dataset"] == arm.dataset

    def test_an_arm_with_neither_records_null(self, tmp_path, monkeypatch):
        """E1's arms leave the batch at the launcher's default; null says so."""
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(
                _build_log([_render(0, 0, _PHASE_AFTER_TRAIN, 1.5)])
            )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(arm, tmp_path, results, dry_run=False)
        record = json.loads(results.read_text().splitlines()[0])
        assert record["global_batch_size"] is None
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py::TestRunArmRecordsTheTrace -q -p no:cacheprovider
```
Expected: FAIL with `KeyError: 'nll_trace'`

- [ ] **Step 3: Implement**

In `run_arm`, after the existing `log_text` read, add the trace parse, and add
the three fields to the `append_result` dict:

```python
    nll, accuracy, per_dataset, steps = (None, None, {}, None)
    trace_points: list = []
    trace_ok: bool | None = None
    trace_why: str | None = None
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if metric == "accuracy":
            accuracy, steps, per_dataset = parse_final_accuracy(log_text, RL_EVAL_DATASETS)
        else:
            nll, steps = parse_final_nll(log_text)
            trace_points = parse_trace(log_text)
            ok, why = trace_is_consistent(trace_points)
            trace_ok, trace_why = ok, (why or None)
```

and inside the record dict, next to `"steps"`:

```python
            # The whole curve, not only its last point: C1's departure step is
            # unrecoverable from a scalar, and logs/ is gitignored.
            "nll_trace": [p._asdict() for p in trace_points] or None,
            "trace_consistent": trace_ok,
            "trace_warning": trace_why,
            # C3 groups by batch size; without this the batch an E2 arm ran at
            # survives only inside its name.
            "global_batch_size": arm.global_batch_size,
            "dataset": arm.dataset,
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py -q -p no:cacheprovider
```
Expected: PASS, no regressions in the existing `TestRunArm*` classes.

- [ ] **Step 5: Commit**

```bash
git add tools/lora_regret/sweep.py tests/fast/utils/test_lora_regret_sweep.py
git commit -m "feat(lora_regret): record the NLL trace and its consistency per arm"
```

---

### Task 3: make `--dry-run` print a command line that is safe to paste

**Files:**
- Modify: `tools/lora_regret/sweep.py:230-256` (`run_arm`)
- Modify: `tests/fast/utils/test_lora_regret_sweep.py:449` (`TestDryRunOutput`)

**Interfaces:**
- Produces: no API change; `run_arm`'s dry-run stdout gains four keys

`run_arm` sets `LAUNCHER_NAME`, `RUN_LOG`, `WANDB_GROUP` and `SAVE_DIR` on the
real invocation but prints only `arm_env(arm)`. Pasting a printed line therefore
runs the arm against the launcher's *shared default* `SAVE_DIR` — hazard #1 of
the runbook's §14, reintroduced by the tool whose purpose is previewing what
will run.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_sweep.py`:

```python
class TestDryRunPrintsAPasteableCommand:
    """A previewed command must be the command, including its isolation.

    The launcher's default SAVE_DIR is one directory per recipe, so two arms
    pasted from a dry run would overwrite each other's checkpoints -- the
    runbook's hazard #1, arriving via the preview tool.
    """

    def test_the_sweep_set_variables_are_in_the_printed_line(self, tmp_path, capsys):
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True)
        line = capsys.readouterr().out.strip()
        assert f"SAVE_DIR={tmp_path}/orbit_ckpts/lora_regret/{arm.name}" in line
        assert f"RUN_LOG={tmp_path}/logs/lora_regret/{arm.name}.log" in line
        assert f"LAUNCHER_NAME={arm.name}" in line
        assert "WANDB_GROUP=lora-regret-sft" in line
        # and still the arm's own knobs
        assert "LORA_RANK=16" in line
        assert line.endswith("bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh")

    def test_rl_arms_are_previewed_against_the_rl_launcher_and_group(self, tmp_path, capsys):
        arm = Arm("lora-r1-all-lr1e-05-s0", "lora", 1, None, ALL_MODULES, 1e-5, 0)
        run_arm(
            arm, tmp_path, tmp_path / "r.jsonl", dry_run=True,
            launcher=sweep.RL_LAUNCHER, metric="accuracy",
        )
        line = capsys.readouterr().out.strip()
        assert "WANDB_GROUP=lora-regret-rl" in line
        assert line.endswith(f"bash {sweep.RL_LAUNCHER}")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py::TestDryRunPrintsAPasteableCommand -q -p no:cacheprovider
```
Expected: FAIL — `assert 'SAVE_DIR=...' in line`

- [ ] **Step 3: Implement**

Restructure the top of `run_arm` so one dict is both what is exported and what
is printed:

```python
    log_path = repo_root / "logs" / "lora_regret" / f"{arm.name}.log"
    # One dict, used for both the real environment and the dry-run preview --
    # so a previewed line cannot omit the per-arm SAVE_DIR that keeps
    # concurrent arms from overwriting each other.
    overrides = dict(arm_env(arm))
    overrides.update(
        {
            "LAUNCHER_NAME": arm.name,
            "RUN_LOG": str(log_path),
            "WANDB_GROUP": f"lora-regret-{'rl' if metric == 'accuracy' else 'sft'}",
            "SAVE_DIR": str(repo_root / "orbit_ckpts" / "lora_regret" / arm.name),
        }
    )
    env = dict(os.environ)
    env.update(overrides)
    cmd = ["bash", str(repo_root / launcher)]
    if dry_run:
        printed = " ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        print(f"{printed} bash {launcher}")
        return
```

- [ ] **Step 4: Fix the pre-existing `TestDryRunOutput` expectations**

That class asserts the old, shorter line. Update its assertions to `in line`
membership checks rather than equality — the additions are correct, so the test
should pin what must be present, not the exact string. Run:

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py -q -p no:cacheprovider
```
Expected: PASS, including both dry-run classes.

- [ ] **Step 5: Commit**

```bash
git add tools/lora_regret/sweep.py tests/fast/utils/test_lora_regret_sweep.py
git commit -m "fix(lora_regret): print the per-arm SAVE_DIR in the dry-run command"
```

---

### Task 4: `adapter_params` — the number E3 and E5 rest on

**Files:**
- Modify: `tools/lora_regret/arms.py` (add `adapter_param_count`)
- Modify: `tools/lora_regret/sweep.py` (add `--num-layers`, thread it into `run_arm`)
- Modify: `tests/fast/utils/test_lora_regret_sweep.py` (add a class)

**Interfaces:**
- Consumes: `orbit.utils.peft_param_match.{megatron_module_shapes,
  lora_param_count_for_modules, oft_param_count_for_modules}`
- Produces: `arms.adapter_param_count(arm, hidden_size, ffn_size, num_layers,
  qkv_output_size=LLAMA31_8B_QKV_OUTPUT) -> int | None`; `sweep.run_arm` gains a
  keyword `adapter_params: int | None = None`; CLI gains required `--num-layers`

The formula is **verified, not assumed**: measured 2026-07-30 against the
complete adapter at
`/lustre/fast/fast/zqiu/tmp/smoke_ckpt_20260730/iter_0000001/adapter/`, which
holds exactly 570,425,344 bf16 parameters over 256 tensors and 32 layers. The
test pins that constant so a refactor cannot drift away from a checkpoint that
is no longer read.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_sweep.py`:

```python
from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT, adapter_param_count

# Counted from the real adapter written by the 2026-07-30 smoke:
# 256 tensors, 32 layers, all bf16. Analytic and measured agree exactly, and
# E3's and E5's matched-parameter claims rest on that agreement.
SMOKE_R256_ALL_MODULES_PARAMS = 570_425_344


class TestAdapterParamCount:
    def test_matches_the_real_r256_adapter(self):
        arm = Arm("lora-r256-all-lr0.00025-s0", "lora", 256, None, ALL_MODULES, 2.5e-4, 0)
        assert (
            adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)
            == SMOKE_R256_ALL_MODULES_PARAMS
        )

    def test_attention_only_counts_only_attention_modules(self):
        arm = Arm("lora-r256-attn-lr0.00025-s0", "lora", 256, None, ATTN_MODULES, 2.5e-4, 0)
        # linear_qkv 256*(4096+6144) + linear_proj 256*(4096+4096), times 32.
        assert adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT) == (
            256 * (4096 + 6144) + 256 * (4096 + 4096)
        ) * 32

    def test_full_finetuning_has_no_adapter(self):
        arm = Arm("full-na-na-lr2.5e-05-s0", "full", None, None, "", 2.5e-5, 0)
        assert adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT) is None

    def test_oft_uses_the_block_size_not_a_rank(self):
        arm = Arm("oft-b64-all-lr0.0001-s0", "oft", None, 64, ALL_MODULES, 1e-4, 0)
        count = adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)
        assert count > 0
        # OFT's count follows d_in and ignores d_out, so it must NOT equal the
        # LoRA count for any rank that happens to share the arm's tag.
        lora = Arm("lora-r64-all-x", "lora", 64, None, ALL_MODULES, 1e-4, 0)
        assert count != adapter_param_count(lora, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)

    def test_an_unknown_target_module_raises(self):
        arm = Arm("lora-r16-na-x", "lora", 16, None, "linear_nonexistent", 1e-4, 0)
        with pytest.raises(ValueError, match="no known module"):
            adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)


class TestLedgerCarriesAdapterParams:
    def test_the_record_reports_the_count(self, tmp_path, monkeypatch):
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(
                _build_log([_render(0, 0, _PHASE_AFTER_TRAIN, 1.5)])
            )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        arm = Arm("lora-r256-all-lr0.00025-s0", "lora", 256, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(
            arm, tmp_path, results, dry_run=False,
            adapter_params=SMOKE_R256_ALL_MODULES_PARAMS,
        )
        record = json.loads(results.read_text().splitlines()[0])
        assert record["adapter_params"] == SMOKE_R256_ALL_MODULES_PARAMS
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py::TestAdapterParamCount -q -p no:cacheprovider
```
Expected: collection error — `ImportError: cannot import name 'adapter_param_count'`

- [ ] **Step 3: Implement `adapter_param_count` in `arms.py`**

`arms.py` already imports `megatron_module_shapes` and
`oft_param_count_for_modules` (lines 33–41) — **do not re-add them**. Only one
name is missing; add it to the existing block in alphabetical position:

```python
    lora_param_count_for_modules,
```

so the block reads `ATTENTION_MODULES, lora_param_count_for_modules,
matched_mlp_rank, matched_oft_block_size, megatron_module_shapes,
oft_block_size_matching_params, oft_lora_match_report,
oft_param_count_for_modules`. Then, after `arm_env`:

```python
def adapter_param_count(
    arm: Arm,
    hidden_size: int,
    ffn_size: int,
    num_layers: int,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
) -> int | None:
    """Trainable adapter parameters for this arm, or None for full fine-tuning.

    Analytic rather than read back from a written checkpoint, so it is available
    at dry-run time -- before compute is spent -- and so E3's and E5's
    matched-parameter claims can be checked against the arm that is *about* to
    run. Verified exact against the real 2026-07-30 r256 adapter
    (570,425,344 parameters); see the plan's Task 4.

    `None` for `full` arms is meaningful, not missing: full fine-tuning has no
    adapter, and recording 0 would read as "an adapter with no parameters".
    """
    if arm.method == "full":
        return None
    shapes = megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)
    wanted = [name.strip() for name in arm.target_modules.split(",") if name.strip()]
    selected = {name: shape for name, shape in shapes.items() if name in wanted}
    if not selected:
        raise ValueError(
            f"arm {arm.name!r} targets no known module: {arm.target_modules!r} "
            f"(known: {sorted(shapes)})"
        )
    if arm.method == "lora":
        per_layer = lora_param_count_for_modules(arm.rank, selected)
    elif arm.method == "oft":
        per_layer = oft_param_count_for_modules(arm.oft_block_size, selected)
    else:
        raise ValueError(f"unknown method {arm.method!r}")
    return per_layer * num_layers
```

- [ ] **Step 4: Thread it through `sweep.py`**

Add the parameter to `run_arm`'s signature after `metric`:

```python
    adapter_params: int | None = None,
```

and replace the hardcoded `"adapter_params": None,` in the record dict with
`"adapter_params": adapter_params,`.

Add the CLI argument next to `--ffn-size`:

```python
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="Decoder layers in the model (32 for Llama-3.1-8B). Required rather "
        "than defaulted, like --hidden-size and --ffn-size: a wrong value makes "
        "every adapter_params in the ledger wrong by a constant factor.",
    )
```

and in `main`'s run loop:

```python
    for i, arm in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {arm.name}", file=sys.stderr)
        run_arm(
            arm, repo_root, args.results, args.dry_run,
            launcher=launcher, metric=metric,
            adapter_params=adapter_param_count(
                arm, args.hidden_size, args.ffn_size, args.num_layers
            ),
        )
```

Import `adapter_param_count` from `tools.lora_regret.arms` at the top of
`sweep.py`.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py -q -p no:cacheprovider
```
Expected: PASS.

- [ ] **Step 6: Verify the CLI still builds every matrix**

```bash
for m in e1 e2 e3 e4 e5scout sft82; do
  echo -n "$m = "
  python -m tools.lora_regret.sweep --matrix $m --hidden-size 4096 --ffn-size 14336 \
    --num-layers 32 --dry-run 2>/dev/null | wc -l
done
```
Expected: `e1 = 40`, `e2 = 36`, `e3 = 20`, `e4 = 16`, `e5scout = 5`, `sft82 = 82`

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py tests/fast/utils/test_lora_regret_sweep.py
git commit -m "feat(lora_regret): record realized adapter parameter counts per arm"
```

---

### Task 5: `p3_check.py` — assert the DP>1 reduction

**Files:**
- Create: `tools/lora_regret/p3_check.py`
- Create: `tests/fast/utils/test_lora_regret_p3_check.py`

**Interfaces:**
- Consumes: `trace.parse_trace_file`, `trace.NllPoint` (Task 1)
- Produces: `compare_traces(a, b, decimals=6) -> list[str]` (empty means equal),
  `main()` exiting 0 or 1

P3 gates every FullFT number in the campaign: the held-out NLL reduces
`(sum_neg_logprob, n_tokens)` over the DP group only, DP=1 makes that reduction
a no-op, and P0 forces DP>1 for every FullFT arm. Acceptance is equality at the
sixth decimal across three fields in two logs — a comparison a human does badly.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_lora_regret_p3_check.py`:

```python
"""P3: the DP>1 held-out NLL reduction must equal the DP=1 answer.

A differing `tokens` means the reduction double-counts or drops a shard, and no
amount of averaging fixes the FullFT numbers downstream -- so this exits
non-zero rather than warning.
"""

import pytest

from tools.lora_regret.p3_check import compare_traces
from tools.lora_regret.trace import PHASE_AFTER_TRAIN, PHASE_BEFORE_TRAIN, NllPoint


def _point(step, nll, phase=PHASE_AFTER_TRAIN, tokens=308760, samples=1000):
    return NllPoint(step, step, phase, nll, nll + 0.2, tokens, samples)


class TestCompareTraces:
    def test_identical_traces_compare_equal(self):
        trace = [_point(0, 1.209810, PHASE_BEFORE_TRAIN), _point(1, 1.194836)]
        assert compare_traces(trace, list(trace)) == []

    def test_a_differing_nll_is_reported_with_both_values(self):
        a = [_point(1, 1.194836)]
        b = [_point(1, 1.194837)]
        problems = compare_traces(a, b)
        assert len(problems) == 1
        assert "1.194836" in problems[0] and "1.194837" in problems[0]

    def test_a_differing_token_count_names_the_shard_failure(self):
        a = [_point(1, 1.194836, tokens=308760)]
        b = [_point(1, 1.194836, tokens=617520)]
        problems = compare_traces(a, b)
        assert len(problems) == 1
        assert "tokens" in problems[0]
        assert "shard" in problems[0]

    def test_nll_equality_is_to_six_decimals_not_exact_float(self):
        """The logs print %.6f, so comparing beyond six decimals compares noise."""
        assert compare_traces([_point(1, 1.1948360000001)], [_point(1, 1.194836)]) == []

    def test_a_missing_measurement_is_a_problem_not_a_silent_skip(self):
        a = [_point(0, 1.2, PHASE_BEFORE_TRAIN), _point(1, 1.1)]
        b = [_point(1, 1.1)]
        problems = compare_traces(a, b)
        assert any("only in" in p for p in problems)

    def test_two_empty_traces_are_a_problem_not_a_pass(self):
        """Two runs that logged nothing must not read as two runs that agreed."""
        problems = compare_traces([], [])
        assert problems
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_p3_check.py -q -p no:cacheprovider
```
Expected: collection error — no module `tools.lora_regret.p3_check`

- [ ] **Step 3: Implement**

Create `tools/lora_regret/p3_check.py`:

```python
"""P3: assert the DP>1 held-out NLL reduction matches the DP=1 answer.

    python -m tools.lora_regret.p3_check logs/p3_dp1_*.log logs/p3_dp4_*.log

The eval reduces `(sum_neg_logprob, n_tokens)` over the **DP group only** --
TP/PP replicas hold identical samples, DP shards hold different token counts.
That code has never executed at DP>1, and P0 forces DP>1 for every FullFT arm,
so every FullFT number in the campaign is downstream of this check.

Exits 1 on any mismatch. A differing `tokens` in particular means the reduction
is double-counting or dropping a shard; the correct response is to stop, not to
average.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.lora_regret.trace import NllPoint, parse_trace_file


def compare_traces(
    dp1: list[NllPoint],
    dpn: list[NllPoint],
    decimals: int = 6,
) -> list[str]:
    """Problems found, empty if the two traces agree. Never raises.

    Measurements are paired by `(phase, step)` rather than by position: the two
    runs may log at different wall-clock moments and interleave differently, but
    a measurement at the same phase and step is the same measurement.

    `nll` is compared to `decimals` places because train.py prints `%.6f` --
    comparing the parsed floats exactly would compare digits the log never
    carried.
    """
    if not dp1 or not dpn:
        return [
            f"empty trace: dp1 has {len(dp1)} measurements, dpN has {len(dpn)}; "
            "two runs that logged nothing are not two runs that agreed"
        ]
    left = {(p.phase, p.step): p for p in dp1}
    right = {(p.phase, p.step): p for p in dpn}
    problems: list[str] = []
    for key in sorted(set(left) - set(right)):
        problems.append(f"{key[0]} step={key[1]}: only in the dp1 log")
    for key in sorted(set(right) - set(left)):
        problems.append(f"{key[0]} step={key[1]}: only in the dpN log")
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        where = f"{key[0]} step={key[1]}"
        if round(a.nll, decimals) != round(b.nll, decimals):
            problems.append(f"{where}: nll {a.nll:.{decimals}f} != {b.nll:.{decimals}f}")
        if a.tokens != b.tokens:
            problems.append(
                f"{where}: tokens {a.tokens} != {b.tokens} -- the DP reduction is "
                "double-counting or dropping a shard"
            )
        if a.samples != b.samples:
            problems.append(
                f"{where}: samples {a.samples} != {b.samples} -- the held-out set "
                "differs between the two runs, so the comparison is not a DP test"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dp1_log", type=Path, help="log from the GPUS_PER_NODE=1 run")
    parser.add_argument("dpn_log", type=Path, help="log from the GPUS_PER_NODE=N run")
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()

    dp1 = parse_trace_file(args.dp1_log)
    dpn = parse_trace_file(args.dpn_log)
    print(f"dp1: {len(dp1)} measurements from {args.dp1_log}")
    print(f"dpN: {len(dpn)} measurements from {args.dpn_log}")

    problems = compare_traces(dp1, dpn, args.decimals)
    if problems:
        print("\nP3 FAILED -- do not trust any FullFT number:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"\nP3 PASSED: {len(dp1)} measurements identical to {args.decimals} decimals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_p3_check.py -q -p no:cacheprovider
```
Expected: PASS

- [ ] **Step 5: Prove the CLI runs, using the committed fixture as both sides**

```bash
python -m tools.lora_regret.p3_check \
  tests/fast/fixtures/lora_regret/smoke_lora_r256_eval_lines.log \
  tests/fast/fixtures/lora_regret/smoke_lora_r256_eval_lines.log
echo "exit=$?"
```
Expected: `P3 PASSED: 3 measurements identical to 6 decimals`, `exit=0`

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/p3_check.py tests/fast/utils/test_lora_regret_p3_check.py
git commit -m "feat(lora_regret): assert the DP>1 held-out NLL reduction matches DP=1"
```

---

### Task 6: `analyze.py` — σ, argmins, and the edge-of-grid rule

**Files:**
- Create: `tools/lora_regret/analyze.py`
- Create: `tests/fast/utils/test_lora_regret_analyze.py`

**Interfaces:**
- Consumes: ledger records written by `sweep.run_arm` (Tasks 2 and 4)
- Produces: `load_records(paths, *, seed=0, require_ok=True) -> list[dict]`,
  `sigma(records) -> float`, `ArmKey = tuple[str, int | None, str]`
  (method, size, target_modules),
  `argmins(records) -> dict[ArmKey, dict]`,
  `lr_grids(records) -> dict[ArmKey, list[float]]`,
  `edge_of_grid(records) -> dict[ArmKey, str]`

Three rules are implemented once here and reused by every claim in Task 7.

**Grid points are seed 0 only.** E1-0's replicates live in the same ledger
directory and are not grid points; the runbook records a synthetic-ledger
measurement where dropping this filter let a replicate at LR 9.95e-4 steal
r256's argmin from the real 2.5e-4.

**Edge-of-grid is an error, not a note.** An argmin at either end of its LR grid
means the true optimum may lie outside it, so the ratio is not quotable. The
runbook's instruction is **re-centre, not extend**, and the message says so.

**An inconsistent trace disqualifies an arm.** `trace_consistent is False` means
the held-out set changed size mid-run, so that arm's NLL is not comparable.

- [ ] **Step 1: Write the failing tests**

Create `tests/fast/utils/test_lora_regret_analyze.py`:

```python
"""Reading the ledger into claims.

Every detector here has a case it must REJECT. A detector with only passing
cases is untested, and these decide whether ~800 GPU-hours produced a result or
an artifact.
"""

import json

import pytest

from tools.lora_regret.analyze import (
    argmins,
    edge_of_grid,
    load_records,
    lr_grids,
    sigma,
)


ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
ATTN = "linear_qkv,linear_proj"
FULL_KEY = ("full", None, "")


def _key(method, size, modules=ALL):
    """The 3-tuple ArmKey. target_modules is part of it because E3 runs
    `lora r256 attn` and `lora r256 all` in one matrix."""
    return (method, size, modules)


def _record(method, rank, lr, nll, seed=0, status="ok", modules=None, **extra):
    record = {
        "arm": f"{method}-r{rank}-all-lr{lr:g}-s{seed}",
        "method": method,
        "rank": rank,
        "oft_block_size": None,
        "target_modules": ("" if method == "full" else ALL) if modules is None else modules,
        "lr": lr,
        "seed": seed,
        "metric": "nll",
        "test_nll": nll,
        "status": status,
        "trace_consistent": True,
        "trace_warning": None,
        "nll_trace": None,
        "adapter_params": None,
        "global_batch_size": None,
        "dataset": None,
        "steps": 2000,
    }
    record.update(extra)
    return record


def _ledger(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


class TestLoadRecords:
    def test_failed_arms_are_dropped(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 2.5e-4, 1.5),
            _record("lora", 16, 5.0e-4, 1.4, status="failed"),
        ])
        assert len(load_records([path])) == 1

    def test_non_zero_seeds_are_dropped_by_default(self, tmp_path):
        """E1-0's replicates share a ledger directory and are not grid points.

        The runbook records the concrete failure: a seed-1 replicate at
        LR 9.95e-4 stealing r256's argmin from the real 2.5e-4.
        """
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 256, 2.5e-4, 1.50),
            _record("lora", 256, 2.5e-4, 1.49, seed=1),
        ])
        assert [r["seed"] for r in load_records([path])] == [0]
        assert len(load_records([path], seed=None)) == 2

    def test_an_inconsistent_trace_disqualifies_the_arm(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 2.5e-4, 1.5),
            _record("lora", 16, 5.0e-4, 1.4, trace_consistent=False,
                    trace_warning="samples=[992, 1000]"),
        ])
        kept = load_records([path])
        assert [r["lr"] for r in kept] == [2.5e-4]

    def test_a_glob_reads_every_shard(self, tmp_path):
        _ledger(tmp_path, "e1_lora_a.jsonl", [_record("lora", 1, 2.5e-4, 1.9)])
        _ledger(tmp_path, "e1_lora_b.jsonl", [_record("lora", 16, 2.5e-4, 1.6)])
        assert len(load_records([str(tmp_path / "e1_*.jsonl")])) == 2


class TestSigma:
    def test_is_the_standard_deviation_of_the_replicates(self, tmp_path):
        path = _ledger(tmp_path, "s.jsonl", [
            _record("lora", 256, 2.5e-4, 1.200000, seed=0),
            _record("lora", 256, 2.5e-4, 1.201000, seed=1),
            _record("lora", 256, 2.5e-4, 1.202000, seed=2),
        ])
        assert sigma(load_records([path], seed=None)) == pytest.approx(0.001, rel=1e-6)

    def test_refuses_fewer_than_three_replicates(self, tmp_path):
        path = _ledger(tmp_path, "s.jsonl", [
            _record("lora", 256, 2.5e-4, 1.20, seed=0),
            _record("lora", 256, 2.5e-4, 1.21, seed=1),
        ])
        with pytest.raises(ValueError, match="at least 3"):
            sigma(load_records([path], seed=None))


class TestArgmins:
    def test_picks_the_lowest_nll_per_arm(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 1.0e-4, 1.60),
            _record("lora", 16, 2.5e-4, 1.50),
            _record("lora", 16, 5.0e-4, 1.55),
            _record("full", None, 2.5e-5, 1.45),
        ])
        best = argmins(load_records([path]))
        assert best[_key("lora", 16)]["lr"] == 2.5e-4
        assert best[FULL_KEY]["lr"] == 2.5e-5

    def test_same_rank_different_placement_are_different_arms(self, tmp_path):
        """E3's collision case. A (method, rank) key would report one r256.

        `lora r256 attention-only` and `lora r256 all-modules` are both in the
        e3 matrix, and C4 is precisely the comparison between placements -- so
        collapsing them would delete the claim while appearing to answer it.
        """
        path = _ledger(tmp_path, "e3.jsonl", [
            _record("lora", 256, 2.5e-4, 1.50, modules=ALL),
            _record("lora", 256, 2.5e-4, 1.44, modules=ATTN),
        ])
        best = argmins(load_records([path]))
        assert len(best) == 2
        assert best[_key("lora", 256, ALL)]["test_nll"] == 1.50
        assert best[_key("lora", 256, ATTN)]["test_nll"] == 1.44


class TestEdgeOfGrid:
    def _grid(self, tmp_path, best_index):
        lrs = [1.0e-4, 1.5e-4, 2.5e-4, 4.0e-4, 6.3e-4]
        records = [
            _record("lora", 16, lr, 1.5 + (0.0 if i == best_index else 0.1))
            for i, lr in enumerate(lrs)
        ]
        return load_records([_ledger(tmp_path, "a.jsonl", records)])

    def test_fires_on_the_lowest_grid_point(self, tmp_path):
        flagged = edge_of_grid(self._grid(tmp_path, 0))
        assert _key("lora", 16) in flagged
        assert "re-centre" in flagged[_key("lora", 16)]

    def test_fires_on_the_highest_grid_point(self, tmp_path):
        assert _key("lora", 16) in edge_of_grid(self._grid(tmp_path, 4))

    def test_silent_one_grid_point_in(self, tmp_path):
        """The non-tautology case: an interior argmin must NOT be flagged."""
        assert edge_of_grid(self._grid(tmp_path, 1)) == {}
        assert edge_of_grid(self._grid(tmp_path, 3)) == {}

    def test_a_single_point_grid_is_flagged(self, tmp_path):
        """One LR is simultaneously the lowest and highest point tried."""
        records = load_records([_ledger(tmp_path, "a.jsonl", [_record("lora", 16, 2.5e-4, 1.5)])])
        assert _key("lora", 16) in edge_of_grid(records)


class TestLrGrids:
    def test_reports_the_sorted_grid_actually_run(self, tmp_path):
        path = _ledger(tmp_path, "a.jsonl", [
            _record("lora", 16, 5.0e-4, 1.55),
            _record("lora", 16, 1.0e-4, 1.60),
        ])
        assert lr_grids(load_records([path]))[_key("lora", 16)] == [1.0e-4, 5.0e-4]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
python -m pytest tests/fast/utils/test_lora_regret_analyze.py -q -p no:cacheprovider
```
Expected: collection error — no module `tools.lora_regret.analyze`

- [ ] **Step 3: Implement the core of `analyze.py`**

Create `tools/lora_regret/analyze.py` with this content (Task 7 appends the
claim readers and the CLI to the same file):

```python
"""Read the sweep ledgers into the campaign's claims.

Every difference this module prints is in units of sigma, measured by E1-0 --
never off absolute loss values. The constant Orbit-vs-HF precision offset
(0.0032 nats) cancels in every ratio, ordering and curve-shape claim the
campaign makes, and cancels in nothing else.
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

# (method, size, target_modules). `size` is the rank for LoRA, the block size
# for OFT, and None for full fine-tuning.
#
# target_modules is part of the key and must NOT be dropped: E3 runs
# `lora r256 attention-only` and `lora r256 all-modules` in the same matrix, so
# a (method, rank) key would silently collapse two different arms into one and
# report whichever happened to score better as "the r256 argmin". That is the
# exact class of bug the seed-0 filter exists to prevent, one axis over.
ArmKey = tuple[str, int | None, str]


def load_records(
    paths,
    *,
    seed: int | None = 0,
    require_ok: bool = True,
    metric: str = "nll",
) -> list[dict]:
    """Ledger records worth analysing, from files or globs.

    `seed=0` is the default and is not cosmetic: E1-0's replicates live in the
    same ledger directory at seeds 1 and 2 and are *not* grid points. Measured
    on a synthetic ledger, dropping this filter let a replicate at LR 9.95e-4
    win r256's argmin away from the real 2.5e-4 purely because that one run
    happened to score better. Pass `seed=None` to read replicates, which is
    what `sigma` wants and nothing else does.

    Arms whose trace was inconsistent are dropped: a held-out set that changed
    size mid-run makes that arm's NLL incomparable to the others.
    """
    records: list[dict] = []
    for entry in paths:
        matches = sorted(glob.glob(str(entry))) or [str(entry)]
        for match in matches:
            path = Path(match)
            if not path.exists():
                raise FileNotFoundError(f"no ledger at {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated final line from an interrupted write
                if require_ok and record.get("status") != "ok":
                    continue
                if seed is not None and record.get("seed") != seed:
                    continue
                if record.get("metric", "nll") != metric:
                    continue
                if record.get("trace_consistent") is False:
                    continue
                records.append(record)
    return records


def arm_key(record: dict) -> ArmKey:
    size = record.get("oft_block_size") if record["method"] == "oft" else record.get("rank")
    return (record["method"], size, record.get("target_modules") or "")


def score(record: dict, metric: str = "nll") -> float:
    return record["accuracy"] if metric == "accuracy" else record["test_nll"]


def sigma(records: list[dict]) -> float:
    """Seed-to-seed standard deviation, from E1-0's replicates.

    Load with `seed=None`: the replicates are seeds 0, 1 and 2 of one
    configuration, and the default seed-0 filter would leave one point.
    """
    values = [score(r) for r in records]
    if len(values) < 3:
        raise ValueError(
            f"sigma needs at least 3 replicates, got {len(values)}. "
            "Run E1-0 (runbook section 7) and load with seed=None."
        )
    return statistics.stdev(values)


def lr_grids(records: list[dict]) -> dict[ArmKey, list[float]]:
    """The learning rates actually run, per arm, sorted."""
    grids: dict[ArmKey, set[float]] = {}
    for record in records:
        grids.setdefault(arm_key(record), set()).add(record["lr"])
    return {key: sorted(values) for key, values in grids.items()}


def argmins(records: list[dict], metric: str = "nll") -> dict[ArmKey, dict]:
    """The best-scoring record per arm.

    Lower is better for NLL, higher for accuracy -- the direction is chosen by
    `metric` rather than assumed, because E4's ledgers score by accuracy.
    """
    better = (lambda a, b: a > b) if metric == "accuracy" else (lambda a, b: a < b)
    best: dict[ArmKey, dict] = {}
    for record in records:
        key = arm_key(record)
        if key not in best or better(score(record, metric), score(best[key], metric)):
            best[key] = record
    return best


def edge_of_grid(records: list[dict], metric: str = "nll") -> dict[ArmKey, str]:
    """Arms whose argmin sits on a boundary of the grid that was run.

    An argmin at either end means the true optimum may lie outside the grid, so
    any ratio quoted from it is a lower bound on an unknown. The runbook's rule
    is to **re-centre, not extend**: extending keeps the old points at the wrong
    spacing and leaves the grid asymmetric about the new estimate.

    A one-point grid is flagged, because a single LR is simultaneously the
    lowest and the highest that was tried.
    """
    grids = lr_grids(records)
    flagged: dict[ArmKey, str] = {}
    for key, best in argmins(records, metric).items():
        grid = grids[key]
        if best["lr"] in (grid[0], grid[-1]):
            flagged[key] = (
                f"argmin LR {best['lr']:g} is on the edge of the grid "
                f"[{grid[0]:g} .. {grid[-1]:g}] ({len(grid)} points); "
                "re-centre the grid on it and re-run before quoting a ratio"
            )
    return flagged
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_analyze.py -q -p no:cacheprovider
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/lora_regret/analyze.py tests/fast/utils/test_lora_regret_analyze.py
git commit -m "feat(lora_regret): read sigma, argmins and edge-of-grid from the ledgers"
```

---

### Task 7: `analyze.py` — the claim readers and the CLI

**Files:**
- Modify: `tools/lora_regret/analyze.py` (append)
- Modify: `tests/fast/utils/test_lora_regret_analyze.py` (append)

**Interfaces:**
- Consumes: everything from Task 6, plus `trace.NllPoint` / `trace.parse_trace_file`
- Produces: `departure_steps(traces, sigma_value, *, threshold_sigma=2.0,
  consecutive=3) -> dict[str, int | None]`, `lr_band(records, sigma_value,
  metric) -> dict[ArmKey, tuple[float, float]]`, and a `main()` with
  subcommands `sigma argmins c1 c2 c3 c4 c5 c6 all`

**C1's departure rule**, stated once: per rank, the first step at which its NLL
exceeds the pointwise minimum across all arms by more than 2σ for **three
consecutive** logging intervals. A rank that never departs and a rank whose run
was too short look identical, so the reader reports the step budget beside every
departure point and says `no departure within N steps` rather than leaving a
blank.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fast/utils/test_lora_regret_analyze.py`:

```python
from tools.lora_regret.analyze import (
    batch_gaps,
    departure_steps,
    lr_band,
    placement_deltas,
)
from tools.lora_regret.trace import PHASE_AFTER_TRAIN, NllPoint


def _trace(nlls, start=1):
    return [
        NllPoint(i, i, PHASE_AFTER_TRAIN, nll, nll, 308760, 1000)
        for i, nll in enumerate(nlls, start=start)
    ]


class TestDepartureSteps:
    SIGMA = 0.001

    def test_an_arm_that_tracks_the_envelope_never_departs(self):
        traces = {"r512": _trace([1.5, 1.4, 1.3, 1.2]), "r256": _trace([1.5, 1.4, 1.3, 1.2])}
        assert departure_steps(traces, self.SIGMA) == {"r512": None, "r256": None}

    def test_reports_the_first_step_of_three_consecutive_excursions(self):
        # r1 exceeds the envelope by 10 sigma from step 2 onward.
        traces = {
            "r512": _trace([1.50, 1.40, 1.30, 1.20, 1.10]),
            "r1": _trace([1.50, 1.41, 1.31, 1.21, 1.11]),
        }
        assert departure_steps(traces, self.SIGMA)["r1"] == 2

    def test_does_not_fire_on_two_consecutive_excursions(self):
        """The non-tautology case: the rule says three, so two must not count."""
        traces = {
            "r512": _trace([1.50, 1.40, 1.30, 1.20, 1.10]),
            "r1": _trace([1.50, 1.41, 1.31, 1.20, 1.10]),
        }
        assert departure_steps(traces, self.SIGMA)["r1"] is None

    def test_an_excursion_under_two_sigma_does_not_count(self):
        traces = {
            "r512": _trace([1.5000, 1.4000, 1.3000, 1.2000]),
            "r16": _trace([1.5000, 1.4015, 1.3015, 1.2015]),  # 1.5 sigma
        }
        assert departure_steps(traces, self.SIGMA)["r16"] is None

    def test_an_empty_trace_is_none_not_a_crash(self):
        traces = {"r512": _trace([1.5, 1.4, 1.3]), "r1": []}
        assert departure_steps(traces, self.SIGMA)["r1"] is None


class TestLrBand:
    def test_the_band_spans_every_lr_within_two_sigma_of_the_best(self, tmp_path):
        path = _ledger(tmp_path, "e4.jsonl", [
            _record("lora", 1, 1e-6, 0.30),
            _record("lora", 1, 1e-5, 0.44),
            _record("lora", 1, 1e-4, 0.4395),
            _record("lora", 1, 1e-3, 0.10),
        ])
        records = load_records([path])
        for record in records:  # an accuracy ledger, scored the other direction
            record["metric"] = "accuracy"
            record["accuracy"] = record.pop("test_nll")
        band = lr_band(records, 0.001, metric="accuracy")
        assert band[_key("lora", 1)] == (1e-5, 1e-4)


class TestBatchGaps:
    """C3: the LoRA-minus-FullFT gap at each batch size, in sigma.

    The claim is a gap that GROWS with batch. A constant offset at all three
    batch sizes is not the signature and must be distinguishable from it, which
    means grouping by batch -- impossible until global_batch_size reached the
    ledger (Task 2).
    """

    def test_groups_by_batch_size(self, tmp_path):
        rows = []
        for batch, full_nll, lora_nll in [(32, 1.50, 1.502), (512, 1.40, 1.45)]:
            rows.append(_record("full", None, 2.5e-5, full_nll, global_batch_size=batch))
            rows.append(_record("lora", 256, 2.5e-4, lora_nll, global_batch_size=batch))
        gaps = batch_gaps(load_records([_ledger(tmp_path, "e2.jsonl", rows)]), 0.001)
        assert gaps[(32, _key("lora", 256))] == pytest.approx(2.0, abs=1e-6)
        assert gaps[(512, _key("lora", 256))] == pytest.approx(50.0, abs=1e-6)

    def test_a_batch_with_no_fullft_arm_is_skipped_not_guessed(self, tmp_path):
        rows = [_record("lora", 256, 2.5e-4, 1.45, global_batch_size=512)]
        gaps = batch_gaps(load_records([_ledger(tmp_path, "e2.jsonl", rows)]), 0.001)
        assert gaps == {}


class TestPlacementDeltas:
    """C4: NLL(attention) - NLL(MLP) at matched parameters, in sigma."""

    def test_pairs_attention_against_mlp(self, tmp_path):
        rows = [
            _record("lora", 256, 2.5e-4, 1.500, modules=ATTN),
            _record("lora", 92, 2.5e-4, 1.503, modules="linear_fc1,linear_fc2"),
        ]
        deltas = placement_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001)
        assert deltas["attn(r256) - mlp(r92)"] == pytest.approx(-3.0, abs=1e-6)

    def test_no_mlp_arm_yields_no_comparison(self, tmp_path):
        rows = [_record("lora", 256, 2.5e-4, 1.500, modules=ATTN)]
        assert placement_deltas(load_records([_ledger(tmp_path, "e3.jsonl", rows)]), 0.001) == {}
```

- [ ] **Step 2: Run them to verify they fail**

```bash
python -m pytest tests/fast/utils/test_lora_regret_analyze.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'departure_steps'`

- [ ] **Step 3: Implement the readers**

Append to `tools/lora_regret/analyze.py`:

```python
def departure_steps(
    traces: dict[str, list],
    sigma_value: float,
    *,
    threshold_sigma: float = 2.0,
    consecutive: int = 3,
) -> dict[str, int | None]:
    """Per arm, the step at which it leaves the shared envelope -- C1's number.

    The envelope is the pointwise minimum NLL across all arms at each step. An
    arm departs at the first step of the first run of `consecutive` steps where
    it sits more than `threshold_sigma` sigma above that envelope. Requiring a
    run of three is what keeps a single noisy eval from reading as a departure.

    `None` means "no departure within this arm's trace", which is NOT the same
    as "does not depart" -- the caller must print the step budget alongside.
    """
    envelope: dict[int, float] = {}
    for points in traces.values():
        for point in points:
            step = point.step
            if step not in envelope or point.nll < envelope[step]:
                envelope[step] = point.nll

    limit = threshold_sigma * sigma_value
    departures: dict[str, int | None] = {}
    for name, points in traces.items():
        run_start: int | None = None
        run_length = 0
        departures[name] = None
        for point in sorted(points, key=lambda p: p.step):
            if point.nll - envelope[point.step] > limit:
                if run_start is None:
                    run_start = point.step
                run_length += 1
                if run_length >= consecutive:
                    departures[name] = run_start
                    break
            else:
                run_start, run_length = None, 0
    return departures


def lr_band(
    records: list[dict],
    sigma_value: float,
    metric: str = "nll",
    *,
    threshold_sigma: float = 2.0,
) -> dict[ArmKey, tuple[float, float]]:
    """Per arm, the lowest and highest LR scoring within `threshold_sigma` of its best.

    C5's second half is about the *width* of the performant band, which is a
    separate checkable statement from peak parity: LoRA's band being wider is a
    claim that survives even if the peaks tie.
    """
    best = argmins(records, metric)
    bands: dict[ArmKey, tuple[float, float]] = {}
    for key, top in best.items():
        top_score = score(top, metric)
        within = [
            r["lr"]
            for r in records
            if arm_key(r) == key
            and abs(score(r, metric) - top_score) <= threshold_sigma * sigma_value
        ]
        bands[key] = (min(within), max(within))
    return bands


def batch_gaps(
    records: list[dict],
    sigma_value: float,
) -> dict[tuple[int | None, ArmKey], float]:
    """C3: `best_LoRA(batch) - best_FullFT(batch)` at each batch size, in sigma.

    The claim is a gap that *grows* with batch -- a gap absent at 32 and present
    at 512 is the signature, a constant offset at all three is not -- so the
    comparison has to be made within each batch size, never pooled. A batch with
    no FullFT arm is skipped rather than compared against another batch's
    baseline: that would attribute a batch-size effect to a placement it never
    had.
    """
    by_batch: dict[int | None, list[dict]] = {}
    for record in records:
        by_batch.setdefault(record.get("global_batch_size"), []).append(record)
    gaps: dict[tuple[int | None, ArmKey], float] = {}
    for batch, group in by_batch.items():
        best = argmins(group)
        baseline = next((v for k, v in best.items() if k[0] == "full"), None)
        if baseline is None:
            continue
        for key, record in best.items():
            if key[0] == "full":
                continue
            gaps[(batch, key)] = (record["test_nll"] - baseline["test_nll"]) / sigma_value
    return gaps


def placement_deltas(records: list[dict], sigma_value: float) -> dict[str, float]:
    """C4: `NLL(attention) - NLL(MLP)` at matched parameters, in sigma.

    Pairs each attention-only arm with each MLP-only arm, labelled by both
    ranks, because the matched pair is `attention r256` against `MLP r92` and
    the post's own pair (`r256`/`r128`) is deliberately in the same matrix -- if
    the two disagree, the disagreement is parameter accounting rather than
    physics, and collapsing them to one number would hide exactly that.
    """
    from orbit.utils.peft_param_match import ATTENTION_MODULES, MLP_MODULES

    attn_set, mlp_set = set(ATTENTION_MODULES), set(MLP_MODULES)

    def modules_of(key: ArmKey) -> set[str]:
        return {name for name in key[2].split(",") if name}

    best = argmins(records)
    attn = {k: v for k, v in best.items() if modules_of(k) == attn_set}
    mlp = {k: v for k, v in best.items() if modules_of(k) == mlp_set}
    deltas: dict[str, float] = {}
    for attn_key, attn_record in attn.items():
        for mlp_key, mlp_record in mlp.items():
            label = f"attn(r{attn_key[1]}) - mlp(r{mlp_key[1]})"
            deltas[label] = (attn_record["test_nll"] - mlp_record["test_nll"]) / sigma_value
    return deltas
```

- [ ] **Step 4: Implement the CLI**

Append to `tools/lora_regret/analyze.py`:

```python
_MODULE_SHORT = {
    "linear_qkv,linear_proj,linear_fc1,linear_fc2": "all",
    "linear_qkv,linear_proj": "attn",
    "linear_fc1,linear_fc2": "mlp",
}


def _fmt_key(key: ArmKey) -> str:
    method, size, modules = key
    if method == "full":
        return "full"
    label = "b" if method == "oft" else "r"
    return f"{method} {label}{size} {_MODULE_SHORT.get(modules, modules)}"


def _load_traces(records: list[dict], log_dir: Path) -> tuple[dict[str, list], dict[str, str]]:
    """Traces per arm, preferring the ledger's own field over re-reading a log.

    Reports the source per arm: a silently-empty trace and a silently-truncated
    one both read as "no departure", so which file the number came from is part
    of the answer.
    """
    from tools.lora_regret.trace import NllPoint, parse_trace_file

    traces: dict[str, list] = {}
    sources: dict[str, str] = {}
    for record in records:
        name = record["arm"]
        if record.get("nll_trace"):
            traces[name] = [NllPoint(**point) for point in record["nll_trace"]]
            sources[name] = "ledger"
            continue
        log_path = log_dir / f"{name}.log"
        if log_path.exists():
            traces[name] = parse_trace_file(log_path)
            sources[name] = str(log_path)
        else:
            traces[name] = []
            sources[name] = "MISSING -- no nll_trace field and no log"
    return traces, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["sigma", "argmins", "c1", "c2", "c3", "c4", "c5", "c6", "all"],
    )
    parser.add_argument("--ledgers", nargs="+", required=True, help="paths or globs")
    parser.add_argument(
        "--sigma-ledger",
        nargs="+",
        default=None,
        help="E1-0's replicate ledger. Required by every claim but 'sigma' itself, "
        "unless --sigma is given.",
    )
    parser.add_argument("--sigma", type=float, default=None, help="override the measured sigma")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/lora_regret"))
    parser.add_argument(
        "--allow-edge-argmin",
        action="store_true",
        help="quote claims that depend on an argmin sitting on a grid edge. Off by "
        "default: the runbook's rule is to re-centre and re-run first.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Both metrics, loaded separately. An E4 ledger carries metric="accuracy"
    # and test_nll=null, so loading only the nll view and bailing on empty would
    # make `analyze c5 --ledgers results/e4_*.jsonl` exit before it ran.
    records = load_records(args.ledgers)
    acc_records = load_records(args.ledgers, metric="accuracy")
    if not records and not acc_records:
        print("no usable records in the given ledgers", file=sys.stderr)
        return 1

    if args.command == "sigma":
        value = sigma(load_records(args.ledgers, seed=None))
        print(f"sigma = {value:.6f} nats  (n={len(load_records(args.ledgers, seed=None))})")
        return 0

    sigma_value = args.sigma
    if sigma_value is None and args.sigma_ledger:
        sigma_value = sigma(load_records(args.sigma_ledger, seed=None))
    if sigma_value is None and args.command != "argmins":
        print(
            "no sigma: pass --sigma-ledger results/e1_0_sigma.jsonl or --sigma VALUE. "
            "Every difference this campaign claims is quoted in units of sigma, and "
            "the Qwen3-era 0.000992 does not transfer to Llama-3.1-8B / Tulu3.",
            file=sys.stderr,
        )
        return 2

    flagged = edge_of_grid(records)
    best = argmins(records)
    grids = lr_grids(records)
    order = lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])  # noqa: E731

    if args.command in ("argmins", "all"):
        print(f"{'arm':22} {'argmin_lr':<11} {'nll':<10} {'adapter_params':>15}  grid")
        for key, record in sorted(best.items(), key=order):
            grid = grids[key]
            params = record.get("adapter_params")
            print(
                f"{_fmt_key(key):22} {record['lr']:<11g} {record['test_nll']:<10.6f} "
                f"{params if params is not None else '-':>15}  "
                f"[{grid[0]:g} .. {grid[-1]:g}]"
                + ("   EDGE OF GRID" if key in flagged else "")
            )
    if flagged and not args.allow_edge_argmin:
        print("\nedge-of-grid arms -- re-centre and re-run before quoting:", file=sys.stderr)
        for key, why in flagged.items():
            print(f"  {_fmt_key(key)}: {why}", file=sys.stderr)
        if args.command != "argmins":
            return 3

    all_modules = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
    if args.command in ("c2", "all"):
        lora = best.get(("lora", 256, all_modules))
        full = best.get(("full", None, ""))
        if lora and full:
            print(f"\nC2: argmin_LR(LoRA r256) / argmin_LR(FullFT) = {lora['lr'] / full['lr']:.2f}")
            print("    the post predicts 9.8, rising toward 15 for runs under ~100 steps")
            edges = [best.get(("lora", r, all_modules)) for r in (4, 512)]
            if all(edges):
                lrs = [record["lr"] for record in edges]
                print(
                    f"    rank 4 vs 512 argmin spread = {max(lrs) / min(lrs):.2f}x "
                    "(the tighter claim is < 2x)"
                )

    if args.command in ("c1", "all"):
        traces, sources = _load_traces(records, args.log_dir)
        departures = departure_steps(traces, sigma_value)
        print("\nC1: departure from the envelope (>2 sigma for 3 consecutive evals)")
        for name in sorted(departures):
            budget = max((p.step for p in traces[name]), default=0)
            where = departures[name]
            verdict = f"step {where}" if where is not None else f"no departure within {budget} steps"
            print(f"    {name:34} {verdict:38} [trace: {sources[name]}]")

    if args.command in ("c3", "all"):
        gaps = batch_gaps(records, sigma_value)
        if gaps:
            print(f"\nC3: best_LoRA - best_FullFT per batch (sigma = {sigma_value:.6f})")
            print("    the claim is a gap that GROWS with batch; a constant offset is not it")
            for (batch, key), delta in sorted(gaps.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
                print(f"    batch {str(batch):>4}  {_fmt_key(key):22} {delta:+8.2f} sigma")

    if args.command in ("c4", "all"):
        deltas = placement_deltas(records, sigma_value)
        if deltas:
            print(f"\nC4: placement at matched parameters (sigma = {sigma_value:.6f})")
            for label, delta in sorted(deltas.items()):
                print(f"    {label:28} {delta:+8.2f} sigma")

    if args.command in ("c6", "all"):
        oft = {k: v for k, v in best.items() if k[0] == "oft"}
        if oft:
            print(f"\nC6: OFT against LoRA at matched parameters (sigma = {sigma_value:.6f})")
            for key, record in sorted(oft.items(), key=order):
                ratio = record.get("matched_ratio")
                params = record.get("adapter_params")
                # Mind the direction: an OFT arm carrying slightly FEWER
                # parameters that still keeps up strengthens the finding, while
                # one carrying fewer and losing is confounded, not informative.
                suffix = f"  matched_ratio={ratio:.3f}" if ratio is not None else "  matched_ratio=?"
                print(
                    f"    {_fmt_key(key):22} nll={record['test_nll']:.6f} "
                    f"params={params}{suffix}"
                )

    if args.command in ("c5", "all"):
        acc = acc_records
        if acc:
            print("\nC5: peak accuracy and performant-LR band")
            peaks = argmins(acc, metric="accuracy")
            bands = lr_band(acc, sigma_value, metric="accuracy")
            for key in sorted(peaks, key=lambda k: (k[0], k[1] or 0, k[2])):
                low, high = bands[key]
                print(
                    f"    {_fmt_key(key):22} peak={peaks[key]['accuracy']:.4f} "
                    f"band=[{low:g} .. {high:g}] ({high / low:.0f}x wide)"
                )
            print(
                "    NOTE: sigma for accuracy has never been measured. These deltas "
                "are raw and none of them is resolved. Measuring it means an E1-0 "
                "for accuracy: 3 seeds of one E4 arm."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add `import argparse` and `import sys` to the module's top-level imports (Task 6
created it with only `glob`, `json`, `statistics` and `pathlib`).

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_analyze.py -q -p no:cacheprovider
```
Expected: PASS

- [ ] **Step 6: Prove the CLI runs end to end on a synthetic ledger**

```bash
python - <<'PY'
import json, pathlib
pathlib.Path("/tmp/lr_demo").mkdir(exist_ok=True)
ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
rows = []
for lr, nll in [(1e-4, 1.60), (1.5e-4, 1.54), (2.5e-4, 1.50), (4e-4, 1.53), (6.3e-4, 1.58)]:
    rows.append({"arm": f"lora-r256-all-lr{lr:g}-s0", "method": "lora", "rank": 256,
                 "oft_block_size": None, "target_modules": ALL,
                 "lr": lr, "seed": 0, "metric": "nll", "test_nll": nll, "status": "ok",
                 "trace_consistent": True, "adapter_params": 570425344, "steps": 2000,
                 "global_batch_size": None, "dataset": "tulu3"})
for lr, nll in [(1e-5, 1.52), (2.5e-5, 1.47), (6.3e-5, 1.51)]:
    rows.append({"arm": f"full-na-na-lr{lr:g}-s0", "method": "full", "rank": None,
                 "oft_block_size": None, "target_modules": "",
                 "lr": lr, "seed": 0, "metric": "nll", "test_nll": nll, "status": "ok",
                 "trace_consistent": True, "adapter_params": None, "steps": 2000,
                 "global_batch_size": None, "dataset": "tulu3"})
pathlib.Path("/tmp/lr_demo/e1.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
sig = [{"arm": f"lora-r256-all-lr0.00025-s{s}", "method": "lora", "rank": 256, "lr": 2.5e-4,
        "oft_block_size": None, "target_modules": ALL,
        "seed": s, "metric": "nll", "test_nll": 1.500 + s*0.001, "status": "ok",
        "trace_consistent": True} for s in (0, 1, 2)]
pathlib.Path("/tmp/lr_demo/sigma.jsonl").write_text("".join(json.dumps(r)+"\n" for r in sig))
PY
python -m tools.lora_regret.analyze c2 --ledgers /tmp/lr_demo/e1.jsonl \
  --sigma-ledger /tmp/lr_demo/sigma.jsonl
```
Expected: `C2: argmin_LR(LoRA r256) / argmin_LR(FullFT) = 10.00`, exit 0.

Then prove the guard bites — the same command with the r256 argmin moved to the
grid edge must exit 3:

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("/tmp/lr_demo/e1_edge.jsonl")
rows = [json.loads(l) for l in pathlib.Path("/tmp/lr_demo/e1.jsonl").read_text().splitlines()]
for r in rows:  # make the lowest LoRA LR the winner
    if r["method"] == "lora":
        r["test_nll"] = 1.40 if r["lr"] == 1e-4 else 1.60
p.write_text("".join(json.dumps(r)+"\n" for r in rows))
PY
python -m tools.lora_regret.analyze c2 --ledgers /tmp/lr_demo/e1_edge.jsonl \
  --sigma-ledger /tmp/lr_demo/sigma.jsonl; echo "exit=$?"
```
Expected: `EDGE OF GRID` on stderr and `exit=3`.

- [ ] **Step 7: Commit**

```bash
git add tools/lora_regret/analyze.py tests/fast/utils/test_lora_regret_analyze.py
git commit -m "feat(lora_regret): read C1-C6 off the ledgers in units of sigma"
```

---

### Task 8: the `e1long` matrix

**Files:**
- Modify: `tools/lora_regret/arms.py` (`Arm`, `arm_env`, `e1long_arms`, `MATRICES`)
- Modify: `tests/fast/utils/test_lora_regret_sweep.py` (add a class)

**Interfaces:**
- Consumes: a plain `dict[(method, rank), float]` of learning rates — **not**
  `analyze.ArmKey`, which is a 3-tuple including `target_modules`. E1's arms are
  all-modules without exception, so the placement axis carries no information
  here; Task 9's `argmins_from` does the projection and refuses it if a ledger
  actually holds two placements at one rank. Keeping the narrower key means
  `arms.py` does not import `analyze.py`
- Produces: `arms.e1long_arms(argmins: dict[tuple[str, int | None], float],
  seed: int = 0) -> list[Arm]`; `Arm` gains `full_epoch: bool = False` and
  `eval_nll_interval: int | None = None`; `MATRICES["e1long"]`

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_sweep.py`:

```python
from tools.lora_regret.arms import E1LONG_EVAL_INTERVAL, e1long_arms

E1LONG_ARGMINS = {
    ("full", None): 2.5e-5,
    ("lora", 1): 5.0e-4,
    ("lora", 4): 4.0e-4,
    ("lora", 16): 2.5e-4,
    ("lora", 64): 2.5e-4,
    ("lora", 128): 2.5e-4,
    ("lora", 256): 2.5e-4,
    ("lora", 512): 1.5e-4,
}


class TestE1LongMatrix:
    def test_one_arm_per_rank_at_its_own_argmin(self):
        arms = e1long_arms(E1LONG_ARGMINS)
        assert len(arms) == 8
        by_key = {(a.method, a.rank): a for a in arms}
        assert set(by_key) == set(E1LONG_ARGMINS)
        assert by_key[("lora", 512)].lr == 1.5e-4
        assert by_key[("full", None)].lr == 2.5e-5

    def test_every_arm_runs_a_full_epoch(self):
        assert all(a.full_epoch for a in e1long_arms(E1LONG_ARGMINS))

    def test_num_rollout_is_emptied_not_omitted(self):
        """A NUM_ROLLOUT=2000 left exported from E1-1 must not shorten the curve.

        The launcher spells it ${NUM_ROLLOUT:-$((...))} -- the colon form -- so an
        empty value re-derives the full epoch, while omitting the key would let
        the stale export through and turn a 29,323-step curve into a 2,000-step
        one. Every rank would then look like it never departs.
        """
        env = arm_env(e1long_arms(E1LONG_ARGMINS)[0])
        assert env["NUM_ROLLOUT"] == ""

    def test_the_eval_interval_is_about_one_percent_of_the_epoch(self):
        env = arm_env(e1long_arms(E1LONG_ARGMINS)[0])
        assert env["EVAL_NLL_INTERVAL"] == str(E1LONG_EVAL_INTERVAL)
        assert 250 <= E1LONG_EVAL_INTERVAL <= 350

    def test_ordinary_arms_set_neither_knob(self):
        """The non-tautology case: e1's arms must be unchanged by this."""
        env = arm_env(e1_arms()[0])
        assert "NUM_ROLLOUT" not in env
        assert "EVAL_NLL_INTERVAL" not in env

    def test_a_missing_rank_is_refused(self):
        partial = {k: v for k, v in E1LONG_ARGMINS.items() if k != ("lora", 512)}
        with pytest.raises(ValueError, match="missing"):
            e1long_arms(partial)

    def test_arms_train_on_tulu3(self):
        assert all(a.dataset == "tulu3" for a in e1long_arms(E1LONG_ARGMINS))
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py::TestE1LongMatrix -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'e1long_arms'`

- [ ] **Step 3: Implement in `arms.py`**

Add two fields to the `Arm` dataclass, after `dataset`:

```python
    # E1-2 only. The long curves must run a full Tulu3 epoch, and the launcher
    # derives that itself -- but only if NUM_ROLLOUT is unset or empty.
    full_epoch: bool = False
    # Explicit so the long curves get ~100 trace points instead of the
    # launcher's default of 10, which would cost 37 h of eval per arm.
    eval_nll_interval: int | None = None
```

Add the constant near the other centres:

```python
# One Tulu3 epoch is (939,343 - 1,000 held out) / 32 = 29,323 optimizer steps,
# and ~1% of that is 293 -- about 100 trace points, which is what C1's departure
# detector needs, for ~1.9 h of eval against ~70 h of training. At the
# launcher's default of 10 the same arm would spend ~55 h evaluating.
E1LONG_EVAL_INTERVAL = 293
E1LONG_RANKS = (1, 4, 16, 64, 128, 256, 512)
```

Add the matrix builder after `e1_arms`:

```python
def e1long_arms(
    argmins: dict[tuple[str, int | None], float],
    seed: int = 0,
) -> list[Arm]:
    """E1-2: the long learning curves that decide C1.

    Eight runs -- one per E1 arm, each at *that arm's own* argmin LR from E1-1,
    each a full Tulu3 epoch. Eight rather than forty precisely because E1-1 has
    already located the learning rates: run at a shared LR instead and a rank
    that departs early is indistinguishable from a rank whose LR was too high.

    `argmins` maps `(method, rank)` to a learning rate. A missing key raises
    rather than being skipped: eight arms silently becoming five would look like
    a completed stage.
    """
    wanted: list[tuple[str, int | None]] = [("full", None)] + [("lora", r) for r in E1LONG_RANKS]
    missing = [key for key in wanted if key not in argmins]
    if missing:
        raise ValueError(
            f"e1long is missing an argmin for {missing}; run E1-1 to completion first "
            "(runbook section 8)"
        )
    arms: list[Arm] = []
    for method, rank in wanted:
        lr = argmins[(method, rank)]
        modules = "" if method == "full" else ALL_MODULES
        tag = "na" if method == "full" else f"r{rank}"
        arms.append(
            Arm(
                _name(method, tag, modules, lr, seed, extra="long"),
                method,
                rank,
                None,
                modules,
                lr,
                seed,
                dataset="tulu3",
                full_epoch=True,
                eval_nll_interval=E1LONG_EVAL_INTERVAL,
            )
        )
    return arms
```

In `arm_env`, immediately after the `env = {"LR": ..., "SEED": ...}` line:

```python
    if arm.full_epoch:
        # The EMPTY STRING, not an omitted key. The launcher spells it
        # ${NUM_ROLLOUT:-$((...))} -- the colon form re-derives on an empty
        # value, so this both requests the full epoch and immunises the arm
        # against a NUM_ROLLOUT=2000 left exported in the shell from E1-1.
        env["NUM_ROLLOUT"] = ""
    if arm.eval_nll_interval is not None:
        env["EVAL_NLL_INTERVAL"] = str(arm.eval_nll_interval)
```

Register the matrix. Every lambda in `MATRICES` gains an `argmins` keyword so
the dispatch stays uniform:

```python
MATRICES = {
    "sft82": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: sft_arms(hidden, ffn, seed=seed),
    "e1": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1_arms(seed=seed),
    "e1long": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e1long_arms(argmins, seed=seed),
    "e2": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e2_arms(seed=seed),
    "e3": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e3_arms(hidden, ffn, seed=seed),
    "e4": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e4_arms(seed=seed),
    "e5scout": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e5_scout_arms(hidden, ffn, seed=seed),
    "e5": lambda hidden, ffn, seed, oft_lr_centre=None, argmins=None: e5_arms(
        hidden, ffn, seed=seed, oft_lr_centre=oft_lr_centre
    ),
}
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py -q -p no:cacheprovider
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/lora_regret/arms.py tests/fast/utils/test_lora_regret_sweep.py
git commit -m "feat(lora_regret): add the e1long matrix for the full-epoch curves"
```

---

### Task 9: `--argmins-from`, and the guards that make it fail closed

**Files:**
- Modify: `tools/lora_regret/sweep.py` (`main`, plus `MATRIX_LAUNCHERS` / `MATRIX_METRICS`)
- Modify: `tests/fast/utils/test_lora_regret_sweep.py` (add a class)

**Interfaces:**
- Consumes: `analyze.load_records`, `analyze.argmins`, `analyze.edge_of_grid`
  (Task 6); `arms.e1long_arms` (Task 8)
- Produces: `sweep.argmins_from(patterns, allow_edge) -> dict[(method, rank), float]`
  — the 2-tuple `e1long_arms` takes, projected from `analyze`'s 3-tuple key;
  CLI gains `--argmins-from` and `--allow-edge-argmin`

Two fail-closed guards. Fewer than eight recovered argmins means a partial
ledger, and running three arms that look like a completed stage is worse than
stopping. An argmin on a grid edge means the LR that E1-2 would spend ~70 GPU
hours at is a boundary value, not an optimum — the most expensive place in the
campaign to act on an unchecked number.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_lora_regret_sweep.py`:

```python
class TestArgminsFrom:
    def _ledger(self, tmp_path, rows):
        path = tmp_path / "e1.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def _row(self, method, rank, lr, nll, seed=0):
        return {
            "arm": f"{method}-r{rank}-all-lr{lr:g}-s{seed}", "method": method, "rank": rank,
            "oft_block_size": None,
            "target_modules": "" if method == "full" else ALL_MODULES,
            "lr": lr, "seed": seed, "metric": "nll", "test_nll": nll, "status": "ok",
            "trace_consistent": True, "global_batch_size": None, "dataset": None,
        }

    def _complete(self):
        rows = []
        for lr, nll in [(1e-5, 1.52), (2.5e-5, 1.47), (6.3e-5, 1.51)]:
            rows.append(self._row("full", None, lr, nll))
        for rank in (1, 4, 16, 64, 128, 256, 512):
            for lr, nll in [(1e-4, 1.60), (2.5e-4, 1.50), (6.3e-4, 1.58)]:
                rows.append(self._row("lora", rank, lr, nll))
        return rows

    def test_recovers_one_lr_per_arm(self, tmp_path):
        path = self._ledger(tmp_path, self._complete())
        found = sweep.argmins_from([str(path)], allow_edge=False)
        assert len(found) == 8
        assert found[("lora", 256)] == 2.5e-4
        assert found[("full", None)] == 2.5e-5

    def test_a_partial_ledger_is_refused(self, tmp_path):
        """Three arms that look like a completed stage is the failure to avoid."""
        rows = [r for r in self._complete() if r["rank"] in (None, 1, 4)]
        path = self._ledger(tmp_path, rows)
        with pytest.raises(SystemExit):
            sweep.argmins_from([str(path)], allow_edge=False)

    def test_an_edge_of_grid_argmin_is_refused(self, tmp_path):
        rows = self._complete()
        for row in rows:  # make r512's lowest LR win
            if row["rank"] == 512:
                row["test_nll"] = 1.40 if row["lr"] == 1e-4 else 1.60
        path = self._ledger(tmp_path, rows)
        with pytest.raises(SystemExit):
            sweep.argmins_from([str(path)], allow_edge=False)

    def test_the_edge_override_lets_it_through(self, tmp_path):
        rows = self._complete()
        for row in rows:
            if row["rank"] == 512:
                row["test_nll"] = 1.40 if row["lr"] == 1e-4 else 1.60
        path = self._ledger(tmp_path, rows)
        found = sweep.argmins_from([str(path)], allow_edge=True)
        assert found[("lora", 512)] == 1e-4


class TestE1LongCliGuards:
    def _run(self, tmp_path, extra):
        return subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.sweep",
             "--hidden-size", "4096", "--ffn-size", "14336", "--num-layers", "32",
             "--dry-run", *extra],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )

    def test_e1long_without_argmins_exits_two(self, tmp_path):
        result = self._run(tmp_path, ["--matrix", "e1long"])
        assert result.returncode == 2
        assert "--argmins-from" in result.stderr

    def test_argmins_from_on_another_matrix_exits_two(self, tmp_path):
        result = self._run(tmp_path, ["--matrix", "e1", "--argmins-from", "results/x.jsonl"])
        assert result.returncode == 2
        assert "e1long" in result.stderr
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py::TestArgminsFrom -q -p no:cacheprovider
```
Expected: `AttributeError: module 'tools.lora_regret.sweep' has no attribute 'argmins_from'`

- [ ] **Step 3: Implement in `sweep.py`**

Register the new matrix in both dispatch tables:

```python
MATRIX_LAUNCHERS = {
    "sft82": LAUNCHER,
    "e1": LAUNCHER,
    "e1long": LAUNCHER,
    "e2": LAUNCHER,
    ...
}
MATRIX_METRICS = {
    "sft82": "nll",
    "e1": "nll",
    "e1long": "nll",
    "e2": "nll",
    ...
}
```

Add the helper, above `main`:

```python
def argmins_from(patterns: list[str], allow_edge: bool) -> dict[tuple[str, int | None], float]:
    """Each E1 arm's argmin learning rate, read from the E1-1 ledgers.

    Fails closed twice, because E1-2 is ~70 GPU-hours per arm and both failures
    are silent otherwise:

    - Fewer than 8 arms recovered means a partial ledger. Running the 3 arms
      that happen to be there would produce a stage that *looks* complete.
    - An argmin on a grid edge means the LR is a boundary value rather than an
      optimum. Spending 70 hours at it is the single most expensive way to act
      on an unchecked number, so it is refused unless overridden.
    """
    from tools.lora_regret.analyze import argmins, edge_of_grid, load_records

    records = load_records(patterns)
    best = argmins(records)
    # analyze keys on (method, size, target_modules); e1long keys on
    # (method, rank), because every E1 arm is all-modules and the long curves
    # inherit that. Project, and refuse to guess if the ledger actually holds
    # two placements at one rank -- that is an E3 ledger, not an E1 one.
    found: dict[tuple[str, int | None], float] = {}
    for (method, size, modules), record in best.items():
        key = (method, size)
        if key in found:
            sys.exit(
                f"--argmins-from found more than one placement for {key} "
                f"(latest: {modules!r}). These ledgers mix placements, so there is no "
                "single argmin per rank; point it at the E1 ledgers only."
            )
        found[key] = record["lr"]
    if len(found) < 8:
        sys.exit(
            f"--argmins-from recovered only {len(found)} arms from {patterns}: "
            f"{sorted(found)}. E1-2 needs all 8 (FullFT plus ranks "
            "1, 4, 16, 64, 128, 256, 512); finish E1-1 first."
        )
    flagged = edge_of_grid(records)
    if flagged and not allow_edge:
        lines = "\n".join(f"  {key}: {why}" for key, why in flagged.items())
        sys.exit(
            "--argmins-from refuses an edge-of-grid argmin:\n"
            f"{lines}\n"
            "Re-centre the grid and re-run those arms, or pass --allow-edge-argmin "
            "to spend ~70 GPU-hours per arm on a boundary value anyway."
        )
    return found
```

Add the CLI arguments beside `--oft-lr-centre`:

```python
    parser.add_argument(
        "--argmins-from",
        nargs="+",
        default=None,
        help="E1-1 ledger paths or globs. Required by --matrix e1long and "
        "meaningless elsewhere: the long curves only mean anything at each "
        "rank's own argmin learning rate, which E1-1 is what finds.",
    )
    parser.add_argument(
        "--allow-edge-argmin",
        action="store_true",
        help="Let --argmins-from accept an argmin sitting on a grid edge.",
    )
```

Add the guards next to the existing `e5` pair:

```python
    if args.matrix == "e1long" and args.argmins_from is None:
        parser.error(
            "--matrix e1long requires --argmins-from; run --matrix e1 to completion "
            "first and point this at its ledgers"
        )
    if args.matrix != "e1long" and args.argmins_from is not None:
        parser.error(f"--argmins-from is only meaningful for --matrix e1long, not {args.matrix}")
```

And thread the result into the matrix call:

```python
    recovered = (
        argmins_from(args.argmins_from, args.allow_edge_argmin) if args.argmins_from else None
    )
    arms = MATRICES[args.matrix](
        args.hidden_size, args.ffn_size, args.seed, args.oft_lr_centre, recovered
    )
```

Note the positional call: `MATRICES` lambdas take
`(hidden, ffn, seed, oft_lr_centre, argmins)` after Task 8.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_sweep.py -q -p no:cacheprovider
```
Expected: PASS

- [ ] **Step 5: Prove the end-to-end path on the synthetic ledger from Task 7**

```bash
python -m tools.lora_regret.sweep --matrix e1long --hidden-size 4096 --ffn-size 14336 \
  --num-layers 32 --argmins-from /tmp/lr_demo/e1.jsonl --dry-run 2>&1 | head -3
```
Expected: exits non-zero with `recovered only 2 arms` — the demo ledger has
FullFT and r256 only, which is exactly the partial-ledger guard firing.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/sweep.py tests/fast/utils/test_lora_regret_sweep.py
git commit -m "feat(lora_regret): drive E1-2 from the E1-1 argmins through the sweep"
```

---

### Task 10: `preflight.py` — fail before the reservation, not inside it

**Files:**
- Create: `tools/lora_regret/preflight.py`
- Create: `tests/fast/utils/test_lora_regret_preflight.py`

**Interfaces:**
- Consumes: `arms.MATRICES`, `arms.DATA_DIR`
- Produces: `Check` (NamedTuple: `name, ok, detail`),
  `check_data(data_dir) -> list[Check]`, `check_checkpoints(hf, megatron) -> list[Check]`,
  `check_matrices(hidden, ffn) -> list[Check]`, `check_env() -> list[Check]`,
  `check_gpus(stage) -> list[Check]`, `STAGE_GPU_REQUIREMENTS`

The env check asserts `__file__ is not None`, not merely that the import
succeeds. The failure mode this guards against — a venv of broken symlinks
pointing into a cleared uv cache — **imports successfully**: Python treats a
directory with no loadable `__init__.py` as a namespace package, so the symptom
is a missing attribute, not an `ImportError`.

- [ ] **Step 1: Write the failing test**

Create `tests/fast/utils/test_lora_regret_preflight.py`:

```python
"""Preflight fails on the ground, not in the air.

Everything here is checkable without a GPU and without the real data, because
the point is to run it *before* an allocation exists.
"""

import json

import pytest

from tools.lora_regret.preflight import (
    STAGE_GPU_REQUIREMENTS,
    Check,
    check_checkpoints,
    check_data,
    check_matrices,
)

EXPECTED_ROWS = {
    "tulu3_train.jsonl": 938_343,
    "tulu3_test.jsonl": 1_000,
}


class TestCheckData:
    def test_a_missing_split_fails_by_name(self, tmp_path):
        failures = [c for c in check_data(tmp_path) if not c.ok]
        assert any("tulu3_train.jsonl" in c.detail for c in failures)

    def test_a_truncated_split_fails_even_though_it_exists(self, tmp_path):
        """Existence is not enough: a short split silently changes E1's denominator."""
        (tmp_path / "tulu3_test.jsonl").write_text(
            "".join(json.dumps({"prompt": []}) + "\n" for _ in range(999))
        )
        checks = {c.name: c for c in check_data(tmp_path)}
        assert not checks["tulu3_test.jsonl"].ok
        assert "999" in checks["tulu3_test.jsonl"].detail
        assert "1000" in checks["tulu3_test.jsonl"].detail

    def test_a_correct_split_passes(self, tmp_path):
        (tmp_path / "tulu3_test.jsonl").write_text(
            "".join(json.dumps({"prompt": []}) + "\n" for _ in range(1000))
        )
        checks = {c.name: c for c in check_data(tmp_path)}
        assert checks["tulu3_test.jsonl"].ok

    def test_all_nine_splits_are_checked(self, tmp_path):
        assert len(check_data(tmp_path)) == 9


class TestCheckCheckpoints:
    def test_a_missing_megatron_checkpoint_fails(self, tmp_path):
        checks = {c.name: c for c in check_checkpoints(tmp_path, tmp_path / "nope")}
        assert not checks["megatron_load"].ok

    def test_a_megatron_dir_without_the_iteration_file_fails(self, tmp_path):
        (tmp_path / "mg").mkdir()
        checks = {c.name: c for c in check_checkpoints(tmp_path, tmp_path / "mg")}
        assert not checks["megatron_load"].ok
        assert "latest_checkpointed_iteration.txt" in checks["megatron_load"].detail

    def test_a_complete_megatron_dir_passes(self, tmp_path):
        (tmp_path / "mg").mkdir()
        (tmp_path / "mg" / "latest_checkpointed_iteration.txt").write_text("0")
        checks = {c.name: c for c in check_checkpoints(tmp_path, tmp_path / "mg")}
        assert checks["megatron_load"].ok


class TestCheckMatrices:
    def test_every_matrix_builds_at_its_documented_count(self):
        checks = {c.name: c for c in check_matrices(4096, 14336)}
        assert checks["matrix:e1"].ok and "40" in checks["matrix:e1"].detail
        assert checks["matrix:e2"].ok and "36" in checks["matrix:e2"].detail
        assert checks["matrix:e3"].ok and "20" in checks["matrix:e3"].detail
        assert checks["matrix:e4"].ok and "16" in checks["matrix:e4"].detail
        assert checks["matrix:e5scout"].ok and "5" in checks["matrix:e5scout"].detail
        assert checks["matrix:e5"].ok and "50" in checks["matrix:e5"].detail

    def test_a_matrix_that_raises_is_reported_not_propagated(self, monkeypatch):
        """A broken matrix must fail the preflight, not crash it.

        Preflight's whole value is telling you every problem at once; an
        uncaught exception in the third matrix hides the fourth.
        """
        import tools.lora_regret.preflight as preflight

        def boom(*_args, **_kwargs):
            raise ValueError("hidden_size and ffn_size must be positive")

        monkeypatch.setitem(preflight.MATRICES, "e1", boom)
        checks = {c.name: c for c in check_matrices(4096, 14336)}
        assert not checks["matrix:e1"].ok
        assert "ValueError" in checks["matrix:e1"].detail
        assert checks["matrix:e2"].ok  # the rest still ran

    def test_a_wrong_count_fails_even_though_the_matrix_builds(self, monkeypatch):
        """Not a tautology: the counts are pinned, not read back from the builder."""
        import tools.lora_regret.preflight as preflight

        monkeypatch.setitem(preflight.EXPECTED_ARMS, "e1", 41)
        checks = {c.name: c for c in check_matrices(4096, 14336)}
        assert not checks["matrix:e1"].ok
        assert "40 arms, expected 41" in checks["matrix:e1"].detail


class TestStageRequirements:
    def test_fullft_needs_four_gpus(self):
        assert STAGE_GPU_REQUIREMENTS["e1-full"] == 4

    def test_p3_needs_at_least_two(self):
        """DP=1 makes the reduction a no-op, so a 1-GPU 'P3' proves nothing."""
        assert STAGE_GPU_REQUIREMENTS["p3"] >= 2

    def test_rl_needs_eight(self):
        assert STAGE_GPU_REQUIREMENTS["e4"] == 8
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/fast/utils/test_lora_regret_preflight.py -q -p no:cacheprovider
```
Expected: collection error — no module `tools.lora_regret.preflight`

- [ ] **Step 3: Implement**

Create `tools/lora_regret/preflight.py`:

```python
"""Audit everything a reservation can discover expensively -- before it starts.

    python -m tools.lora_regret.preflight --stage e1-lora

Each check prints what it found. Exits non-zero if any required check failed,
so this can gate a job script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

from tools.lora_regret.arms import DATA_DIR, MATRICES

HF_CKPT = "/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B"
# Note: still under the *old* repo's path. Verified present (15 GB) on
# 2026-07-30; it is a cross-repo dependency rather than a break, which is
# exactly why it is checked here rather than assumed.
MEGATRON_LOAD = "/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints/Llama-3.1-8B_torch_dist"

# Measured counts from the 2026-07-30 materialization, not expectations. MATH is
# 7,498 rather than 7,500 because two number_theory rows carry an empty \boxed{}
# and an empty label can never be earned honestly.
EXPECTED_ROWS = {
    "tulu3_train.jsonl": 938_343,
    "tulu3_test.jsonl": 1_000,
    "openthoughts3_train.jsonl": 10_000,
    "openthoughts3_test.jsonl": 100,
    "math_train.jsonl": 7_498,
    "math_test.jsonl": 5_000,
    "gsm8k_train.jsonl": 7_473,
    "gsm8k_test.jsonl": 1_319,
    "math_gsm8k_train.jsonl": 14_971,
}

EXPECTED_ARMS = {"e1": 40, "e2": 36, "e3": 20, "e4": 16, "e5scout": 5, "e5": 50, "sft82": 82}

# What each stage needs before it is worth starting. P3 is 2 rather than 1
# because DP=1 makes the reduction it tests a no-op; FullFT is 4 for the
# 32 GB + 96 GB/N optimizer-state arithmetic the launcher enforces.
STAGE_GPU_REQUIREMENTS = {
    "smoke": 1,
    "e1-lora": 1,
    "e3": 1,
    "e5": 1,
    "p3": 2,
    "e1-full": 4,
    "e2-full": 4,
    "e4": 8,
}


class Check(NamedTuple):
    name: str
    ok: bool
    detail: str


def check_env() -> list[Check]:
    """Imports, and that each module has a real file behind it.

    `__file__ is not None` is the load-bearing half. The failure this guards
    against -- a venv of symlinks into a cleared uv cache -- *imports
    successfully*: Python treats a directory with no loadable __init__.py as a
    namespace package, so it presents as a missing attribute, not an ImportError.
    """
    checks = []
    for name in ("torch", "transformers", "megatron.core", "orbit"):
        try:
            module = __import__(name, fromlist=["__file__"])
            path = getattr(module, "__file__", None)
            version = getattr(module, "__version__", "?")
            if path is None:
                checks.append(Check(f"import:{name}", False,
                                    "imported as a namespace package with no __file__ -- "
                                    "the venv's symlinks are dangling; rebuild per INSTALL.md"))
            else:
                checks.append(Check(f"import:{name}", True, f"{version} at {path}"))
        except Exception as exc:  # noqa: BLE001 -- report any import failure verbatim
            checks.append(Check(f"import:{name}", False, f"{type(exc).__name__}: {exc}"))
    return checks


def check_gpus(stage: str | None) -> list[Check]:
    try:
        import torch

        count = torch.cuda.device_count()
        names = {torch.cuda.get_device_name(i) for i in range(count)}
    except Exception as exc:  # noqa: BLE001
        return [Check("gpus", False, f"could not query CUDA: {exc}")]
    detail = f"{count} device(s): {', '.join(sorted(names)) or 'none'}"
    if stage is None:
        return [Check("gpus", True, detail)]
    needed = STAGE_GPU_REQUIREMENTS[stage]
    return [Check("gpus", count >= needed, f"{detail}; stage {stage!r} needs >= {needed}")]


def check_checkpoints(hf_ckpt: str | Path, megatron_load: str | Path) -> list[Check]:
    hf_path, mg_path = Path(hf_ckpt), Path(megatron_load)
    checks = [
        Check("hf_checkpoint", hf_path.is_dir(),
              f"{hf_path}" if hf_path.is_dir() else f"missing: {hf_path}")
    ]
    marker = mg_path / "latest_checkpointed_iteration.txt"
    if not mg_path.is_dir():
        checks.append(Check("megatron_load", False, f"missing: {mg_path}"))
    elif not marker.exists():
        checks.append(Check("megatron_load", False,
                            f"{mg_path} exists but has no latest_checkpointed_iteration.txt"))
    else:
        checks.append(Check("megatron_load", True,
                            f"{mg_path} at iteration {marker.read_text().strip()}"))
    return checks


def check_data(data_dir: str | Path) -> list[Check]:
    """Row counts, not just existence.

    A truncated split silently changes the denominator of every E1 number, and
    it is indistinguishable from a good one by `ls`.
    """
    root = Path(data_dir)
    checks = []
    for name, expected in EXPECTED_ROWS.items():
        path = root / name
        if not path.exists():
            checks.append(Check(name, False, f"missing: {path}"))
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle)
        checks.append(
            Check(name, rows == expected,
                  f"{rows} rows" if rows == expected else f"{rows} rows, expected {expected}")
        )
    return checks


def check_matrices(hidden_size: int, ffn_size: int) -> list[Check]:
    """Every matrix builds, at the count the runbook documents.

    A matrix that raises does so here, in a second, rather than after Ray has
    started on a reserved node. `e1long` is excluded: it needs a real E1-1
    ledger, so its guard is tested by the sweep's own CLI instead.
    """
    checks = []
    for name, expected in EXPECTED_ARMS.items():
        try:
            centre = 1e-4 if name == "e5" else None
            built = MATRICES[name](hidden_size, ffn_size, 0, centre, None)
            checks.append(
                Check(f"matrix:{name}", len(built) == expected,
                      f"{len(built)} arms" if len(built) == expected
                      else f"{len(built)} arms, expected {expected}")
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"matrix:{name}", False, f"{type(exc).__name__}: {exc}"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_GPU_REQUIREMENTS), default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--hf-checkpoint", default=HF_CKPT)
    parser.add_argument("--megatron-load", default=MEGATRON_LOAD)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-size", type=int, default=14336)
    parser.add_argument("--skip-gpu", action="store_true", help="for CPU-only preflight")
    args = parser.parse_args()

    checks: list[Check] = []
    checks += check_env()
    if not args.skip_gpu:
        checks += check_gpus(args.stage)
    checks += check_checkpoints(args.hf_checkpoint, args.megatron_load)
    checks += check_data(args.data_dir)
    checks += check_matrices(args.hidden_size, args.ffn_size)

    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL':>4}] {check.name:{width}}  {check.detail}")

    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed -- do not start the reservation:", file=sys.stderr)
        for check in failed:
            print(f"  {check.name}: {check.detail}", file=sys.stderr)
        return 1
    print(f"\nall {len(checks)} checks passed"
          + (f" for stage {args.stage!r}" if args.stage else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/fast/utils/test_lora_regret_preflight.py -q -p no:cacheprovider
```
Expected: PASS

- [ ] **Step 5: Run preflight for real against the materialized data**

```bash
python -m tools.lora_regret.preflight --skip-gpu
```
Expected: every check `ok` — 4 imports, 2 checkpoints, 9 data files, 7 matrices —
and `all 22 checks passed`. This reads 3.4 GB of JSONL to count rows and takes
about a minute.

- [ ] **Step 6: Commit**

```bash
git add tools/lora_regret/preflight.py tests/fast/utils/test_lora_regret_preflight.py
git commit -m "feat(lora_regret): add the preflight audit for a reservation"
```

---

### Task 11: update the runbook to use the new tools

**Files:**
- Modify: `docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md`

No code and no tests — this is the task that makes the previous ten reachable by
an operator. Every command below must be pasted from a section above, not
retyped, so the doc cannot drift from what was built.

- [ ] **Step 1: Add a preflight step as the new §1.5**

After the §1 environment block, before §2:

````markdown
## 1.5 Preflight — run this first, every time

```bash
python -m tools.lora_regret.preflight --stage e1-lora
```

Checks the four imports have real files behind them (the dangling-symlink venv
imports *successfully*), the GPU count against the stage, both checkpoints, all
nine splits **at their row counts**, and that every matrix builds. Exits 1 with
the specific failure. Pass `--skip-gpu` to run it from a login node.
````

- [ ] **Step 2: Replace §4's hand-comparison with `p3_check`**

Replace the `grep 'eval/test_nll' logs/p3_dp1_*.log logs/p3_dp4_*.log` line and
the prose acceptance paragraph with:

````markdown
```bash
python -m tools.lora_regret.p3_check logs/p3_dp1_*.log logs/p3_dp4_*.log
```

**Acceptance is the exit code.** It pairs measurements by `(phase, step)` and
asserts `nll` equal to six decimals with `tokens` and `samples` exactly equal.
A differing `tokens` means the reduction is double-counting or dropping a shard;
it says so, and the correct response is to stop — every FullFT number downstream
is wrong, and no amount of averaging fixes it.
````

- [ ] **Step 3: Replace §8's E1-2 heredoc with the `e1long` matrix**

Delete the entire `python - <<'PY' | tee /tmp/e1_2_cmds.sh` block and its
following paragraph, replacing them with:

````markdown
```bash
python -m tools.lora_regret.sweep --matrix e1long \
  --hidden-size 4096 --ffn-size 14336 --num-layers 32 \
  --argmins-from 'results/e1_*.jsonl' \
  --results results/e1_2.jsonl
```

E1-2 goes through the same driver as every other stage, so it gets the per-arm
`SAVE_DIR`, the resume ledger and uniform result records — which matter most
here, on 70-hour arms.

`--argmins-from` fails closed twice. Fewer than 8 arms recovered means a partial
E1-1 ledger, and running the 3 that happen to be there would look like a
completed stage. An argmin on a grid edge means the LR is a boundary value
rather than an optimum, and E1-2 is the most expensive place in the campaign to
act on an unchecked number — `--allow-edge-argmin` overrides it if you have
decided otherwise.

`NUM_ROLLOUT` is set to the **empty string** by these arms, not omitted: the
launcher's `${NUM_ROLLOUT:-...}` re-derives the full epoch on an empty value,
which also immunises the stage against a `NUM_ROLLOUT=2000` left exported in
your shell from E1-1. `EVAL_NLL_INTERVAL` is 293, ~1% of the epoch.
````

- [ ] **Step 4: Replace §13's argmin heredoc with `analyze`**

Replace the `python - <<'PY'` argmin block with:

````markdown
```bash
python -m tools.lora_regret.analyze argmins --ledgers 'results/e1_*.jsonl'
python -m tools.lora_regret.analyze all \
  --ledgers 'results/e1_*.jsonl' --sigma-ledger results/e1_0_sigma.jsonl
```

The seed-0 filter, the edge-of-grid rule and the σ units are built in rather
than restated per reading. `argmins` marks edge-of-grid arms and still prints;
every *claim* subcommand exits 3 rather than quoting one, unless
`--allow-edge-argmin` is passed.
````

- [ ] **Step 5: Add the new tools to §"What is ready"**

Add these rows to the table at the top:

````markdown
| Preflight audit | `tools/lora_regret/preflight.py` |
| P3 DP-equality check | `tools/lora_regret/p3_check.py` |
| NLL trace extraction | `tools/lora_regret/trace.py` |
| σ, argmins, C1-C6 readings | `tools/lora_regret/analyze.py` |
````

- [ ] **Step 6: Add `--num-layers 32` to every `sweep` invocation in the doc**

It is now a required argument. Search the runbook for
`python -m tools.lora_regret.sweep` and add `--num-layers 32` to each. Verify
none was missed:

```bash
grep -n 'tools.lora_regret.sweep' docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md \
  | grep -v 'num-layers'
```
Expected: no output.

- [ ] **Step 7: Update the E1-2 row of §5's execution table**

Change the `§8 | E1-2: long curves` row's "Gated by" cell from
`E1-1's argmins` to `E1-1's argmins (via --argmins-from)`.

- [ ] **Step 8: Run the full suite one last time**

```bash
python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -3
```
Expected: 502 + the new tests passed, 0 failed.

- [ ] **Step 9: Commit**

```bash
git add docs/superpowers/plans/2026-07-30-lora-without-regret-runbook.md
git commit -m "docs(runbook): drive P3, E1-2 and the readings through the new tools"
```

---

## Self-review notes

**Spec coverage.** Gap 1 → Task 10. Gap 2 → Task 3. Gap 3 → Task 5. Gap 4 →
Tasks 6 and 7. Gap 5 → Task 4. Gap 6 → Tasks 1 and 2. The `e1long` matrix and
`--argmins-from` from spec §5 → Tasks 8 and 9. The runbook rewrite that spec §5
requires ("the §8 heredoc generator is deleted") → Task 11.

**Five defects found in a second review pass and fixed above.** Recorded because
each was invisible from the prose and only surfaced by checking the plan against
the code:

1. **C3 was unreadable.** `Arm` carries `global_batch_size` and `e2_arms` sets
   it, but `run_arm`'s record dropped it — so the batch size each E2 arm ran at
   survived only inside the arm's name. C3 groups by batch; without the field it
   cannot be computed at all. Task 2 now writes `global_batch_size` and
   `dataset`, with tests.
2. **`ArmKey` collapsed two E3 arms into one.** With a `(method, rank)` key,
   `lora r256 attention-only` and `lora r256 all-modules` — both in the `e3`
   matrix — share a key, so `argmins` would report whichever scored better as
   "the r256 argmin". C4 is *precisely* the comparison between those placements,
   so the bug would have deleted the claim while appearing to answer it. The key
   is now `(method, size, target_modules)`, and `argmins_from` projects to
   `(method, rank)` for E1 while refusing to guess if a ledger mixes placements.
3. **`analyze c5` exited before running.** `main` loaded with the default
   `metric="nll"` and bailed on an empty result; an E4 ledger is entirely
   `metric="accuracy"`, so pointing `c5` at `results/e4_*.jsonl` — the only
   sensible invocation — returned 1. Both views are now loaded and the guard
   fires only if both are empty.
4. **C3 and C4 were not implemented.** The first draft emitted a generic
   "delta vs FullFT" table that grouped by nothing, so it could not distinguish
   C3's *growing* gap from a constant offset, and never computed
   attention-minus-MLP at all. Now `batch_gaps` and `placement_deltas`, each
   with its own tests including a case it must refuse.
5. **Task 4 would have duplicated two imports.** `arms.py` already imports
   `megatron_module_shapes` and `oft_param_count_for_modules`; only
   `lora_param_count_for_modules` is missing.

**Naming consistency.** `ArmKey = tuple[str, int | None, str]` is used
identically across `analyze.py` (Tasks 6 and 7), and `sweep.argmins_from`
(Task 9) documents its projection to the 2-tuple `e1long_arms` (Task 8) takes.
`score(record, metric)` is the single accessor for both `test_nll` and
`accuracy`. `Check(name, ok, detail)` is the only preflight return shape.

**Known limitation, stated rather than hidden.** `placement_deltas` pairs every
attention-only arm with every MLP-only arm and labels each pair by both ranks.
That is deliberate — E3 deliberately contains two candidate matched pairs
(`r256`/`r92` from Orbit's fused layout and the post's own `r256`/`r128`) and
collapsing them to one number would hide a disagreement that is itself the
finding. It does mean the output has one row per pair rather than one row.
