# Math OFT BS128 Lower-Learning-Rate Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one resumable 8-GPU launcher that runs Math OFT BS128 at exactly `1e-7, 3e-7, 1e-6, 3e-6, 1e-5` under the unchanged E4 protocol.

**Architecture:** Add a focused five-arm matrix rather than changing E4's established grids. Route that matrix through the existing RL sweep metadata and preflight, then add a thin shell wrapper that selects the complete matrix and delegates execution and resume behavior to `campaign.sh`.

**Tech Stack:** Python 3.12, pytest, Bash, Orbit's `tools.lora_regret` sweep framework.

## Global Constraints

- Matrix key is exactly `e4oftb128low`; arm label is exactly `oftlow`.
- Learning rates are exactly `(1e-7, 3e-7, 1e-6, 3e-6, 1e-5)` in ascending order.
- Every arm is Math, OFT, BS128, seed 0 by default, and targets `linear_qkv,linear_proj,linear_fc1,linear_fc2`.
- Every arm uses the existing E4 RL launcher and accuracy metric on 8 GPUs for 150 rollouts.
- The dedicated ledger is `results/e4_math_oft_b128_low_lr.jsonl`.
- Existing E4 matrices, launchers, LR grids, and ledgers remain unchanged.
- Successful rows resume by skipping; failed or missing rows rerun through existing `campaign.sh` behavior.
- Implement with test-driven development: observe each new test fail before adding its production code.
- The design-time local `.venv` lacks pytest. Execute tests only in an environment with the repository's test dependencies; do not mutate the shared training environment merely to satisfy a local test command.

---

## File structure

- `tools/lora_regret/arms.py`: fixed LR tuple, focused arm builder, and matrix registration.
- `tools/lora_regret/sweep.py`: RL launcher, accuracy parser, and distinct W&B routing.
- `tools/lora_regret/preflight.py`: expected five-arm count and eight-GPU floor.
- `scripts/lora_regret/campaign.sh`: wrapper-selected preflight stage with `e4` as the backward-compatible default.
- `scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh`: operator entry point and dedicated ledger owner.
- `tests/fast/utils/test_math_oft_b128_low_lr_sweep.py`: structural, routing, preflight, and launcher contract tests.

### Task 1: Add the focused matrix and runtime metadata

**Files:**
- Create: `tests/fast/utils/test_math_oft_b128_low_lr_sweep.py`
- Modify: `tools/lora_regret/arms.py:1-30`
- Modify: `tools/lora_regret/arms.py:780-856`
- Modify: `tools/lora_regret/arms.py:1235-1287`
- Modify: `tools/lora_regret/sweep.py:65-94`
- Modify: `tools/lora_regret/sweep.py:148-162`
- Modify: `tools/lora_regret/preflight.py:39-67`

**Interfaces:**
- Produces: `E4_MATH_OFT_B128_LOW_LRS: tuple[float, ...]`.
- Produces: `e4_math_oft_b128_low_arms(hidden_size: int = 4096, ffn_size: int = 14336, seed: int = 0, qkv_output_size: int = 6144) -> list[Arm]`.
- Produces: registry and runtime metadata entries keyed by `e4oftb128low`.
- Consumes: `ALL_MODULES`, `Arm`, `_name`, `megatron_module_shapes`, `oft_lora_match_report`, and the standard matrix-builder calling convention.

- [ ] **Step 1: Write the failing focused matrix tests**

Create `tests/fast/utils/test_math_oft_b128_low_lr_sweep.py` with:

```python
from tools.lora_regret.arms import (
    ALL_MODULES,
    E4_MATH_OFT_B128_LOW_LRS,
    MATRICES,
    e4_arms,
    e4_math_oft_b128_low_arms,
)
from tools.lora_regret.preflight import EXPECTED_ARMS, STAGE_GPU_REQUIREMENTS
from tools.lora_regret.sweep import (
    MATRIX_LAUNCHERS,
    MATRIX_METRICS,
    MATRIX_PROJECTS,
    RL_LAUNCHER,
    wandb_project,
)

HIDDEN, FFN, QKV = 4096, 14336, 6144
EXPECTED_LRS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
EXPECTED_NAMES = (
    "oftlow-b128-all-math-lr1e-07-s0",
    "oftlow-b128-all-math-lr3e-07-s0",
    "oftlow-b128-all-math-lr1e-06-s0",
    "oftlow-b128-all-math-lr3e-06-s0",
    "oftlow-b128-all-math-lr1e-05-s0",
)


def _arms():
    return e4_math_oft_b128_low_arms(HIDDEN, FFN, seed=0, qkv_output_size=QKV)


def test_the_matrix_is_exactly_the_requested_five_point_sweep():
    arms = _arms()
    assert E4_MATH_OFT_B128_LOW_LRS == EXPECTED_LRS
    assert tuple(arm.lr for arm in arms) == EXPECTED_LRS
    assert tuple(arm.name for arm in arms) == EXPECTED_NAMES
    assert len({arm.name for arm in arms}) == 5


def test_every_arm_holds_capacity_dataset_placement_and_seed_fixed():
    arms = _arms()
    assert {arm.method for arm in arms} == {"oft"}
    assert {arm.oft_block_size for arm in arms} == {128}
    assert {arm.dataset for arm in arms} == {"math"}
    assert {arm.target_modules for arm in arms} == {ALL_MODULES}
    assert {arm.seed for arm in arms} == {0}
    assert all(arm.matched_ratio is not None for arm in arms)


def test_the_focused_arm_names_do_not_collide_with_original_e4():
    assert not ({arm.name for arm in _arms()} & {arm.name for arm in e4_arms()})


def test_the_matrix_registry_builds_the_same_five_arms():
    registered = MATRICES["e4oftb128low"](HIDDEN, FFN, QKV, 0, None, None)
    assert registered == _arms()


def test_runtime_metadata_routes_the_focused_matrix_as_e4_rl():
    assert MATRIX_LAUNCHERS["e4oftb128low"] == RL_LAUNCHER
    assert MATRIX_METRICS["e4oftb128low"] == "accuracy"
    assert MATRIX_PROJECTS["e4oftb128low"] == "rl-b128-low-lr"
    assert wandb_project("e4oftb128low", dataset="math", method="oft") == "math-rl-b128-low-lr-oft"
    assert EXPECTED_ARMS["e4oftb128low"] == 5
    assert STAGE_GPU_REQUIREMENTS["e4oftb128low"] == 8
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q tests/fast/utils/test_math_oft_b128_low_lr_sweep.py
```

Expected: collection fails because `E4_MATH_OFT_B128_LOW_LRS` and `e4_math_oft_b128_low_arms` do not exist.

- [ ] **Step 3: Implement the fixed arm builder**

First add this matrix to `tools/lora_regret/arms.py`'s module overview so the
enumerated matrix documentation stays complete:

```text
* ``e4oftb128low`` -- the focused Math OFT BS128 lower-LR sweep that follows
  up E4's weak `3e-5` result without changing E4's established grid.
```

Then, immediately after `E4_OFT_BLOCK_LADDER`, add:

```python
E4_MATH_OFT_B128_LOW_LRS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5)


def e4_math_oft_b128_low_arms(
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
) -> list[Arm]:
    """Focused lower-LR Math sweep for the E4 OFT BS128 cell."""
    shapes = megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)
    selected_shapes = {
        name: shape for name, shape in shapes.items() if name in ALL_MODULES.split(",")
    }
    report = oft_lora_match_report(128, selected_shapes)
    return [
        Arm(
            _name("oftlow", "b128", ALL_MODULES, lr, seed, extra="math"),
            "oft",
            None,
            128,
            ALL_MODULES,
            lr,
            seed,
            dataset="math",
            matched_ratio=report["ratio"],
        )
        for lr in E4_MATH_OFT_B128_LOW_LRS
    ]
```

Register it in `MATRICES`:

```python
"e4oftb128low": (
    lambda hidden, ffn, qkv_output, seed, oft_lr_centre=None, argmins=None:
    e4_math_oft_b128_low_arms(hidden, ffn, seed=seed, qkv_output_size=qkv_output)
),
```

- [ ] **Step 4: Implement the runtime metadata**

Add these entries directly to their existing dictionary literals:

```python
# tools/lora_regret/sweep.py
"e4oftb128low": RL_LAUNCHER,       # MATRIX_LAUNCHERS
"e4oftb128low": "accuracy",       # MATRIX_METRICS
"e4oftb128low": "rl-b128-low-lr", # MATRIX_PROJECTS

# tools/lora_regret/preflight.py
"e4oftb128low": 5,  # EXPECTED_ARMS
"e4oftb128low": 8,  # STAGE_GPU_REQUIREMENTS
```

- [ ] **Step 5: Run the focused and neighboring tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/fast/utils/test_math_oft_b128_low_lr_sweep.py tests/fast/utils/test_lora_regret_preflight.py tests/fast/utils/test_lora_regret_arms_coverage.py
```

Expected: all collected tests pass, including `set(EXPECTED_ARMS) == set(MATRICES) - {"e1long"}`.

- [ ] **Step 6: Commit the matrix**

```bash
git add tools/lora_regret/arms.py tools/lora_regret/sweep.py tools/lora_regret/preflight.py tests/fast/utils/test_math_oft_b128_low_lr_sweep.py
git commit -m "feat(oft): add BS128 lower-LR matrix"
```

### Task 2: Add the dedicated resumable launcher

**Files:**
- Modify: `tests/fast/utils/test_math_oft_b128_low_lr_sweep.py`
- Modify: `scripts/lora_regret/campaign.sh:44-66`
- Modify: `scripts/lora_regret/campaign.sh:99-110`
- Create: `scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh`

**Interfaces:**
- Consumes: matrix key `e4oftb128low` and its five `oftlow` arms from Task 1.
- Produces: `bash scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh`.
- Produces: optional `PREFLIGHT_STAGE`, defaulting to `e4` for every existing caller.
- Produces: `results/e4_math_oft_b128_low_lr.jsonl`.

- [ ] **Step 1: Add failing launcher-contract tests**

Append these imports and tests to the focused test file:

```python
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "lora_regret"
WRAPPER = SCRIPT_DIR / "run_e4_math_oft_b128_low_lr_8gpu.sh"


def test_the_dedicated_wrapper_selects_the_complete_focused_matrix():
    text = WRAPPER.read_text(encoding="utf-8")
    match = re.search(r"METHOD_RE='([^']+)'", text)
    assert match
    pattern = re.compile(match.group(1))
    selected = [arm.name for arm in _arms() if pattern.search(arm.name)]
    assert selected == list(EXPECTED_NAMES)
    assert "MATRIX=e4oftb128low" in text
    assert "RESULTS=results/e4_math_oft_b128_low_lr.jsonl" in text
    assert "EXPECT_ARMS=5" in text
    assert "ALLOW_OFT=1" in text
    assert "PREFLIGHT_STAGE=e4oftb128low" in text
    assert 'source "${HERE}/e4_protocol.sh"' in text


def test_campaign_preflight_stage_is_configurable_without_changing_the_default():
    text = (SCRIPT_DIR / "campaign.sh").read_text(encoding="utf-8")
    assert "PREFLIGHT_STAGE=${PREFLIGHT_STAGE:-e4}" in text
    assert 'preflight (stage ${PREFLIGHT_STAGE})' in text
    assert '--stage "${PREFLIGHT_STAGE}"' in text
```

- [ ] **Step 2: Run the launcher tests and verify RED**

Run:

```bash
python -m pytest -q tests/fast/utils/test_math_oft_b128_low_lr_sweep.py::test_the_dedicated_wrapper_selects_the_complete_focused_matrix tests/fast/utils/test_math_oft_b128_low_lr_sweep.py::test_campaign_preflight_stage_is_configurable_without_changing_the_default
```

Expected: the wrapper test fails because the file is absent; the campaign test fails because the stage is hardcoded.

- [ ] **Step 3: Make the campaign preflight stage configurable**

Initialize the knob beside `SKIP_PREFLIGHT`:

```bash
PREFLIGHT_STAGE=${PREFLIGHT_STAGE:-e4}
```

Replace the hardcoded preflight block with:

```bash
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    say "preflight (stage ${PREFLIGHT_STAGE})"
    if ! python -m tools.lora_regret.preflight --stage "${PREFLIGHT_STAGE}"; then
        echo "preflight failed -- fix it before spending the node." >&2
        exit 1
    fi
fi
```

Add `PREFLIGHT_STAGE=e4` to the documented campaign knobs. Existing wrappers set nothing and continue using stage `e4`.

- [ ] **Step 4: Create the dedicated wrapper**

Create `scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh`:

```bash
#!/usr/bin/env bash
#
# Focused E4 Math OFT BS128 lower-learning-rate sweep:
# 1e-7, 3e-7, 1e-6, 3e-6, 1e-5. Book a whole 8-GPU node.
#
#   source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
#   cd /fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
#
# Resumable: successful rows in the dedicated ledger are skipped. Use one
# writer for this RESULTS file.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env MATRIX=e4oftb128low METHOD_RE='^oftlow-b128-all-math-lr' RESULTS=results/e4_math_oft_b128_low_lr.jsonl EXPECT_ARMS=5 ALLOW_OFT=1 PREFLIGHT_STAGE=e4oftb128low bash "${HERE}/campaign.sh" "$@"
```

Then run:

```bash
chmod +x scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
```

- [ ] **Step 5: Run launcher and compatibility tests**

```bash
python -m pytest -q tests/fast/utils/test_math_oft_b128_low_lr_sweep.py tests/fast/utils/test_lora_regret_lr_columns.py tests/fast/utils/test_lora_regret_preflight.py
bash -n scripts/lora_regret/campaign.sh
bash -n scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
```

Expected: all pytest tests pass and both `bash -n` commands exit 0.

- [ ] **Step 6: Verify the real dry-run selection**

Run in the same dependency-complete environment used for the focused tests:

```bash
SKIP_PREFLIGHT=1 DRY_RUN=1 bash scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh > /tmp/e4oftb128low-dry-run.stdout 2> /tmp/e4oftb128low-dry-run.stderr
```

Then run:

```bash
grep -F "5 arms selected, 5 to run -> results/e4_math_oft_b128_low_lr.jsonl" /tmp/e4oftb128low-dry-run.stdout
grep -F "oftlow-b128-all-math-lr1e-07-s0" /tmp/e4oftb128low-dry-run.stdout
grep -F "oftlow-b128-all-math-lr3e-07-s0" /tmp/e4oftb128low-dry-run.stdout
grep -F "oftlow-b128-all-math-lr1e-06-s0" /tmp/e4oftb128low-dry-run.stdout
grep -F "oftlow-b128-all-math-lr3e-06-s0" /tmp/e4oftb128low-dry-run.stdout
grep -F "oftlow-b128-all-math-lr1e-05-s0" /tmp/e4oftb128low-dry-run.stdout
```

Expected: every `grep` exits 0; stdout contains no BS8, BS1024, GSM8K, FullFT, or LoRA arm.

- [ ] **Step 7: Run final static verification**

```bash
git diff --check
git status --short
```

Expected: `git diff --check` exits 0; status lists only the Task 2 launcher, campaign, and focused test changes.

- [ ] **Step 8: Commit the launcher**

```bash
git add scripts/lora_regret/campaign.sh scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh tests/fast/utils/test_math_oft_b128_low_lr_sweep.py
git commit -m "feat(oft): add BS128 lower-LR sweep launcher"
```

## Final verification before handoff

Run from the feature worktree in an environment with pytest installed:

```bash
python -m pytest -q tests/fast/utils/test_math_oft_b128_low_lr_sweep.py tests/fast/utils/test_lora_regret_arms_coverage.py tests/fast/utils/test_lora_regret_lr_columns.py tests/fast/utils/test_lora_regret_preflight.py tests/fast/utils/test_lora_regret_sweep.py
bash -n scripts/lora_regret/campaign.sh
bash -n scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
git diff --check HEAD~2..HEAD
git status --short --branch
```

Expected: all tests and shell checks pass; the worktree is clean; the branch contains the design and plan commits plus the two implementation commits.
