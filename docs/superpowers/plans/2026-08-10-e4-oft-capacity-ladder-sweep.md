# E4 OFT Capacity-Ladder Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-capacity OFT sweep at LoRA lr0-lr6 for both E4 dataset panels, exposed through fourteen resumable 8-GPU launch wrappers.

**Architecture:** Keep the new OFT arms in the existing `e4` matrix so they inherit the established E4 launcher, dataset routing, protocol, and ledger behavior. Generate fixed-block `b8`, `b128`, and `b1024` cells over the seven-point `2e-6` through `4e-4` scout window, then make each wrapper select one dataset/LR column containing exactly those three capacities.

**Tech Stack:** Python 3 dataclasses and pytest, Bash launch wrappers, existing `tools.lora_regret` sweep/campaign infrastructure.

## Global Constraints

- The OFT block ladder is exactly `(8, 128, 1024)`; do not add `b64`.
- The seven learning rates are exactly `(2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4)`; do not add lr7 (`1e-3`).
- Create seven GSM8K wrappers and seven Math wrappers; every wrapper selects three arms and has a unique ledger.
- Every OFT arm targets `linear_qkv,linear_proj,linear_fc1,linear_fc2`, uses seed 0, and retains the `oftscout-b<block>-all-<dataset>-lr...-s0` identity.
- Reuse `MATRIX=e4` and `scripts/lora_regret/e4_protocol.sh`; do not change the RL launcher, rollout budget, evaluation cadence, checkpoint policy, W&B entity, or scheduler resources.
- Keep `campaign.sh`'s OFT rejection as the default for existing FullFT/LoRA wrappers; only the new dedicated OFT ledgers may opt in with `ALLOW_OFT=1`.
- Preserve the existing fourteen FullFT/LoRA wrappers and their 56-arm partition.
- This implementation is local campaign tooling only: do not synchronize to a cluster, allocate Condor nodes, or start training.

---

### Task 1: Define and verify the E4 OFT arm ladder

**Files:**
- Modify: `tests/fast/utils/test_lora_regret_arms_coverage.py`
- Modify: `tools/lora_regret/arms.py`

**Interfaces:**
- Consumes: `oft_lr_values(centre, n, step_decades, span, sig_figs) -> tuple[list[float], bool]`, `oft_lora_match_report(block_size, module_shapes) -> dict`, and `Arm`.
- Produces: `E4_OFT_BLOCK_LADDER: tuple[int, int, int]` and `e4_arms(...) -> list[Arm]` containing 42 OFT arms plus the existing 56 non-OFT arms.

- [ ] **Step 1: Add failing E4 ladder tests**

Add a focused class to `tests/fast/utils/test_lora_regret_arms_coverage.py`:

```python
class TestE4OftCapacityLadder:
    EXPECTED_LRS = {2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4}

    def test_the_block_ladder_is_the_selected_low_middle_high_sweep(self):
        from tools.lora_regret.arms import E4_OFT_BLOCK_LADDER

        assert E4_OFT_BLOCK_LADDER == (8, 128, 1024)

    def test_each_dataset_has_three_blocks_on_the_lora_lr0_lr6_window(self):
        from tools.lora_regret.arms import RL_DATASETS, e4_arms

        oft = [arm for arm in e4_arms() if arm.method == "oft"]
        assert len(oft) == 42
        for dataset in RL_DATASETS:
            panel = [arm for arm in oft if arm.dataset == dataset]
            assert {arm.oft_block_size for arm in panel} == {8, 128, 1024}
            assert {arm.lr for arm in panel} == self.EXPECTED_LRS
            assert len(panel) == 21

    def test_every_arm_is_an_all_modules_scout_with_recorded_match(self):
        from tools.lora_regret.arms import ALL_MODULES, e4_arms

        oft = [arm for arm in e4_arms() if arm.method == "oft"]
        assert all(arm.name.startswith("oftscout-") for arm in oft)
        assert {arm.target_modules for arm in oft} == {ALL_MODULES}
        assert all(arm.matched_ratio is not None for arm in oft)

    def test_the_capacity_reports_remain_visible_and_stable(self):
        from orbit.utils.peft_param_match import megatron_module_shapes, oft_lora_match_report
        from tools.lora_regret.arms import E4_OFT_BLOCK_LADDER

        shapes = megatron_module_shapes(HIDDEN, FFN, QKV)
        reports = [oft_lora_match_report(block, shapes) for block in E4_OFT_BLOCK_LADDER]
        assert [(r["block_size"], r["lora_rank"]) for r in reports] == [
            (8, 1), (128, 24), (1024, 196)
        ]
        assert [r["ratio"] for r in reports] == pytest.approx(
            [1.3382352941, 1.0116421569, 0.9978241297]
        )
```

Change the general different-grid test so the deliberately copied completed
E4 lr0-lr6 scout window is exempt while retaining all span checks. The E4
matrix's LoRA cell itself is lr1-lr7, so this is an exception to the rule, not
an equality assertion:

```python
if matrix not in {"e4", "e4place"}:
    assert set(oft_lrs) != set(lora_lrs)
```

Update the expected E4 count from 70 to 98 and update the SFT/RL block-ceiling assertion so E4's RL block set is `{8, 128, 1024}`.

- [ ] **Step 2: Run the focused tests and verify the new assertions fail**

Run:

```bash
pytest -q tests/fast/utils/test_lora_regret_arms_coverage.py -k 'E4OftCapacityLadder or oft_grid_is_never_loras_grid or new_counts or sft_and_rl_now_reach_the_same_block'
```

Expected: failures because `E4_OFT_BLOCK_LADDER` does not exist, E4 has one OFT block and 70 arms, and its OFT grid differs from LoRA's.

- [ ] **Step 3: Implement fixed-block E4 OFT cells**

In `tools/lora_regret/arms.py`, set the selected RL scout span and declare the ladder next to the E4 matrix:

```python
RL_OFT_SCOUT_SPAN = (2e-6, 4e-4)

E4_OFT_BLOCK_LADDER = (8, 128, 1024)
```

Replace E4's single rank-derived `_oft_cell(...)` with fixed-block cells:

```python
    shapes = megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)
    selected_shapes = {
        name: shape for name, shape in shapes.items() if name in ALL_MODULES.split(",")
    }
    oft_lrs, scouting = oft_lr_values(
        oft_lr_centre,
        RL_GRID_POINTS,
        step_decades=RL_STEP_DECADES,
        span=RL_OFT_SCOUT_SPAN,
        sig_figs=RL_SIG_FIGS,
    )
    label = "oftscout" if scouting else "oft"
    for dataset in datasets:
        for block_size in E4_OFT_BLOCK_LADDER:
            report = oft_lora_match_report(block_size, selected_shapes)
            for lr in oft_lrs:
                arms.append(
                    Arm(
                        _name(label, f"b{block_size}", ALL_MODULES, lr, seed, extra=dataset),
                        "oft",
                        None,
                        block_size,
                        ALL_MODULES,
                        lr,
                        seed,
                        dataset=dataset,
                        matched_ratio=report["ratio"],
                    )
                )
```

Keep the FullFT and LoRA construction unchanged. Update E4's docstring/comments to state 98 total arms and the explicit low/middle/high ladder rather than a single rank-derived block.

- [ ] **Step 4: Run the focused arm tests and verify they pass**

Run the same focused command from Step 2.

Expected: all selected tests pass, including 42 OFT arms, the exact block/LR ladders, and stable capacity reports.

- [ ] **Step 5: Commit the arm model change**

```bash
git add tools/lora_regret/arms.py tests/fast/utils/test_lora_regret_arms_coverage.py
git commit -m "feat(lora-regret): add e4 oft capacity ladder"
```

---

### Task 2: Add and verify the fourteen OFT column wrappers

**Files:**
- Modify: `tests/fast/utils/test_lora_regret_lr_columns.py`
- Modify: `scripts/lora_regret/campaign.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr0_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr1_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr2_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr3_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr4_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr5_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_gsm8k_oft_lr6_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr0_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr1_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr2_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr4_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr5_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_lr6_8gpu.sh`

**Interfaces:**
- Consumes: E4 arm names `oftscout-b<block>-all-<dataset>-lr<value>-s0`, `e4_protocol.sh`, and `campaign.sh` environment variables.
- Produces: `campaign.sh` opt-in variable `ALLOW_OFT` (default `0`) and fourteen executable wrappers that set `MATRIX=e4`, `METHOD_RE`, `RESULTS`, `EXPECT_ARMS=3`, and `ALLOW_OFT=1` before invoking `campaign.sh`.

- [ ] **Step 1: Isolate the original wrapper set and add failing OFT wrapper tests**

Replace the broad original-script glob in `tests/fast/utils/test_lora_regret_lr_columns.py` with an explicit list:

```python
SCRIPTS = [
    SCRIPT_DIR / f"run_e4_{dataset}_lr{column}_8gpu.sh"
    for dataset in ("gsm8k", "math")
    for column in range(1, 8)
]
OFT_LRS = (2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4)
OFT_SCRIPTS = [
    SCRIPT_DIR / f"run_e4_{dataset}_oft_lr{column}_8gpu.sh"
    for dataset in ("gsm8k", "math")
    for column in range(7)
]
```

Add helpers and tests that parse those wrappers using the existing `_pattern` mechanism:

```python
def _oft_selected(path: Path):
    return [arm for arm in _arms() if arm.method == "oft" and _pattern(path).search(arm.name)]

def test_the_fourteen_oft_scripts_exist():
    assert all(path.is_file() for path in OFT_SCRIPTS)

def test_each_oft_script_selects_one_dataset_lr_and_three_blocks():
    for path in OFT_SCRIPTS:
        selected = _oft_selected(path)
        dataset = path.name.split("_")[2]
        column = int(re.search(r"_oft_lr(\d)_", path.name).group(1))
        assert len(selected) == 3, (path.name, [arm.name for arm in selected])
        assert {arm.dataset for arm in selected} == {dataset}
        assert {arm.lr for arm in selected} == {OFT_LRS[column]}
        assert {arm.oft_block_size for arm in selected} == {8, 128, 1024}
        assert {arm.target_modules for arm in selected} == {ALL_MODULES}
        assert all(arm.name.startswith("oftscout-") for arm in selected)

def test_the_oft_scripts_partition_all_forty_two_arms_once():
    selected = [arm.name for path in OFT_SCRIPTS for arm in _oft_selected(path)]
    expected = {arm.name for arm in _arms() if arm.method == "oft"}
    assert len(selected) == len(set(selected)) == 42
    assert set(selected) == expected

def test_oft_scripts_use_unique_ledgers_three_arm_guards_and_shared_protocol():
    texts = [path.read_text(encoding="utf-8") for path in OFT_SCRIPTS]
    ledgers = {re.search(r"RESULTS=(\S+)", text).group(1) for text in texts}
    assert ledgers == {
        f"results/e4_{dataset}_oft_lr{column}.jsonl"
        for dataset in ("gsm8k", "math")
        for column in range(7)
    }
    assert all("EXPECT_ARMS=3" in text for text in texts)
    assert all("ALLOW_OFT=1" in text for text in texts)
    assert all('source "${HERE}/e4_protocol.sh"' in text for text in texts)
```

Import `ALL_MODULES` with the existing E4 arm functions.

Add an execution-boundary test for the campaign guard. The test creates a fake
`python` executable under `tmp_path`, puts it first on `PATH`, sets
`VIRTUAL_ENV`, `SKIP_PREFLIGHT=1`, and `DRY_RUN=1`, then runs one new wrapper.
The fake reports three selected `PEFT_METHOD=oft` commands. Assert that the
wrapper exits 0 and reaches the campaign's `dry run -- launcher commands only`
message. Add a companion invocation of `campaign.sh` without `ALLOW_OFT=1` and
assert it still exits nonzero with `REFUSING: the selection contains OFT arms`.
This exercises the real wrapper/campaign control flow without importing the GPU
stack or starting a training process.

- [ ] **Step 2: Run the LR-column tests and verify wrapper tests fail**

Run:

```bash
pytest -q tests/fast/utils/test_lora_regret_lr_columns.py -k 'oft or seven_scripts_partition_e4_exactly or one_script_per_grid_point_per_panel'
```

Expected: OFT existence/coverage tests fail because the fourteen files do not exist; the opt-in campaign test fails because OFT is still unconditionally refused; the original-wrapper enumeration and 56-arm partition continue to pass.

- [ ] **Step 3: Make the campaign's OFT support explicit and opt-in**

In `scripts/lora_regret/campaign.sh`, add the default beside the other wrapper
knobs:

```bash
ALLOW_OFT=${ALLOW_OFT:-0}
```

Narrow the existing refusal so unchanged FullFT/LoRA wrappers retain their
protection while dedicated OFT wrappers may proceed:

```bash
if [[ "${ALLOW_OFT}" != "1" ]] && printf '%s' "${PLAN}" | grep -q "PEFT_METHOD=oft"; then
    echo "REFUSING: the selection contains OFT arms; ${METHOD_RE} is wrong." >&2
    exit 1
fi
```

Document `ALLOW_OFT=0` in the script's knob list as an opt-in restricted to a
dedicated OFT ledger.

- [ ] **Step 4: Create one wrapper for every dataset/LR column**

Create all fourteen scripts with the same structure, substituting the dataset, zero-based LR column, exact LR token, and ledger. For example, `run_e4_gsm8k_oft_lr0_8gpu.sh` is:

```bash
#!/usr/bin/env bash
#
# E4 OFT, gsm8k panel, learning-rate column 0 of 6: b8/b128/b1024 at 2e-06.
# Book a WHOLE 8-GPU node.
#
#   source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_gsm8k_oft_lr0_8gpu.sh
#
# The fourteen OFT wrappers partition 42 arms: two datasets x seven learning
# rates x three capacities. This ledger is resumable; rerunning skips arms with
# status "ok". Use only one writer per RESULTS file.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env MATRIX=e4 METHOD_RE='^oftscout-b(8|128|1024)-all-gsm8k-lr2e\-06-s' RESULTS=results/e4_gsm8k_oft_lr0.jsonl EXPECT_ARMS=3 ALLOW_OFT=1 \
    bash "${HERE}/campaign.sh" "$@"
```

Use these exact column/token pairs in both dataset sets:

```text
lr0 -> 2e\-06
lr1 -> 5e\-06
lr2 -> 1e\-05
lr3 -> 3e\-05
lr4 -> 7e\-05
lr5 -> 0\.0002
lr6 -> 0\.0004
```

Make every new script executable.

- [ ] **Step 5: Run wrapper tests and shell syntax checks**

Run:

```bash
pytest -q tests/fast/utils/test_lora_regret_lr_columns.py
bash -n scripts/lora_regret/run_e4_gsm8k_oft_lr{0,1,2,3,4,5,6}_8gpu.sh
bash -n scripts/lora_regret/run_e4_math_oft_lr{0,1,2,3,4,5,6}_8gpu.sh
```

Expected: all LR-column tests pass and both `bash -n` commands exit 0.

- [ ] **Step 6: Commit the launch wrappers**

```bash
git add tests/fast/utils/test_lora_regret_lr_columns.py scripts/lora_regret/campaign.sh scripts/lora_regret/run_e4_*_oft_lr*_8gpu.sh
git commit -m "feat(lora-regret): add e4 oft lr column scripts"
```

---

### Task 3: Verify the complete local campaign tooling

**Files:**
- Verify: `tools/lora_regret/arms.py`
- Verify: `tests/fast/utils/test_lora_regret_arms_coverage.py`
- Verify: `tests/fast/utils/test_lora_regret_lr_columns.py`
- Verify: `scripts/lora_regret/run_e4_*_oft_lr*_8gpu.sh`

**Interfaces:**
- Consumes: the fixed E4 arm ladder and fourteen wrappers from Tasks 1-2.
- Produces: a verified implementation commit with no remote execution side effects.

- [ ] **Step 1: Run the focused E4 test files together**

```bash
pytest -q tests/fast/utils/test_lora_regret_arms_coverage.py tests/fast/utils/test_lora_regret_lr_columns.py
```

Expected: all tests pass.

- [ ] **Step 2: Run the broader LoRA-regret fast-test subset**

```bash
pytest -q tests/fast/utils/test_lora_regret_*.py
```

Expected: all available LoRA-regret fast tests pass. If the repository environment cannot import pytest or an existing optional dependency, record the exact missing dependency and run the import-free arm/wrapper assertions plus every available focused test.

- [ ] **Step 3: Validate every new wrapper and the exact file count**

```bash
bash -n scripts/lora_regret/run_e4_gsm8k_oft_lr{0,1,2,3,4,5,6}_8gpu.sh
bash -n scripts/lora_regret/run_e4_math_oft_lr{0,1,2,3,4,5,6}_8gpu.sh
test "$(find scripts/lora_regret -maxdepth 1 -name 'run_e4_*_oft_lr[0-6]_8gpu.sh' | wc -l | tr -d ' ')" = 14
```

Expected: all commands exit 0 and exactly fourteen wrappers are present.

- [ ] **Step 4: Inspect the final diff and confirm there is no cluster mutation**

```bash
git diff --check HEAD^
git status --short
git log --oneline -3
```

Expected: no whitespace errors; only the planned arm, test, script, spec/plan changes are present; no results, Condor submissions, W&B artifacts, or remote state changes exist.

- [ ] **Step 5: Record any final verification-only adjustment**

If Step 1-4 required a source or test correction, stage only those planned files and commit it:

```bash
git add tools/lora_regret/arms.py tests/fast/utils/test_lora_regret_arms_coverage.py tests/fast/utils/test_lora_regret_lr_columns.py scripts/lora_regret/run_e4_*_oft_lr*_8gpu.sh
git commit -m "test(lora-regret): verify e4 oft sweep coverage"
```

If no correction was required, leave the two feature commits unchanged.
