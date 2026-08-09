# E4 LoRA LR0 Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible GSM8K and Math LoRA-only E4 wrappers at learning rate `2e-6`.

**Architecture:** A dedicated `e4lr0` matrix owns six arms without changing the established seven-column E4 grid. Two dataset-specific wrappers filter that matrix into independent three-arm ledgers while reusing the existing E4 protocol and campaign runner.

**Tech Stack:** Python arm enumeration and pytest; Bash launch wrappers; Git push/pull synchronization.

## Global Constraints

- The learning rate is exactly `2e-6`.
- The ranks are exactly 1, 16, and 256, with LoRA targeting all modules.
- GSM8K and Math use separate wrappers and result ledgers.
- Existing E4 learning-rate columns and their 70-arm matrix remain unchanged.
- No allocation or training is started by this implementation.

---

### Task 1: Add the isolated LR0 arm matrix and routing

**Files:**
- Modify: `tests/fast/utils/test_lora_regret_sweep.py`
- Modify: `tests/fast/utils/test_lora_regret_preflight.py`
- Modify: `tools/lora_regret/arms.py`
- Modify: `tools/lora_regret/sweep.py`
- Modify: `tools/lora_regret/preflight.py`

**Interfaces:**
- Produces: `e4lr0_arms(seed: int = 0, datasets: tuple[str, ...] = RL_DATASETS) -> list[Arm]`
- Produces: matrix key `e4lr0`, routed through the RL launcher with accuracy scoring and the `rl-rank` W&B project stem.

- [ ] **Step 1: Write failing matrix and routing tests**

```python
def test_e4lr0_is_six_lora_arms_at_two_e_minus_six():
    arms = e4lr0_arms()
    assert len(arms) == 6
    assert {a.dataset for a in arms} == {"gsm8k", "math"}
    assert {a.rank for a in arms} == {1, 16, 256}
    assert {a.method for a in arms} == {"lora"}
    assert {a.lr for a in arms} == {2e-6}
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/fast/utils/test_lora_regret_sweep.py tests/fast/utils/test_lora_regret_preflight.py`

Expected: collection fails because `e4lr0_arms` and the `e4lr0` routing do not exist.

- [ ] **Step 3: Implement the matrix and routing**

Add `e4lr0_arms`, register it in `MATRICES`, and add `e4lr0` to `MATRIX_LAUNCHERS`, `MATRIX_METRICS`, `MATRIX_PROJECTS`, and `EXPECTED_ARMS` with values `RL_LAUNCHER`, `accuracy`, `rl-rank`, and `6` respectively.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/fast/utils/test_lora_regret_sweep.py tests/fast/utils/test_lora_regret_preflight.py`

Expected: all tests pass.

### Task 2: Add dataset wrappers and partition tests

**Files:**
- Modify: `tests/fast/utils/test_lora_regret_lr_columns.py`
- Create: `scripts/lora_regret/run_e4_gsm8k_lr0_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_lr0_8gpu.sh`

**Interfaces:**
- Produces: GSM8K ledger `results/e4_gsm8k_lr0.jsonl`
- Produces: Math ledger `results/e4_math_lr0.jsonl`

- [ ] **Step 1: Write failing wrapper tests**

```python
def test_lr0_scripts_partition_the_lr0_matrix():
    selected = [name for path in LR0_SCRIPTS for name in _lr0_selected(path)]
    assert len(LR0_SCRIPTS) == 2
    assert len(selected) == len(set(selected)) == 6
    assert set(selected) == {arm.name for arm in e4lr0_arms()}
```

Also assert each wrapper selects one dataset, three ranks, no FullFT/OFT arm,
has `EXPECT_ARMS=3`, sources `e4_protocol.sh`, and writes a unique LR0 ledger.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/fast/utils/test_lora_regret_lr_columns.py`

Expected: failure because both LR0 wrapper files are missing.

- [ ] **Step 3: Create the wrappers**

Each executable wrapper sources `e4_protocol.sh` and invokes:

```bash
exec env MATRIX=e4lr0 METHOD_RE='^lora-r(1|16|256)-all-DATASET-lr2e\-06-s' \
    RESULTS=results/e4_DATASET_lr0.jsonl EXPECT_ARMS=3 \
    bash "${HERE}/campaign.sh" "$@"
```

with `DATASET` replaced by `gsm8k` or `math`.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run: `pytest -q tests/fast/utils/test_lora_regret_lr_columns.py`

Run: `bash -n scripts/lora_regret/run_e4_gsm8k_lr0_8gpu.sh scripts/lora_regret/run_e4_math_lr0_8gpu.sh`

Expected: both commands pass.

### Task 3: Verify, commit, push, and synchronize

**Files:**
- Include the implementation plan and all Task 1-2 changes in one implementation commit.

**Interfaces:**
- Produces: one clean commit present on local, GitHub, and `/fast/zqiu/orbit-iclr/orbit`.

- [ ] **Step 1: Run focused verification**

Run: `pytest -q tests/fast/utils/test_lora_regret_sweep.py tests/fast/utils/test_lora_regret_preflight.py tests/fast/utils/test_lora_regret_lr_columns.py`

Run: `git diff --check`

- [ ] **Step 2: Commit and push**

```bash
git add docs/superpowers/plans/2026-08-09-e4-lr0-extension.md \
  tools/lora_regret/arms.py tools/lora_regret/sweep.py tools/lora_regret/preflight.py \
  tests/fast/utils/test_lora_regret_sweep.py tests/fast/utils/test_lora_regret_preflight.py \
  tests/fast/utils/test_lora_regret_lr_columns.py scripts/lora_regret/run_e4_gsm8k_lr0_8gpu.sh \
  scripts/lora_regret/run_e4_math_lr0_8gpu.sh
git commit -m "feat(lora-regret): add the lr0 LoRA point"
git push origin feat/lora-without-regret
```

- [ ] **Step 3: Pull and verify the cluster checkout**

Run `git pull --ff-only origin feat/lora-without-regret` in
`/fast/zqiu/orbit-iclr/orbit`, then verify its HEAD equals local HEAD and both
LR0 wrapper files are present. Preserve all untracked result ledgers.
