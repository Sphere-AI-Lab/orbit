# Math OFT BS128 Refinement Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two resumable launchers for six Math OFT BS128 learning rates and run the two three-arm columns concurrently on two whole-node H100 allocations.

**Architecture:** A new `e4oftb128refine` matrix owns six disjoint `oftrefine` arms. The common matrix is registered once with the existing RL/accuracy/preflight stack, while two thin shell launchers select three arms each and write separate ledgers. A behavioral pytest module exercises both the Python registry and the real shell-to-campaign boundary with a fake Python executable, so selection and ledger ownership are tested without starting the GPU stack.

**Tech Stack:** Python 3.12, pytest, Bash, Orbit's `tools.lora_regret` experiment framework, Git, HTCondor, and the `control-remote-condor` controller.

## Global Constraints

- The exact learning-rate tuple is `(5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 2e-5)` in that order.
- Every arm is Math, OFT block size 128, seed 0, and targets `linear_qkv,linear_proj,linear_fc1,linear_fc2`.
- Every arm uses the unchanged E4 protocol: Llama-3.1-8B, 150 rollouts, accuracy evaluation, and eight GPUs.
- The matrix key is `e4oftb128refine`; arm names begin with `oftrefine-b128-all-math-`.
- Launcher A owns `5e-6, 6e-6, 7e-6` and `results/e4_math_oft_b128_refine_a.jsonl`.
- Launcher B owns `8e-6, 9e-6, 2e-5` and `results/e4_math_oft_b128_refine_b.jsonl`.
- W&B routes to `math-rl-b128-refine-lr-oft`.
- Do not change the completed `e4oftb128low` matrix, launcher, or ledger.
- Use the existing repository environment only. Do not create or synchronize a task-specific Python environment.
- Observe the new behavioral test failing before changing production files.
- Launch only after the tested source is merged and pushed to `feat/lora-without-regret` and the shared cluster checkout is fast-forwarded to that exact commit.
- Submit exactly two whole-node H100 allocations. Refresh capacity and obtain user approval for the exact numeric bid before submission.
- Launch through the two project-native scripts without adding a training wrapper or a remote scientific qualification suite.

---

### Task 1: Add the Behavioral Contract

**Files:**
- Create: `tests/fast/utils/test_math_oft_b128_refinement_sweep.py`

**Interfaces:**
- Consumes: existing `Arm`, `ALL_MODULES`, `MATRICES`, `e4_arms`, `e4_math_oft_b128_low_arms`, campaign shell contract, and registry dictionaries.
- Produces: a failing executable specification for `e4_math_oft_b128_refine_arms`, `e4oftb128refine`, and both launchers.

- [ ] **Step 1: Write the failing matrix and registry tests**

Create the test module with these literal expectations and assertions:

```python
"""Behavioral contract for the Math OFT BS128 refinement sweep."""

import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.lora_regret.arms import (
    ALL_MODULES,
    MATRICES,
    e4_arms,
    e4_math_oft_b128_low_arms,
)

HIDDEN, FFN, QKV = 4096, 14336, 6144
SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "lora_regret"
WRAPPER_A = SCRIPT_DIR / "run_e4_math_oft_b128_refine_a_8gpu.sh"
WRAPPER_B = SCRIPT_DIR / "run_e4_math_oft_b128_refine_b_8gpu.sh"
EXPECTED_LRS = (5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 2e-5)
EXPECTED_NAMES = (
    "oftrefine-b128-all-math-lr5e-06-s0",
    "oftrefine-b128-all-math-lr6e-06-s0",
    "oftrefine-b128-all-math-lr7e-06-s0",
    "oftrefine-b128-all-math-lr8e-06-s0",
    "oftrefine-b128-all-math-lr9e-06-s0",
    "oftrefine-b128-all-math-lr2e-05-s0",
)
SPLITS = (
    (
        WRAPPER_A,
        "results/e4_math_oft_b128_refine_a.jsonl",
        EXPECTED_NAMES[:3],
    ),
    (
        WRAPPER_B,
        "results/e4_math_oft_b128_refine_b.jsonl",
        EXPECTED_NAMES[3:],
    ),
)


def _arms():
    from tools.lora_regret.arms import e4_math_oft_b128_refine_arms

    return e4_math_oft_b128_refine_arms(
        HIDDEN, FFN, seed=0, qkv_output_size=QKV
    )


def test_matrix_builds_the_six_literal_math_bs128_arms():
    arms = _arms()

    assert tuple(arm.lr for arm in arms) == EXPECTED_LRS
    assert tuple(arm.name for arm in arms) == EXPECTED_NAMES
    assert {arm.method for arm in arms} == {"oft"}
    assert {arm.oft_block_size for arm in arms} == {128}
    assert {arm.target_modules for arm in arms} == {ALL_MODULES}
    assert {arm.dataset for arm in arms} == {"math"}
    assert {arm.seed for arm in arms} == {0}
    assert all(arm.matched_ratio is not None for arm in arms)


def test_matrix_names_are_disjoint_from_prior_e4_and_low_lr_arms():
    names = {arm.name for arm in _arms()}

    assert not names & {arm.name for arm in e4_arms()}
    assert not names & {arm.name for arm in e4_math_oft_b128_low_arms()}


def test_registry_builds_the_same_six_arms():
    registered = MATRICES["e4oftb128refine"](
        HIDDEN, FFN, QKV, 0, None, None
    )

    assert registered == _arms()


def test_matrix_routes_through_the_rl_accuracy_stack():
    from tools.lora_regret.preflight import EXPECTED_ARMS, STAGE_GPU_REQUIREMENTS
    from tools.lora_regret.sweep import (
        MATRIX_LAUNCHERS,
        MATRIX_METRICS,
        MATRIX_PROJECTS,
        RL_LAUNCHER,
        wandb_project,
    )

    assert MATRIX_LAUNCHERS["e4oftb128refine"] == RL_LAUNCHER
    assert MATRIX_METRICS["e4oftb128refine"] == "accuracy"
    assert MATRIX_PROJECTS["e4oftb128refine"] == "rl-b128-refine-lr"
    assert wandb_project("e4oftb128refine", None, "math", "oft") == (
        "math-rl-b128-refine-lr-oft"
    )
    assert EXPECTED_ARMS["e4oftb128refine"] == 6
    assert STAGE_GPU_REQUIREMENTS["e4oftb128refine"] == 8
```

- [ ] **Step 2: Add the fake campaign boundary and parametrized launcher test**

Append this controlled fake. It reports only the three arms owned by the selected ledger and records the exact environment passed through the real `campaign.sh`:

```python
def _fake_campaign_python(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        r'''#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.lora_regret.preflight" ]]; then
    printf 'preflight\t%s\n' "${4:-}" >> "${CAPTURE_FILE}"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.lora_regret.sweep" ]]; then
    printf 'sweep\t%s\t%s\t%s\t%s\t%s\n' \
        "${MATRIX:-}" "${METHOD_RE:-}" "${RESULTS:-}" \
        "${EXPECT_ARMS:-}" "${ALLOW_OFT:-}" >> "${CAPTURE_FILE}"
    case "${RESULTS:-}" in
        *refine_a.jsonl)
            printf '%s\n' \
                'ARM=oftrefine-b128-all-math-lr5e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr6e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr7e-06-s0 PEFT_METHOD=oft'
            ;;
        *refine_b.jsonl)
            printf '%s\n' \
                'ARM=oftrefine-b128-all-math-lr8e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr9e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr2e-05-s0 PEFT_METHOD=oft'
            ;;
        *) exit 98 ;;
    esac
    printf '3 arms selected, 0 already done, 3 to run\n' >&2
    exit 0
fi
exit 99
''',
        encoding="utf-8",
    )
    python.chmod(0o755)
    return fake_bin


def _campaign_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = _fake_campaign_python(tmp_path)
    capture = tmp_path / "campaign-boundary.tsv"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VIRTUAL_ENV": str(tmp_path / "venv"),
            "CUDA_HOME": str(tmp_path),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "SKIP_PREFLIGHT": "0",
            "DRY_RUN": "1",
            "CAPTURE_FILE": str(capture),
        }
    )
    return env, capture


@pytest.mark.parametrize(("wrapper", "ledger", "expected_names"), SPLITS)
def test_each_wrapper_owns_three_arms_and_drives_the_real_campaign(
    tmp_path: Path,
    wrapper: Path,
    ledger: str,
    expected_names: tuple[str, ...],
):
    env, capture = _campaign_env(tmp_path)

    result = subprocess.run(
        ["bash", str(wrapper), "--model", "llama3.1-8b"],
        cwd=SCRIPT_DIR.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    rows = [line.split("\t") for line in capture.read_text().splitlines()]
    assert rows[0] == ["preflight", "e4oftb128refine"]
    _, matrix, method_re, results, expected_arms, allow_oft = rows[1]
    assert matrix == "e4oftb128refine"
    assert results == ledger
    assert expected_arms == "3"
    assert allow_oft == "1"
    assert [arm.name for arm in _arms() if re.search(method_re, arm.name)] == list(
        expected_names
    )
    assert all(name in result.stdout for name in expected_names)
    assert not ({*EXPECTED_NAMES} - {*expected_names}) & set(result.stdout.split())
    assert "3 arms selected, 3 to run" in result.stdout
```

- [ ] **Step 3: Run dependency-free syntax checks locally**

Run:

```bash
python3 -m py_compile tests/fast/utils/test_math_oft_b128_refinement_sweep.py
git diff --check
```

Expected: both commands exit 0. Do not claim the pytest contract has run locally; the existing local `.venv` lacks both pytest and torch.

- [ ] **Step 4: Commit and publish the RED contract on the task branch**

```bash
git add tests/fast/utils/test_math_oft_b128_refinement_sweep.py
git commit -m "test(oft): specify BS128 refinement sweep"
git push -u origin codex/math-oft-b128-refine
```

Expected: the pushed task ref points to a commit containing the design, plan, and failing test but no production changes.

---

### Task 2: Acquire Two Whole-Node H100 Allocations and Observe RED

**Files:**
- No repository files change.
- Create remotely: `/fast/zqiu/orbit-iclr/orbit/.worktrees/math-oft-b128-refine`

**Interfaces:**
- Consumes: pushed `codex/math-oft-b128-refine`, two reachable login aliases, and a user-approved numeric H100 bid exposed as `APPROVED_H100_BID`.
- Produces: two managed eight-H100 sessions and durable proof that the new focused test fails before production implementation.

- [ ] **Step 1: Refresh controller inventory and H100 capacity**

Run the current controller with its 30-second default host deadline:

```bash
CONTROL=/Users/zqiu/Documents/GitHub/agent-skills/personal/control-remote-condor/scripts/condor_control.py
python3 "$CONTROL" sessions-all
```

Choose two reachable login aliases with the least managed-session load. For each selected alias run:

```bash
python3 "$CONTROL" --host "$LOGIN_A" check-connection --attempts 3
python3 "$CONTROL" --host "$LOGIN_A" probe
python3 "$CONTROL" --host "$LOGIN_A" jobs
python3 "$CONTROL" --host "$LOGIN_A" capacity

python3 "$CONTROL" --host "$LOGIN_B" check-connection --attempts 3
python3 "$CONTROL" --host "$LOGIN_B" probe
python3 "$CONTROL" --host "$LOGIN_B" jobs
python3 "$CONTROL" --host "$LOGIN_B" capacity
```

Expected: both aliases are stable, inventories are valid, and capacity reports at least two complete eight-GPU H100 nodes. If fewer than two complete nodes are available, stop and report capacity rather than changing GPU type.

- [ ] **Step 2: Obtain and validate the exact bid**

Present the fresh H100 minimum price and ask the user for one exact bid applying to both allocations. After approval:

```bash
: "${APPROVED_H100_BID:?set to the exact user-approved integer bid}"
case "$APPROVED_H100_BID" in
    ''|*[!0-9]*) echo "bid must be a positive integer" >&2; exit 2 ;;
esac
test "$APPROVED_H100_BID" -gt 0
```

No submission occurs before this succeeds.

- [ ] **Step 3: Render and submit exactly two managed interactive bids**

```bash
SESSION_A=codex-oft-b128-refine-a
SESSION_B=codex-oft-b128-refine-b
REQ='CUDADeviceName=="NVIDIA H100 80GB HBM3"'

python3 "$CONTROL" --host "$LOGIN_A" ensure-session "$SESSION_A"
python3 "$CONTROL" --host "$LOGIN_B" ensure-session "$SESSION_B"

PLAN_A=$(python3 "$CONTROL" --host "$LOGIN_A" plan-interactive \
    --bid "$APPROVED_H100_BID" --cpus 32 --gpus 8 --disk 1000G \
    --memory 1000000 --requirements "$REQ")
PLAN_B=$(python3 "$CONTROL" --host "$LOGIN_B" plan-interactive \
    --bid "$APPROVED_H100_BID" --cpus 32 --gpus 8 --disk 1000G \
    --memory 1000000 --requirements "$REQ")
BID_CMD_A=$(printf '%s\n' "$PLAN_A" | sed -n 's/^command=//p')
BID_CMD_B=$(printf '%s\n' "$PLAN_B" | sed -n 's/^command=//p')
printf '%s\n' "$PLAN_A" "$PLAN_B"
test -n "$BID_CMD_A"
test -n "$BID_CMD_B"

python3 "$CONTROL" --host "$LOGIN_A" send "$SESSION_A" -- sh -lc "$BID_CMD_A"
python3 "$CONTROL" --host "$LOGIN_B" send "$SESSION_B" -- sh -lc "$BID_CMD_B"
```

After Condor prints each job ID, record it without guessing:

```bash
python3 "$CONTROL" --host "$LOGIN_A" mark-job --bid "$APPROVED_H100_BID" \
    "$SESSION_A" "$JOB_A"
python3 "$CONTROL" --host "$LOGIN_B" mark-job --bid "$APPROVED_H100_BID" \
    "$SESSION_B" "$JOB_B"
python3 "$CONTROL" --host "$LOGIN_A" job "$JOB_A"
python3 "$CONTROL" --host "$LOGIN_B" job "$JOB_B"
```

Expected: both jobs reach running state on distinct complete H100 nodes and each compute shell sees eight H100 GPUs.

- [ ] **Step 4: Create the dedicated remote worktree from the RED task ref**

Send this to session A only:

```bash
set -euo pipefail
ROOT=/fast/zqiu/orbit-iclr/orbit
REMOTE_WT="$ROOT/.worktrees/math-oft-b128-refine"
git -C "$ROOT" fetch --no-tags origin \
    codex/math-oft-b128-refine:refs/remotes/origin/codex/math-oft-b128-refine
test ! -e "$REMOTE_WT"
git -C "$ROOT" worktree add --detach "$REMOTE_WT" \
    refs/remotes/origin/codex/math-oft-b128-refine
test -z "$(git -C "$REMOTE_WT" status --porcelain)"
```

Use `control-remote-condor send`; do not run this project command on a login node.

- [ ] **Step 5: Run the new focused test and verify RED**

Inside allocation A:

```bash
set -o pipefail
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /fast/zqiu/orbit-iclr/orbit/.worktrees/math-oft-b128-refine
source env.sh
PYTHONPATH="$PWD" python -m pytest -q \
    tests/fast/utils/test_math_oft_b128_refinement_sweep.py \
    2>&1 | tee /tmp/math-oft-b128-refine-red.log
test "${PIPESTATUS[0]}" -ne 0
rg -n 'e4_math_oft_b128_refine_arms|e4oftb128refine|No such file' \
    /tmp/math-oft-b128-refine-red.log
```

Expected: the test fails because the new builder, registry key, or launchers do not exist. A dependency/import/infrastructure failure does not satisfy RED; fix only the test execution environment and rerun before production edits.

---

### Task 3: Implement the Matrix, Registries, and Launchers

**Files:**
- Modify: `tools/lora_regret/arms.py`
- Modify: `tools/lora_regret/sweep.py`
- Modify: `tools/lora_regret/preflight.py`
- Create: `scripts/lora_regret/run_e4_math_oft_b128_refine_a_8gpu.sh`
- Create: `scripts/lora_regret/run_e4_math_oft_b128_refine_b_8gpu.sh`
- Test: `tests/fast/utils/test_math_oft_b128_refinement_sweep.py`

**Interfaces:**
- Consumes: the RED contract from Task 1 and existing `_name`, `oft_lora_match_report`, `campaign.sh`, and `e4_protocol.sh` interfaces.
- Produces: `E4_MATH_OFT_B128_REFINE_LRS`, `e4_math_oft_b128_refine_arms`, the `e4oftb128refine` registry route, and two executable launchers.

- [ ] **Step 1: Add the matrix constant and builder**

In `tools/lora_regret/arms.py`, add the matrix to the module documentation, then define:

```python
E4_MATH_OFT_B128_REFINE_LRS = (5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 2e-5)


def e4_math_oft_b128_refine_arms(
    hidden_size: int = LLAMA31_8B_HIDDEN,
    ffn_size: int = LLAMA31_8B_FFN,
    seed: int = 0,
    qkv_output_size: int = LLAMA31_8B_QKV_OUTPUT,
) -> list[Arm]:
    """Math OFT BS128 learning-rate refinement under the E4 protocol."""
    shapes = megatron_module_shapes(hidden_size, ffn_size, qkv_output_size)
    selected_shapes = {
        name: shape for name, shape in shapes.items() if name in ALL_MODULES.split(",")
    }
    report = oft_lora_match_report(128, selected_shapes)
    return [
        Arm(
            _name("oftrefine", "b128", ALL_MODULES, lr, seed, extra="math"),
            "oft",
            None,
            128,
            ALL_MODULES,
            lr,
            seed,
            dataset="math",
            matched_ratio=report["ratio"],
        )
        for lr in E4_MATH_OFT_B128_REFINE_LRS
    ]
```

Register it in `MATRICES`:

```python
"e4oftb128refine": (
    lambda hidden, ffn, qkv_output, seed, oft_lr_centre=None, argmins=None:
    e4_math_oft_b128_refine_arms(
        hidden, ffn, seed=seed, qkv_output_size=qkv_output
    )
),
```

- [ ] **Step 2: Register the RL route, metric, project, arm count, and GPU floor**

Add these exact entries:

```python
# tools/lora_regret/sweep.py
MATRIX_LAUNCHERS["e4oftb128refine"] = RL_LAUNCHER
MATRIX_METRICS["e4oftb128refine"] = "accuracy"
MATRIX_PROJECTS["e4oftb128refine"] = "rl-b128-refine-lr"

# tools/lora_regret/preflight.py
EXPECTED_ARMS["e4oftb128refine"] = 6
STAGE_GPU_REQUIREMENTS["e4oftb128refine"] = 8
```

Place the literal entries alongside `e4oftb128low` in each dictionary rather than assigning them after dictionary construction.

- [ ] **Step 3: Create launcher A**

Create `scripts/lora_regret/run_e4_math_oft_b128_refine_a_8gpu.sh`:

```bash
#!/usr/bin/env bash
# Math OFT BS128 refinement A: 5e-6, 6e-6, 7e-6 on one 8-GPU node.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4oftb128refine \
    METHOD_RE='^oftrefine-b128-all-math-lr(5e-06|6e-06|7e-06)-s0$' \
    RESULTS=results/e4_math_oft_b128_refine_a.jsonl \
    EXPECT_ARMS=3 \
    ALLOW_OFT=1 \
    PREFLIGHT_STAGE=e4oftb128refine \
    bash "${HERE}/campaign.sh" "$@"
```

- [ ] **Step 4: Create launcher B**

Create `scripts/lora_regret/run_e4_math_oft_b128_refine_b_8gpu.sh`:

```bash
#!/usr/bin/env bash
# Math OFT BS128 refinement B: 8e-6, 9e-6, 2e-5 on one 8-GPU node.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4oftb128refine \
    METHOD_RE='^oftrefine-b128-all-math-lr(8e-06|9e-06|2e-05)-s0$' \
    RESULTS=results/e4_math_oft_b128_refine_b.jsonl \
    EXPECT_ARMS=3 \
    ALLOW_OFT=1 \
    PREFLIGHT_STAGE=e4oftb128refine \
    bash "${HERE}/campaign.sh" "$@"
```

Mark both files executable.

- [ ] **Step 5: Run local static verification**

```bash
bash -n scripts/lora_regret/run_e4_math_oft_b128_refine_a_8gpu.sh
bash -n scripts/lora_regret/run_e4_math_oft_b128_refine_b_8gpu.sh
python3 -m py_compile \
    tools/lora_regret/arms.py \
    tools/lora_regret/sweep.py \
    tools/lora_regret/preflight.py \
    tests/fast/utils/test_math_oft_b128_refinement_sweep.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit and push the minimal implementation**

```bash
git add \
    tools/lora_regret/arms.py \
    tools/lora_regret/sweep.py \
    tools/lora_regret/preflight.py \
    scripts/lora_regret/run_e4_math_oft_b128_refine_a_8gpu.sh \
    scripts/lora_regret/run_e4_math_oft_b128_refine_b_8gpu.sh
git commit -m "feat(oft): add BS128 refinement sweep"
git push origin codex/math-oft-b128-refine
```

Expected: the test commit remains before the production commit in history.

---

### Task 4: Run GREEN and Regression Tests on Allocation A

**Files:**
- No new repository files.

**Interfaces:**
- Consumes: the pushed implementation commit and the remote worktree created in Task 2.
- Produces: focused and regression test evidence at the exact task-branch commit.

- [ ] **Step 1: Fast-forward the clean remote worktree**

Inside allocation A:

```bash
set -euo pipefail
REMOTE_WT=/fast/zqiu/orbit-iclr/orbit/.worktrees/math-oft-b128-refine
git -C "$REMOTE_WT" fetch --no-tags origin \
    codex/math-oft-b128-refine:refs/remotes/origin/codex/math-oft-b128-refine
git -C "$REMOTE_WT" merge --ff-only \
    refs/remotes/origin/codex/math-oft-b128-refine
test -z "$(git -C "$REMOTE_WT" status --porcelain)"
```

- [ ] **Step 2: Run the focused GREEN test**

```bash
set -o pipefail
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /fast/zqiu/orbit-iclr/orbit/.worktrees/math-oft-b128-refine
source env.sh
PYTHONPATH="$PWD" python -m pytest -q \
    tests/fast/utils/test_math_oft_b128_refinement_sweep.py \
    2>&1 | tee /tmp/math-oft-b128-refine-green.log
test "${PIPESTATUS[0]}" -eq 0
```

Expected: all focused cases pass, including both real shell launchers through the fake campaign boundary.

- [ ] **Step 3: Run the registry regression set**

```bash
set -o pipefail
PYTHONPATH="$PWD" python -m pytest -q \
    tests/fast/utils/test_math_oft_b128_refinement_sweep.py \
    tests/fast/utils/test_math_oft_b128_low_lr_sweep.py \
    tests/fast/utils/test_lora_regret_arms_coverage.py \
    tests/fast/utils/test_lora_regret_sweep.py \
    tests/fast/utils/test_lora_regret_preflight.py \
    2>&1 | tee /tmp/math-oft-b128-refine-regression.log
test "${PIPESTATUS[0]}" -eq 0
```

Expected: pytest exits 0 with no failed, error, or deselected tests.

- [ ] **Step 4: Verify exact source and clean state**

```bash
test "$(git rev-parse HEAD)" = "$(git rev-parse refs/remotes/origin/codex/math-oft-b128-refine)"
test -z "$(git status --porcelain)"
git diff --check HEAD^
```

Expected: the tested commit is exactly the pushed task ref and the worktree is clean.

---

### Task 5: Review, Fast-Forward the Feature Branch, and Publish It

**Files:**
- Existing task-branch files only.

**Interfaces:**
- Consumes: exact green task commit from Task 4.
- Produces: `origin/feat/lora-without-regret` and `/fast/zqiu/orbit-iclr/orbit` at the same tested commit.

- [ ] **Step 1: Run final local review gates**

```bash
git status --short
git diff --check feat/lora-without-regret..codex/math-oft-b128-refine
git log --oneline --decorate feat/lora-without-regret..codex/math-oft-b128-refine
```

Verify the range contains only the design, implementation plan, behavioral test, registry changes, and two launchers. Reject unrelated files.

- [ ] **Step 2: Fast-forward the local feature branch**

From the main checkout `/Users/zqiu/Documents/GitHub/orbit-iclr/orbit`:

```bash
set -euo pipefail
git fetch --no-tags origin \
    feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
test "$(git symbolic-ref --short HEAD)" = feat/lora-without-regret
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/lora-without-regret)"
git merge --ff-only codex/math-oft-b128-refine
```

- [ ] **Step 3: Push and verify the exact server-side SHA**

```bash
FINAL_SHA=$(git rev-parse HEAD)
git push origin feat/lora-without-regret
test "$(git ls-remote origin refs/heads/feat/lora-without-regret | awk '{print $1}')" = "$FINAL_SHA"
```

- [ ] **Step 4: Fast-forward the shared cluster checkout once**

On idle allocation A, before either training command starts:

```bash
set -euo pipefail
ROOT=/fast/zqiu/orbit-iclr/orbit
git -C "$ROOT" fetch --no-tags origin \
    feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
test "$(git -C "$ROOT" symbolic-ref --short HEAD)" = feat/lora-without-regret
test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)"
git -C "$ROOT" merge --ff-only refs/remotes/origin/feat/lora-without-regret
test "$(git -C "$ROOT" rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)"
```

Preserve untracked `results/`, `logs/`, and `wandb/` artifacts. Stop on a tracked edit or wrong branch; do not reset, stash, or clean the shared checkout.

---

### Task 6: Launch Both Three-Arm Campaigns

**Files:**
- Runtime outputs only in the project-native `results/`, `logs/lora_regret/`, and `wandb/` locations.

**Interfaces:**
- Consumes: two running managed H100 sessions and the shared checkout at `FINAL_SHA`.
- Produces: two concurrent three-arm Math OFT BS128 campaigns with independent ledgers.

- [ ] **Step 1: Launch column A exactly once**

```bash
python3 "$CONTROL" --host "$LOGIN_A" send "$SESSION_A" -- \
    bash -lc 'source /fast/zqiu/orbit-iclr/orbit_env/bin/activate && cd /fast/zqiu/orbit-iclr/orbit && exec bash scripts/lora_regret/run_e4_math_oft_b128_refine_a_8gpu.sh'
```

- [ ] **Step 2: Launch column B exactly once**

```bash
python3 "$CONTROL" --host "$LOGIN_B" send "$SESSION_B" -- \
    bash -lc 'source /fast/zqiu/orbit-iclr/orbit_env/bin/activate && cd /fast/zqiu/orbit-iclr/orbit && exec bash scripts/lora_regret/run_e4_math_oft_b128_refine_b_8gpu.sh'
```

- [ ] **Step 3: Confirm each campaign crossed its selection boundary**

```bash
python3 "$CONTROL" --host "$LOGIN_A" capture "$SESSION_A"
python3 "$CONTROL" --host "$LOGIN_B" capture "$SESSION_B"
```

Expected in each pane: `3 arms selected` and `running 3 arms sequentially on 8 GPUs`, or a resume message showing fewer TODO arms because accepted rows already exist. If a script exits, preserve its logs and report the first error; do not add a wrapper or relaunch the healthy peer.

- [ ] **Step 4: Monitor without sending input**

Use controller `panes`, `capture`, and `job` only for read-only status. Do not poll logs continuously. Report meaningful transitions: first arm started, arm completed and ledger row appended, next arm started, or terminal failure/completion.

- [ ] **Step 5: Verify terminal ledgers and summarize results**

After both scripts return, parse the two JSONL ledgers and require the latest row for every expected arm to have `status == "ok"`, `metric == "accuracy"`, a non-null final accuracy, and 150 completed rollouts. Report one six-row table ordered by learning rate and compare it with the prior BS128 points `1e-7` through `1e-5` and `3e-5`.

Do not infer hardware timing differences from the two nodes; this is a single-seed accuracy refinement.
