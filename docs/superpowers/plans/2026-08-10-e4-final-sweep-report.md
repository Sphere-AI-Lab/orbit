# E4 GSM8K and Math Final Sweep Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one verified, self-contained HTML experiment record for the completed E4 GSM8K and Math FullFT-versus-LoRA sweeps.

**Architecture:** Treat the remote JSONL ledgers and supporting logs as authoritative evidence, reduce them into deduplicated per-arm result tables, and author one Markdown record that supersedes the partial GSM8K report. Render the Markdown with the repository's established HTML-report workflow, regenerate the index, and commit only the derived report artifacts.

**Tech Stack:** Bash/SSH for one bounded evidence snapshot, Python project analysis utilities, Markdown, the stdlib-only `htmlreport.py` renderer, Git.

## Global Constraints

- Do not access `mpi1`; use exactly one connection attempt to the selected stable login host.
- Do not launch training, evaluation, W&B synchronization, or any new Condor allocation.
- Do not commit raw result ledgers, logs, checkpoints, credentials, or run-store snapshots.
- Deduplicate by arm and prefer successful rows over stale failed attempts.
- Report the two intentionally abandoned Math LoRA `1e-3` arms as negative outcomes, not missing successes.
- Do not infer missing trajectories or claim statistical significance from the single seed.

---

### Task 1: Capture and validate authoritative experiment evidence

**Files:**
- Read: `/fast/zqiu/orbit-iclr/orbit/results/e4_gsm8k_lr*.jsonl`
- Read: `/fast/zqiu/orbit-iclr/orbit/results/e4_math_lr*.jsonl`
- Read: `/fast/zqiu/orbit-iclr/orbit/logs/lora_regret/`
- Create temporarily: a bounded local snapshot below `/private/tmp/`

**Interfaces:**
- Consumes: completed remote E4 ledgers, remote Git provenance, and supporting run logs.
- Produces: a locally inspected, uncommitted evidence snapshot with exactly 31 successful GSM8K arms and 29 successful Math arms, plus two documented Math LR7 LoRA abandonments.

- [ ] **Step 1: Verify the selected login host once**

Run the Condor controller's `check-connection --attempts 1` for `mpi2`, without probing or retrying any failed host.

- [ ] **Step 2: Record remote provenance and copy one bounded snapshot**

Record the remote checkout's branch, commit, and status, then copy only the 16 E4 JSONL ledgers and the minimum log evidence required for endpoint/trajectory validation into one temporary local directory.

- [ ] **Step 3: Validate successful-arm counts and uniqueness**

Filter rows where `status == "ok"`, deduplicate by `arm`, and assert:

```text
GSM8K: lr0=3, lr1..lr7=4 each, total=31
Math:  lr0=3, lr1..lr6=4 each, lr7=2, total=29
```

Fail the task if a successful arm has conflicting endpoint records or any expected count differs.

- [ ] **Step 4: Derive report-ready tables**

Produce endpoint accuracy, wall time, parameter count, step count, per-dataset accuracy, and available evaluation checkpoints for every successful arm. Keep stale failures and the abandoned Math arms in a separate negative-results inventory.

### Task 2: Author and render the final experiment record

**Files:**
- Create: `docs/reports/_src/2026-08-10-e4-gsm8k-math-panel.md`
- Create: `docs/reports/2026-08-10-e4-gsm8k-math-panel.html`
- Modify: `docs/reports/index.html` (renderer-generated)
- Delete: `docs/reports/_src/2026-08-08-e4-gsm8k-panel.md`
- Delete: `docs/reports/2026-08-08-e4-gsm8k-panel.html`

**Interfaces:**
- Consumes: the validated Task 1 tables and provenance.
- Produces: one Markdown experiment source, one self-contained HTML artifact, and an index containing only the final combined E4 record.

- [ ] **Step 1: Write the Markdown source**

Include executive results, setup, exact commands, complete GSM8K and Math endpoint matrices, per-arm details, supported trajectory evidence, interpretation, negative outcomes, limitations, and next steps. Distinguish training completion from W&B synchronization state.

- [ ] **Step 2: Remove the superseded partial record**

Delete the August 8 GSM8K-only Markdown and HTML files so the report index cannot present duplicate records for the same E4 experiment.

- [ ] **Step 3: Render and regenerate the index**

Run:

```bash
python3 /Users/zqiu/Documents/GitHub/agent-skills/personal/html-reports/scripts/htmlreport.py render \
  docs/reports/_src/2026-08-10-e4-gsm8k-math-panel.md \
  -o docs/reports/2026-08-10-e4-gsm8k-math-panel.html \
  --repo . --meta scheduler=HTCondor --meta login_host=mpi2 --index
```

Read stderr and resolve every warning before proceeding.

### Task 3: Verify, commit, and publish

**Files:**
- Verify: `docs/reports/_src/2026-08-10-e4-gsm8k-math-panel.md`
- Verify: `docs/reports/2026-08-10-e4-gsm8k-math-panel.html`
- Verify: `docs/reports/index.html`

**Interfaces:**
- Consumes: Task 2 report artifacts.
- Produces: a reviewed commit pushed to `origin/feat/lora-without-regret`.

- [ ] **Step 1: Run structural verification**

Run:

```bash
grep -c 'class="math-block unparsed"' docs/reports/2026-08-10-e4-gsm8k-math-panel.html
grep -o 'src="[^d][^"]*"' docs/reports/2026-08-10-e4-gsm8k-math-panel.html
git diff --check
```

Expect zero unparsed math blocks, no non-data sources, and no whitespace errors.

- [ ] **Step 2: Reconcile the rendered numbers**

Parse the Markdown endpoint tables and compare every arm/lr/accuracy tuple against the validated ledger reduction. Verify the index links to the combined report and not the deleted partial report.

- [ ] **Step 3: Inspect visual layout**

Open the local HTML report and inspect the executive summary, both endpoint matrices, long per-arm tables, callouts, provenance, and narrow-width table scrolling. Fix any clipping, broken anchors, or unreadable density.

- [ ] **Step 4: Review and commit**

Review `git diff --stat`, `git diff --check`, and the complete Markdown diff, then commit the plan and report artifacts with:

```bash
git add docs/superpowers/plans/2026-08-10-e4-final-sweep-report.md docs/reports
git commit -m "docs(lora-regret): publish the final e4 sweep results"
```

- [ ] **Step 5: Push and verify the remote ref**

Push `feat/lora-without-regret` to `origin`, then verify the remote branch resolves to the new local `HEAD`.
