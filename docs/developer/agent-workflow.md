---
title: Agent Workflow
description: Plan-first collaboration conventions for Codex, Claude Code, and human implementers.
---

# Agent workflow

Use this workflow when an agent is helping with non-trivial changes.

## Record plans in docs

- Prefer repo docs over assistant memory for project-specific agreements.
- Put subsystem-specific implementation plans next to that subsystem's docs, for example
  `scripts/slurm/docs/plan-notes/<topic>.md`.
- Keep `debug-notes/` for incident reports and investigation notes; keep
  `plan-notes/` for decision-complete implementation plans. Both are gitignored
  scratch space. Upstream-sync records are the exception: they live in the
  git-tracked `scripts/slurm/docs/sync-records/` (see its README) so every sync's
  context ships in its PR.
- Use a personal skill only when the workflow should apply across many repos, not for a
  single-repo engineering convention.

## Plan first

1. Explore with non-mutating commands until the current implementation is understood.
2. Write a decision-complete plan: goal, exact behavior, files/interfaces, failure modes,
   tests, and assumptions.
3. Save the plan before implementation starts, so Codex, Claude Code, or a human can
   execute the same spec.

## Branch and worktree

- Create a dedicated branch and sibling worktree for each implementation stream.
- Use short topic branch names without an agent prefix, for example
  `nccl-ib-healthcheck`.
- Put sibling worktrees under `/data/home/<user>/workspace/<repo>-worktree/<topic>`.
- Do not rely on uncommitted changes from another worktree unless the plan explicitly
  says so.
- Do not copy `.claude/scheduled_tasks.lock` between worktrees. It is Claude Code
  runtime state for `/loop` and scheduled tasks; if a worktree needs a monitor, let
  Claude Code recreate the lock in that worktree.

## Execution

- The executor may be Codex, Claude Code, or a human.
- The executor may edit files and run validation only after the plan is accepted or the
  user explicitly asks for execution.
- Do not commit unless the user explicitly asks for a commit.
- If implementation reveals a high-impact ambiguity, stop and update the plan instead of
  making a hidden decision.
