# OFT Tiny Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the tested tiny-block OFT SGLang runtime on Orbit's stable SGLang line, pin Orbit to that exact runtime, prove the first-adapter-update transport fix with real colocated BS8 training, and then run the three-arm eight-GPU Math OFT LR3 campaign.

**Architecture:** Treat the work as four independently auditable source transitions separated by hard runtime gates: a pure fast-forward of SGLang's stable ref, a fresh three-file Orbit dependency-pin commit, an isolated three-file Orbit transport commit, and a source-clean campaign branch. All GPU work runs in the existing managed Condor allocation only after its identity is revalidated, and all evidence begins in the canonical run store outside every repository.

**Tech Stack:** Git and project-local worktrees, Python 3.12 and pytest, uv lock/sync, SGLang/Triton/CUDA, Orbit/Megatron/Ray, Bash launchers, HTCondor through the `control-remote-condor` controller, tmux, JSONL ledgers, and offline Weights & Biases.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-11-oft-tiny-runtime-integration-design.md` at Orbit commit `e2bc1cacaecb23c2df89c2f6f4be7f3498f2464a`.
- Execute this as one sequential plan because every later stage consumes the exact commit admitted by the previous stage. Do not parallelize source mutations, environment mutation, transport training, or the campaign.
- SGLang stable starts at `89ea43812ec6fb161fe29902a6c6f1fbefb524dd` on `orbit-sgl-v0.5.9`; the already completed and pushed feature tip is `b52394d22fc4b686016943efc47cce6fb892cef2` on `codex/oft-bs4`.
- Orbit integration targets `feat/lora-without-regret`, not `main`. Start the pin branch from the committed implementation-plan tip whose parent is the reviewed spec commit, not the historical pre-spec commit.
- Keep SGLang's in-repository `sgl-kernel/` tree at `5e56c1fc833007052c45c707eed1665dfb8de508` across the stable fast-forward. Separately, keep Orbit's external `sgl-kernel` source pin at `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`. Do not change Megatron-Bridge source or its pin.
- Do not merge Orbit `codex/oft-bs4` wholesale. Recreate the pin as one fresh commit, then carry only transport commit `ccc351678d7ccfa8a41a48d57fb064dcb3be0e2e` on a separate validation branch.
- Use project-local worktrees beneath `.worktrees/`. Reject a pre-existing path or branch with ambiguous ownership instead of resetting or reusing it.
- Local task worktrees are authoritative. Remote worktrees are execution-only, must be clean, and must resolve to a pushed task branch at the exact local commit.
- Use `/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-runtime-integration-control` only as the clean remote Git control clone from which task worktrees are created. Do not run tests or training from that control clone, the stale `/fast/zqiu/orbit-iclr/orbit` checkout, or any dirty historical checkout.
- Every remote project command and GPU command runs inside the managed Condor allocation, never on a login node. Login-node commands may only inspect Git state, create run-store paths, synchronize source/evidence, and control the allocated tmux pane.
- Treat Condor's inherited `CUDA_VISIBLE_DEVICES` as the scheduler authority. For one- or two-GPU gates, select the first one or two comma-separated assigned entries without replacing UUIDs or remapping to assumed physical indices. For the full campaign, preserve the eight-entry value byte-for-byte.
- Before every remote operation, run the bounded four-login inventory. Partial inventory is acceptable and must be disclosed. Because the known allocation and tmux session belong to `mpi1`, an inaccessible `mpi1` is a hard stop for that operation; do not retry or reroute it in the same turn.
- The known allocation is job `17451507`, tmux session `codex-orbit-oft-training-smoke`, last observed on `i402` with eight B200 GPUs at bid 60. Reuse it only if the controller proves that exact job/session is still running and idle. Do not submit a replacement automatically. A new allocation requires a fresh capacity report and explicit approval of a new numeric bid.
- Before each remote launch, report the resolved remote run directory, local mirror, stdout path, stderr or combined-log path, provenance path, and completion path.
- Create every execution ID as `$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)` and create each run directory with collision-failing `mkdir`, not `mkdir -p` against an existing final directory.
- The canonical remote root is `${XDG_STATE_HOME:-$HOME/.local/state}/remote-cluster-runs`; the canonical local mirror is `/Users/zqiu/.local/state/remote-cluster-runs`. Run output starts there before the process starts, not in a repository and not copied there only afterward.
- Every run records local and remote paths, repository URL, branch, full SHA, clean state, import paths, environment identity, login host, execution host, Condor job, GPU model/count, tmux session, exact command, UTC timestamps, exit code, and verification result.
- Completion and acceptance files are published atomically through a same-directory temporary file followed by `mv`. Submission, process startup, or a live tmux pane is not completion evidence.
- Preserve failed evidence and stop at the failing gate. Do not force-push, rewrite a branch, broaden cleanup, modify installed `site-packages`, or substitute a different checkout.
- Commit messages are concise conventional-commit lines with no AI attribution.

### Shared-environment test-lease protocol

Every process that imports or executes from `/fast/zqiu/orbit-iclr/orbit_env` outside an already-owned mutation lease must publish a canonical `test-*` lease for its full process lifetime. Appendix A and Appendix B embed this protocol. For the inline Task 5 and Task 7 pytest orchestrations, set `ENV_ROOT`, `TEST_EXECUTION_ID`, `TEST_WORKTREE`, `TEST_COMMAND`, and `TEST_OWNER_SNAPSHOT` to the exact values named by that task, then run this block in the same foreground shell immediately before activating the environment:

```bash
RESERVATIONS="${ENV_ROOT}.remote-dev-reservations-v1"
GATE="$RESERVATIONS/gate"
TEST_LEASE="$RESERVATIONS/test-${TEST_EXECUTION_ID}"
TEST_LEASE_ARMED=0
TEST_HOST=
TEST_PID=
TEST_START_ID=
acquire_test_lease() {
  local created=0 rc=0 temporary
  mkdir -p "$RESERVATIONS" || return 1
  mkdir "$GATE" || return 1
  if [[ -e "$TEST_LEASE" ]] ||
     [[ -n "$(find "$RESERVATIONS" -mindepth 1 -maxdepth 1 \
       -type d \( -name 'test-*' -o -name 'mutation-*' \) -print -quit)" ]]; then
    rmdir "$GATE" 2>/dev/null || :
    return 1
  fi
  TEST_HOST="$(hostname -f)"
  TEST_PID="$$"
  TEST_START_ID="$(awk '{print $22}' "/proc/$TEST_PID/stat")"
  temporary="$(mktemp "$TEST_OWNER_SNAPSHOT.tmp.XXXXXX")" || rc=1
  if ((rc == 0)); then
    printf 'host=%s\npid=%s\nstart_id=%s\nagent=%s\nworktree=%s\ncommand=%s\nreserved_cpus=%s\nexecution_id=%s\n' \
      "$TEST_HOST" "$TEST_PID" "$TEST_START_ID" root "$TEST_WORKTREE" "$TEST_COMMAND" \
      1 "$TEST_EXECUTION_ID" >"$temporary" || rc=1
  fi
  if ((rc == 0)); then
    mv "$temporary" "$TEST_OWNER_SNAPSHOT" || rc=1
  else
    rm -f -- "${temporary:-}"
  fi
  if ((rc == 0)); then
    mkdir "$TEST_LEASE" && created=1 || rc=1
  fi
  if ((rc == 0)); then
    cp "$TEST_OWNER_SNAPSHOT" "$TEST_LEASE/owner" || rc=1
  fi
  if ((rc != 0 && created == 1)); then
    rm -f -- "$TEST_LEASE/owner"
    rmdir "$TEST_LEASE" 2>/dev/null || :
  fi
  rmdir "$GATE" || rc=1
  ((rc == 0)) || return 1
  TEST_LEASE_ARMED=1
}
release_test_lease() {
  local cleanup_rc=0 live_start=
  mkdir "$GATE" || return 1
  if [[ ! -f "$TEST_LEASE/owner" ]] ||
     ! cmp -s "$TEST_OWNER_SNAPSHOT" "$TEST_LEASE/owner"; then
    cleanup_rc=1
  elif [[ "$(hostname -f)" != "$TEST_HOST" ]] || [[ "$$" != "$TEST_PID" ]] ||
       [[ ! -r "/proc/$TEST_PID/stat" ]]; then
    cleanup_rc=1
  else
    live_start="$(awk '{print $22}' "/proc/$TEST_PID/stat")"
    if [[ "$live_start" != "$TEST_START_ID" ]] ||
       [[ "$(find "$TEST_LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" != owner ]]; then
      cleanup_rc=1
    else
      rm "$TEST_LEASE/owner" || cleanup_rc=1
      ((cleanup_rc != 0)) || rmdir "$TEST_LEASE" || cleanup_rc=1
    fi
  fi
  rmdir "$GATE" || cleanup_rc=1
  return "$cleanup_rc"
}
test_lease_exit_cleanup() {
  local original_rc=$? cleanup_rc=0
  trap - EXIT
  set +e
  if ((TEST_LEASE_ARMED == 1)); then
    release_test_lease
    cleanup_rc=$?
    ((cleanup_rc == 0)) && TEST_LEASE_ARMED=0
  fi
  if ((original_rc == 0 && cleanup_rc != 0)); then
    original_rc=74
  fi
  exit "$original_rc"
}
acquire_test_lease || return 75
trap test_lease_exit_cleanup EXIT
```

The mutation admission blocks while any `test-*` exists, and test admission blocks while any `test-*` or `mutation-*` exists. Test and mutation owners are therefore serialized: at most one process may use or mutate the shared environment at a time. A controller accepts a terminal run only after the exact test lease and `gate` are absent. An identity mismatch is never cleaned automatically.

## Source and Evidence Map

### SGLang source

- Stable checkout: `/Users/zqiu/Documents/GitHub/sglang-spherelab`
- Tested feature worktree: `/Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4`
- Remote execution worktree: `/fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4`
- Source changes in this plan: none. Move `orbit-sgl-v0.5.9` by fast-forward only after the fresh GPU gate.

### Orbit pin unit

- Branch: `codex/oft-runtime-pin`
- Local worktree: `/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-runtime-pin`
- Remote worktree: `/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin`
- Branch key: `codex-oft-runtime-pin-8fc8ac57`
- Modify exactly: `pyproject.toml`, `uv.lock`, `tests/fast/utils/test_lora_regret_arms_coverage.py`

### Orbit transport unit

- Branch: `codex/oft-ipc-validation`
- Local worktree: `/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-ipc-validation`
- Remote worktree: `/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation`
- Branch key: `codex-oft-ipc-validation-6ddedda9`
- Carry exactly: `orbit/backends/megatron_utils/peft_transport/backends/ipc.py`, `orbit/backends/sglang_utils/sglang_engine.py`, `tests/test_peft_ipc_transport.py`

### Campaign source and evidence

- Branch: `codex/math-oft-lr3`
- Local worktree: `/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/math-oft-lr3`
- Remote worktree: `/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3`
- Branch key: `codex-math-oft-lr3-140979bf`
- Repository source changes: none. Run-owned wrappers, links, ledgers, logs, W&B data, and completion evidence live only in the run store.

---

### Task 1: Revalidate the allocation and freshly test the SGLang feature tip

**Files:**
- Create outside repositories: SGLang Stage 1 run-store evidence only.

**Interfaces:**
- Consumes: job `17451507`, tmux session `codex-orbit-oft-training-smoke`, clean remote SGLang worktree at `b52394d22fc4b686016943efc47cce6fb892cef2`.
- Produces: a terminal focused-suite record proving exactly 189 tests passed from the intended source on one B200 GPU.

- [ ] **Step 1: Run the bounded controller inventory and gate `mpi1` once**

Run locally:

```bash
CONTROL=/Users/zqiu/Documents/GitHub/agent-skills/personal/control-remote-condor/scripts/condor_control.py
python3 "$CONTROL" sessions-all --host-timeout 8
python3 "$CONTROL" --host mpi1 check-connection --attempts 1
python3 "$CONTROL" --host mpi1 probe
python3 "$CONTROL" --host mpi1 jobs
python3 "$CONTROL" --host mpi1 job 17451507
python3 "$CONTROL" --host mpi1 capture codex-orbit-oft-training-smoke --lines 200
```

Expected: `mpi1` is reachable; job `17451507` is running on `i402`; the allocation exposes eight B200 GPUs; the managed pane is an idle compute-node shell with no test or training child. Record any unavailable `mpi2` through `mpi4` inventory entries as partial coverage without failing this gate.

Stop if `mpi1` is unavailable, the job is terminal or unknown, the execution host changed without scheduler evidence, fewer than eight B200 GPUs are assigned, or the pane is busy. Do not send a command, submit another job, or retry a different login in that turn.

- [ ] **Step 2: Verify the remote execution worktree without modifying it**

Run:

```bash
ssh mpi1 'git -C /fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4 rev-parse --show-toplevel; git -C /fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4 remote get-url origin; git -C /fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4 symbolic-ref --short HEAD; git -C /fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4 rev-parse HEAD; git -C /fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4 status --porcelain'
```

Expected, in order: the exact worktree path, `https://github.com/Sphere-AI-Lab/sglang.git`, branch `codex/oft-bs4`, SHA `b52394d22fc4b686016943efc47cce6fb892cef2`, and empty status. Any mismatch is a hard stop; do not repair or reset this worktree during the gate.

- [ ] **Step 3: Resolve and report the fresh evidence paths before launch**

Resolve these values in the local controller shell so the same absolute strings are available to the login command, tmux send, and later rsync:

```bash
EXECUTION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
REMOTE_RUN_ROOT="/lustre/home/zqiu/.local/state/remote-cluster-runs/mpi1/sglang/oft-bs4-a6a55a65/${EXECUTION_ID}/stage-1-focused-oft"
LOCAL_RUN_ROOT="/Users/zqiu/.local/state/remote-cluster-runs/mpi1/sglang/oft-bs4-a6a55a65/${EXECUTION_ID}/stage-1-focused-oft"
ssh mpi1 mkdir -p "$(dirname "$(dirname "$REMOTE_RUN_ROOT")")"
ssh mpi1 mkdir "$(dirname "$REMOTE_RUN_ROOT")"
ssh mpi1 mkdir "$REMOTE_RUN_ROOT"
printf '%s\n' "$EXECUTION_ID" "$REMOTE_RUN_ROOT"
```

The local mirror is `/Users/zqiu/.local/state/remote-cluster-runs/mpi1/sglang/oft-bs4-a6a55a65/${EXECUTION_ID}/stage-1-focused-oft`. Before sending the test, report:

- stdout: `${REMOTE_RUN_ROOT}/stdout.log`
- stderr: `${REMOTE_RUN_ROOT}/stderr.log`
- JUnit: `${REMOTE_RUN_ROOT}/pytest.xml`
- preflight: `${REMOTE_RUN_ROOT}/preflight.log`
- provenance: `${REMOTE_RUN_ROOT}/provenance.txt`
- wrapper: `${REMOTE_RUN_ROOT}/run_stage1_gate.sh`
- wrapper hash: `${REMOTE_RUN_ROOT}/run_stage1_gate.sha256`
- completion: `${REMOTE_RUN_ROOT}/completion.status`

Write initial provenance before the process starts. Install the exact executable wrapper from Appendix C at `${REMOTE_RUN_ROOT}/run_stage1_gate.sh`, substitute no paths inside it, hash it to `run_stage1_gate.sha256`, and record that hash. The wrapper's compute-side preflight records `hostname`, `nvidia-smi -L`, Python executable/version, Git URL/branch/SHA/status, the exact command, and an import check that `sglang.__file__` is beneath `/fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4/python`.

Materialize the Appendix C fenced body byte-for-byte at `/tmp/run_stage1_gate.sh` with `apply_patch`, then install and admit that exact artifact before sending it:

```bash
set -euo pipefail
bash -n /tmp/run_stage1_gate.sh
WRAPPER_SHA="$(shasum -a 256 /tmp/run_stage1_gate.sh | awk '{print $1}')"
rsync -a -- /tmp/run_stage1_gate.sh \
  "mpi1:$REMOTE_RUN_ROOT/run_stage1_gate.sh"
ssh mpi1 sh -s -- "$REMOTE_RUN_ROOT" "$EXECUTION_ID" "$WRAPPER_SHA" <<'REMOTE'
set -eu
run_dir=$1
execution_id=$2
expected_wrapper_sha=$3
wrapper="$run_dir/run_stage1_gate.sh"
remote_wrapper_sha="$(sha256sum "$wrapper" | awk '{print $1}')"
test "$remote_wrapper_sha" = "$expected_wrapper_sha"
chmod 700 "$wrapper"
hash_tmp="$(mktemp "$run_dir/run_stage1_gate.sha256.tmp.XXXXXX")"
printf '%s  %s\n' "$remote_wrapper_sha" run_stage1_gate.sh >"$hash_tmp"
mv "$hash_tmp" "$run_dir/run_stage1_gate.sha256"
provenance_tmp="$(mktemp "$run_dir/provenance.txt.tmp.XXXXXX")"
printf '%s\n' \
  "state=prepared" \
  "created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "execution_id=$execution_id" \
  "login_host=mpi1" \
  "expected_execution_host=i402" \
  "condor_job_id=17451507" \
  "tmux_session=codex-orbit-oft-training-smoke" \
  "allocation=8xB200" \
  "allocation_bid=60" \
  "repository_url=https://github.com/Sphere-AI-Lab/sglang.git" \
  "source_worktree=/fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4" \
  "source_branch=codex/oft-bs4" \
  "source_sha=b52394d22fc4b686016943efc47cce6fb892cef2" \
  "remote_run_root=$run_dir" \
  "wrapper=$wrapper" \
  "wrapper_sha256=$remote_wrapper_sha" \
  "launch_argv=bash $wrapper" \
  >"$provenance_tmp"
mv "$provenance_tmp" "$run_dir/provenance.txt"
REMOTE
test "$(ssh mpi1 sha256sum "$REMOTE_RUN_ROOT/run_stage1_gate.sh" | awk '{print $1}')" = \
  "$WRAPPER_SHA"
```

`apply_patch` is used only to materialize the reviewed appendix in `/tmp`; it does not edit a repository. Stop if syntax validation, transfer, either hash comparison, `chmod`, or either atomic publication fails.

- [ ] **Step 4: Run the exact SGLang focused suite inside the allocation**

Send the Appendix C wrapper once, using the absolute value retained in the local controller shell:

```bash
python3 "$CONTROL" --host mpi1 send codex-orbit-oft-training-smoke \
  -- bash "$REMOTE_RUN_ROOT/run_stage1_gate.sh"
```

Appendix C preserves pytest's actual status, verifies JUnit counts and stdout, performs the Git postflight, and atomically publishes completion. Expected: exit 0, `189 passed`, zero failures, zero errors, zero skips, and zero deselections. Existing warning text is recorded but does not fail the gate. Do not infer completion from the controller send result.

- [ ] **Step 5: Take one bounded local snapshot and independently verify the evidence**

Run once after completion:

```bash
mkdir -p "$(dirname "$LOCAL_RUN_ROOT")"
rsync -a -- \
  "mpi1:$REMOTE_RUN_ROOT/" \
  "$LOCAL_RUN_ROOT/"
```

Expected: every declared evidence file exists locally, the wrapper hash matches, JUnit and stdout agree on 189 passed with no skip/failure/error, provenance names the exact source and job, and completion is terminal. A live pane or controller exit alone is not acceptance.

---

### Task 2: Fast-forward and publish `orbit-sgl-v0.5.9`

**Files:**
- Modify no files and create no commit; move only the stable branch ref by fast-forward.

**Interfaces:**
- Consumes: accepted Task 1 evidence for `b52394d22fc4b686016943efc47cce6fb892cef2`.
- Produces: remote `orbit-sgl-v0.5.9` resolving to that exact tested SHA.

- [ ] **Step 1: Recheck both local SGLang worktrees and fetch only the two relevant refs**

Run:

```bash
set -euo pipefail
SGLANG_ROOT=/Users/zqiu/Documents/GitHub/sglang-spherelab
SGLANG_FEATURE_WT="$SGLANG_ROOT/.worktrees/oft-bs4"
test -z "$(git -C "$SGLANG_ROOT" status --porcelain)"
test "$(git -C "$SGLANG_ROOT" symbolic-ref --short HEAD)" = orbit-sgl-v0.5.9
test "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" = \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd
test -z "$(git -C "$SGLANG_FEATURE_WT" status --porcelain)"
test "$(git -C "$SGLANG_FEATURE_WT" symbolic-ref --short HEAD)" = codex/oft-bs4
test "$(git -C "$SGLANG_FEATURE_WT" rev-parse HEAD)" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
git -C "$SGLANG_ROOT" fetch origin \
  refs/heads/orbit-sgl-v0.5.9:refs/remotes/origin/orbit-sgl-v0.5.9 \
  refs/heads/codex/oft-bs4:refs/remotes/origin/codex/oft-bs4
test "$(git -C "$SGLANG_ROOT" rev-parse \
  refs/remotes/origin/orbit-sgl-v0.5.9)" = \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd
test "$(git -C "$SGLANG_ROOT" rev-parse \
  refs/remotes/origin/codex/oft-bs4)" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
```

Expected: stable is clean on `orbit-sgl-v0.5.9` at `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`; feature is clean on `codex/oft-bs4` at `b52394d22fc4b686016943efc47cce6fb892cef2`; both remote refs match those same values. Stop if either remote moved.

- [ ] **Step 2: Prove the audited ancestry and unchanged kernel surface**

Run:

```bash
set -euo pipefail
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab merge-base --is-ancestor \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd \
  b52394d22fc4b686016943efc47cce6fb892cef2
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-list --left-right --count \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd...b52394d22fc4b686016943efc47cce6fb892cef2
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab diff --quiet \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd \
  b52394d22fc4b686016943efc47cce6fb892cef2 \
  -- sgl-kernel python/pyproject.toml
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd:sgl-kernel
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse \
  b52394d22fc4b686016943efc47cce6fb892cef2:sgl-kernel
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd:sgl-kernel)" = \
  5e56c1fc833007052c45c707eed1665dfb8de508
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse \
  b52394d22fc4b686016943efc47cce6fb892cef2:sgl-kernel)" = \
  5e56c1fc833007052c45c707eed1665dfb8de508
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-list \
  --left-right --count \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd...b52394d22fc4b686016943efc47cce6fb892cef2)" = \
  '0	26'
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab diff --check \
  89ea43812ec6fb161fe29902a6c6f1fbefb524dd..b52394d22fc4b686016943efc47cce6fb892cef2
```

Expected: ancestry succeeds, divergence is `0 26`, the quiet diff exits zero, both `sgl-kernel` trees are `5e56c1fc833007052c45c707eed1665dfb8de508`, and `diff --check` is empty.

- [ ] **Step 3: Move the stable branch without creating a commit**

Run from the stable checkout:

```bash
set -euo pipefail
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab merge --ff-only \
  b52394d22fc4b686016943efc47cce6fb892cef2
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse HEAD)" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab symbolic-ref --short HEAD)" = \
  orbit-sgl-v0.5.9
test -z "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab status --porcelain)"
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4 rev-parse HEAD)" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
test -z "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4 status --porcelain)"
```

Expected: fast-forward from `89ea43812` to `b52394d22`, branch still `orbit-sgl-v0.5.9`, clean status. Do not run `git commit`, cherry-pick, or a non-fast-forward merge.

- [ ] **Step 4: Push only the stable ref and verify it server-side**

Run:

```bash
set -euo pipefail
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab push origin \
  refs/heads/orbit-sgl-v0.5.9:refs/heads/orbit-sgl-v0.5.9
git -C /Users/zqiu/Documents/GitHub/sglang-spherelab ls-remote --exit-code \
  origin refs/heads/orbit-sgl-v0.5.9
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab \
  ls-remote origin refs/heads/orbit-sgl-v0.5.9 | awk '{print $1}')" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab rev-parse HEAD)" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
test -z "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab status --porcelain)"
test "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4 rev-parse HEAD)" = \
  b52394d22fc4b686016943efc47cce6fb892cef2
test -z "$(git -C /Users/zqiu/Documents/GitHub/sglang-spherelab/.worktrees/oft-bs4 status --porcelain)"
```

Expected remote line:

```text
b52394d22fc4b686016943efc47cce6fb892cef2	refs/heads/orbit-sgl-v0.5.9
```

If push is rejected, stop. Never retry with force or force-with-lease. Task 3 may begin only when the fresh GPU result and server-side stable ref name the same SHA.

---

### Task 3: Create the fresh Orbit pin-only commit with a failing contract first

**Files:**
- Modify: `tests/fast/utils/test_lora_regret_arms_coverage.py`
- Modify: `pyproject.toml`
- Modify mechanically through uv: `uv.lock`

**Interfaces:**
- Consumes: published SGLang stable SHA `b52394d22fc4b686016943efc47cce6fb892cef2`.
- Produces: branch `codex/oft-runtime-pin` with one fresh pin-only commit whose diff contains exactly the three listed files.

- [ ] **Step 1: Revalidate the Orbit target and create the isolated pin worktree**

Run:

```bash
set -euo pipefail
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit fetch origin \
  refs/heads/feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
INTEGRATION_BASE="$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse HEAD)"
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit symbolic-ref --short HEAD)" = \
  feat/lora-without-regret
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse "$INTEGRATION_BASE^")" = \
  e2bc1cacaecb23c2df89c2f6f4be7f3498f2464a
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit show \
  "$INTEGRATION_BASE:docs/superpowers/plans/2026-08-11-oft-tiny-runtime-integration.md" | \
  sed -n '1p')" = '# OFT Tiny Runtime Integration Implementation Plan'
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit diff --name-only \
  e2bc1cacaecb23c2df89c2f6f4be7f3498f2464a.."$INTEGRATION_BASE")" = \
  docs/superpowers/plans/2026-08-11-oft-tiny-runtime-integration.md
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse origin/feat/lora-without-regret)" = \
  a7a87ddaa6130d2cc522770dd640d17938e4d250
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit merge-base --is-ancestor \
  origin/feat/lora-without-regret HEAD
test -z "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit status --porcelain)"
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit check-ignore -q .worktrees
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit worktree add \
  /Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-runtime-pin \
  -b codex/oft-runtime-pin "$INTEGRATION_BASE"
```

Report base `feat/lora-without-regret` at the committed implementation-plan tip `INTEGRATION_BASE`, task branch `codex/oft-runtime-pin`, and the worktree path before editing. Stop if the remote target moved, the branch/path already exists ambiguously, or the base is dirty.

- [ ] **Step 2: Add the lock-contract test without changing the pin**

Append this method to `TestOftBlockCeilingUnderRl` using `apply_patch`:

```python
    def test_sglang_runtime_supports_power_of_two_blocks_from_four(self):
        import tomllib
        from pathlib import Path

        from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

        expected_sha = "b52394d22fc4b686016943efc47cce6fb892cef2"
        supported = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        assert supported[0] == 4
        assert all(block & (block - 1) == 0 for block in supported)
        assert supported[-1] == OFT_MAX_BLOCK_SGLANG

        repo = Path(__file__).resolve().parents[3]
        config = tomllib.loads((repo / "pyproject.toml").read_text())
        sources = config["tool"]["uv"]["sources"]
        pins = config["tool"]["orbit"]["release"]["backend-pins"]
        lock = tomllib.loads((repo / "uv.lock").read_text())
        packages = {package["name"]: package for package in lock["package"]}
        orbit_requires = {
            requirement["name"]: requirement
            for requirement in packages["orbit"]["metadata"]["requires-dist"]
        }
        sglang_git = "https://github.com/Sphere-AI-Lab/sglang.git"
        kernel_sha = "9c83ae8be07cbb1eb6898ce608ae244e3be375b4"
        bridge_sha = "85c84cbc26d4c983a3d6e46c804f02e2a99af5a2"
        assert sources["sglang"]["rev"] == expected_sha
        assert pins["sglang"]["tested-ref"] == expected_sha
        assert packages["sglang"]["source"]["git"] == (
            f"{sglang_git}?subdirectory=python&rev={expected_sha}#{expected_sha}"
        )
        assert orbit_requires["sglang"]["git"] == (
            f"{sglang_git}?subdirectory=python&rev={expected_sha}"
        )
        assert sources["sgl-kernel"]["rev"] == kernel_sha
        assert packages["sgl-kernel"]["source"]["git"] == (
            f"{sglang_git}?subdirectory=sgl-kernel&rev={kernel_sha}#{kernel_sha}"
        )
        assert orbit_requires["sgl-kernel"]["git"] == (
            f"{sglang_git}?subdirectory=sgl-kernel&rev={kernel_sha}"
        )
        assert sources["megatron-bridge"]["rev"] == bridge_sha
        assert packages["megatron-bridge"]["source"]["git"].endswith(
            f"rev={bridge_sha}#{bridge_sha}"
        )
        assert orbit_requires["megatron-bridge"]["git"].endswith(
            f"rev={bridge_sha}"
        )
```

- [ ] **Step 3: Run the new test alone and prove RED against the old pin**

The local checkout lacks the project pytest environment, so run this dependency-free import harness from the pin worktree:

```bash
/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.venv/bin/python - <<'PY'
import importlib.util
import sys
import types
from pathlib import Path

pytest_stub = types.SimpleNamespace(
    mark=types.SimpleNamespace(parametrize=lambda *args, **kwargs: lambda fn: fn)
)
sys.modules["pytest"] = pytest_stub
path = Path("tests/fast/utils/test_lora_regret_arms_coverage.py").resolve()
spec = importlib.util.spec_from_file_location("pin_contract", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TestOftBlockCeilingUnderRl().test_sglang_runtime_supports_power_of_two_blocks_from_four()
PY
```

Expected: assertion failure on the old SGLang source pin because `pyproject.toml` still names `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`. A missing-module/import error is not the required RED. If the test passes before the pin changes, stop because it is not mutation-sensitive.

- [ ] **Step 4: Update only the SGLang runtime source and release pin**

Use `apply_patch` in `pyproject.toml` to:

1. Replace the SGLang source revision with `b52394d22fc4b686016943efc47cce6fb892cef2`.
2. Replace `[tool.orbit.release.backend-pins.sglang].tested-ref` with the same SHA.
3. Update the adjacent SGLang comment to say this pure-Python range now supports power-of-two OFT blocks from 4 through 1024, keeps BS16+ dot paths, and defaults QKV BS4/8 to the measured legacy path.
4. Leave `sgl-kernel`, Megatron-Bridge, Megatron-Core, and every other source entry untouched.

Rerun the Step 3 harness before touching `uv.lock`. Expected: assertion failure on `packages["sglang"]["source"]["git"]` because the lock still names `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`. This second RED proves the lockfile half of the contract.

Regenerate only SGLang's locked package from the worktree:

```bash
uv lock --upgrade-package sglang
```

Expected SGLang lock identity: source URL contains `rev=b52394d22fc4b686016943efc47cce6fb892cef2#b52394d22fc4b686016943efc47cce6fb892cef2`; package version resolves to `0.0.0.dev9909+gb52394d22`. No `sgl-kernel` URL or version changes.

- [ ] **Step 5: Prove GREEN and audit the complete pin diff**

Rerun the import harness from Step 3. Expected: exit 0.

Then run:

```bash
uv lock --check
git diff --check
git diff --name-only
git diff -- pyproject.toml uv.lock tests/fast/utils/test_lora_regret_arms_coverage.py
```

The name-only output must be exactly:

```text
pyproject.toml
tests/fast/utils/test_lora_regret_arms_coverage.py
uv.lock
```

Inspect every `sglang` and `sgl-kernel` occurrence. Require all runtime/release/lock SGLang refs to be `b52394d22fc4b686016943efc47cce6fb892cef2`, every `sgl-kernel` ref to remain `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`, and Megatron-Bridge to remain `85c84cbc26d4c983a3d6e46c804f02e2a99af5a2`.

- [ ] **Step 6: Commit the pin as one auditable unit**

Run:

```bash
set -euo pipefail
git add pyproject.toml uv.lock tests/fast/utils/test_lora_regret_arms_coverage.py
git diff --cached --check
git diff --cached --name-only
git commit -m "build(deps): pin tiny OFT runtime"
PIN_COMMIT="$(git rev-parse HEAD)"
PIN_BASE="$(git rev-parse "$PIN_COMMIT^")"
test "$(git symbolic-ref --short HEAD)" = codex/oft-runtime-pin
test "$(git rev-parse "$PIN_BASE^")" = \
  e2bc1cacaecb23c2df89c2f6f4be7f3498f2464a
test "$(git diff --name-only \
  e2bc1cacaecb23c2df89c2f6f4be7f3498f2464a.."$PIN_BASE")" = \
  docs/superpowers/plans/2026-08-11-oft-tiny-runtime-integration.md
test "$(git show "$PIN_BASE":docs/superpowers/plans/2026-08-11-oft-tiny-runtime-integration.md | sed -n '1p')" = \
  '# OFT Tiny Runtime Integration Implementation Plan'
test "$(git rev-list --count "$PIN_BASE".."$PIN_COMMIT")" = 1
test "$(git diff --name-only "$PIN_BASE".."$PIN_COMMIT")" = \
  $'pyproject.toml\ntests/fast/utils/test_lora_regret_arms_coverage.py\nuv.lock'
test -z "$(git status --porcelain)"
```

Expected: one pin commit on `codex/oft-runtime-pin` after the committed plan, exactly three changed files, clean worktree. Persist the printed `PIN_COMMIT` and `PIN_BASE` in the Task 4 provenance; later tasks re-resolve rather than assuming shell variables survive.

---

### Task 4: Publish the pin, reconcile the shared environment, and admit the pin to Orbit

**Files:**
- Modify no additional repository files.
- Create outside repositories: pin validation and shared-environment evidence.

**Interfaces:**
- Consumes: `PIN_COMMIT` from Task 3 and published SGLang stable ref.
- Produces: pushed `feat/lora-without-regret` containing the pin-only commit and a shared environment resolving to the same SGLang/sgl-kernel identities.

- [ ] **Step 1: Recheck upstream, push the task branch, and create a clean remote worktree**

Run locally:

```bash
set -euo pipefail
ORBIT_ROOT=/Users/zqiu/Documents/GitHub/orbit-iclr/orbit
PIN_WORKTREE="$ORBIT_ROOT/.worktrees/oft-runtime-pin"
PIN_COMMIT="$(git -C "$PIN_WORKTREE" rev-parse HEAD)"
PIN_BASE="$(git -C "$PIN_WORKTREE" rev-parse "$PIN_COMMIT^")"
test "$(git -C "$PIN_WORKTREE" symbolic-ref --short HEAD)" = codex/oft-runtime-pin
test -z "$(git -C "$PIN_WORKTREE" status --porcelain)"
test "$(git -C "$ORBIT_ROOT" rev-parse feat/lora-without-regret)" = "$PIN_BASE"
git -C "$ORBIT_ROOT" fetch origin \
  refs/heads/feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
test "$(git -C "$ORBIT_ROOT" rev-parse origin/feat/lora-without-regret)" = \
  a7a87ddaa6130d2cc522770dd640d17938e4d250
git -C "$PIN_WORKTREE" push -u origin codex/oft-runtime-pin
test "$(git -C "$PIN_WORKTREE" ls-remote origin refs/heads/codex/oft-runtime-pin | awk '{print $1}')" = \
  "$PIN_COMMIT"
```

On `mpi1`, verify the control clone is clean and belongs to the Orbit origin, fetch `codex/oft-runtime-pin`, reject an existing ambiguous destination, and create:

```bash
git -C /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-runtime-integration-control \
  worktree add \
  /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin \
  --track -b codex/oft-runtime-pin origin/codex/oft-runtime-pin
```

Before this command, require the control clone's top level, clean state, and origin `https://github.com/Sphere-AI-Lab/orbit.git`; fetch the exact task branch; require both the destination and local `refs/heads/codex/oft-runtime-pin` to be absent. If either exists, stop instead of deleting it. After creation verify remote URL, worktree path, branch `codex/oft-runtime-pin`, exact `PIN_COMMIT`, and empty status.

- [ ] **Step 2: Revalidate the allocation and reserve exclusive shared-environment mutation**

Repeat Task 1 Step 1 and inspect every active user Condor job and reachable managed session. Require the same idle job/session and hard-stop on any other live or ambiguous job that may use `/fast/zqiu/orbit-iclr/orbit_env`. From the safe remote control clone, list every Orbit worktree and correlate it with the job/session inventory; inspect `requires-python`, `pyproject.toml`, and `uv.lock` for every worktree supporting a live job or admitted test. Proceed only if the new locked environment satisfies all of them. Record that compatibility matrix in provenance. This cluster-wide audit is the primary admission check; a process scan on `i402` is supplemental.

Create one fresh execution ID and these sibling run labels before acquiring the lease:

```text
${XDG_STATE_HOME:-$HOME/.local/state}/remote-cluster-runs/mpi1/orbit/codex-oft-runtime-pin-8fc8ac57/${EXECUTION_ID}/shared-env-pin-sync/
${XDG_STATE_HOME:-$HOME/.local/state}/remote-cluster-runs/mpi1/orbit/codex-oft-runtime-pin-8fc8ac57/${EXECUTION_ID}/pin-focused-tests/
```

The environment label predeclares `command.txt`, `uv-version.txt`, `dry-run.json`, `dry-run.stderr.log`, `sync.stdout.log`, `sync.stderr.log`, `before.freeze`, `after.freeze`, `freeze.diff`, `imports.json`, `orbit-direct-url.before`, `orbit-direct-url.after`, `provenance.json`, and `completion.status`. The test label predeclares `command.txt`, `stdout.log`, `stderr.log`, `pytest.xml`, `provenance.json`, and `completion.status`. Report both remote directories and identical local-mirror suffixes before mutation.

Resolve and create them in the local controller shell, retaining the printed absolute values for the one compute-side orchestration process that performs Steps 2 through 5:

```bash
EXECUTION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
REMOTE_EXEC_ROOT="/lustre/home/zqiu/.local/state/remote-cluster-runs/mpi1/orbit/codex-oft-runtime-pin-8fc8ac57/${EXECUTION_ID}"
ENV_RUN_DIR="$REMOTE_EXEC_ROOT/shared-env-pin-sync"
PIN_TEST_RUN_DIR="$REMOTE_EXEC_ROOT/pin-focused-tests"
ssh mpi1 mkdir -p "$(dirname "$REMOTE_EXEC_ROOT")"
ssh mpi1 mkdir "$REMOTE_EXEC_ROOT"
ssh mpi1 mkdir "$ENV_RUN_DIR"
ssh mpi1 mkdir "$PIN_TEST_RUN_DIR"
printf '%s\n' "$EXECUTION_ID" "$ENV_RUN_DIR" "$PIN_TEST_RUN_DIR"
```

Inside the allocation, canonicalize the environment and use the task-worktree protocol's common reservation namespace. Admission and later release are serialized by an atomic `gate` directory; the mutation lease remains for the whole dry-run/sync/verification window:

```bash
set -euo pipefail
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
RESERVATIONS="${ENV_ROOT}.remote-dev-reservations-v1"
GATE="$RESERVATIONS/gate"
LEASE="$RESERVATIONS/mutation-${EXECUTION_ID}"
OWNER_SNAPSHOT="$ENV_RUN_DIR/mutation-lease-owner"
HOST="$(hostname -f)"
PID="$$"
START_ID="$(awk '{print $22}' "/proc/$PID/stat")"
GATE_OWNED=0
MUTATION_LEASE_ARMED=0
LEASE_DIR_CREATED=0
OWNER_TEMP=
release_owned_gate() {
  local cleanup_rc=0
  ((GATE_OWNED == 1)) || return 0
  if [[ -e "$GATE/owner" || -L "$GATE/owner" ]]; then
    if [[ -f "$OWNER_SNAPSHOT" ]] && cmp -s "$OWNER_SNAPSHOT" "$GATE/owner"; then
      rm "$GATE/owner" || cleanup_rc=1
    else
      cleanup_rc=1
    fi
  fi
  if ((cleanup_rc == 0)); then
    rmdir "$GATE" || cleanup_rc=1
    ((cleanup_rc != 0)) || GATE_OWNED=0
  fi
  return "$cleanup_rc"
}
release_owned_mutation_lease() {
  local cleanup_rc=0 live_start=
  if ((GATE_OWNED == 0)); then
    mkdir "$GATE" || return 1
    GATE_OWNED=1
  fi
  if [[ ! -e "$GATE/owner" && ! -L "$GATE/owner" ]]; then
    cp "$OWNER_SNAPSHOT" "$GATE/owner" || cleanup_rc=1
  elif ! cmp -s "$OWNER_SNAPSHOT" "$GATE/owner"; then
    cleanup_rc=1
  fi
  if [[ ! -f "$LEASE/owner" ]] || ! cmp -s "$OWNER_SNAPSHOT" "$LEASE/owner"; then
    cleanup_rc=1
  elif [[ "$(hostname -f)" != "$HOST" ]] || [[ "$$" != "$PID" ]] ||
       [[ ! -r "/proc/$PID/stat" ]]; then
    cleanup_rc=1
  else
    live_start="$(awk '{print $22}' "/proc/$PID/stat")"
    if [[ "$live_start" != "$START_ID" ]] ||
       [[ "$(find "$LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" != owner ]]; then
      cleanup_rc=1
    else
      rm "$LEASE/owner" || cleanup_rc=1
      if ((cleanup_rc == 0)); then
        rmdir "$LEASE" || cleanup_rc=1
        if ((cleanup_rc == 0)); then
          MUTATION_LEASE_ARMED=0
          LEASE_DIR_CREATED=0
        fi
      fi
    fi
  fi
  release_owned_gate || cleanup_rc=1
  return "$cleanup_rc"
}
release_owned_unarmed_lease_dir() {
  ((LEASE_DIR_CREATED == 1 && GATE_OWNED == 1)) || return 1
  if [[ -f "$LEASE/owner" ]] && cmp -s "$OWNER_SNAPSHOT" "$LEASE/owner" &&
     [[ "$(find "$LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" == owner ]]; then
    rm "$LEASE/owner" || return 1
  elif [[ -n "$(find "$LEASE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    return 1
  fi
  rmdir "$LEASE" || return 1
  LEASE_DIR_CREATED=0
}
mutation_exit_cleanup() {
  local original_rc=$? cleanup_rc=0
  trap - EXIT
  set +e
  if ((MUTATION_LEASE_ARMED == 1)); then
    release_owned_mutation_lease
    cleanup_rc=$?
  elif ((LEASE_DIR_CREATED == 1)); then
    release_owned_unarmed_lease_dir || cleanup_rc=$?
    release_owned_gate || cleanup_rc=1
  elif ((GATE_OWNED == 1)); then
    release_owned_gate
    cleanup_rc=$?
  fi
  [[ -z "$OWNER_TEMP" ]] || rm -f -- "$OWNER_TEMP"
  if ((original_rc == 0 && cleanup_rc != 0)); then
    original_rc=74
  fi
  exit "$original_rc"
}
trap mutation_exit_cleanup EXIT
mkdir -p "$RESERVATIONS"
mkdir "$GATE"
GATE_OWNED=1
OWNER_TEMP="$(mktemp "$OWNER_SNAPSHOT.tmp.XXXXXX")"
printf 'host=%s\npid=%s\nstart_id=%s\nagent=%s\nworktree=%s\ncommand=%s\nreserved_cpus=%s\nexecution_id=%s\n' \
  "$HOST" "$PID" "$START_ID" root \
  /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin \
  'uv sync --locked --inexact --extra allinone --no-install-project' \
  1 "$EXECUTION_ID" >"$OWNER_TEMP"
mv "$OWNER_TEMP" "$OWNER_SNAPSHOT"
OWNER_TEMP=
cp "$OWNER_SNAPSHOT" "$GATE/owner"
if [[ -n "$(find "$RESERVATIONS" -mindepth 1 -maxdepth 1 -type d \
     \( -name 'test-*' -o -name 'mutation-*' \) -print)" ]] ||
   [[ -n "$(pgrep -af '/fast/zqiu/orbit-iclr/orbit_env/bin/(python|ray|torchrun)' || true)" ]]; then
  exit 75
fi
mkdir "$LEASE"
LEASE_DIR_CREATED=1
cp "$OWNER_SNAPSHOT" "$LEASE/owner"
MUTATION_LEASE_ARMED=1
rm "$GATE/owner"
rmdir "$GATE"
GATE_OWNED=0
```

If `gate` or any recognizable lease exists, verify its owner identity under the gate following `develop-on-remote-clusters/references/task-worktree-protocol.md`; never delete a live or ambiguous record. This plan does not authorize automatic stale pruning. Record the owner record, job, source SHAs, active-job/session audit, and intended command in provenance immediately after admission.

- [ ] **Step 3: Dry-run the locked shared-environment reconciliation**

From the remote pin worktree inside the allocation:

```bash
cd /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT"
export UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source "$ENV_ROOT/bin/activate"
source env.sh
test "$ORBIT_VENV" = "$ENV_ROOT"
UV_BIN=/home/zqiu/.local/bin/uv
"$UV_BIN" --version >"$ENV_RUN_DIR/uv-version.txt"
grep -Eq '^uv 0\.10\.11([[:space:]]|$)' "$ENV_RUN_DIR/uv-version.txt"
OLD_SGLANG_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("sglang"))')"
NEW_SGLANG_VERSION=0.0.0.dev9909+gb52394d22
test "$OLD_SGLANG_VERSION" != "$NEW_SGLANG_VERSION"
"$UV_BIN" pip freeze --strict | LC_ALL=C sort >"$ENV_RUN_DIR/before.freeze"
python -c 'import importlib.metadata as m; print(m.distribution("orbit").read_text("direct_url.json") or "")' \
  >"$ENV_RUN_DIR/orbit-direct-url.before"
"$UV_BIN" --preview-features json-output sync \
  --locked --inexact --extra allinone --no-install-project --dry-run \
  --output-format json \
  >"$ENV_RUN_DIR/dry-run.json" 2>"$ENV_RUN_DIR/dry-run.stderr.log"
python - "$ENV_RUN_DIR/dry-run.json" "$PWD" "$ENV_ROOT" \
  "$OLD_SGLANG_VERSION" "$NEW_SGLANG_VERSION" <<'PY'
import json
import os
import sys

path, project_root, env_root, old, new = sys.argv[1:]
document = json.load(open(path))
assert document["schema"] == {"version": "preview"}, document["schema"]
assert document["target"] == "project", document["target"]
assert document["dry_run"] is True
assert os.path.realpath(document["project"]["path"]) == os.path.realpath(project_root)
assert os.path.realpath(document["project"]["workspace"]["path"]) == os.path.realpath(project_root)
assert os.path.realpath(document["sync"]["environment"]["path"]) == os.path.realpath(env_root)
assert document["sync"]["action"] == "check", document["sync"]["action"]
assert document["lock"]["action"] == "check", document["lock"]
assert os.path.realpath(document["lock"]["path"]) == os.path.realpath(
    os.path.join(project_root, "uv.lock")
)
changes = document["sync"]["changes"]
assert len(changes) == 2, changes
assert all(set(change) == {"name", "version", "action"}
           for change in changes), changes
actual = {(change["name"], change["version"], change["action"])
          for change in changes}
assert actual == {
    ("sglang", old, "uninstalled"),
    ("sglang", new, "installed"),
}, changes
PY
```

The exact `uv --version` string is first captured in provenance; if the cluster binary differs, stop and inspect that binary's help and one isolated no-op schema probe before changing this parser. `json-output` is a preview feature in uv 0.10.11, so both the feature flag and schema assertions are mandatory. Capture stdout/stderr in a fresh run label `shared-env-pin-sync`. Hard stop if the dry run proposes installing, removing, rebuilding, or changing anything other than `sglang`. In particular, it must not rebuild `sgl-kernel`, alter CUDA/Torch/Megatron packages, or install Orbit from the temporary worktree.

- [ ] **Step 4: Apply the exact same sync and verify installed identities**

Run the same command without `--dry-run`:

```bash
"$UV_BIN" sync --locked --inexact --extra allinone --no-install-project \
  >"$ENV_RUN_DIR/sync.stdout.log" 2>"$ENV_RUN_DIR/sync.stderr.log"
"$UV_BIN" pip freeze --strict | LC_ALL=C sort >"$ENV_RUN_DIR/after.freeze"
diff -u "$ENV_RUN_DIR/before.freeze" "$ENV_RUN_DIR/after.freeze" \
  >"$ENV_RUN_DIR/freeze.diff" || test "$?" -eq 1
python -c 'import importlib.metadata as m; print(m.distribution("orbit").read_text("direct_url.json") or "")' \
  >"$ENV_RUN_DIR/orbit-direct-url.after"
cmp "$ENV_RUN_DIR/orbit-direct-url.before" "$ENV_RUN_DIR/orbit-direct-url.after"
```

Require the two freeze files, after removing only lines beginning `sglang @ `, to be byte-identical; require exactly one old and one new `sglang @ ` line; require the new line to contain `b52394d22fc4b686016943efc47cce6fb892cef2`. With `PYTHONPATH=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin`, run this probe into `imports.json`:

```bash
grep -v '^sglang @ ' "$ENV_RUN_DIR/before.freeze" >"$ENV_RUN_DIR/before.without-sglang"
grep -v '^sglang @ ' "$ENV_RUN_DIR/after.freeze" >"$ENV_RUN_DIR/after.without-sglang"
cmp "$ENV_RUN_DIR/before.without-sglang" "$ENV_RUN_DIR/after.without-sglang"
test "$(grep -c '^sglang @ ' "$ENV_RUN_DIR/before.freeze")" -eq 1
test "$(grep -c '^sglang @ ' "$ENV_RUN_DIR/after.freeze")" -eq 1
grep -Fq b52394d22fc4b686016943efc47cce6fb892cef2 \
  "$ENV_RUN_DIR/after.freeze"
```

```python
import importlib.metadata as md
import json
import os
import sys

import orbit
import sglang

env_root = os.path.realpath("/fast/zqiu/orbit-iclr/orbit_env")
orbit_root = os.path.realpath(
    "/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin"
)
expected = {
    "sglang": ("b52394d22fc4b686016943efc47cce6fb892cef2", "python"),
    "sgl-kernel": ("9c83ae8be07cbb1eb6898ce608ae244e3be375b4", "sgl-kernel"),
    "megatron-bridge": ("85c84cbc26d4c983a3d6e46c804f02e2a99af5a2", None),
    "megatron-core": ("00eb75b0c803b0fc8e5413d736529d9d3b82b6bd", None),
}
assert os.path.commonpath([os.path.realpath(orbit.__file__), orbit_root]) == orbit_root
assert os.path.commonpath([os.path.realpath(sglang.__file__), env_root]) == env_root
assert os.path.realpath(sys.prefix) == env_root
records = {}
for name, (sha, subdirectory) in expected.items():
    direct = json.loads(md.distribution(name).read_text("direct_url.json"))
    if subdirectory is not None:
        assert direct["subdirectory"] == subdirectory
    assert direct["vcs_info"]["commit_id"] == sha
    assert direct["vcs_info"]["requested_revision"] == sha
    records[name] = direct
records["orbit_file"] = orbit.__file__
records["sglang_file"] = sglang.__file__
records["sys_prefix"] = sys.prefix
print(json.dumps(records, indent=2, sort_keys=True))
```

Atomically publish the environment-sync completion only after every assertion passes. Keep the owned mutation lease through the focused tests in Step 5 so no other test or mutation can observe a half-verified environment. A successful uv exit without matching import/direct-URL/freeze evidence is failure.

- [ ] **Step 5: Run the focused Orbit pin/configuration suite in the allocation**

Create a new run label `pin-focused-tests` beneath the same branch/execution identity and run:

```bash
cd /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT" UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source "$ENV_ROOT/bin/activate"
source env.sh
test "$ORBIT_VENV" = "$ENV_ROOT"
test_rc=0
PYTHONPATH=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-runtime-pin \
python -m pytest -q \
  --junitxml="$PIN_TEST_RUN_DIR/pytest.xml" \
  tests/fast/utils/test_lora_regret_arms_coverage.py \
  tests/fast/utils/test_lora_regret_lr_columns.py \
  tests/fast/utils/test_lora_regret_preflight.py \
  tests/fast/utils/test_peft_param_match.py \
  tests/fast/utils/test_lora_arguments.py \
  >"$PIN_TEST_RUN_DIR/stdout.log" 2>"$PIN_TEST_RUN_DIR/stderr.log" \
  || test_rc=$?
verification_rc=0
test "$test_rc" -eq 0 || verification_rc=1
python - "$PIN_TEST_RUN_DIR/pytest.xml" <<'PY' || verification_rc=1
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = list(root) if root.tag == "testsuites" else [root]
for field in ("failures", "errors", "skipped"):
    assert sum(int(suite.attrib.get(field, 0)) for suite in suites) == 0, field
PY
grep -Fq test_sglang_runtime_supports_power_of_two_blocks_from_four \
  "$PIN_TEST_RUN_DIR/pytest.xml" || verification_rc=1
test "$(git rev-parse HEAD)" = "$PIN_COMMIT" || verification_rc=1
test -z "$(git status --porcelain)" || verification_rc=1
final_rc=$test_rc
((final_rc == 0 && verification_rc != 0)) && final_rc=1
status_tmp="$(mktemp "$PIN_TEST_RUN_DIR/completion.status.tmp.XXXXXX")"
printf 'test_exit_code=%s\nverification_exit_code=%s\nfinal_exit_code=%s\n' \
  "$test_rc" "$verification_rc" "$final_rc" >"$status_tmp"
mv "$status_tmp" "$PIN_TEST_RUN_DIR/completion.status"
test "$final_rc" -eq 0
```

Expected: exit 0, no failures/errors/skips, exact pin-contract test collected. Recheck remote SHA and clean state, atomically publish completion, and take one bounded local snapshot of the environment and test labels.

After both labels are terminal, end the one foreground orchestration process with its real status. Its installed `EXIT` trap atomically reacquires `gate`, requires `LEASE/owner` to be byte-identical to `mutation-lease-owner` (which contains the saved host/PID/start identity, execution ID, worktree, and command), removes only that owner file and owned lease, and then removes the gate. A failing task status is preserved; cleanup failure changes only an otherwise-successful status to 74. Never remove a lease whose identity differs. The controller independently requires the exact lease path and `gate` to be absent before accepting the run. Steps 2 through 5 run in one foreground orchestration process so the recorded PID/start identity remains valid for release.

- [ ] **Step 6: Fast-forward the Orbit target to the proven pin and push it**

After remote evidence is locally verified, run:

```bash
set -euo pipefail
ORBIT_ROOT=/Users/zqiu/Documents/GitHub/orbit-iclr/orbit
PIN_WORKTREE="$ORBIT_ROOT/.worktrees/oft-runtime-pin"
PIN_COMMIT="$(git -C "$PIN_WORKTREE" rev-parse HEAD)"
PIN_BASE="$(git -C "$PIN_WORKTREE" rev-parse "$PIN_COMMIT^")"
git -C "$ORBIT_ROOT" fetch origin \
  refs/heads/feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse origin/feat/lora-without-regret)" = \
  a7a87ddaa6130d2cc522770dd640d17938e4d250
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse HEAD)" = \
  "$PIN_BASE"
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit symbolic-ref --short HEAD)" = \
  feat/lora-without-regret
test -z "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit status --porcelain)"
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit merge --ff-only codex/oft-runtime-pin
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse HEAD)" = "$PIN_COMMIT"
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit symbolic-ref --short HEAD)" = \
  feat/lora-without-regret
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit push origin \
  refs/heads/feat/lora-without-regret:refs/heads/feat/lora-without-regret
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit ls-remote --exit-code \
  origin refs/heads/feat/lora-without-regret
test "$(git -C "$ORBIT_ROOT" ls-remote origin refs/heads/feat/lora-without-regret | awk '{print $1}')" = \
  "$PIN_COMMIT"
test -z "$(git -C "$ORBIT_ROOT" status --porcelain)"
```

Expected: a pure fast-forward through the reviewed docs and pin commit; remote target resolves to `PIN_COMMIT`; local target is clean. If upstream moved, stop and re-audit instead of merging or forcing.

---

### Task 5: Isolate the three-file transport fix and run its focused tests

**Files:**
- Modify through cherry-pick: `orbit/backends/megatron_utils/peft_transport/backends/ipc.py`
- Modify through cherry-pick: `orbit/backends/sglang_utils/sglang_engine.py`
- Create through cherry-pick: `tests/test_peft_ipc_transport.py`
- Create outside repositories: transport test evidence.

**Interfaces:**
- Consumes: pushed `PIN_COMMIT` and transport source commit `ccc351678d7ccfa8a41a48d57fb064dcb3be0e2e`.
- Produces: a pushed, isolated validation branch with exact transport blobs and six passing focused tests.

- [ ] **Step 1: Create the validation worktree from the pushed pin**

Run locally:

```bash
set -euo pipefail
ORBIT_ROOT=/Users/zqiu/Documents/GitHub/orbit-iclr/orbit
git -C "$ORBIT_ROOT" fetch origin \
  refs/heads/feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
PIN_COMMIT="$(git -C "$ORBIT_ROOT" rev-parse feat/lora-without-regret)"
test "$(git rev-parse feat/lora-without-regret)" = "$PIN_COMMIT"
test "$(git rev-parse origin/feat/lora-without-regret)" = "$PIN_COMMIT"
test -z "$(git status --porcelain)"
git check-ignore -q .worktrees
git worktree add \
  /Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-ipc-validation \
  -b codex/oft-ipc-validation "$PIN_COMMIT"
```

Report the pin base, task branch, and worktree path before mutation.

- [ ] **Step 2: Cherry-pick and prove the branch contains only the approved transport unit**

Run in the validation worktree:

```bash
PIN_COMMIT="$(git rev-parse feat/lora-without-regret)"
git cherry-pick -x ccc351678d7ccfa8a41a48d57fb064dcb3be0e2e
TRANSPORT_SHA="$(git rev-parse HEAD)"
test "$(git rev-parse "$TRANSPORT_SHA^")" = "$PIN_COMMIT"
git diff --check "$PIN_COMMIT".."$TRANSPORT_SHA"
git diff --name-only "$PIN_COMMIT".."$TRANSPORT_SHA"
```

Expected name-only output:

```text
orbit/backends/megatron_utils/peft_transport/backends/ipc.py
orbit/backends/sglang_utils/sglang_engine.py
tests/test_peft_ipc_transport.py
```

For each file, require `git rev-parse "$TRANSPORT_SHA:$file"` to equal `git rev-parse "ccc351678d7ccfa8a41a48d57fb064dcb3be0e2e:$file"`. Require empty status. Any conflict, extra file, or blob mismatch is a hard stop.

- [ ] **Step 3: Push the validation branch and create its remote execution worktree**

Run locally:

```bash
TRANSPORT_SHA="$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-ipc-validation rev-parse HEAD)"
git push -u origin codex/oft-ipc-validation
test "$(git ls-remote origin refs/heads/codex/oft-ipc-validation | awk '{print $1}')" = \
  "$TRANSPORT_SHA"
```

In the same local controller shell, retain the validated 40-hex `TRANSPORT_SHA` and inject that exact value into one `mpi1` login-shell operation. From the verified remote control clone, fetch the just-pushed ref into its explicit remote-tracking ref, assert equality, and only then create the execution worktree on the dedicated branch:

```bash
set -euo pipefail
CONTROL=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-runtime-integration-control
DESTINATION=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation
test "$TRANSPORT_SHA" = "$(printf '%s' "$TRANSPORT_SHA" | grep -E '^[0-9a-f]{40}$')"
git -C "$CONTROL" fetch origin \
  refs/heads/codex/oft-ipc-validation:refs/remotes/origin/codex/oft-ipc-validation
test "$(git -C "$CONTROL" rev-parse refs/remotes/origin/codex/oft-ipc-validation)" = \
  "$TRANSPORT_SHA"
test ! -e "$DESTINATION"
test -z "$(git -C "$CONTROL" branch --list codex/oft-ipc-validation)"
git -C "$CONTROL" \
  worktree add \
  "$DESTINATION" \
  --track -b codex/oft-ipc-validation origin/codex/oft-ipc-validation
test "$(git -C "$DESTINATION" rev-parse HEAD)" = "$TRANSPORT_SHA"
test "$(git -C "$DESTINATION" symbolic-ref --short HEAD)" = \
  codex/oft-ipc-validation
test -z "$(git -C "$DESTINATION" status --porcelain)"
```

Reject any existing ambiguous path or local task branch. Also verify the execution worktree's origin URL is `https://github.com/Sphere-AI-Lab/orbit.git`. A missing or stale remote-tracking ref is a hard stop, never a reason to create from another ref.

- [ ] **Step 4: Run the six focused tests with durable evidence**

After repeating the allocation gate, create one execution ID and the `transport-tests` label in the local controller shell:

```bash
EXECUTION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
REMOTE_EXEC_ROOT="/lustre/home/zqiu/.local/state/remote-cluster-runs/mpi1/orbit/codex-oft-ipc-validation-6ddedda9/${EXECUTION_ID}"
TRANSPORT_TEST_RUN_DIR="$REMOTE_EXEC_ROOT/transport-tests"
ssh mpi1 mkdir -p "$(dirname "$REMOTE_EXEC_ROOT")"
ssh mpi1 mkdir "$REMOTE_EXEC_ROOT"
ssh mpi1 mkdir "$TRANSPORT_TEST_RUN_DIR"
```

Report the remote path and local mirror with the identical suffix. Record `EXECUTION_ID` and `REMOTE_EXEC_ROOT` in `provenance.json`; Task 6 must re-read these recorded values rather than assume shell state persists. Create `command.txt`, `stdout.log`, `stderr.log`, `provenance.json`, `test-lease-owner`, and the eventual `completion.status` path before launch. Write the authoritative local `TRANSPORT_SHA` plus a newline to `$TRANSPORT_TEST_RUN_DIR/expected-orbit-sha`. Materialize the exact shared-environment test-lease block above byte-for-byte as `$TRANSPORT_TEST_RUN_DIR/test-lease-protocol.sh`, validate it with `bash -n`, and record/verify its SHA-256 before launch. Inside the allocation, keep the full block in one foreground shell and source that admitted helper before activating the environment:

```bash
cd /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation
EXPECTED_TRANSPORT_SHA="$(tr -d '\n' <"$TRANSPORT_TEST_RUN_DIR/expected-orbit-sha")"
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
TEST_EXECUTION_ID="${EXECUTION_ID}-transport-tests"
TEST_WORKTREE=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation
TEST_COMMAND='python -m pytest -q tests/test_peft_ipc_transport.py'
TEST_OWNER_SNAPSHOT="$TRANSPORT_TEST_RUN_DIR/test-lease-owner"
source "$TRANSPORT_TEST_RUN_DIR/test-lease-protocol.sh" || exit $?
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT" UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source "$ENV_ROOT/bin/activate"
source env.sh
test "$ORBIT_VENV" = "$ENV_ROOT"
test_rc=0
PYTHONPATH=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation \
python -m pytest -q --junitxml="$TRANSPORT_TEST_RUN_DIR/pytest.xml" \
  tests/test_peft_ipc_transport.py \
  >"$TRANSPORT_TEST_RUN_DIR/stdout.log" \
  2>"$TRANSPORT_TEST_RUN_DIR/stderr.log" || test_rc=$?
verification_rc=0
test "$test_rc" -eq 0 || verification_rc=1
python - "$TRANSPORT_TEST_RUN_DIR/pytest.xml" <<'PY' || verification_rc=1
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = list(root) if root.tag == "testsuites" else [root]
totals = {field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
          for field in ("tests", "failures", "errors", "skipped")}
assert totals == {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}, totals
PY
grep -Eq '(^|[^0-9])6 passed([^0-9]|$)' "$TRANSPORT_TEST_RUN_DIR/stdout.log" \
  || verification_rc=1
test "$(git rev-parse HEAD)" = "$EXPECTED_TRANSPORT_SHA" || verification_rc=1
test -z "$(git status --porcelain)" || verification_rc=1
lease_release_rc=0
release_test_lease || lease_release_rc=$?
if ((lease_release_rc == 0)); then
  TEST_LEASE_ARMED=0
else
  verification_rc=1
fi
final_rc=$test_rc
((final_rc == 0 && verification_rc != 0)) && final_rc=1
status_tmp="$(mktemp "$TRANSPORT_TEST_RUN_DIR/completion.status.tmp.XXXXXX")"
printf 'test_exit_code=%s\nverification_exit_code=%s\nfinal_exit_code=%s\n' \
  "$test_rc" "$verification_rc" "$final_rc" >"$status_tmp"
mv "$status_tmp" "$TRANSPORT_TEST_RUN_DIR/completion.status"
test "$final_rc" -eq 0
```

Expected: exactly `6 passed`, exit 0, no skip/failure/error, exact `TRANSPORT_SHA`, clean post-test status, and successful owned test-lease release before completion publication. Atomically publish completion and take one bounded local snapshot. A unit-test pass permits Task 6 but does not permit target-branch integration.

The `EXIT` trap remains only as failure/interruption fallback. The green path releases and disarms the lease before computing or publishing `final_exit_code`; require both the exact `test-${EXECUTION_ID}-transport-tests` lease and `gate` absent before accepting the snapshot.

---

### Task 6: Prove the first and subsequent adapter updates in real two-GPU BS8 training

**Files:**
- Modify no repository files.
- Create in the run store: `run_bs8_smoke.sh`, its SHA-256, smoke data, logs, imports, timings, provenance, acceptance, completion, W&B data, Ray data, and temporary runtime files.

**Interfaces:**
- Consumes: clean remote transport worktree at `TRANSPORT_SHA`, shared SGLang runtime at `b52394d22fc4b686016943efc47cce6fb892cef2`, two GPUs from the existing eight-B200 allocation.
- Produces: ordered evidence for adapter versions 1 and 2 with generation and actor training between them.

- [ ] **Step 1: Revalidate source, environment, allocation, and the fresh smoke run directory**

Repeat the bounded controller/job/session gate. In the allocation, require:

- eight assigned B200 GPUs and `CUDA_VISIBLE_DEVICES` preserved by Condor;
- remote worktree branch `codex/oft-ipc-validation` at `TRANSPORT_SHA`, clean;
- `orbit.__file__` beneath the transport worktree under an explicit `PYTHONPATH`;
- SGLang installed direct URL at `b52394d22fc4b686016943efc47cce6fb892cef2`;
- `sgl-kernel` installed direct URL at `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`;
- Megatron-Bridge and Megatron-Core installed direct URLs at `85c84cbc26d4c983a3d6e46c804f02e2a99af5a2` and `00eb75b0c803b0fc8e5413d736529d9d3b82b6bd`, with no inherited `PYTHONPATH`, `MEGATRON_PATH`, or `MEGATRON_BRIDGE_ROOT` override;
- real training and evaluation inputs readable.

Read `EXECUTION_ID` and `REMOTE_EXEC_ROOT` from the accepted transport-test provenance, require the root's branch/SHA to match the current validation branch, and create a collision-failing sibling `oft-bs8-2gpu-smoke` directory. Before launch, report the remote/local directories and these paths:

```text
run_bs8_smoke.sh
run_bs8_smoke.sha256
console.log
orbit.log
imports.json
environment.txt
timings.txt
provenance.json
acceptance.status
completion.status
smoke_math_test.jsonl
wandb/
ray/
tmp/
expected-orbit-sha
```

Write the exact accepted `TRANSPORT_SHA` plus a newline to `expected-orbit-sha`. Copy exactly the first four rows of `/lustre/fast/fast/groups/ei-slm/data/lora_regret/math_test.jsonl` to the run-owned `smoke_math_test.jsonl` and require `wc -l` to report 4.

- [ ] **Step 2: Write and hash the external validation wrapper**

Create `run_bs8_smoke.sh` in the run directory, not in the Git worktree. It must:

1. Use `set -uo pipefail` and resolve its own run directory.
2. Re-exec itself once through `env -i`, preserving only `HOME`, `USER`, `LOGNAME`, a fixed system `PATH`, and Condor's exact `CUDA_VISIBLE_DEVICES`; every recipe/runtime variable below is then set explicitly.
3. Install INT/TERM traps that forward the signal only to the launcher's process group, wait a bounded grace period, escalate that process group only if needed, reap the launcher, and atomically publish interrupted status.
4. Unset inherited Ray addresses and all fixed Ray port variables plus `MASTER_ADDR`.
5. Read Condor's eight-entry `CUDA_VISIBLE_DEVICES`, preserve its entry spelling/order, and narrow only this command to its first two entries.
6. Set `PYTHONPATH` to exactly the remote transport worktree and clear Megatron path overrides.
7. Unset `ORBIT_VENV`, export `UV_PROJECT_ENVIRONMENT`, `UV_LINK_MODE=copy`, and `CUDA_HOME=/is/software/nvidia/cuda-13.2`, activate `/fast/zqiu/orbit-iclr/orbit_env`, then source that worktree's `env.sh` and require `ORBIT_VENV` to equal the canonical shared environment.
8. Invoke the unchanged production launcher under `timeout --signal=TERM --kill-after=120s 90m`.
9. Capture the launcher's merged stdout/stderr as `console.log` while also retaining `RUN_LOG` from Orbit. The production launcher itself merges streams, so this plan does not claim a false separation.
10. Run the acceptance checks only after the launcher and logger are reaped.
11. Publish `acceptance.status` and `completion.status` atomically; preserve the launcher's actual exit code and set `final_exit_code=1` when verification fails.

Set these exact overrides inside the wrapper:

```bash
ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Condor did not assign GPUs}"
IFS=',' read -r -a ASSIGNED_GPUS <<< "$ALLOCATED_CUDA_VISIBLE_DEVICES"
test "${#ASSIGNED_GPUS[@]}" -eq 8
CUDA_VISIBLE_DEVICES="${ASSIGNED_GPUS[0]},${ASSIGNED_GPUS[1]}"
GPUS_PER_NODE=2
RAY_NUM_GPUS=2
ROLLOUT_NUM_GPUS_PER_ENGINE=2
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1
PEFT_METHOD=oft
OFT_BLOCK_SIZE=8
TARGET_MODULES=linear_qkv,linear_proj,linear_fc1,linear_fc2
LR=3e-5
SEED=0
NUM_ROLLOUT=2
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
ROLLOUT_MAX_RESPONSE_LEN=256
EVAL_MAX_RESPONSE_LEN=256
EVAL_INTERVAL=999999
SAVE_INTERVAL=
WANDB_MODE=offline
DISABLE_EVAL=0
ENABLE_WANDB=1
ORBIT_DRY_RUN_ARGV=0
ORBIT_COLOCATE=1
ROLLOUT_NUM_GPUS=0
ADVANTAGE_ESTIMATOR=grpo
ORBIT_RAY_LIFECYCLE=private
ORBIT_LOG_WEIGHT_SYNC=1
TRAIN_JSONL=/lustre/fast/fast/groups/ei-slm/data/lora_regret/math_train.jsonl
MATH_TEST_JSONL="$RUN_DIR/smoke_math_test.jsonl"
EVAL_DATASETS=math
RL_EXTRA_ARGS="--disable-grpo-std-normalization --skip-eval-before-train --wandb-dir $RUN_DIR/wandb"
RUN_LOG="$RUN_DIR/orbit.log"
SAVE_DIR="$RUN_DIR/checkpoints"
WANDB_DIR="$RUN_DIR/wandb"
RAY_TEMP_DIR="$RUN_DIR/ray"
ORBIT_TMPDIR="$RUN_DIR/tmp"
```

The final command is exactly:

```bash
bash examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh
```

Hash the immutable wrapper with `sha256sum` before launch and store the digest in `run_bs8_smoke.sha256` and `provenance.json`.

Use the exact executable body in Appendix A; the requirements and environment block above explain its contract but do not permit a different lifecycle implementation.

- [ ] **Step 3: Launch once in the managed tmux pane and monitor without duplicate commands**

Send the wrapper's absolute path once through the controller. Record start UTC/epoch before sending. For an ordinary status check, perform one scheduler query and one bounded run-store snapshot; never loop tmux captures, SSH, or rsync. Do not start a second wrapper if output is slow or the controller returns before training completes.

- [ ] **Step 4: Enforce the ordered real-training acceptance gate**

Require launcher exit 0, no `pidfd_getfd` anywhere in `orbit.log`, `console.log`, or `ray/**`, and these exact substrings in `orbit.log`:

```text
weight_sync stage=update_weights_complete rank=0 world_size=2 weight_version=1
startup: actor update_weights done elapsed=
rollout 0: generate done elapsed=
rollout 0: actor train done elapsed=
weight_sync stage=update_weights_complete rank=0 world_size=2 weight_version=2
rollout 0: actor update_weights done elapsed=
progress rollout=1/1 completed=2/2 remaining=0
Training driver exited with code 0
```

Extract the first matching line number for each phase and require:

```text
version 1 update < rollout 0 generation < rollout 0 actor training < version 2 update
```

Write the timing lines for startup/rollout update, generation, training, and shutdown to `timings.txt`. `acceptance.status` must record every boolean, line number, and every path searched for `pidfd_getfd`. Only launcher exit 0 plus all markers, ordering, and negative checks produces `final_exit_code=0`.

- [ ] **Step 5: Snapshot and review the complete smoke evidence**

After terminal completion, take one bounded rsync snapshot of the transport execution directory without `--delete`. Independently compare the wrapper hash, provenance SHA/imports, acceptance booleans, line ordering, and completion exit codes. Server initialization, adapter-manager construction, or reaching the first update call is not acceptance.

---

### Task 7: Admit the proven transport commit and prepare the final campaign source

**Files:**
- Add the already tested three-file transport commit to `feat/lora-without-regret` by fast-forward.
- Modify no dependency files; the final shared-environment sync must be a no-op.

**Interfaces:**
- Consumes: smoke `final_exit_code=0` at `TRANSPORT_SHA`.
- Produces: pushed final Orbit target, clean campaign branch/worktrees, and final verified runtime imports.

- [ ] **Step 1: Run the integrated focused Orbit suite at the tested transport tip**

Create a collision-failing sibling `integrated-focused-tests` label beneath the accepted transport execution root, set `INTEGRATED_TEST_RUN_DIR` to its absolute path, and write the authoritative local `TRANSPORT_SHA` to `expected-orbit-sha`. Retain the original transport `EXECUTION_ID`. Materialize, syntax-check, hash, and verify the same exact lease helper as `$INTEGRATED_TEST_RUN_DIR/test-lease-protocol.sh`. Inside the allocation, run from the remote transport worktree in one foreground shell and source the admitted helper before activating the environment:

```bash
cd /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation
EXPECTED_TRANSPORT_SHA="$(tr -d '\n' <"$INTEGRATED_TEST_RUN_DIR/expected-orbit-sha")"
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
TEST_EXECUTION_ID="${EXECUTION_ID}-integrated-focused-tests"
TEST_WORKTREE=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation
TEST_COMMAND='python -m pytest -q integrated-focused-tests'
TEST_OWNER_SNAPSHOT="$INTEGRATED_TEST_RUN_DIR/test-lease-owner"
source "$INTEGRATED_TEST_RUN_DIR/test-lease-protocol.sh" || exit $?
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT" UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source "$ENV_ROOT/bin/activate"
source env.sh
test "$ORBIT_VENV" = "$ENV_ROOT"
test_rc=0
PYTHONPATH=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation \
python -m pytest -q \
  --junitxml="$INTEGRATED_TEST_RUN_DIR/pytest.xml" \
  tests/test_peft_ipc_transport.py \
  tests/fast/utils/test_lora_regret_arms_coverage.py \
  tests/fast/utils/test_lora_regret_lr_columns.py \
  tests/fast/utils/test_lora_regret_preflight.py \
  tests/fast/utils/test_peft_param_match.py \
  tests/fast/utils/test_lora_arguments.py \
  >"$INTEGRATED_TEST_RUN_DIR/stdout.log" \
  2>"$INTEGRATED_TEST_RUN_DIR/stderr.log" || test_rc=$?
verification_rc=0
test "$test_rc" -eq 0 || verification_rc=1
python - "$INTEGRATED_TEST_RUN_DIR/pytest.xml" <<'PY' || verification_rc=1
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = list(root) if root.tag == "testsuites" else [root]
for field in ("failures", "errors", "skipped"):
    assert sum(int(suite.attrib.get(field, 0)) for suite in suites) == 0, field
PY
test "$(git rev-parse HEAD)" = "$EXPECTED_TRANSPORT_SHA" || verification_rc=1
test -z "$(git status --porcelain)" || verification_rc=1
lease_release_rc=0
release_test_lease || lease_release_rc=$?
if ((lease_release_rc == 0)); then
  TEST_LEASE_ARMED=0
else
  verification_rc=1
fi
final_rc=$test_rc
((final_rc == 0 && verification_rc != 0)) && final_rc=1
status_tmp="$(mktemp "$INTEGRATED_TEST_RUN_DIR/completion.status.tmp.XXXXXX")"
printf 'test_exit_code=%s\nverification_exit_code=%s\nfinal_exit_code=%s\n' \
  "$test_rc" "$verification_rc" "$final_rc" >"$status_tmp"
mv "$status_tmp" "$INTEGRATED_TEST_RUN_DIR/completion.status"
test "$final_rc" -eq 0
```

Expected: exit 0, no failures/errors/skips, clean `TRANSPORT_SHA`, and successful owned lease release before completion publication. The ownership-verifying `EXIT` trap is fallback only; require `test-${EXECUTION_ID}-integrated-focused-tests` and `gate` absent, then snapshot and verify terminal evidence before changing the target ref.

- [ ] **Step 2: Recheck upstream and fast-forward the Orbit target**

Run locally:

```bash
set -euo pipefail
ORBIT_ROOT=/Users/zqiu/Documents/GitHub/orbit-iclr/orbit
TRANSPORT_WORKTREE="$ORBIT_ROOT/.worktrees/oft-ipc-validation"
TRANSPORT_SHA="$(git -C "$TRANSPORT_WORKTREE" rev-parse HEAD)"
PIN_COMMIT="$(git -C "$TRANSPORT_WORKTREE" rev-parse "$TRANSPORT_SHA^")"
git -C "$ORBIT_ROOT" fetch origin \
  refs/heads/feat/lora-without-regret:refs/remotes/origin/feat/lora-without-regret
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse origin/feat/lora-without-regret)" = "$PIN_COMMIT"
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse feat/lora-without-regret)" = "$PIN_COMMIT"
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit symbolic-ref --short HEAD)" = \
  feat/lora-without-regret
test -z "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit status --porcelain)"
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit merge --ff-only codex/oft-ipc-validation
FINAL_SHA="$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit rev-parse HEAD)"
test "$FINAL_SHA" = "$TRANSPORT_SHA"
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit symbolic-ref --short HEAD)" = \
  feat/lora-without-regret
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit push origin \
  refs/heads/feat/lora-without-regret:refs/heads/feat/lora-without-regret
test "$(git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit ls-remote origin refs/heads/feat/lora-without-regret | awk '{print $1}')" = \
  "$FINAL_SHA"
```

Expected: pure fast-forward to the exact smoke-tested commit and clean status. If remote moved, stop; do not merge unrelated history or force-push.

- [ ] **Step 3: Create and push the source-clean campaign branch/worktree**

Run:

```bash
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit worktree add \
  /Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/math-oft-lr3 \
  -b codex/math-oft-lr3 "$FINAL_SHA"
git -C /Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/math-oft-lr3 push \
  -u origin codex/math-oft-lr3
```

Verify the branch points exactly to `FINAL_SHA`, has no additional commit, is clean, and resolves server-side to `FINAL_SHA`. In the same local controller shell, retain and inject that validated 40-hex value into one `mpi1` login-shell operation. From the remote control clone, fetch the just-pushed campaign ref into its explicit remote-tracking ref, assert equality, and then create the campaign execution worktree:

```bash
set -euo pipefail
CONTROL=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-runtime-integration-control
DESTINATION=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3
test "$FINAL_SHA" = "$(printf '%s' "$FINAL_SHA" | grep -E '^[0-9a-f]{40}$')"
git -C "$CONTROL" fetch origin \
  refs/heads/codex/math-oft-lr3:refs/remotes/origin/codex/math-oft-lr3
test "$(git -C "$CONTROL" rev-parse refs/remotes/origin/codex/math-oft-lr3)" = \
  "$FINAL_SHA"
test ! -e "$DESTINATION"
test -z "$(git -C "$CONTROL" branch --list codex/math-oft-lr3)"
git -C "$CONTROL" \
  worktree add \
  "$DESTINATION" \
  --track -b codex/math-oft-lr3 origin/codex/math-oft-lr3
test "$(git -C "$DESTINATION" rev-parse HEAD)" = "$FINAL_SHA"
test "$(git -C "$DESTINATION" symbolic-ref --short HEAD)" = codex/math-oft-lr3
test -z "$(git -C "$DESTINATION" status --porcelain)"
```

Reject a pre-existing ambiguous path or local task branch. Also verify the execution worktree's origin URL is `https://github.com/Sphere-AI-Lab/orbit.git`. A missing or stale remote-tracking ref is a hard stop, never a reason to create from another ref.

- [ ] **Step 4: Reconcile the shared environment from the final lock and require a no-op**

Repeat Task 4's cluster-wide active-job/session audit. Then resolve a fresh `shared-env-final-sync` label in the local controller shell and create it collision-failingly:

```bash
FINAL_ENV_EXECUTION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
REMOTE_FINAL_ENV_ROOT="/lustre/home/zqiu/.local/state/remote-cluster-runs/mpi1/orbit/codex-math-oft-lr3-140979bf/${FINAL_ENV_EXECUTION_ID}"
FINAL_ENV_RUN_DIR="$REMOTE_FINAL_ENV_ROOT/shared-env-final-sync"
LOCAL_FINAL_ENV_RUN_DIR="/Users/zqiu/.local/state/remote-cluster-runs/mpi1/orbit/codex-math-oft-lr3-140979bf/${FINAL_ENV_EXECUTION_ID}/shared-env-final-sync"
ssh mpi1 mkdir -p "$(dirname "$REMOTE_FINAL_ENV_ROOT")"
ssh mpi1 mkdir "$REMOTE_FINAL_ENV_ROOT"
ssh mpi1 mkdir "$FINAL_ENV_RUN_DIR"
printf '%s\n' "$FINAL_ENV_EXECUTION_ID" "$FINAL_ENV_RUN_DIR" \
  "$LOCAL_FINAL_ENV_RUN_DIR"
```

Predeclare `command.txt`, `uv-version.txt`, `dry-run.json`, `dry-run.stderr.log`, `sync.stdout.log`, `sync.stderr.log`, `before.freeze`, `after.freeze`, `imports.json`, `orbit-direct-url.before`, `orbit-direct-url.after`, `mutation-lease-owner`, `provenance.json`, and `completion.status`. Retain the resolved absolute strings when installing the one foreground compute orchestration. At its start, set `EXECUTION_ID=$FINAL_ENV_EXECUTION_ID`, set `ENV_RUN_DIR=$FINAL_ENV_RUN_DIR`, and acquire the exclusive mutation lease with this complete block:

```bash
set -euo pipefail
: "${FINAL_ENV_EXECUTION_ID:?pass the retained execution ID to this orchestration}"
: "${FINAL_ENV_RUN_DIR:?pass the retained absolute run directory to this orchestration}"
EXECUTION_ID="$FINAL_ENV_EXECUTION_ID"
ENV_RUN_DIR="$FINAL_ENV_RUN_DIR"
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
RESERVATIONS="${ENV_ROOT}.remote-dev-reservations-v1"
GATE="$RESERVATIONS/gate"
LEASE="$RESERVATIONS/mutation-${EXECUTION_ID}"
OWNER_SNAPSHOT="$ENV_RUN_DIR/mutation-lease-owner"
HOST="$(hostname -f)"
PID="$$"
START_ID="$(awk '{print $22}' "/proc/$PID/stat")"
GATE_OWNED=0
MUTATION_LEASE_ARMED=0
LEASE_DIR_CREATED=0
OWNER_TEMP=
release_owned_gate() {
  local cleanup_rc=0
  ((GATE_OWNED == 1)) || return 0
  if [[ -e "$GATE/owner" || -L "$GATE/owner" ]]; then
    if [[ -f "$OWNER_SNAPSHOT" ]] && cmp -s "$OWNER_SNAPSHOT" "$GATE/owner"; then
      rm "$GATE/owner" || cleanup_rc=1
    else
      cleanup_rc=1
    fi
  fi
  if ((cleanup_rc == 0)); then
    rmdir "$GATE" || cleanup_rc=1
    ((cleanup_rc != 0)) || GATE_OWNED=0
  fi
  return "$cleanup_rc"
}
release_owned_mutation_lease() {
  local cleanup_rc=0 live_start=
  if ((GATE_OWNED == 0)); then
    mkdir "$GATE" || return 1
    GATE_OWNED=1
  fi
  if [[ ! -e "$GATE/owner" && ! -L "$GATE/owner" ]]; then
    cp "$OWNER_SNAPSHOT" "$GATE/owner" || cleanup_rc=1
  elif ! cmp -s "$OWNER_SNAPSHOT" "$GATE/owner"; then
    cleanup_rc=1
  fi
  if [[ ! -f "$LEASE/owner" ]] || ! cmp -s "$OWNER_SNAPSHOT" "$LEASE/owner"; then
    cleanup_rc=1
  elif [[ "$(hostname -f)" != "$HOST" ]] || [[ "$$" != "$PID" ]] ||
       [[ ! -r "/proc/$PID/stat" ]]; then
    cleanup_rc=1
  else
    live_start="$(awk '{print $22}' "/proc/$PID/stat")"
    if [[ "$live_start" != "$START_ID" ]] ||
       [[ "$(find "$LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" != owner ]]; then
      cleanup_rc=1
    else
      rm "$LEASE/owner" || cleanup_rc=1
      if ((cleanup_rc == 0)); then
        rmdir "$LEASE" || cleanup_rc=1
        if ((cleanup_rc == 0)); then
          MUTATION_LEASE_ARMED=0
          LEASE_DIR_CREATED=0
        fi
      fi
    fi
  fi
  release_owned_gate || cleanup_rc=1
  return "$cleanup_rc"
}
release_owned_unarmed_lease_dir() {
  ((LEASE_DIR_CREATED == 1 && GATE_OWNED == 1)) || return 1
  if [[ -f "$LEASE/owner" ]] && cmp -s "$OWNER_SNAPSHOT" "$LEASE/owner" &&
     [[ "$(find "$LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" == owner ]]; then
    rm "$LEASE/owner" || return 1
  elif [[ -n "$(find "$LEASE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    return 1
  fi
  rmdir "$LEASE" || return 1
  LEASE_DIR_CREATED=0
}
mutation_exit_cleanup() {
  local original_rc=$? cleanup_rc=0
  trap - EXIT
  set +e
  if ((MUTATION_LEASE_ARMED == 1)); then
    release_owned_mutation_lease
    cleanup_rc=$?
  elif ((LEASE_DIR_CREATED == 1)); then
    release_owned_unarmed_lease_dir || cleanup_rc=$?
    release_owned_gate || cleanup_rc=1
  elif ((GATE_OWNED == 1)); then
    release_owned_gate
    cleanup_rc=$?
  fi
  [[ -z "$OWNER_TEMP" ]] || rm -f -- "$OWNER_TEMP"
  if ((original_rc == 0 && cleanup_rc != 0)); then
    original_rc=74
  fi
  exit "$original_rc"
}
trap mutation_exit_cleanup EXIT
mkdir -p "$RESERVATIONS"
mkdir "$GATE"
GATE_OWNED=1
OWNER_TEMP="$(mktemp "$OWNER_SNAPSHOT.tmp.XXXXXX")"
printf 'host=%s\npid=%s\nstart_id=%s\nagent=%s\nworktree=%s\ncommand=%s\nreserved_cpus=%s\nexecution_id=%s\n' \
  "$HOST" "$PID" "$START_ID" root \
  /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3 \
  'uv sync --locked --inexact --extra allinone --no-install-project' \
  1 "$EXECUTION_ID" >"$OWNER_TEMP"
mv "$OWNER_TEMP" "$OWNER_SNAPSHOT"
OWNER_TEMP=
cp "$OWNER_SNAPSHOT" "$GATE/owner"
if [[ -n "$(find "$RESERVATIONS" -mindepth 1 -maxdepth 1 -type d \
     \( -name 'test-*' -o -name 'mutation-*' \) -print)" ]] ||
   [[ -n "$(pgrep -af '/fast/zqiu/orbit-iclr/orbit_env/bin/(python|ray|torchrun)' || true)" ]]; then
  exit 75
fi
mkdir "$LEASE"
LEASE_DIR_CREATED=1
cp "$OWNER_SNAPSHOT" "$LEASE/owner"
MUTATION_LEASE_ARMED=1
rm "$GATE/owner"
rmdir "$GATE"
GATE_OWNED=0
```

If any mutation/test lease or matching environment process exists, release `gate` without touching that record and stop. Otherwise, write initial provenance atomically and keep this mutation lease through the entire no-op sync and verification. From the remote campaign worktree, initialize the environment in the same order and run:

```bash
cd /lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3
FINAL_SHA="$(git rev-parse HEAD)"
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT" UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source "$ENV_ROOT/bin/activate"
source env.sh
test "$ORBIT_VENV" = "$ENV_ROOT"
UV_BIN=/home/zqiu/.local/bin/uv
"$UV_BIN" --version >"$FINAL_ENV_RUN_DIR/uv-version.txt"
grep -Eq '^uv 0\.10\.11([[:space:]]|$)' "$FINAL_ENV_RUN_DIR/uv-version.txt"
"$UV_BIN" pip freeze --strict | LC_ALL=C sort >"$FINAL_ENV_RUN_DIR/before.freeze"
python -c 'import importlib.metadata as m; print(m.distribution("orbit").read_text("direct_url.json") or "")' \
  >"$FINAL_ENV_RUN_DIR/orbit-direct-url.before"
"$UV_BIN" --preview-features json-output sync \
  --locked --inexact --extra allinone --no-install-project --dry-run \
  --output-format json \
  >"$FINAL_ENV_RUN_DIR/dry-run.json" 2>"$FINAL_ENV_RUN_DIR/dry-run.stderr.log"
python - "$FINAL_ENV_RUN_DIR/dry-run.json" "$PWD" "$ENV_ROOT" <<'PY'
import json
import os
import sys

path, project_root, env_root = sys.argv[1:]
document = json.load(open(path))
assert document["schema"] == {"version": "preview"}, document["schema"]
assert document["target"] == "project", document["target"]
assert document["dry_run"] is True
assert os.path.realpath(document["project"]["path"]) == os.path.realpath(project_root)
assert os.path.realpath(document["project"]["workspace"]["path"]) == os.path.realpath(project_root)
assert os.path.realpath(document["sync"]["environment"]["path"]) == os.path.realpath(env_root)
assert document["sync"]["action"] == "check", document["sync"]["action"]
assert document["lock"]["action"] == "check", document["lock"]
assert os.path.realpath(document["lock"]["path"]) == os.path.realpath(
    os.path.join(project_root, "uv.lock")
)
assert document["sync"]["changes"] == [], document["sync"]["changes"]
PY
```

Expected: no package changes because the transport commit does not alter the lock. Then run the same command without `--dry-run`, capture after-freeze and Orbit direct URL, and require both pairs to be byte-identical:

```bash
"$UV_BIN" sync --locked --inexact --extra allinone --no-install-project \
  >"$FINAL_ENV_RUN_DIR/sync.stdout.log" 2>"$FINAL_ENV_RUN_DIR/sync.stderr.log"
"$UV_BIN" pip freeze --strict | LC_ALL=C sort >"$FINAL_ENV_RUN_DIR/after.freeze"
python -c 'import importlib.metadata as m; print(m.distribution("orbit").read_text("direct_url.json") or "")' \
  >"$FINAL_ENV_RUN_DIR/orbit-direct-url.after"
cmp "$FINAL_ENV_RUN_DIR/before.freeze" "$FINAL_ENV_RUN_DIR/after.freeze"
cmp "$FINAL_ENV_RUN_DIR/orbit-direct-url.before" \
  "$FINAL_ENV_RUN_DIR/orbit-direct-url.after"
PYTHONPATH=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3 \
python - "$FINAL_ENV_RUN_DIR/imports.json" "$FINAL_SHA" <<'PY'
import importlib.metadata as md
import json
import os
import sys

import orbit
import megatron.bridge
import megatron.core
import sglang

destination, orbit_sha = sys.argv[1:]
env_root = os.path.realpath("/fast/zqiu/orbit-iclr/orbit_env")
orbit_root = os.path.realpath(
    "/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3"
)
assert os.path.commonpath([os.path.realpath(orbit.__file__), orbit_root]) == orbit_root
assert os.path.commonpath([os.path.realpath(sglang.__file__), env_root]) == env_root
assert os.path.realpath(sys.prefix) == env_root
records = {"orbit_file": orbit.__file__, "orbit_sha": orbit_sha,
           "sglang_file": sglang.__file__, "sys_prefix": sys.prefix,
           "megatron_bridge_file": megatron.bridge.__file__,
           "megatron_core_file": megatron.core.__file__}
for name, sha, subdirectory in (
    ("sglang", "b52394d22fc4b686016943efc47cce6fb892cef2", "python"),
    ("sgl-kernel", "9c83ae8be07cbb1eb6898ce608ae244e3be375b4", "sgl-kernel"),
    ("megatron-bridge", "85c84cbc26d4c983a3d6e46c804f02e2a99af5a2", None),
    ("megatron-core", "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd", None),
):
    direct = json.loads(md.distribution(name).read_text("direct_url.json"))
    if subdirectory is not None:
        assert direct["subdirectory"] == subdirectory
    assert direct["vcs_info"]["commit_id"] == sha
    assert direct["vcs_info"]["requested_revision"] == sha
    records[name] = direct
temporary = destination + ".tmp"
with open(temporary, "w") as stream:
    json.dump(records, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, destination)
PY
test "$(git symbolic-ref --short HEAD)" = codex/math-oft-lr3
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git status --porcelain)"
```

- Orbit import beneath `/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3` under explicit `PYTHONPATH`;
- SGLang direct URL at `b52394d22fc4b686016943efc47cce6fb892cef2`;
- `sgl-kernel` direct URL at `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`;
- Git branch/SHA/status exactly `codex/math-oft-lr3`, `FINAL_SHA`, clean.

Atomically publish provenance and completion only after every assertion succeeds, then end the one foreground orchestration process. Its installed `EXIT` trap must release only the byte-identical owned mutation lease; the controller then requires `LEASE` and `GATE` absent before snapshotting the complete label. Any mismatch or cleanup failure blocks Task 8.

---

### Task 8: Launch and verify the eight-GPU Math OFT LR3 campaign

**Files:**
- Modify no tracked repository files.
- Create in the campaign run store: external lifecycle wrapper, run-owned symlinks/exclude file, ledger, arm logs, offline W&B data, provenance, events, acceptance, and completion.

**Interfaces:**
- Consumes: final pushed Orbit `FINAL_SHA`, verified shared environment, and all eight GPUs of the revalidated B200 allocation.
- Produces: terminal rows for BS8, BS128, and BS1024 at LR `3e-5`, seed 0, with durable source/runtime evidence.

- [ ] **Step 1: Revalidate the whole-node allocation and create the fresh campaign run root**

Repeat the bounded inventory, `mpi1`, job, and tmux checks. Require job `17451507` still running and all eight B200 GPUs idle. If the allocation is gone, stop and report capacity; obtain explicit approval for a new numeric eight-GPU bid before any submission. Bid 60 from the old allocation is not reusable authority.

Resolve:

```bash
EXECUTION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 4)"
REMOTE_CAMPAIGN_ROOT="/lustre/home/zqiu/.local/state/remote-cluster-runs/mpi1/orbit/codex-math-oft-lr3-140979bf/${EXECUTION_ID}/math-oft-lr3-8gpu"
LOCAL_CAMPAIGN_ROOT="/Users/zqiu/.local/state/remote-cluster-runs/mpi1/orbit/codex-math-oft-lr3-140979bf/${EXECUTION_ID}/math-oft-lr3-8gpu"
ssh mpi1 mkdir -p "$(dirname "$(dirname "$REMOTE_CAMPAIGN_ROOT")")"
ssh mpi1 mkdir "$(dirname "$REMOTE_CAMPAIGN_ROOT")"
ssh mpi1 mkdir "$REMOTE_CAMPAIGN_ROOT"
ssh mpi1 mkdir "$REMOTE_CAMPAIGN_ROOT/results"
ssh mpi1 mkdir "$REMOTE_CAMPAIGN_ROOT/arm-logs"
ssh mpi1 mkdir "$REMOTE_CAMPAIGN_ROOT/wandb"
ssh mpi1 mkdir "$REMOTE_CAMPAIGN_ROOT/ray"
```

Retain both absolute variables in the local controller shell. Before launch report combined `console.log`, `provenance.json`, `recipe.env`, `campaign.events.jsonl`, `acceptance.status`, `completion.status`, ledger, arm-log, and W&B paths. The production launcher merges stdout/stderr; `console.log` is therefore the authoritative combined stream.

- [ ] **Step 2: Bind the production launcher's relative outputs into the run store without dirtying tracked source**

In the fresh remote campaign worktree, first record clean tracked status. Then configure only this worktree:

```bash
git config extensions.worktreeConfig true
git config --worktree core.excludesFile "$REMOTE_CAMPAIGN_ROOT/git-excludes"
```

The run-owned `git-excludes` file contains exactly:

```text
/results/e4_math_oft_lr3.jsonl
/logs/lora_regret
/wandb
```

Require the three destinations to be absent, create parent directories as needed, and link them exactly:

```bash
set -euo pipefail
CAMPAIGN_WT=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3
cd "$CAMPAIGN_WT"
test ! -e results/e4_math_oft_lr3.jsonl
test ! -L results/e4_math_oft_lr3.jsonl
test ! -e logs/lora_regret
test ! -L logs/lora_regret
test ! -e wandb
test ! -L wandb
mkdir -p results logs
printf '%s\n' \
  /results/e4_math_oft_lr3.jsonl \
  /logs/lora_regret \
  /wandb >"$REMOTE_CAMPAIGN_ROOT/git-excludes"
ln -s "$REMOTE_CAMPAIGN_ROOT/results/e4_math_oft_lr3.jsonl" \
  results/e4_math_oft_lr3.jsonl
ln -s "$REMOTE_CAMPAIGN_ROOT/arm-logs" logs/lora_regret
ln -s "$REMOTE_CAMPAIGN_ROOT/wandb" wandb
test "$(readlink results/e4_math_oft_lr3.jsonl)" = \
  "$REMOTE_CAMPAIGN_ROOT/results/e4_math_oft_lr3.jsonl"
test "$(readlink logs/lora_regret)" = "$REMOTE_CAMPAIGN_ROOT/arm-logs"
test "$(readlink wandb)" = "$REMOTE_CAMPAIGN_ROOT/wandb"
test -z "$(git status --porcelain)"
```

Record each link target and the worktree-specific exclude configuration in provenance. These links are run-owned execution plumbing, not source changes. Retain them while the campaign or any evidence audit is active. At final worktree retirement, remove only these three verified links and unset this worktree's `core.excludesFile`; never remove their run-store targets.

- [ ] **Step 3: Write, hash, and preflight the external campaign wrapper**

Create a run-store wrapper with the same signal forwarding, process-group isolation, launcher reaping, atomic status publication, and re-entry protection required in Task 6. It must:

- re-exec once through `env -i`, preserving only `HOME`, `USER`, `LOGNAME`, a fixed system `PATH`, and Condor's exact GPU mask;
- unset inherited Ray addresses/ports and `MASTER_ADDR`;
- preserve Condor's `CUDA_VISIBLE_DEVICES` byte-for-byte and require it to contain exactly eight assigned entries;
- set `GPUS_PER_NODE=8`;
- set `PYTHONPATH` to exactly the remote campaign worktree and clear `MEGATRON_PATH`/`MEGATRON_BRIDGE_ROOT`;
- unset `ORBIT_VENV`, export the shared environment/CUDA variables, activate `/fast/zqiu/orbit-iclr/orbit_env`, then source the campaign worktree's `env.sh` and verify `ORBIT_VENV`;
- set `WANDB_MODE=offline`;
- set `RAY_TEMP_DIR=$REMOTE_CAMPAIGN_ROOT/ray`;
- set `RL_EXTRA_ARGS="--disable-grpo-std-normalization --wandb-dir $REMOTE_CAMPAIGN_ROOT/wandb"` so the production launcher actually forwards the W&B directory;
- capture merged launcher stdout/stderr to the declared run-store `console.log`;
- append start/arm/terminal events to `campaign.events.jsonl`;
- invoke exactly `bash scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh` from the remote campaign worktree;
- preserve the launcher exit code and atomically publish terminal status after ledger verification.

Export this complete protocol block so stale tmux values cannot turn the campaign into the earlier smoke or enable global Ray cleanup:

```bash
GPUS_PER_NODE=8
RAY_NUM_GPUS=8
ORBIT_RAY_LIFECYCLE=private
ORBIT_LOG_WEIGHT_SYNC=1
MODEL=llama3.1-8b
SEED=0
ROLLOUT_SEED=0
NUM_ROLLOUT=150
GLOBAL_BATCH_SIZE=1024
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=32
EVAL_INTERVAL=25
SAVE_INTERVAL=
WANDB_MODE=offline
WANDB_AUTOSYNC=0
EPS_CLIP=1e9
EPS_CLIP_HIGH=1e9
ROLLOUT_MAX_RESPONSE_LEN=2048
EVAL_MAX_RESPONSE_LEN=2048
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
ROLLOUT_NUM_GPUS_PER_ENGINE=2
SKIP_PREFLIGHT=0
DRY_RUN=0
DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
RM_TYPE=math
ROLLOUT_TEMPERATURE=1.0
ROLLOUT_STOP=$'\n\nProblem:'
LR_DECAY_STYLE=constant
WEIGHT_DECAY=0.0
ADAM_BETA1=0.9
ADAM_BETA2=0.999
KL_LOSS_COEF=0.0
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEF=0.0
N_SAMPLES_PER_EVAL_PROMPT=1
CONTEXT_PARALLEL_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=1
EXPERT_TENSOR_PARALLEL_SIZE=1
MAX_TOKENS_PER_GPU=16384
RECOMPUTE_NUM_LAYERS=1
SGLANG_MEM_FRACTION_STATIC=0.75
SGLANG_MAX_RUNNING_REQUESTS=128
SGLANG_MAX_TOTAL_TOKENS=262144
RAY_NUM_CPUS=16
OFT_EPS=6e-5
DISABLE_EVAL=0
ENABLE_WANDB=1
ORBIT_DRY_RUN_ARGV=0
ORBIT_COLOCATE=1
ROLLOUT_NUM_GPUS=0
ADVANTAGE_ESTIMATOR=grpo
HF_CKPT=/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B
MEGATRON_LOAD=/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints/Llama-3.1-8B_torch_dist
MODEL_ARGS_FILE=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3/orbit_plugins/model_args/llama3.1-8B-Instruct.sh
ORBIT_ENTRYPOINT=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3/train.py
RL_EXTRA_ARGS="--disable-grpo-std-normalization --wandb-dir $REMOTE_CAMPAIGN_ROOT/wandb"
RAY_TEMP_DIR="$REMOTE_CAMPAIGN_ROOT/ray"
```

Before invoking the campaign wrapper, unset `TRAIN_JSONL`, `TEST_JSONL`, `MATH_TEST_JSONL`, `GSM8K_TEST_JSONL`, `EVAL_DATASETS`, `WANDB_PROJECT`, `WANDB_GROUP`, `WANDB_RUN_NAME`, `LAUNCHER_NAME`, `PEFT_METHOD`, `OFT_BLOCK_SIZE`, `TARGET_MODULES`, and `LR`; each arm must derive its canonical dataset, method, rank, LR, W&B grouping, and unique run name from the selected arm definition. Record the complete allowlist above plus the cleared-variable list in provenance and assert their effective values in each arm log or ledger configuration.

Hash the wrapper before launch. Preflight must record and verify:

- remote URL, `codex/math-oft-lr3`, exact `FINAL_SHA`, clean status;
- Orbit import beneath the campaign worktree;
- SGLang and `sgl-kernel` direct URLs at the approved SHAs;
- Megatron-Bridge and Megatron-Core direct URLs at the approved SHAs, with their runtime import paths recorded after clearing all external path overrides;
- eight B200 GPUs visible;
- model/checkpoint/data paths readable;
- ledger absent or empty and no other writer using its path;
- the production wrapper selects exactly three arms in a dry run, with the exact three arm names from Step 5.

Use the exact executable body in Appendix B; the bullets and protocol block above are its acceptance rationale, not an invitation to reimplement it.

- [ ] **Step 4: Launch the production campaign once**

Send the run-store wrapper's absolute path once to `codex-orbit-oft-training-smoke`. Initial output must include:

```text
3 arms selected, 3 to run -> results/e4_math_oft_lr3.jsonl
running 3 arms sequentially on 8 GPUs
```

For an ordinary status check, perform one scheduler query and one bounded run-store snapshot. Never loop tmux captures, SSH, or rsync. Do not start a second writer, replace the symlink, or interpret the first arm's start as campaign success; tmux output is not authoritative evidence.

- [ ] **Step 5: Verify all three terminal ledger rows**

After the wrapper terminates, parse the JSONL by `arm`, taking the latest row for each arm. The exact accepted set is:

```text
oftscout-b8-all-math-lr3e-05-s0
oftscout-b128-all-math-lr3e-05-s0
oftscout-b1024-all-math-lr3e-05-s0
```

For each latest row require:

- `status == "ok"`;
- `method == "oft"`;
- the arm-to-block-size mapping is exact: `b8 -> 8`, `b128 -> 128`, and `b1024 -> 1024`;
- `target_modules == "linear_qkv,linear_proj,linear_fc1,linear_fc2"` for every arm;
- `lr == 3e-5` and `seed == 0`;
- `metric == "accuracy"`;
- `accuracy` is not null;
- `steps == 149`;
- `len(rollout_seconds) == 150`;
- `gpus == 8`;
- `probe_rollouts` is null.

Also require wrapper exit 0, exact arm-set equality, no unexpected fourth row in the selection, and all three arm logs present/naming their arm. Independently enumerate nonempty `**/offline-run-*/run-*.wandb` files recursively below the run-owned W&B root, require exactly three distinct parent directories with one run file apiece, parse each binary record through the installed W&B datastore/protobuf API, and require the resulting `display_name` set to equal the three-arm set bijectively. Do not derive W&B directories from filtered arm logs. Publish every row/log/W&B bijection check in `acceptance.status`; only all-green checks yield campaign `final_exit_code=0`.

- [ ] **Step 6: Snapshot terminal evidence and hand off the result**

Take one bounded rsync snapshot of the complete campaign execution directory without `--delete`. Verify local ledger hashes, wrapper hash, source/import provenance, acceptance, completion, and arm logs. Report per-arm accuracy and duration, total campaign duration, exact source SHAs, and clickable local evidence paths. Keep the allocation/session unless the user separately asks to release it; successful training does not authorize scheduler cleanup.

---

## Final Acceptance

The plan is complete only when all of these are simultaneously true:

1. SGLang `orbit-sgl-v0.5.9` resolves server-side to freshly tested `b52394d22fc4b686016943efc47cce6fb892cef2`.
2. Orbit `feat/lora-without-regret` contains a fresh three-file pin commit followed by the exact, real-training-proven three-file transport commit.
3. Orbit still pins `sgl-kernel` to `9c83ae8be07cbb1eb6898ce608ae244e3be375b4` and Megatron-Bridge to `85c84cbc26d4c983a3d6e46c804f02e2a99af5a2`.
4. The shared environment imports SGLang from the exact stable SHA and Orbit from the selected clean worktree; no temporary worktree became its editable Orbit installation.
5. The BS8 smoke proves adapter update version 1, generation, actor training, and adapter update version 2 in order, with terminal exit 0 and no `pidfd_getfd` failure.
6. The full campaign's latest ledger rows for BS8, BS128, and BS1024 all satisfy the terminal acceptance contract.
7. Every remote action has a terminal durable run-store record and bounded local snapshot.

If any item fails, preserve the evidence and leave the next source ref or campaign gate untouched.

## Appendix A: Exact BS8 Transport Smoke Wrapper

Install this body unchanged as `run_bs8_smoke.sh`; its location determines `RUN_DIR`.

```bash
#!/usr/bin/env bash
set -uo pipefail

if [[ "${OFT_WRAPPER_SANITIZED:-0}" != 1 ]]; then
    exec /usr/bin/env -i \
        HOME="$HOME" \
        USER="${USER:-zqiu}" \
        LOGNAME="${LOGNAME:-${USER:-zqiu}}" \
        PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Condor did not assign GPUs}" \
        OFT_WRAPPER_SANITIZED=1 \
        /bin/bash "$0" "$@"
fi

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ORBIT_WT=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
LOCK_DIR="$RUN_DIR/.run-lock"
COMPLETION="$RUN_DIR/completion.status"
ACCEPTANCE="$RUN_DIR/acceptance.status"
CONSOLE="$RUN_DIR/console.log"
RUN_LOG="$RUN_DIR/orbit.log"
SIGNAL_GRACE_SECONDS=10
active_pid=
started_epoch="$(date +%s)"
RESERVATIONS="${ENV_ROOT}.remote-dev-reservations-v1"
GATE="$RESERVATIONS/gate"
TEST_EXECUTION_ID="$(basename "$(dirname "$RUN_DIR")")-bs8-smoke"
TEST_LEASE="$RESERVATIONS/test-${TEST_EXECUTION_ID}"
TEST_OWNER_SNAPSHOT="$RUN_DIR/test-lease-owner"
TEST_LEASE_ARMED=0
TEST_HOST=
TEST_PID=
TEST_START_ID=

atomic_lines() {
    local destination=$1
    shift
    local temporary
    temporary="$(mktemp "${destination}.tmp.XXXXXX")" || return 74
    if ! printf '%s\n' "$@" >"$temporary"; then
        rm -f -- "$temporary"
        return 74
    fi
    if ! mv -- "$temporary" "$destination"; then
        rm -f -- "$temporary"
        return 74
    fi
}

publish_completion() {
    local launcher_rc=$1 verification_rc=$2 final_rc=$3 state=$4
    atomic_lines "$COMPLETION" \
        "state=$state" \
        "launcher_exit_code=$launcher_rc" \
        "verification_exit_code=$verification_rc" \
        "final_exit_code=$final_rc" \
        "duration_seconds=$(($(date +%s) - started_epoch))" \
        "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

acquire_test_lease() {
    local created=0 rc=0 temporary
    mkdir -p "$RESERVATIONS" || return 1
    mkdir "$GATE" || return 1
    if [[ -e "$TEST_LEASE" ]] ||
       [[ -n "$(find "$RESERVATIONS" -mindepth 1 -maxdepth 1 \
           -type d \( -name 'test-*' -o -name 'mutation-*' \) -print -quit)" ]]; then
        rmdir "$GATE" 2>/dev/null || :
        return 1
    fi
    TEST_HOST="$(hostname -f)"
    TEST_PID="$$"
    TEST_START_ID="$(awk '{print $22}' "/proc/$TEST_PID/stat")"
    temporary="$(mktemp "$TEST_OWNER_SNAPSHOT.tmp.XXXXXX")" || rc=1
    if ((rc == 0)); then
        printf 'host=%s\npid=%s\nstart_id=%s\nagent=%s\nworktree=%s\ncommand=%s\nreserved_cpus=%s\nexecution_id=%s\n' \
            "$TEST_HOST" "$TEST_PID" "$TEST_START_ID" root "$ORBIT_WT" \
            'bash examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh' \
            1 "$TEST_EXECUTION_ID" >"$temporary" || rc=1
    fi
    if ((rc == 0)); then
        mv "$temporary" "$TEST_OWNER_SNAPSHOT" || rc=1
    else
        rm -f -- "${temporary:-}"
    fi
    if ((rc == 0)); then
        mkdir "$TEST_LEASE" && created=1 || rc=1
    fi
    if ((rc == 0)); then
        cp "$TEST_OWNER_SNAPSHOT" "$TEST_LEASE/owner" || rc=1
    fi
    if ((rc != 0 && created == 1)); then
        rm -f -- "$TEST_LEASE/owner"
        rmdir "$TEST_LEASE" 2>/dev/null || :
    fi
    rmdir "$GATE" || rc=1
    ((rc == 0)) || return 1
    TEST_LEASE_ARMED=1
}

release_test_lease() {
    local cleanup_rc=0 live_start=
    mkdir "$GATE" || return 1
    if [[ ! -f "$TEST_LEASE/owner" ]] ||
       ! cmp -s "$TEST_OWNER_SNAPSHOT" "$TEST_LEASE/owner"; then
        cleanup_rc=1
    elif [[ "$(hostname -f)" != "$TEST_HOST" ]] || [[ "$$" != "$TEST_PID" ]] ||
         [[ ! -r "/proc/$TEST_PID/stat" ]]; then
        cleanup_rc=1
    else
        live_start="$(awk '{print $22}' "/proc/$TEST_PID/stat")"
        if [[ "$live_start" != "$TEST_START_ID" ]] ||
           [[ "$(find "$TEST_LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" != owner ]]; then
            cleanup_rc=1
        else
            rm "$TEST_LEASE/owner" || cleanup_rc=1
            ((cleanup_rc != 0)) || rmdir "$TEST_LEASE" || cleanup_rc=1
        fi
    fi
    rmdir "$GATE" || cleanup_rc=1
    return "$cleanup_rc"
}

test_lease_exit_cleanup() {
    local original_rc=$? cleanup_rc=0
    trap - EXIT
    set +e
    if ((TEST_LEASE_ARMED == 1)); then
        release_test_lease
        cleanup_rc=$?
        ((cleanup_rc == 0)) && TEST_LEASE_ARMED=0
    fi
    if ((original_rc == 0 && cleanup_rc != 0)); then
        original_rc=74
    fi
    exit "$original_rc"
}

terminate_group() {
    local requested_signal=$1
    [[ -n "$active_pid" ]] || return 0
    kill "-$requested_signal" -- "-$active_pid" 2>/dev/null \
        || kill "-$requested_signal" -- "$active_pid" 2>/dev/null \
        || :
    local deadline=$((SECONDS + SIGNAL_GRACE_SECONDS))
    while kill -0 "$active_pid" 2>/dev/null && ((SECONDS < deadline)); do
        sleep 0.1
    done
    if kill -0 "$active_pid" 2>/dev/null; then
        kill -KILL -- "-$active_pid" 2>/dev/null \
            || kill -KILL -- "$active_pid" 2>/dev/null \
            || :
    fi
}

handle_signal() {
    local signal_name=$1 signal_rc=143 launcher_rc=143
    [[ "$signal_name" == INT ]] && signal_rc=130
    trap '' INT TERM
    if [[ -n "$active_pid" ]]; then
        terminate_group "$signal_name"
        wait "$active_pid" 2>/dev/null
        launcher_rc=$?
        [[ "$launcher_rc" -eq 0 ]] && launcher_rc=$signal_rc
    fi
    atomic_lines "$ACCEPTANCE" \
        "state=interrupted" "signal=$signal_name" "accepted=0" || :
    publish_completion "$launcher_rc" 1 "$signal_rc" interrupted || :
    rmdir "$LOCK_DIR" 2>/dev/null || :
    exit "$signal_rc"
}

line_of() {
    local needle=$1
    awk -v needle="$needle" 'index($0, needle) { print NR; exit }' "$RUN_LOG"
}

if [[ -e "$COMPLETION" || -L "$COMPLETION" ]]; then
    printf 'refusing completed smoke directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
if ! mkdir "$LOCK_DIR"; then
    printf 'refusing active or stale smoke lock: %s\n' "$LOCK_DIR" >&2
    exit 2
fi
if ! acquire_test_lease; then
    atomic_lines "$ACCEPTANCE" "state=test_lease_failed" "accepted=0"
    publish_completion 0 1 1 test_lease_failed
    rmdir "$LOCK_DIR"
    exit 1
fi
trap test_lease_exit_cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

verification_rc=0
launcher_rc=0
expected_orbit_sha="$(tr -d '\n' <"$RUN_DIR/expected-orbit-sha")"
if [[ ! "$expected_orbit_sha" =~ ^[0-9a-f]{40}$ ]]; then
    verification_rc=1
fi
if [[ "$(wc -l <"$RUN_DIR/smoke_math_test.jsonl")" -ne 4 ]]; then
    verification_rc=1
fi

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a assigned_gpus <<<"$ALLOCATED_CUDA_VISIBLE_DEVICES"
declare -A seen_gpus=()
if (( ${#assigned_gpus[@]} != 8 )); then
    verification_rc=1
fi
for gpu in "${assigned_gpus[@]}"; do
    if [[ -z "$gpu" ]]; then
        verification_rc=1
        continue
    fi
    if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
        verification_rc=1
    fi
    seen_gpus[$gpu]=1
done
if ((verification_rc != 0)); then
    atomic_lines "$ACCEPTANCE" "state=preflight_failed" "accepted=0"
    publish_completion 0 1 1 preflight_failed
    rmdir "$LOCK_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${assigned_gpus[0]},${assigned_gpus[1]}"
export GPUS_PER_NODE=2 RAY_NUM_GPUS=2 ROLLOUT_NUM_GPUS_PER_ENGINE=2
export TENSOR_MODEL_PARALLEL_SIZE=2 PIPELINE_MODEL_PARALLEL_SIZE=1
export PEFT_METHOD=oft OFT_BLOCK_SIZE=8
export TARGET_MODULES=linear_qkv,linear_proj,linear_fc1,linear_fc2
export LR=3e-5 SEED=0 NUM_ROLLOUT=2
export ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=2
export ROLLOUT_MAX_RESPONSE_LEN=256 EVAL_MAX_RESPONSE_LEN=256
export EVAL_INTERVAL=999999 SAVE_INTERVAL=""
export EPS_CLIP=1e9 EPS_CLIP_HIGH=1e9
export WANDB_MODE=offline WANDB_AUTOSYNC=0
export ORBIT_RAY_LIFECYCLE=private ORBIT_LOG_WEIGHT_SYNC=1
export DISABLE_EVAL=0 ENABLE_WANDB=1 ORBIT_DRY_RUN_ARGV=0
export ORBIT_COLOCATE=1 ROLLOUT_NUM_GPUS=0 ADVANTAGE_ESTIMATOR=grpo
export ORBIT_LOG_FILTER=1 ORBIT_RAY_LOG_TO_DRIVER=1 PARITY_CHECK=0
export TRAIN_JSONL=/lustre/fast/fast/groups/ei-slm/data/lora_regret/math_train.jsonl
export MATH_TEST_JSONL="$RUN_DIR/smoke_math_test.jsonl" EVAL_DATASETS=math
export RL_EXTRA_ARGS="--disable-grpo-std-normalization --skip-eval-before-train --wandb-dir $RUN_DIR/wandb"
export RUN_LOG SAVE_DIR="$RUN_DIR/checkpoints" WANDB_DIR="$RUN_DIR/wandb"
export RAY_TEMP_DIR="$RUN_DIR/ray" ORBIT_TMPDIR="$RUN_DIR/tmp"
export LAUNCHER_NAME=oft_bs8_transport_smoke WANDB_RUN_NAME=oft-bs8-transport-smoke
export PYTHONPATH="$ORBIT_WT"
unset MEGATRON_PATH MEGATRON_BRIDGE_ROOT
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT" UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
unset ORBIT_RAY_ADDRESS RAY_ADDRESS RAY_PORT RAY_HEAD_PORT RAY_CLIENT_SERVER_PORT
unset RAY_DASHBOARD_PORT RAY_DASHBOARD_AGENT_LISTEN_PORT
unset RAY_DASHBOARD_AGENT_GRPC_PORT RAY_GCS_SERVER_PORT RAY_METRICS_EXPORT_PORT
unset RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT RAY_NODE_MANAGER_PORT
unset RAY_OBJECT_MANAGER_PORT RAY_RUNTIME_ENV_AGENT_PORT MASTER_ADDR
unset ORBIT_DEBUG_MODE ORBIT_DRIVER_DEBUG ORBIT_LAUNCHER_XTRACE
unset STAGE_HF_CKPT_TO STAGE_MEGATRON_CKPT_TO
unset CRITIC_NUM_GPUS_PER_NODE CRITIC_NUM_NODES

mkdir -p "$SAVE_DIR" "$WANDB_DIR" "$RAY_TEMP_DIR" "$ORBIT_TMPDIR" \
    || verification_rc=1
source "$ENV_ROOT/bin/activate" || verification_rc=1
cd "$ORBIT_WT" || verification_rc=1
source env.sh || verification_rc=1
test "${ORBIT_VENV:-}" = "$ENV_ROOT" || verification_rc=1
command -v setsid >/dev/null || verification_rc=1
command -v timeout >/dev/null || verification_rc=1
test "$(git symbolic-ref --short HEAD)" = codex/oft-ipc-validation \
    || verification_rc=1
test "$(git rev-parse HEAD)" = "$expected_orbit_sha" || verification_rc=1
test -z "$(git status --porcelain)" || verification_rc=1

python - "$RUN_DIR/imports.json" "$expected_orbit_sha" <<'PY'
import importlib.metadata as md
import json
import os
import sys

import orbit
import megatron.bridge
import megatron.core
import sglang

destination, orbit_sha = sys.argv[1:]
env_root = os.path.realpath("/fast/zqiu/orbit-iclr/orbit_env")
orbit_root = os.path.realpath(
    "/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-oft-ipc-validation"
)
assert os.path.commonpath([os.path.realpath(orbit.__file__), orbit_root]) == orbit_root
assert os.path.commonpath([os.path.realpath(sglang.__file__), env_root]) == env_root
assert os.path.realpath(sys.prefix) == env_root
records = {"orbit_file": orbit.__file__, "orbit_sha": orbit_sha,
           "sglang_file": sglang.__file__, "sys_prefix": sys.prefix,
           "megatron_bridge_file": megatron.bridge.__file__,
           "megatron_core_file": megatron.core.__file__}
for name, sha, subdirectory in (
    ("sglang", "b52394d22fc4b686016943efc47cce6fb892cef2", "python"),
    ("sgl-kernel", "9c83ae8be07cbb1eb6898ce608ae244e3be375b4", "sgl-kernel"),
    ("megatron-bridge", "85c84cbc26d4c983a3d6e46c804f02e2a99af5a2", None),
    ("megatron-core", "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd", None),
):
    direct = json.loads(md.distribution(name).read_text("direct_url.json"))
    if subdirectory is not None:
        assert direct["subdirectory"] == subdirectory
    assert direct["vcs_info"]["commit_id"] == sha
    assert direct["vcs_info"]["requested_revision"] == sha
    records[name] = direct
temporary = destination + ".tmp"
with open(temporary, "w") as stream:
    json.dump(records, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, destination)
PY
[[ "$?" -eq 0 ]] || verification_rc=1
{
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
    printf 'ray_num_gpus=%s\n' "$RAY_NUM_GPUS"
    printf 'rollout_num_gpus_per_engine=%s\n' "$ROLLOUT_NUM_GPUS_PER_ENGINE"
    printf 'tensor_model_parallel_size=%s\n' "$TENSOR_MODEL_PARALLEL_SIZE"
    printf 'pipeline_model_parallel_size=%s\n' "$PIPELINE_MODEL_PARALLEL_SIZE"
    printf 'peft_method=%s\n' "$PEFT_METHOD"
    printf 'oft_block_size=%s\n' "$OFT_BLOCK_SIZE"
    printf 'num_rollout=%s\n' "$NUM_ROLLOUT"
    printf 'rollout_seed=%s\n' "${ROLLOUT_SEED:-$SEED}"
    printf 'orbit_worktree=%s\n' "$ORBIT_WT"
    printf 'orbit_sha=%s\n' "$expected_orbit_sha"
    printf 'environment_root=%s\n' "$ENV_ROOT"
    printf 'python=%s\n' "$(command -v python)"
    python --version
} >"$RUN_DIR/environment.txt" 2>&1 || verification_rc=1
if ((verification_rc != 0)); then
    atomic_lines "$ACCEPTANCE" "state=preflight_failed" "accepted=0"
    publish_completion 0 1 1 preflight_failed
    rmdir "$LOCK_DIR"
    exit 1
fi

setsid timeout --signal=TERM --kill-after=120s 90m \
    bash examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh \
    >"$CONSOLE" 2>&1 &
active_pid=$!
wait "$active_pid"
launcher_rc=$?
active_pid=

version1_line="$(line_of 'weight_sync stage=update_weights_complete rank=0 world_size=2 weight_version=1')"
startup_line="$(line_of 'startup: actor update_weights done elapsed=')"
generate_line="$(line_of 'rollout 0: generate done elapsed=')"
train_line="$(line_of 'rollout 0: actor train done elapsed=')"
version2_line="$(line_of 'weight_sync stage=update_weights_complete rank=0 world_size=2 weight_version=2')"
rollout_update_line="$(line_of 'rollout 0: actor update_weights done elapsed=')"
progress_line="$(line_of 'progress rollout=1/1 completed=2/2 remaining=0')"
driver_line="$(line_of 'Training driver exited with code 0')"
for value in "$version1_line" "$startup_line" "$generate_line" "$train_line" \
    "$version2_line" "$rollout_update_line" "$progress_line" "$driver_line"; do
    [[ -n "$value" ]] || verification_rc=1
done
if [[ -n "$version1_line" && -n "$generate_line" && -n "$train_line" \
    && -n "$version2_line" ]]; then
    ((version1_line < generate_line && generate_line < train_line \
        && train_line < version2_line)) || verification_rc=1
fi
pidfd_absent=1
if grep -Fq -- pidfd_getfd "$RUN_LOG" "$CONSOLE" 2>/dev/null \
    || grep -r -Fq -- pidfd_getfd "$RUN_DIR/ray" 2>/dev/null; then
    pidfd_absent=0
    verification_rc=1
fi
timings_tmp="$(mktemp "$RUN_DIR/timings.txt.tmp.XXXXXX")"
grep -E '(startup|rollout [0-9]+|shutdown): (actor update_weights|generate|actor train|eval).*done elapsed=' \
    "$RUN_LOG" >"$timings_tmp" || :
mv "$timings_tmp" "$RUN_DIR/timings.txt"

lease_release_rc=0
release_test_lease
lease_release_rc=$?
if ((lease_release_rc == 0)); then
    TEST_LEASE_ARMED=0
else
    verification_rc=1
fi
lock_release_rc=0
rmdir "$LOCK_DIR" || lock_release_rc=74
if ((lock_release_rc != 0)); then
    verification_rc=1
fi
accepted=0
if ((launcher_rc == 0 && verification_rc == 0)); then
    accepted=1
fi
final_rc=$launcher_rc
if ((final_rc == 0 && verification_rc != 0)); then
    final_rc=1
fi
if ! atomic_lines "$ACCEPTANCE" \
    "state=terminal" \
    "accepted=$accepted" \
    "version1_line=${version1_line:-missing}" \
    "startup_update_line=${startup_line:-missing}" \
    "generate_line=${generate_line:-missing}" \
    "actor_train_line=${train_line:-missing}" \
    "version2_line=${version2_line:-missing}" \
    "rollout_update_line=${rollout_update_line:-missing}" \
    "progress_line=${progress_line:-missing}" \
    "driver_exit_line=${driver_line:-missing}" \
    "test_lease_released=$((lease_release_rc == 0 ? 1 : 0))" \
    "run_lock_released=$((lock_release_rc == 0 ? 1 : 0))" \
    "pidfd_absent=$pidfd_absent" \
    "pidfd_search_paths=orbit.log,console.log,ray/**"; then
    verification_rc=1
    final_rc=74
fi
if ! publish_completion "$launcher_rc" "$verification_rc" "$final_rc" terminal; then
    exit 74
fi
exit "$final_rc"
```

## Appendix C: Exact SGLang Focused-Gate Wrapper

Install this body unchanged as `run_stage1_gate.sh` in the Stage 1 run directory.

```bash
#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SGLANG_WT=/fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4
PYTHON=/fast/zqiu/software/nk-env-cu13/.venv/bin/python
EXPECTED_SHA=b52394d22fc4b686016943efc47cce6fb892cef2
LOCK_DIR="$RUN_DIR/.run-lock"
COMPLETION="$RUN_DIR/completion.status"
active_pid=
started_epoch="$(date +%s)"

atomic_completion() {
    local test_rc=$1 verify_rc=$2 final_rc=$3 state=$4
    local temporary
    temporary="$(mktemp "$COMPLETION.tmp.XXXXXX")" || return 74
    if ! printf '%s\n' \
        "state=$state" \
        "test_exit_code=$test_rc" \
        "verification_exit_code=$verify_rc" \
        "final_exit_code=$final_rc" \
        "duration_seconds=$(($(date +%s) - started_epoch))" \
        "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$temporary"; then
        rm -f -- "$temporary"
        return 74
    fi
    mv -- "$temporary" "$COMPLETION"
}

handle_signal() {
    local signal_name=$1 signal_rc=143 test_rc=143
    [[ "$signal_name" == INT ]] && signal_rc=130
    trap '' INT TERM
    if [[ -n "$active_pid" ]]; then
        kill "-$signal_name" -- "-$active_pid" 2>/dev/null \
            || kill "-$signal_name" -- "$active_pid" 2>/dev/null \
            || :
        local deadline=$((SECONDS + 10))
        while kill -0 "$active_pid" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 0.1
        done
        if kill -0 "$active_pid" 2>/dev/null; then
            kill -KILL -- "-$active_pid" 2>/dev/null \
                || kill -KILL -- "$active_pid" 2>/dev/null \
                || :
        fi
        wait "$active_pid" 2>/dev/null
        test_rc=$?
        [[ "$test_rc" -eq 0 ]] && test_rc=$signal_rc
    fi
    atomic_completion "$test_rc" 1 "$signal_rc" interrupted || :
    rmdir "$LOCK_DIR" 2>/dev/null || :
    exit "$signal_rc"
}

if [[ -e "$COMPLETION" || -L "$COMPLETION" ]]; then
    printf 'refusing completed test directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
if ! mkdir "$LOCK_DIR"; then
    printf 'refusing active or stale test lock: %s\n' "$LOCK_DIR" >&2
    exit 2
fi
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

verification_rc=0
pytest_rc=0
ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a assigned_gpus <<<"$ALLOCATED_CUDA_VISIBLE_DEVICES"
declare -A seen_gpus=()
if (( ${#assigned_gpus[@]} != 8 )); then
    verification_rc=1
fi
for gpu in "${assigned_gpus[@]}"; do
    if [[ -z "$gpu" ]] || [[ -n "${seen_gpus[$gpu]:-}" ]]; then
        verification_rc=1
    else
        seen_gpus[$gpu]=1
    fi
done

{
    printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname -f)"
    printf 'cuda_visible_devices=%s\n' "$ALLOCATED_CUDA_VISIBLE_DEVICES"
    "$PYTHON" --version
    nvidia-smi -L
    git -C "$SGLANG_WT" rev-parse --show-toplevel
    git -C "$SGLANG_WT" remote get-url origin
    git -C "$SGLANG_WT" symbolic-ref --short HEAD
    git -C "$SGLANG_WT" rev-parse HEAD
    git -C "$SGLANG_WT" status --porcelain
    CUDA_VISIBLE_DEVICES="${assigned_gpus[0]:-}" \
    PYTHONPATH="$SGLANG_WT/python" \
    "$PYTHON" - <<'PY'
import os
import sglang
import torch

root = os.path.realpath(
    "/fast/zqiu/software/proj/spherelab/sglang-spherelab-oft-bs4/python"
)
assert os.path.commonpath([os.path.realpath(sglang.__file__), root]) == root
assert torch.cuda.is_available()
print(f"sglang_file={sglang.__file__}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
} >"$RUN_DIR/preflight.log" 2>&1 || verification_rc=1

test "$(git -C "$SGLANG_WT" remote get-url origin)" = \
    https://github.com/Sphere-AI-Lab/sglang.git || verification_rc=1
test "$(git -C "$SGLANG_WT" symbolic-ref --short HEAD)" = codex/oft-bs4 \
    || verification_rc=1
test "$(git -C "$SGLANG_WT" rev-parse HEAD)" = "$EXPECTED_SHA" \
    || verification_rc=1
test -z "$(git -C "$SGLANG_WT" status --porcelain)" || verification_rc=1
command -v setsid >/dev/null || verification_rc=1
if ((verification_rc != 0)); then
    atomic_completion 0 1 1 preflight_failed
    rmdir "$LOCK_DIR"
    exit 1
fi

cd "$SGLANG_WT"
setsid env CUDA_VISIBLE_DEVICES="${assigned_gpus[0]}" \
    PYTHONPATH="$SGLANG_WT/python" \
    "$PYTHON" -m pytest -q -ra \
    --junitxml="$RUN_DIR/pytest.xml" \
    test/srt/oft/test_split_dense_merged_projection_dispatch.py \
    test/srt/oft/test_tiny_block_validation.py \
    test/srt/oft/test_tiny_block_benchmark_report.py \
    test/srt/oft/test_gemm_oft_r_tiled.py \
    test/srt/oft/test_fused_rotate_project_tiled.py \
    test/srt/oft/test_tiny_block_grouped_moe.py \
    test/srt/oft/test_tiny_block_backward_cayley.py \
    test/srt/oft/test_streamed_chunk_limit.py \
    >"$RUN_DIR/stdout.log" 2>"$RUN_DIR/stderr.log" &
active_pid=$!
wait "$active_pid"
pytest_rc=$?
active_pid=

grep -Eq '(^|[^0-9])189 passed([^0-9]|$)' "$RUN_DIR/stdout.log" \
    || verification_rc=1
python_status=0
"$PYTHON" - "$RUN_DIR/pytest.xml" <<'PY' || python_status=$?
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
if root.tag == "testsuites":
    totals = {key: int(root.attrib.get(key, 0))
              for key in ("tests", "failures", "errors", "skipped")}
else:
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
assert totals == {"tests": 189, "failures": 0, "errors": 0, "skipped": 0}, totals
PY
[[ "$python_status" -eq 0 ]] || verification_rc=1
test "$pytest_rc" -eq 0 || verification_rc=1
test "$(git -C "$SGLANG_WT" rev-parse HEAD)" = "$EXPECTED_SHA" \
    || verification_rc=1
test -z "$(git -C "$SGLANG_WT" status --porcelain)" || verification_rc=1

final_rc=$pytest_rc
if ((final_rc == 0 && verification_rc != 0)); then
    final_rc=1
fi
if ! rmdir "$LOCK_DIR"; then
    verification_rc=1
    final_rc=74
fi
if ! atomic_completion "$pytest_rc" "$verification_rc" "$final_rc" terminal; then
    exit 74
fi
exit "$final_rc"
```

## Appendix B: Exact Math OFT LR3 Campaign Wrapper

Install this body unchanged as `run_math_oft_lr3_campaign.sh` in the campaign run root. Before hashing it, write the exact `FINAL_SHA` plus a newline to `expected-orbit-sha` in the same directory.

```bash
#!/usr/bin/env bash
set -uo pipefail

if [[ "${OFT_WRAPPER_SANITIZED:-0}" != 1 ]]; then
    exec /usr/bin/env -i \
        HOME="$HOME" \
        USER="${USER:-zqiu}" \
        LOGNAME="${LOGNAME:-${USER:-zqiu}}" \
        PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Condor did not assign GPUs}" \
        OFT_WRAPPER_SANITIZED=1 \
        /bin/bash "$0" "$@"
fi

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ORBIT_WT=/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3
ENV_ROOT="$(readlink -f /fast/zqiu/orbit-iclr/orbit_env)"
LOCK_DIR="$RUN_DIR/.campaign-lock"
COMPLETION="$RUN_DIR/completion.status"
ACCEPTANCE="$RUN_DIR/acceptance.status"
CONSOLE="$RUN_DIR/console.log"
EVENTS="$RUN_DIR/campaign.events.jsonl"
LEDGER="$RUN_DIR/results/e4_math_oft_lr3.jsonl"
SIGNAL_GRACE_SECONDS=30
active_pid=
started_epoch="$(date +%s)"
RESERVATIONS="${ENV_ROOT}.remote-dev-reservations-v1"
GATE="$RESERVATIONS/gate"
TEST_EXECUTION_ID="$(basename "$(dirname "$RUN_DIR")")-math-oft-lr3"
TEST_LEASE="$RESERVATIONS/test-${TEST_EXECUTION_ID}"
TEST_OWNER_SNAPSHOT="$RUN_DIR/test-lease-owner"
TEST_LEASE_ARMED=0
TEST_HOST=
TEST_PID=
TEST_START_ID=

atomic_lines() {
    local destination=$1
    shift
    local temporary
    temporary="$(mktemp "${destination}.tmp.XXXXXX")" || return 74
    if ! printf '%s\n' "$@" >"$temporary"; then
        rm -f -- "$temporary"
        return 74
    fi
    if ! mv -- "$temporary" "$destination"; then
        rm -f -- "$temporary"
        return 74
    fi
}

publish_completion() {
    local launcher_rc=$1 verification_rc=$2 final_rc=$3 state=$4
    atomic_lines "$COMPLETION" \
        "state=$state" \
        "launcher_exit_code=$launcher_rc" \
        "verification_exit_code=$verification_rc" \
        "final_exit_code=$final_rc" \
        "duration_seconds=$(($(date +%s) - started_epoch))" \
        "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

acquire_test_lease() {
    local created=0 rc=0 temporary
    mkdir -p "$RESERVATIONS" || return 1
    mkdir "$GATE" || return 1
    if [[ -e "$TEST_LEASE" ]] ||
       [[ -n "$(find "$RESERVATIONS" -mindepth 1 -maxdepth 1 \
           -type d \( -name 'test-*' -o -name 'mutation-*' \) -print -quit)" ]]; then
        rmdir "$GATE" 2>/dev/null || :
        return 1
    fi
    TEST_HOST="$(hostname -f)"
    TEST_PID="$$"
    TEST_START_ID="$(awk '{print $22}' "/proc/$TEST_PID/stat")"
    temporary="$(mktemp "$TEST_OWNER_SNAPSHOT.tmp.XXXXXX")" || rc=1
    if ((rc == 0)); then
        printf 'host=%s\npid=%s\nstart_id=%s\nagent=%s\nworktree=%s\ncommand=%s\nreserved_cpus=%s\nexecution_id=%s\n' \
            "$TEST_HOST" "$TEST_PID" "$TEST_START_ID" root "$ORBIT_WT" \
            'bash scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh' \
            1 "$TEST_EXECUTION_ID" >"$temporary" || rc=1
    fi
    if ((rc == 0)); then
        mv "$temporary" "$TEST_OWNER_SNAPSHOT" || rc=1
    else
        rm -f -- "${temporary:-}"
    fi
    if ((rc == 0)); then
        mkdir "$TEST_LEASE" && created=1 || rc=1
    fi
    if ((rc == 0)); then
        cp "$TEST_OWNER_SNAPSHOT" "$TEST_LEASE/owner" || rc=1
    fi
    if ((rc != 0 && created == 1)); then
        rm -f -- "$TEST_LEASE/owner"
        rmdir "$TEST_LEASE" 2>/dev/null || :
    fi
    rmdir "$GATE" || rc=1
    ((rc == 0)) || return 1
    TEST_LEASE_ARMED=1
}

release_test_lease() {
    local cleanup_rc=0 live_start=
    mkdir "$GATE" || return 1
    if [[ ! -f "$TEST_LEASE/owner" ]] ||
       ! cmp -s "$TEST_OWNER_SNAPSHOT" "$TEST_LEASE/owner"; then
        cleanup_rc=1
    elif [[ "$(hostname -f)" != "$TEST_HOST" ]] || [[ "$$" != "$TEST_PID" ]] ||
         [[ ! -r "/proc/$TEST_PID/stat" ]]; then
        cleanup_rc=1
    else
        live_start="$(awk '{print $22}' "/proc/$TEST_PID/stat")"
        if [[ "$live_start" != "$TEST_START_ID" ]] ||
           [[ "$(find "$TEST_LEASE" -mindepth 1 -maxdepth 1 -printf '%f\n')" != owner ]]; then
            cleanup_rc=1
        else
            rm "$TEST_LEASE/owner" || cleanup_rc=1
            ((cleanup_rc != 0)) || rmdir "$TEST_LEASE" || cleanup_rc=1
        fi
    fi
    rmdir "$GATE" || cleanup_rc=1
    return "$cleanup_rc"
}

test_lease_exit_cleanup() {
    local original_rc=$? cleanup_rc=0
    trap - EXIT
    set +e
    if ((TEST_LEASE_ARMED == 1)); then
        release_test_lease
        cleanup_rc=$?
        ((cleanup_rc == 0)) && TEST_LEASE_ARMED=0
    fi
    if ((original_rc == 0 && cleanup_rc != 0)); then
        original_rc=74
    fi
    exit "$original_rc"
}

terminate_group() {
    local requested_signal=$1
    [[ -n "$active_pid" ]] || return 0
    kill "-$requested_signal" -- "-$active_pid" 2>/dev/null \
        || kill "-$requested_signal" -- "$active_pid" 2>/dev/null \
        || :
    local deadline=$((SECONDS + SIGNAL_GRACE_SECONDS))
    while kill -0 "$active_pid" 2>/dev/null && ((SECONDS < deadline)); do
        sleep 0.1
    done
    if kill -0 "$active_pid" 2>/dev/null; then
        kill -KILL -- "-$active_pid" 2>/dev/null \
            || kill -KILL -- "$active_pid" 2>/dev/null \
            || :
    fi
}

handle_signal() {
    local signal_name=$1 signal_rc=143 launcher_rc=143
    [[ "$signal_name" == INT ]] && signal_rc=130
    trap '' INT TERM
    if [[ -n "$active_pid" ]]; then
        terminate_group "$signal_name"
        wait "$active_pid" 2>/dev/null
        launcher_rc=$?
        [[ "$launcher_rc" -eq 0 ]] && launcher_rc=$signal_rc
    fi
    atomic_lines "$ACCEPTANCE" \
        "state=interrupted" "signal=$signal_name" "accepted=0" || :
    publish_completion "$launcher_rc" 1 "$signal_rc" interrupted || :
    printf '{"event":"interrupted","signal":"%s","at":"%s"}\n' \
        "$signal_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$EVENTS" || :
    rmdir "$LOCK_DIR" 2>/dev/null || :
    exit "$signal_rc"
}

if [[ -e "$COMPLETION" || -L "$COMPLETION" ]]; then
    printf 'refusing completed campaign directory: %s\n' "$RUN_DIR" >&2
    exit 2
fi
if ! mkdir "$LOCK_DIR"; then
    printf 'refusing active or stale campaign lock: %s\n' "$LOCK_DIR" >&2
    exit 2
fi
if ! acquire_test_lease; then
    atomic_lines "$ACCEPTANCE" "state=test_lease_failed" "accepted=0"
    publish_completion 0 1 1 test_lease_failed
    rmdir "$LOCK_DIR"
    exit 1
fi
trap test_lease_exit_cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

verification_rc=0
launcher_rc=0
expected_orbit_sha="$(tr -d '\n' <"$RUN_DIR/expected-orbit-sha")"
[[ "$expected_orbit_sha" =~ ^[0-9a-f]{40}$ ]] || verification_rc=1

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a assigned_gpus <<<"$ALLOCATED_CUDA_VISIBLE_DEVICES"
declare -A seen_gpus=()
if (( ${#assigned_gpus[@]} != 8 )); then
    verification_rc=1
fi
for gpu in "${assigned_gpus[@]}"; do
    if [[ -z "$gpu" ]]; then
        verification_rc=1
        continue
    fi
    if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
        verification_rc=1
    fi
    seen_gpus[$gpu]=1
done
if ((verification_rc != 0)); then
    atomic_lines "$ACCEPTANCE" "state=preflight_failed" "accepted=0"
    publish_completion 0 1 1 preflight_failed
    rmdir "$LOCK_DIR"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$ALLOCATED_CUDA_VISIBLE_DEVICES"
export GPUS_PER_NODE=8 RAY_NUM_GPUS=8
export ORBIT_RAY_LIFECYCLE=private ORBIT_LOG_WEIGHT_SYNC=1
export MODEL=llama3.1-8b SEED=0 ROLLOUT_SEED=0 NUM_ROLLOUT=150
export GLOBAL_BATCH_SIZE=1024 ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=32
export EVAL_INTERVAL=25 SAVE_INTERVAL=""
export WANDB_MODE=offline WANDB_AUTOSYNC=0
export DISABLE_EVAL=0 ENABLE_WANDB=1 ORBIT_DRY_RUN_ARGV=0
export ORBIT_COLOCATE=1 ROLLOUT_NUM_GPUS=0 ADVANTAGE_ESTIMATOR=grpo
export ORBIT_LOG_FILTER=1 ORBIT_RAY_LOG_TO_DRIVER=1 PARITY_CHECK=0
export EPS_CLIP=1e9 EPS_CLIP_HIGH=1e9
export ROLLOUT_MAX_RESPONSE_LEN=2048 EVAL_MAX_RESPONSE_LEN=2048
export TENSOR_MODEL_PARALLEL_SIZE=1 PIPELINE_MODEL_PARALLEL_SIZE=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2 SKIP_PREFLIGHT=0 DRY_RUN=0
export DATA_DIR=/lustre/fast/fast/groups/ei-slm/data/lora_regret
export RM_TYPE=math ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_STOP=$'\n\nProblem:'
export LR_DECAY_STYLE=constant WEIGHT_DECAY=0.0
export ADAM_BETA1=0.9 ADAM_BETA2=0.999
export KL_LOSS_COEF=0.0 KL_LOSS_TYPE=low_var_kl ENTROPY_COEF=0.0
export N_SAMPLES_PER_EVAL_PROMPT=1
export CONTEXT_PARALLEL_SIZE=1 EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1 MAX_TOKENS_PER_GPU=16384
export RECOMPUTE_NUM_LAYERS=1
export SGLANG_MEM_FRACTION_STATIC=0.75 SGLANG_MAX_RUNNING_REQUESTS=128
export SGLANG_MAX_TOTAL_TOKENS=262144 RAY_NUM_CPUS=16
export OFT_EPS=6e-5
export HF_CKPT=/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B
export MEGATRON_LOAD=/lustre/fast/fast/zqiu/orbit-infra/orbit/checkpoints/Llama-3.1-8B_torch_dist
export MODEL_ARGS_FILE="$ORBIT_WT/orbit_plugins/model_args/llama3.1-8B-Instruct.sh"
export ORBIT_ENTRYPOINT="$ORBIT_WT/train.py"
export RL_EXTRA_ARGS="--disable-grpo-std-normalization --wandb-dir $RUN_DIR/wandb"
export RAY_TEMP_DIR="$RUN_DIR/ray" ORBIT_TMPDIR="$RUN_DIR/tmp"
export PYTHONPATH="$ORBIT_WT"
unset MEGATRON_PATH MEGATRON_BRIDGE_ROOT
unset ORBIT_VENV
export UV_PROJECT_ENVIRONMENT="$ENV_ROOT" UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
unset ORBIT_RAY_ADDRESS RAY_ADDRESS RAY_PORT RAY_HEAD_PORT RAY_CLIENT_SERVER_PORT
unset RAY_DASHBOARD_PORT RAY_DASHBOARD_AGENT_LISTEN_PORT
unset RAY_DASHBOARD_AGENT_GRPC_PORT RAY_GCS_SERVER_PORT RAY_METRICS_EXPORT_PORT
unset RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT RAY_NODE_MANAGER_PORT
unset RAY_OBJECT_MANAGER_PORT RAY_RUNTIME_ENV_AGENT_PORT MASTER_ADDR
unset ORBIT_DEBUG_MODE ORBIT_DRIVER_DEBUG ORBIT_LAUNCHER_XTRACE
unset STAGE_HF_CKPT_TO STAGE_MEGATRON_CKPT_TO
unset CRITIC_NUM_GPUS_PER_NODE CRITIC_NUM_NODES
unset TRAIN_JSONL TEST_JSONL MATH_TEST_JSONL GSM8K_TEST_JSONL EVAL_DATASETS
unset WANDB_PROJECT WANDB_GROUP WANDB_RUN_NAME LAUNCHER_NAME
unset PEFT_METHOD OFT_BLOCK_SIZE TARGET_MODULES LR

mkdir -p "$RAY_TEMP_DIR" "$ORBIT_TMPDIR" || verification_rc=1
source "$ENV_ROOT/bin/activate" || verification_rc=1
cd "$ORBIT_WT" || verification_rc=1
source env.sh || verification_rc=1
test "${ORBIT_VENV:-}" = "$ENV_ROOT" || verification_rc=1
command -v setsid >/dev/null || verification_rc=1
command -v timeout >/dev/null || verification_rc=1
test "$(git symbolic-ref --short HEAD)" = codex/math-oft-lr3 || verification_rc=1
test "$(git rev-parse HEAD)" = "$expected_orbit_sha" || verification_rc=1
test -z "$(git status --porcelain)" || verification_rc=1
test "$(readlink results/e4_math_oft_lr3.jsonl)" = "$LEDGER" || verification_rc=1
test "$(readlink logs/lora_regret)" = "$RUN_DIR/arm-logs" || verification_rc=1
test "$(readlink wandb)" = "$RUN_DIR/wandb" || verification_rc=1
test ! -s "$LEDGER" || verification_rc=1
if pgrep -af 'run_e4_math_oft_lr3_8gpu[.]sh|e4_math_oft_lr3[.]jsonl' \
    >"$RUN_DIR/conflicting-processes.log"; then
    verification_rc=1
fi

recipe_tmp="$(mktemp "$RUN_DIR/recipe.env.tmp.XXXXXX")"
recipe_vars=(
    CUDA_VISIBLE_DEVICES GPUS_PER_NODE RAY_NUM_GPUS MODEL SEED ROLLOUT_SEED
    NUM_ROLLOUT GLOBAL_BATCH_SIZE ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT
    EVAL_INTERVAL SAVE_INTERVAL WANDB_MODE WANDB_AUTOSYNC EPS_CLIP EPS_CLIP_HIGH
    ROLLOUT_MAX_RESPONSE_LEN EVAL_MAX_RESPONSE_LEN TENSOR_MODEL_PARALLEL_SIZE
    PIPELINE_MODEL_PARALLEL_SIZE ROLLOUT_NUM_GPUS_PER_ENGINE RM_TYPE
    ROLLOUT_TEMPERATURE ROLLOUT_STOP LR_DECAY_STYLE WEIGHT_DECAY ADAM_BETA1
    ADAM_BETA2 KL_LOSS_COEF KL_LOSS_TYPE ENTROPY_COEF
    N_SAMPLES_PER_EVAL_PROMPT CONTEXT_PARALLEL_SIZE EXPERT_MODEL_PARALLEL_SIZE
    EXPERT_TENSOR_PARALLEL_SIZE MAX_TOKENS_PER_GPU RECOMPUTE_NUM_LAYERS
    SGLANG_MEM_FRACTION_STATIC SGLANG_MAX_RUNNING_REQUESTS
    SGLANG_MAX_TOTAL_TOKENS RAY_NUM_CPUS OFT_EPS DATA_DIR HF_CKPT
    MEGATRON_LOAD MODEL_ARGS_FILE ORBIT_ENTRYPOINT RL_EXTRA_ARGS RAY_TEMP_DIR
    ORBIT_TMPDIR DISABLE_EVAL ENABLE_WANDB ORBIT_DRY_RUN_ARGV ORBIT_COLOCATE
    ROLLOUT_NUM_GPUS ADVANTAGE_ESTIMATOR ORBIT_LOG_FILTER
    ORBIT_RAY_LOG_TO_DRIVER PARITY_CHECK
)
for variable in "${recipe_vars[@]}"; do
    printf '%s=%q\n' "$variable" "${!variable}" >>"$recipe_tmp"
done
mv "$recipe_tmp" "$RUN_DIR/recipe.env"

python - "$RUN_DIR/imports.json" "$expected_orbit_sha" <<'PY' || verification_rc=1
import importlib.metadata as md
import json
import os
import sys

import orbit
import megatron.bridge
import megatron.core
import sglang

destination, orbit_sha = sys.argv[1:]
env_root = os.path.realpath("/fast/zqiu/orbit-iclr/orbit_env")
orbit_root = os.path.realpath(
    "/lustre/fast/fast/zqiu/software/proj/spherelab/orbit-math-oft-lr3"
)
assert os.path.commonpath([os.path.realpath(orbit.__file__), orbit_root]) == orbit_root
assert os.path.commonpath([os.path.realpath(sglang.__file__), env_root]) == env_root
assert os.path.realpath(sys.prefix) == env_root
records = {"orbit_file": orbit.__file__, "orbit_sha": orbit_sha,
           "sglang_file": sglang.__file__, "sys_prefix": sys.prefix,
           "megatron_bridge_file": megatron.bridge.__file__,
           "megatron_core_file": megatron.core.__file__}
for name, sha, subdirectory in (
    ("sglang", "b52394d22fc4b686016943efc47cce6fb892cef2", "python"),
    ("sgl-kernel", "9c83ae8be07cbb1eb6898ce608ae244e3be375b4", "sgl-kernel"),
    ("megatron-bridge", "85c84cbc26d4c983a3d6e46c804f02e2a99af5a2", None),
    ("megatron-core", "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd", None),
):
    direct = json.loads(md.distribution(name).read_text("direct_url.json"))
    if subdirectory is not None:
        assert direct["subdirectory"] == subdirectory
    assert direct["vcs_info"]["commit_id"] == sha
    assert direct["vcs_info"]["requested_revision"] == sha
    records[name] = direct
temporary = destination + ".tmp"
with open(temporary, "w") as stream:
    json.dump(records, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, destination)
PY

DRY_RUN=1 bash scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh \
    >"$RUN_DIR/dry-run.log" 2>&1 || verification_rc=1
python - "$RUN_DIR/dry-run.log" <<'PY' || verification_rc=1
import shlex
import sys

expected_blocks = {
    "oftscout-b8-all-math-lr3e-05-s0": "8",
    "oftscout-b128-all-math-lr3e-05-s0": "128",
    "oftscout-b1024-all-math-lr3e-05-s0": "1024",
}
expected_targets = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
expected_train = "/lustre/fast/fast/groups/ei-slm/data/lora_regret/math_train.jsonl"
rows = {}
with open(sys.argv[1]) as stream:
    for line in stream:
        tokens = shlex.split(line)
        if "bash" not in tokens:
            continue
        bash_index = tokens.index("bash")
        values = dict(
            token.split("=", 1) for token in tokens[:bash_index] if "=" in token
        )
        arm = values.get("LAUNCHER_NAME")
        if arm in expected_blocks:
            assert arm not in rows, arm
            rows[arm] = values
assert set(rows) == set(expected_blocks), rows
for arm, block_size in expected_blocks.items():
    values = rows[arm]
    assert values["OFT_BLOCK_SIZE"] == block_size, (arm, values)
    assert values["TARGET_MODULES"] == expected_targets, (arm, values)
    assert values["PEFT_METHOD"] == "oft", (arm, values)
    assert values["LR"] == "3e-05", (arm, values)
    assert values["SEED"] == "0", (arm, values)
    assert values["EVAL_DATASETS"] == "math", (arm, values)
    assert values["TRAIN_JSONL"] == expected_train, (arm, values)
PY
grep -Fq -- '3 arms selected, 3 to run -> results/e4_math_oft_lr3.jsonl' \
    "$RUN_DIR/dry-run.log" || verification_rc=1
if ((verification_rc != 0)); then
    atomic_lines "$ACCEPTANCE" "state=preflight_failed" "accepted=0"
    publish_completion 0 1 1 preflight_failed
    rmdir "$LOCK_DIR"
    exit 1
fi

printf '{"event":"start","at":"%s","orbit_sha":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$expected_orbit_sha" >>"$EVENTS"
export DRY_RUN=0
setsid bash scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh >"$CONSOLE" 2>&1 &
active_pid=$!
wait "$active_pid"
launcher_rc=$?
active_pid=

grep -Fq -- '3 arms selected, 3 to run -> results/e4_math_oft_lr3.jsonl' \
    "$CONSOLE" || verification_rc=1
grep -Fq -- 'running 3 arms sequentially on 8 GPUs' "$CONSOLE" \
    || verification_rc=1

ledger_rc=0
python - "$LEDGER" "$RUN_DIR/ledger-summary.json" "$EVENTS" <<'PY' \
    || ledger_rc=$?
import json
import os
import sys

ledger, destination, events = sys.argv[1:]
expected_blocks = {
    "oftscout-b8-all-math-lr3e-05-s0": 8,
    "oftscout-b128-all-math-lr3e-05-s0": 128,
    "oftscout-b1024-all-math-lr3e-05-s0": 1024,
}
expected = set(expected_blocks)
expected_targets = "linear_qkv,linear_proj,linear_fc1,linear_fc2"
with open(ledger) as stream:
    records = [json.loads(line) for line in stream if line.strip()]
assert len(records) == 3, len(records)
latest = {record["arm"]: record for record in records}
assert set(latest) == expected, set(latest)
summary = {}
for arm in sorted(expected):
    row = latest[arm]
    assert row["status"] == "ok"
    assert row["method"] == "oft"
    assert row["oft_block_size"] == expected_blocks[arm]
    assert row["target_modules"] == expected_targets
    assert row["lr"] == 3e-5
    assert row["seed"] == 0
    assert row["metric"] == "accuracy"
    assert row["accuracy"] is not None
    assert row["steps"] == 149
    assert len(row["rollout_seconds"]) == 150
    assert row["gpus"] == 8
    assert row["probe_rollouts"] is None
    summary[arm] = {
        "accuracy": row["accuracy"],
        "seconds": row["seconds"],
        "steps": row["steps"],
        "rollouts": len(row["rollout_seconds"]),
        "oft_block_size": row["oft_block_size"],
        "target_modules": row["target_modules"],
        "accepted": True,
    }
temporary = destination + ".tmp"
with open(temporary, "w") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, destination)
with open(events, "a") as stream:
    for arm in sorted(expected):
        stream.write(json.dumps({"event": "arm_result", "arm": arm,
                                 **summary[arm]}, sort_keys=True) + "\n")
PY
[[ "$ledger_rc" -eq 0 ]] || verification_rc=1

wandb_count="$(find "$RUN_DIR/wandb" -type f -name 'run-*.wandb' -size +0c | wc -l | tr -d ' ')"
wandb_rc=0
python - "$RUN_DIR" "$RUN_DIR/wandb-map.json" <<'PY' || wandb_rc=$?
import json
import os
from pathlib import Path
import sys

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

run_root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2])
wandb_root = (run_root / "wandb").resolve()
expected = {
    "oftscout-b8-all-math-lr3e-05-s0",
    "oftscout-b128-all-math-lr3e-05-s0",
    "oftscout-b1024-all-math-lr3e-05-s0",
}
mapping = {}
for arm in sorted(expected):
    log = run_root / "arm-logs" / f"{arm}.log"
    assert log.is_file() and log.stat().st_size > 0, log
    assert arm in log.read_text(errors="replace"), (arm, log)
run_files = sorted(path.resolve() for path in wandb_root.glob(
    "**/offline-run-*/run-*.wandb"
) if path.is_file() and path.stat().st_size > 0)
assert len(run_files) == 3, run_files
assert len({path.parent for path in run_files}) == 3, run_files
for run_file in run_files:
    directory = run_file.parent
    assert os.path.commonpath([directory, wandb_root]) == str(wandb_root)
    siblings = [path for path in directory.glob("run-*.wandb")
                if path.is_file() and path.stat().st_size > 0]
    assert siblings == [run_file], (directory, siblings)
    store = DataStore()
    store.open_for_scan(str(run_file))
    display_names = set()
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField("run") and record.run.display_name:
            display_names.add(record.run.display_name)
    assert len(display_names) == 1, (run_file, display_names)
    arm = display_names.pop()
    assert arm in expected and arm not in mapping, (arm, mapping)
    mapping[arm] = {
        "offline_run_dir": str(directory),
        "run_file": str(run_file),
        "display_name": arm,
    }
assert set(mapping) == expected, mapping
temporary = destination.with_name(destination.name + ".tmp")
temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY
[[ "$wandb_rc" -eq 0 ]] || verification_rc=1

lease_release_rc=0
release_test_lease
lease_release_rc=$?
if ((lease_release_rc == 0)); then
    TEST_LEASE_ARMED=0
else
    verification_rc=1
fi
lock_release_rc=0
rmdir "$LOCK_DIR" || lock_release_rc=74
if ((lock_release_rc != 0)); then
    verification_rc=1
fi
final_rc=$launcher_rc
if ((final_rc == 0 && verification_rc != 0)); then
    final_rc=1
fi
terminal_event_rc=0
printf '{"event":"terminal","at":"%s","launcher_rc":%d,"verification_rc":%d,"final_rc":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$launcher_rc" "$verification_rc" \
    "$final_rc" >>"$EVENTS" || terminal_event_rc=74
if ((terminal_event_rc != 0)); then
    verification_rc=1
    final_rc=74
fi
accepted=0
if ((launcher_rc == 0 && verification_rc == 0)); then
    accepted=1
fi
if ! atomic_lines "$ACCEPTANCE" \
    "state=terminal" \
    "accepted=$accepted" \
    "ledger_exact_arm_set=$((ledger_rc == 0 ? 1 : 0))" \
    "bs8_row_accepted=$((ledger_rc == 0 ? 1 : 0))" \
    "bs128_row_accepted=$((ledger_rc == 0 ? 1 : 0))" \
    "bs1024_row_accepted=$((ledger_rc == 0 ? 1 : 0))" \
    "wandb_arm_run_bijection=$((wandb_rc == 0 ? 1 : 0))" \
    "wandb_run_file_count=$wandb_count" \
    "arm_logs_and_run_names_accepted=$((wandb_rc == 0 ? 1 : 0))" \
    "test_lease_released=$((lease_release_rc == 0 ? 1 : 0))" \
    "run_lock_released=$((lock_release_rc == 0 ? 1 : 0))" \
    "terminal_event_published=$((terminal_event_rc == 0 ? 1 : 0))"; then
    verification_rc=1
    final_rc=74
fi
if ! publish_completion "$launcher_rc" "$verification_rc" "$final_rc" terminal; then
    exit 74
fi
exit "$final_rc"
```
