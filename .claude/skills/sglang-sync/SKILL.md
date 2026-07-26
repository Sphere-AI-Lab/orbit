---
name: sglang-sync
description: "Advance the thirdparty/sglang submodule to the current sgl-project/sglang@sglang-miles line, mirror it to impossible-inc/sglang, and realign the ACTIVE source pin plus torch-ABI wheels bundle. Run it TOGETHER with /miles-sync (the default — miles code and sglang move as one bundle), or standalone for an sglang-only bump. Hard rules: (1) on a version-bump rebase, re-apply local mirror patches (e.g. the geo3k VLM mrope gate) onto the new target, then advance the mirror by ARCHIVING the old sglang-miles tip (branch + date tag) and force-pushing the rebased tip (as sgl-project itself maintains the branch); STOP on any mirror-only commit you can't classify, a patch that won't re-apply, or an unconfirmed force; (2) pushes (impossible-inc/sglang + the PR) happen ONLY after explicit user approval."
---

# sglang-sync — advance the sglang dependency bundle

Bring `thirdparty/sglang` up to the current miles sglang line and realign the
ACTIVE source pin plus wheel bundle (`MILES_SGLANG_SOURCE_VERSION`,
`MILES_WHEELS_TAG`, torch, router) so a fresh `install_env.sh`
builds a consistent env. Designed to run **together with `/miles-sync`** — the
source line and its torch-compatible prebuilt wheels should move in the same
PR, although the wheel release's SGLang label may lag the source when torch
matches.

## ⚠️⚠️⚠️ HARD RULES ⚠️⚠️⚠️

1. **STOP on mirror-only commits; never silently drop a local patch.** A
   non-fast-forward is EXPECTED on a version bump (upstream rebases `sglang-miles`
   onto the new release tag). Step 3 MUST list the mirror-only commits and you MUST
   get explicit confirmation of which are LOCAL miles patches vs old-upstream-line
   commits. Every local patch (e.g. the geo3k VLM mrope gate) must be RE-APPLIED onto
   the rebased target (Step 3, after checkout) — never lost. STOP if you can't classify
   a commit, a patch won't re-apply cleanly, or it may no longer be needed (ask). The
   mirror's `sglang-miles` is advanced by ARCHIVING the old tip (branch + date tag),
   then FORCE-PUSHING the rebased tip (Step 8) — as sgl-project does, but archived first
   so our SHA-pinned gitlinks still resolve.
2. **No outward pushes until the user says so.** That means BOTH the
   `impossible-inc/sglang` branch push AND the miles-imp PR. Do all local work
   (fetch, ff, gitlink bump, pin edits), show the combined diff, and wait for an
   explicit "push it".
3. **One bundle, one commit.** The sglang gitlink bump +
   `MILES_SGLANG_SOURCE_VERSION` + `MILES_WHEELS_TAG` + any `WHEELS_STACK` row
   go in a single commit (folded into the miles-sync "our changes" commit when
   run together). (The sync-record folder is the one exception: standalone
   runs commit it as its own `[docs] sglang-sync record` commit — Step 7;
   combined runs leave it to miles-sync Step 9.)
4. **`extract_pins.py --check` must end at exit 0 with no `[sglang-sync pending]`**
   — that is the definition of done: ACTIVE source == UPSTREAM image target and
   the wheels are torch-ABI consistent.

## Topology (validated)

- The miles sglang line is the **`sglang-miles` branch on `sgl-project/sglang`**
  (the official sglang repo hosts it; it is *not* a radixark fork). That is the
  submodule's `upstream` remote.
- `thirdparty/sglang` `origin` = `impossible-inc/sglang` (our mirror; what
  `.gitmodules` points at and what fresh clones fetch from). The pinned commit
  MUST exist here, so the mirror is updated **before the miles PR branch is
  published** (the push itself still waits for approval — Step 8).
- `impossible-inc/sglang` is a **private https** remote. The plain `git fetch
  origin` / `git push origin` FAIL non-interactively ("could not read Username") —
  both must use a token-injected URL: `https://x-access-token:$GH_TOKEN@github.com/impossible-inc/sglang.git`.
  Treat the stale `refs/remotes/origin/sglang-miles` as unreliable until you have
  fetched it through that URL.
- **The `sglang-miles` line is rebased upstream on version bumps** (the v0.5.10
  patches are re-applied onto v0.5.12, etc.). So across a bump our old pin is NOT an
  ancestor of the new target. We advance our mirror the way sgl-project does — by
  FORCE-PUSHING `sglang-miles` to the rebased tip (`$NEWPIN` = target + re-applied local
  patches) — but FIRST archive the old tip as a branch (`sglang-miles-<base>-final`) +
  a date tag, because our mirror is consumed by SHA-pinned gitlinks (old miles commits
  pin the old SHA) and the archive keeps it reachable. Within a version (v0.5.12-23 →
  v0.5.12-50) it's a plain fast-forward of `sglang-miles` itself (no archive needed).
- `impossible-inc/sglang@sglang-miles` is **NOT a pure mirror** — it carries local
  miles patches on top of the upstream line. Currently FOUR (as of the v0.5.15 sync):
  1. `[sglang-miles] forward_batch: gate mrope text-only path on rl_on_policy_target`
     (geo3k VLM fix, authored locally; upstream candidate).
  2. `[sglang-miles cu129] bare-metal cu12 dep flavors` (pyproject: cuda-python <13,
     flashinfer [cu12], plain cutlass-dsl, +cu129 local-version pins for
     sglang-kernel/sgl-deep-gemm/torchao — the PyPI default wheels of those three are
     cu13-linked). Mirror-only BY DESIGN — not an upstream candidate (upstream's
     docker line wants the cu13 flavors).
  3. `[sglang-miles] exact multimodal scoring suffix` (preserves the sampled
     response-token suffix while the teacher owns multimodal prefix processing).
  4. `[sglang-miles] qwen-vl pretokenized input IDs` (keeps exact caller IDs on
     Qwen-VL's legacy multimodal processor path).
  The old pause-aware `flush_cache` patch is retired: upstream #31962 now
  provides the required `is_fully_idle(ignore_waiting=self._engine_paused)`
  behavior.
  On every version bump these local patches MUST be re-applied onto the rebased target
  (Step 3) — the new pin is `target + re-applied patch(es)`, not the bare target. Step 3
  surfaces the mirror-only commits and STOPs so you can tell local patches from
  old-upstream-line commits.
- **Verify re-applied patch CONTENT, not commit titles.** sgl-project's own rebase of
  its miles patch stack can silently drop hunks: on the v0.5.13 rebase, the twin of
  "Fix pause-aware weight update deadlocks" carried the right title but only a one-line
  fragment — the flush_cache disjunct was lost to a conflicting upstream refactor. For
  each `[sglang-miles]` patch expected on the new line, grep the rebased TREE for the
  patch's key lines (e.g. `_engine_paused and self.running_batch.is_empty`,
  `rl_on_policy_target` in forward_batch_info) before trusting it survived.
- The torch ABI + prebuilt wheels live in `yueming-yuan/miles-wheels` under
  rolling CUDA/architecture tags such as `cu129-x86_64`; `WHEELS_STACK` in
  `extract_pins.py` records the source line, torch ABI, and router version most
  recently validated for that rolling binary set.

## Usage

```
/sglang-sync                       # discover target from sgl-project/sglang@sglang-miles base version
/sglang-sync cu129-x86_64           # explicit target wheels tag for bare-metal cu12.9
```

## Workflow

### Step 0 — Ensure submodule remotes

```bash
S=thirdparty/sglang
git -C "$S" remote get-url upstream >/dev/null 2>&1 \
    || git -C "$S" remote add upstream https://github.com/sgl-project/sglang.git
git -C "$S" remote get-url origin    # expect https://github.com/impossible-inc/sglang.git
```

### Step 1 — Fetch + analyze the target line

```bash
S=thirdparty/sglang
# Explicit refspec — a bare `fetch upstream sglang-miles` only guarantees FETCH_HEAD;
# this guarantees refs/remotes/upstream/sglang-miles is the fresh tip (not stale).
git -C "$S" fetch upstream sglang-miles:refs/remotes/upstream/sglang-miles

PIN=$(git -C "$S" rev-parse HEAD)
TGT=$(git -C "$S" rev-parse upstream/sglang-miles)
echo "pin:    $(git -C "$S" describe --tags "$PIN")"
echo "target: $(git -C "$S" describe --tags "$TGT")  (+$(git -C "$S" rev-list --count ${PIN}..${TGT}) commits)"

SGLANG_BASE=$(git -C "$S" describe --tags --abbrev=0 "$TGT")          # e.g. v0.5.12
NEW_TORCH=$(git -C "$S" show "$TGT:python/pyproject.toml" | grep -oE '"torch==[^"]+"' | head -1 | tr -d '"' | cut -d= -f3)
echo "sglang base: $SGLANG_BASE   torch: $NEW_TORCH (current $(grep -oE '"torch==[^"]+"' "$S/python/pyproject.toml" | head -1))"
```

Show the user the base-version jump, the torch jump, and commit count. Optionally
skim `git -C "$S" log --oneline ${PIN}..${TGT} | grep -i '\[sglang-miles\]'` for
the miles-specific patches landing.

### Step 2 — Determine + verify the wheels bundle

```bash
ARG="${1:-}"
if [[ -n "$ARG" ]]; then
    TAG="$ARG"
else
    # Rolling tags no longer encode an sglang version. Preserve the ACTIVE
    # CUDA/architecture release unless this sync deliberately changes that
    # platform, in which case pass the new tag explicitly.
    TAG=$(sed -n 's/^MILES_WHEELS_TAG=${MILES_WHEELS_TAG:-\([^}]*\)}$/\1/p' \
        scripts/slurm/setup/pins.env)
fi
echo "[sglang] target wheels tag: $TAG"

# The miles-wheels release for this tag MUST exist (install_env.sh fetches FA/apex/
# sglang_router from it).
gh release view "$TAG" --repo yueming-yuan/miles-wheels --json name,assets \
    --jq '.name, ([.assets[].name] | join(", "))'
```

Rolling release names do not prove a torch ABI. Before updating `WHEELS_STACK`,
require one of these:

1. An upstream image build using the same release and `$NEW_TORCH`.
2. SHA256 equality with a wheel set already validated under `$NEW_TORCH`.
3. A scratch-environment import/GPU smoke for FA2, FA3, and Apex.

Record the release asset fingerprint and the evidence used. If the ABI is not
proven, **STOP** and record `[sglang-sync pending]`.

**A same-version wheels release is NOT required — the bundle may LAG the sglang source
when torch matches.** Bundle wheels (FA2/FA3/apex/router/gateway) are torch-ABI-bound,
not sglang-version-bound; the sglang-version-locked kernels (sglang-kernel,
sgl-deep-gemm) come from `docs.sglang.ai/whl/cuNNN` +cuNNN builds pinned in the fork's
pyproject (SGL_WHL_INDEX_URL), not from the bundle. When the existing rolling
binary set is proven compatible with the new source line's unchanged torch,
keep that tag as ACTIVE and refresh its `WHEELS_STACK` metadata. `install_env.sh`
still fail-closes on any torch mismatch. `[sglang-sync pending]` compares
`MILES_SGLANG_SOURCE_VERSION` directly with `UPSTREAM_SGLANG_IMAGE_TAG`; wheels
tags are not used as a source-version proxy.

### Step 3 — Advance the mirror (rebase-aware; uses the TRUE origin state)

The `sglang-miles` line is rebased upstream on version bumps, so do NOT assume a
fast-forward and do NOT trust the cached `origin/sglang-miles` ref — fetch origin
through the token URL first to learn the real mirror state.

```bash
S=thirdparty/sglang
OURL="https://x-access-token:$GH_TOKEN@github.com/impossible-inc/sglang.git"
# upstream: explicit refspec so the remote-tracking ref (not just FETCH_HEAD) is fresh.
git -C "$S" fetch upstream sglang-miles:refs/remotes/upstream/sglang-miles  # target (public)
TGT=$(git -C "$S" rev-parse refs/remotes/upstream/sglang-miles)
# origin: anonymous token URL (no configured refspec) → only writes FETCH_HEAD, so read
# MIRROR right after, before anything else overwrites it. This is the TRUE mirror state
# (the cached refs/remotes/origin/sglang-miles is stale — origin is private https).
git -C "$S" fetch "$OURL" sglang-miles
MIRROR=$(git -C "$S" rev-parse FETCH_HEAD)

if git -C "$S" merge-base --is-ancestor "$MIRROR" "$TGT"; then
    echo "[sglang] fast-forward: mirror $MIRROR is an ancestor of target $TGT"
    FORCE=0
else
    echo "[sglang] NON-fast-forward (upstream rebased, or mirror diverged)."
    echo "  Commits on the mirror NOT in the target (must all be old-upstream-line,"
    echo "  NOT local patches you authored — else STOP and preserve them):"
    git -C "$S" log --oneline "$TGT".."$MIRROR"
    FORCE=1
fi
```

- `FORCE=0` (ancestor) → clean fast-forward; the Step 8 push needs no `--force`.
- `FORCE=1` (non-ancestor) → expected on a version bump (rebase). **Show the user
  the `$TGT..$MIRROR` list and get explicit confirmation** of which commits are local
  miles patches vs old-upstream-line commits. Local patches are NOT carried forward by
  sgl-project's rebase — they must be re-applied below. **STOP** if you can't classify
  a commit or a patch is ambiguous.

Pin/checkout the target, then RE-APPLY the local patches so the pin = target + patches
(the gitlink in Step 5 records this `$NEWPIN`):

```bash
git -C "$S" checkout --detach "$TGT"      # start from the exact rebased target
# For each LOCAL patch from the $TGT..$MIRROR list that the rebase didn't carry and the
# new base doesn't subsume, re-apply it — and verify it's still needed against the new base:
#   git -C "$S" cherry-pick <sha>          # or hand-port if context changed; preserve --author
NEWPIN=$(git -C "$S" rev-parse HEAD)       # = $TGT if there are NO local patches
```

The submodule HEAD `$NEWPIN` (target + re-applied local patches) is the pin. Step 8
archives the old tip, then force-pushes `sglang-miles` to `$NEWPIN`.

### Step 4 — Refresh the WHEELS_STACK row

```bash
python3 scripts/slurm/setup/extract_pins.py --resolve "$TAG" || true
```

Edit `WHEELS_STACK` in `scripts/slurm/setup/extract_pins.py` after Step 2's ABI
evidence is complete. Update the existing rolling-tag row, or add one only for a
new CUDA/architecture tag:

`"$TAG": {"sglang": "$SGLANG_BASE", "torch": "$NEW_TORCH", "router": "<release router version>"}`

`router` is the version in the `sglang_router-*.whl` asset.

### Step 5 — Bump the gitlink + realign ACTIVE pins

```bash
git add thirdparty/sglang        # records the advanced submodule commit

# Record the ACTIVE source line and select the torch-compatible wheels bundle,
# then re-derive bundle metadata:
sed -i "s|^MILES_SGLANG_SOURCE_VERSION=.*|MILES_SGLANG_SOURCE_VERSION=\${MILES_SGLANG_SOURCE_VERSION:-$SGLANG_BASE}|" scripts/slurm/setup/pins.env
sed -i "s|^MILES_WHEELS_TAG=.*|MILES_WHEELS_TAG=\${MILES_WHEELS_TAG:-$TAG}|" scripts/slurm/setup/pins.env
python3 scripts/slurm/setup/extract_pins.py --write    # re-derives torch/sglang/router + refreshes UPSTREAM_*
```

`--write` reads `TORCH_VERSION` from the now-bumped submodule pyproject and the
derived fields from `WHEELS_STACK[$TAG]`; it **refuses (exit 2)** if they disagree
(submodule not actually bumped, or WHEELS_STACK row wrong). Fix before continuing.

### Step 6 — Confirm done (match what install_env.sh fails closed on)

`extract_pins.py --check` validates source/target alignment plus torch-ABI
consistency. `install_env.sh` also checks the effective bundle and the
`sglang_router` wheel version. So "done" must verify all of them, or you'll
declare success and watch the install abort:

Each check **aborts** on failure (`exit 1`) — otherwise the block ends on a
`git diff` that returns 0 and the flow would commit/push past a failed check.

```bash
# 1. pins self-consistent AND no residual pending. NOTE: --check exits 0 on a bare
#    [sglang-sync pending] (it's deferrable for CI), so the exit code alone does NOT
#    catch it — capture the output and grep. In the together/default flow (and any
#    same-base bump) ACTIVE must == UPSTREAM, i.e. NO pending; if one remains the
#    advance didn't take. (Exception: a deliberate standalone bump AHEAD of a
#    not-yet-bumped miles Dockerfile — only then is a reverse pending acceptable;
#    the operator confirms that case and skips this grep.)
check_out=$(python3 scripts/slurm/setup/extract_pins.py --check 2>&1) \
    || { printf '%s\n' "$check_out" >&2; echo "STOP: --check not clean (ABI/drift)" >&2; exit 1; }
printf '%s\n' "$check_out"
grep -qF '[sglang-sync pending]' <<<"$check_out" \
    && { echo "STOP: [sglang-sync pending] remains — ACTIVE != UPSTREAM, advance incomplete" >&2; exit 1; } || true

# 2. submodule TORCH == the bundle torch (the real ABI safety; install_env.sh fails
#    closed on this). The submodule's sglang BASE may legitimately be AHEAD of the
#    bundle's (bundle-may-lag rule, Step 2) — report it, don't stop.
source scripts/slurm/setup/pins.env
eval "$(python3 scripts/slurm/setup/extract_pins.py --resolve "$TAG")"   # sets MILES_WHEELS_{SGLANG,TORCH,...}_VERSION
sub_torch=$(grep -oE '"torch==[0-9][^"]*"' thirdparty/sglang/python/pyproject.toml | head -1 | tr -d '"' | cut -d= -f3)
[[ "$sub_torch" == "$MILES_WHEELS_TORCH_VERSION" ]] \
    && echo "✓ submodule torch $sub_torch == bundle torch $MILES_WHEELS_TORCH_VERSION (sglang: source $MILES_SGLANG_SOURCE_VERSION / bundle label $MILES_WHEELS_SGLANG_VERSION)" \
    || { echo "STOP: submodule torch $sub_torch != bundle torch $MILES_WHEELS_TORCH_VERSION — ABI mismatch" >&2; exit 1; }

# 3. the release ships sglang_router-$SGLANG_ROUTER_VERSION (install_env.sh fetches it by
#    prefix and FAILS CLOSED — exit 1 — on a version mismatch, so this must hold)
gh release view "$TAG" --repo yueming-yuan/miles-wheels --json assets \
    --jq '[.assets[].name] | map(select(startswith("sglang_router-'"$SGLANG_ROUTER_VERSION"'-"))) | length' \
    | grep -qx 1 \
    && echo "✓ release has sglang_router-$SGLANG_ROUTER_VERSION" \
    || { echo "STOP: release sglang_router != $SGLANG_ROUTER_VERSION — fix WHEELS_STACK router field" >&2; exit 1; }

git diff --cached --stat            # thirdparty/sglang + pins.env + extract_pins.py
```

All three green = the bundle install_env.sh will build is consistent. (Standalone
with ACTIVE deliberately ahead of a not-yet-bumped miles Dockerfile → a *reverse*
pending notice is acceptable; note it.)

### Step 7 — Stage + commit (fold into miles-sync's commit when combined)

Standalone: one commit.

```bash
git add scripts/slurm/setup/pins.env scripts/slurm/setup/extract_pins.py thirdparty/sglang
git commit -m "[sglang] sync sglang-miles $SGLANG_BASE (torch $NEW_TORCH); ACTIVE -> $TAG"
```

When invoked by `/miles-sync`: do NOT commit separately — these staged changes are
folded into miles-sync's single "our changes" commit (Step 6 there), so the PR has
the miles merge + one combined bundle commit.

**Record the event** (git-tracked history — see
`scripts/slurm/docs/sync-records/README.md`; before a standalone run, read ONLY the
newest record there — older ones describe superseded pin states and pollute context):
combined runs write their notes into the
cycle's `sync-records/miles-sync-YYYY-MM-DD/` folder; a **standalone** sglang bump gets
its own `sync-records/sglang-sync-YYYY-MM-DD/` folder. Save anything the next operator
needs: the mirror-only commit classification from Step 3, patch re-apply notes, the
publish script/commands actually run (Step 8), and debug notes for anything that broke.
Commit the folder as `[docs] sglang-sync YYYY-MM-DD record` alongside the push
(combined runs: miles-sync Step 9 commits it).

### Step 8 — Publish: sync branch now, archive + force-advance sglang-miles later (ONLY after explicit approval)

⛔ No outward push without an explicit "push it". The gitlink points at `$NEWPIN`
(= `$TGT` + re-applied local patches), which must be fetchable from
`impossible-inc/sglang` BEFORE the miles PR is published (so the submodule resolves for
anyone who fetches it). Push the dated sync branch first — it anchors the pin (gc-safe +
fetchable + reviewable) before we touch `sglang-miles`:

```bash
OURL="https://x-access-token:$GH_TOKEN@github.com/impossible-inc/sglang.git"
S=thirdparty/sglang
NEWPIN=$(git -C "$S" rev-parse HEAD)
BR="sync-${SGLANG_BASE}-$(date +%Y%m%d)"          # e.g. sync-v0.5.12-20260603
git -C "$S" checkout -B "$BR" "$NEWPIN"            # anchor the pin on a branch (gc-safe)
git -C "$S" push "$OURL" "refs/heads/$BR:refs/heads/$BR"   # NON-force; sglang-miles untouched
```

That push alone makes the pin durable AND resolvable (`git submodule update` fetches
`$NEWPIN` from `$BR`). **Defer here if asked ("branch out but don't PR yet").**

**Open the review PR (`$BR -> sglang-miles`) BEFORE advancing `sglang-miles` — this is
the history track, not optional** (precedent: impossible-inc/sglang PR #1 v0.5.12,
PR #2 v0.5.13). Body per that convention: contents (base jump + each local patch with
one-line rationale), the landing mechanics for this $FORCE case, and the consuming
miles-imp PR. Once the new tip lands, the PR auto-marks **Merged** (its head becomes
the base tip). Ordering matters: after the force-advance, base == head and GitHub
refuses to create the PR — you'd have to briefly reset `sglang-miles` to the old tip
to open it retroactively (v0.5.13 landing had to do exactly that).

Then land it on `sglang-miles`, advancing by Step 3's `$FORCE`:

**`$FORCE=0` (within-version bump — `sglang-miles` IS an ancestor of `$NEWPIN`):** plain
fast-forward. Push directly (or "Rebase and merge" the review PR) — both keep it linear:

```bash
git -C "$S" push "$OURL" "$NEWPIN:refs/heads/sglang-miles"   # fast-forward, no --force
```

**`$FORCE=1` (version bump — upstream rebased `sglang-miles`; `$NEWPIN` diverges, the
GitHub UI can't auto-merge):** force-push `sglang-miles` to `$NEWPIN`, exactly as
sgl-project maintains the branch — but ARCHIVE the old tip FIRST so SHA-pinned gitlinks
(old miles commits) still resolve:

```bash
OLD=$(git -C "$S" ls-remote "$OURL" sglang-miles | awk '{print $1}')   # TRUE current tip
OLD_BASE=$(git -C "$S" describe --tags --abbrev=0 "$OLD"); D=$(date +%Y%m%d)
# 1. archive old tip — branch (robust for `git submodule update`) + date tag
git -C "$S" push "$OURL" "$OLD:refs/heads/sglang-miles-${OLD_BASE}-final"
git -C "$S" tag "sglang-miles-${OLD_BASE}-$D" "$OLD" && git -C "$S" push "$OURL" "sglang-miles-${OLD_BASE}-$D"
# 2. force-advance (lease-guarded to the archived OLD) + date-tag the new tip
git -C "$S" push --force-with-lease=sglang-miles:"$OLD" "$OURL" "$NEWPIN:refs/heads/sglang-miles"
git -C "$S" tag "sglang-miles-${SGLANG_BASE}-$D" "$NEWPIN" && git -C "$S" push "$OURL" "sglang-miles-${SGLANG_BASE}-$D"
```

NOTE: `.gitmodules` sets `branch = sglang-miles`, so `git submodule update --remote`
follows the new tip; default gitlink updates (`PULL_REMOTE=0`) use the pinned SHA.

Then the miles-imp branch + PR (standalone), or hand back to miles-sync Step 9 (combined).

### Test plan (always — torch jumps are heavyweight)

A torch minor/major bump (e.g. 2.9.1 → 2.11.0) can break Megatron-LM / TE / apex /
flash-attn. Put these in the PR body:

- [ ] `bash scripts/slurm/setup/install_env.sh` in a fresh GPU salloc (full rebuild — torch changed).
- [ ] `python scripts/slurm/setup/verify_env.py` passes.
- [ ] Sanity-launch a recipe to first eval.

## Not ready → fall back to pending

If Step 2 finds the wheels release missing, or Step 1 shows `sgl-project/sglang@sglang-miles`
is still on the OLD base (sgl-project hasn't rebased to the version miles wants yet),
the bundle can't be advanced. Do NOT force it. Leave ACTIVE where it is, let
`extract_pins.py --check` keep emitting `[sglang-sync pending]`, and tell the user to
retry sglang-sync once upstream is ready. This is the safe fallback the install-time
ABI guard backstops.

## See also

- [`/miles-sync`](../miles-sync/SKILL.md) — invokes this at its Step 5d pending gate (sync-together).
- [`scripts/slurm/setup/extract_pins.py`](../../../scripts/slurm/setup/extract_pins.py) — `WHEELS_STACK`, `--resolve`, `--check`.
- [`scripts/slurm/docs/sync-records/upstream-sync-design.md`](../../../scripts/slurm/docs/sync-records/upstream-sync-design.md) — ACTIVE/UPSTREAM model + sglang topology.
- [`scripts/slurm/docs/sync-records/README.md`](../../../scripts/slurm/docs/sync-records/README.md) — the tracked sync-history layout + index of past syncs.
