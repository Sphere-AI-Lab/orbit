#!/bin/bash
# sglang-sync Step 8 — publish the v0.5.13 mirror state to impossible-inc/sglang.
# PREPARED 2026-07-04; RUN ONLY AFTER EXPLICIT APPROVAL ("push it").
#
# State at preparation:
#   NEWPIN  = 723ed7d19  (sync-v0.5.13-20260702: 6ee17b436 v0.5.13-35 + mrope
#                         4f3aaf47a + cu129 flavors fe3f5fced + flush restore 723ed7d19)
#   OLD tip = 361f0f375  (mirror sglang-miles, v0.5.12 line, fetched 2026-07-04)
#   NON-fast-forward (version-bump rebase) -> archive old tip, then force-advance.
set -euo pipefail
S=/data/home/xiuyul/workspace/miles-imp/thirdparty/sglang
TOK="${GH_TOKEN:-$(gh auth token)}"
OURL="https://x-access-token:${TOK}@github.com/impossible-inc/sglang.git"
NEWPIN=723ed7d19144ba310bd49977251178253cafc21d
D=$(date +%Y%m%d)

# 0. Re-verify the TRUE mirror tip; refuse to proceed if it moved past what we analyzed.
OLD=$(git -C "$S" ls-remote "$OURL" refs/heads/sglang-miles | awk '{print $1}')
EXPECTED_OLD=361f0f375f67a160f3489ad1af1492882a14749c
[[ "$OLD" == "$EXPECTED_OLD" ]] || {
  echo "STOP: mirror sglang-miles moved ($OLD != $EXPECTED_OLD) — re-run the Step-3 analysis." >&2
  exit 1
}

# 1. Push the dated sync branch (non-force) — anchors NEWPIN: gc-safe, fetchable, reviewable.
git -C "$S" push "$OURL" "refs/heads/sync-v0.5.13-20260702:refs/heads/sync-v0.5.13-20260702"

# 2. Archive the old tip: branch (robust for `git submodule update`) + date tag.
git -C "$S" push "$OURL" "$OLD:refs/heads/sglang-miles-v0.5.12-final"
git -C "$S" tag -f "sglang-miles-v0.5.12-$D" "$OLD"
git -C "$S" push "$OURL" "refs/tags/sglang-miles-v0.5.12-$D"

# 3. Force-advance sglang-miles (lease-guarded to the archived OLD) + date-tag the new tip.
git -C "$S" push --force-with-lease=refs/heads/sglang-miles:"$OLD" "$OURL" "$NEWPIN:refs/heads/sglang-miles"
git -C "$S" tag -f "sglang-miles-v0.5.13-$D" "$NEWPIN"
git -C "$S" push "$OURL" "refs/tags/sglang-miles-v0.5.13-$D"

echo "DONE: sglang-miles -> $NEWPIN; old tip archived as sglang-miles-v0.5.12-final + tag."
echo "Next: push miles-imp branch sync-upstream-20260630 + open the PR (separate approval)."
