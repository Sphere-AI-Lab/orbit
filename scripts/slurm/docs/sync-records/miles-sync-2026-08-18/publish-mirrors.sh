#!/usr/bin/env bash
# miles-sync 2026-08-18 — outward pushes for the combined miles+sglang sync.
# RUN ONLY AFTER EXPLICIT APPROVAL. Order matters: mirrors first, so the
# gitlinks in the miles PR branch resolve for fresh clones.
set -euo pipefail

: "${GH_TOKEN:?set GH_TOKEN for the private impossible-inc https remotes}"

# ---- 1. impossible-inc/sglang: archive old sglang-miles, force-move to the rebased tip
cd "$(git rev-parse --show-toplevel)/thirdparty/sglang"
SGLANG_URL="https://x-access-token:${GH_TOKEN}@github.com/impossible-inc/sglang.git"
OLD_TIP=14f2a7cb11a6580bb9f70a6d6e73e54738cc7db2   # sglang-miles-v0.5.15-20260727-2
NEW_TIP=36982fef0b44e07c80d37b1ef152599493912be3   # v0.5.16 (cb05a44f3) + 7 re-applied local patches

git fetch "$SGLANG_URL" "refs/heads/sglang-miles:refs/remotes/mirror/sglang-miles"
ACTUAL=$(git rev-parse mirror/sglang-miles)
if [ "$ACTUAL" != "$OLD_TIP" ]; then
    echo "FATAL: mirror sglang-miles at $ACTUAL, expected $OLD_TIP — re-inspect before archiving" >&2
    exit 1
fi
# archive (branch + date tag) so SHA-pinned gitlinks in old miles commits stay reachable
git push "$SGLANG_URL" "$OLD_TIP:refs/heads/sglang-miles-v0.5.15-final"
git push "$SGLANG_URL" "$OLD_TIP:refs/tags/sglang-miles-v0.5.15-20260818-archive"
# force-move the line to the rebased tip (as sgl-project maintains the branch)
git push --force-with-lease=sglang-miles:"$OLD_TIP" "$SGLANG_URL" "$NEW_TIP:refs/heads/sglang-miles"
# publish the local re-apply branch name too, for review convenience
git push "$SGLANG_URL" "$NEW_TIP:refs/heads/sync-v0.5.16-20260818"

# ---- 2. impossible-inc/Megatron-Bridge: fast-forward bridge to upstream's TE-2.17 pin
cd ../Megatron-Bridge
BRIDGE_URL="https://x-access-token:${GH_TOKEN}@github.com/impossible-inc/Megatron-Bridge.git"
NEW_BRIDGE=7f0fb3456f8ffe47599b5fd167b454605d85f932
git push "$BRIDGE_URL" "$NEW_BRIDGE:refs/heads/bridge"   # plain push: fast-forward from c092daca

echo "mirrors published; now push the miles branch and open the PR"
