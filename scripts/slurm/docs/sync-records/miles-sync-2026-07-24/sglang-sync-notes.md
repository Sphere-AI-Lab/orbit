# sglang-sync notes — combined with miles-sync-2026-07-24

Target: `sgl-project/sglang@sglang-miles` = `94949da73` = **v0.5.15-31** (rebased line;
old pin `27d5e97c3` on the v0.5.13 line is NOT an ancestor → FORCE path).
New pin: **`38d4bbef5`** = target + 4 re-applied local patches.
torch unchanged (2.11.0) → ACTIVE bundle stays `cu129-x86_64-v0.5.12` (bundle-may-lag).

## Mirror-only commit classification (TGT..MIRROR = 52 commits)

- **5 local patches (ours)** — top of the stack:
  `27d5e97c3` qwen-vl pretokenized IDs (#4) → RE-APPLIED as `38d4bbef5`
  `2b778c2da` exact multimodal scoring suffix (#3) → RE-APPLIED as `cd7052764`
  `723ed7d19` pause-aware flush_cache restore → **DROPPED — subsumed by upstream #31962**
  (`is_fully_idle(ignore_waiting=self._engine_paused)` in v0.5.15's `flush_cache`; verified by content)
  `fe3f5fced` cu12 dep flavors → RE-PORTED as `3425836e1` (versions moved to the v0.5.15 dep set)
  `4f3aaf47a` mrope rl_on_policy_target gate → RE-APPLIED as `77f2c8db9` (verified: v0.5.15
  still lacks the disjunct; `get_rl_on_policy_target` machinery exists on the new base)
- **47 old-upstream-line commits** — the v0.5.13-era sglang-miles patch stack
  ([1/14]..[14/14], cherry-picks, R3/PD/PDMux fixes). Carried by sgl-project's own
  rebase onto v0.5.15; not ours to re-apply.

## Re-apply notes

- **cu12 flavors port** (`3425836e1`): `flashinfer_python[cu12]==0.6.12`,
  `sglang-kernel==0.4.4+cu129`, `sgl-deep-gemm==0.1.4+cu129`, `torchao==0.17.0+cu129`,
  `cuda-python>=12.9.4,<13`, plain `nvidia-cutlass-dsl==4.5.2`. All three +cu129 wheels
  verified present (docs.sglang.ai/whl/cu129 + download.pytorch.org/whl/cu129) BEFORE
  committing. Kept upstream's new deps as-is (flash-attn-4==4.0.0b15, nvidia-mathdx,
  smg-grpc-servicer, flashinfer_cubin).
- **exact scoring suffix** (`cd7052764`): 4 conflict files — all tail-of-parameter-list
  unions with upstream's new `session_id` plumbing; tokenizer_manager's
  TokenizedGenerateReqInput moved to kwargs style, our `scoring_suffix_len` kwarg appended.
  One shipped test needed `session_id=None` on its fake recv_req (v0.5.15's radix-native
  session route reads it).
- Unit tests on final pin: **54 passed + 13 subtests**
  (test_scoring_suffix{,_scheduler,_api}, test_pretokenized_input_ids, test_io_struct).

## Publication status and remaining landing plan

Completed:

- `sync-v0.5.15-20260724` is published on `impossible-inc/sglang`.
- The branch tip is `38d4bbef599e3375f143f48a27c659910ebdd064`,
  exactly matching the Miles gitlink.
- The mirror review is open as
  [impossible-inc/sglang#5](https://github.com/impossible-inc/sglang/pull/5).

Not yet executed:

- After PR #5 review, archive the old line and lease-guarded force-advance
  `sglang-miles`. Do not perform this landing while the PRs are only being
  opened for review.

```bash
S=thirdparty/sglang; NEWPIN=38d4bbef599e3375f143f48a27c659910ebdd064
# 1. review PR sync-v0.5.15-20260724 -> sglang-miles on impossible-inc/sglang (history track,
#    per PR #1 v0.5.12 / PR #2 v0.5.13 precedent) — BEFORE the force-advance
# 2. after review, archive old tip, then lease-guarded force-advance + date tags
OLD=$(git -C $S ls-remote origin sglang-miles | awk '{print $1}')   # expect 27d5e97c3
git -C $S push origin "$OLD:refs/heads/sglang-miles-v0.5.13-final"
git -C $S tag sglang-miles-v0.5.13-20260724 "$OLD" && git -C $S push origin sglang-miles-v0.5.13-20260724
git -C $S push --force-with-lease=sglang-miles:"$OLD" origin "$NEWPIN:refs/heads/sglang-miles"
git -C $S tag sglang-miles-v0.5.15-20260724 "$NEWPIN" && git -C $S push origin sglang-miles-v0.5.15-20260724
```

Note: this clone's submodule origin is ssh (`git@github.com:impossible-inc/sglang.git`)
and this user pushed to it over ssh on 2026-07-23 — the skill's token-URL dance is for
clones where origin is anonymous https.
