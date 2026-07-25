<!-- PR: sync-v0.5.15-20260724 -> sglang-miles on impossible-inc/sglang -->
<!-- Title: [sync] sglang-miles v0.5.13 -> v0.5.15-31 + local patch stack -->

## Summary

Advance the mirror's `sglang-miles` to the current `sgl-project/sglang@sglang-miles`
tip **`94949da73` = v0.5.15-31** (rebased upstream line; torch stays 2.11.0), with the
local miles patch stack re-applied on top. Consumed by impossible-inc/miles-imp's
2026-07-24 upstream sync
([impossible-inc/miles-imp#41](https://github.com/impossible-inc/miles-imp/pull/41),
branch `sync-upstream-20260723`, pin `38d4bbef5`).

## Contents (base + 4 local patches)

| commit | patch | status |
|---|---|---|
| `94949da73` | upstream sglang-miles v0.5.15-31 base | — |
| `77f2c8db9` | [sglang-miles] forward_batch: gate mrope text-only path on rl_on_policy_target | re-applied (verified absent on v0.5.15; upstream candidate) |
| `3425836e1` | [sglang-miles cu129] bare-metal cu12 dep flavors | re-ported to the v0.5.15 dep set (kernel 0.4.4+cu129 / deep-gemm 0.1.4+cu129 / torchao 0.17.0+cu129 / flashinfer[cu12] 0.6.12 — all wheels verified to exist); mirror-only by design |
| `cd7052764` | feat: add exact multimodal scoring suffix (#3) | re-applied; conflicts were parameter-tail unions with v0.5.15's new `session_id` plumbing; upstream candidate |
| `38d4bbef5` | fix(qwen-vl): preserve pretokenized IDs in legacy multimodal loading (#4) | re-applied clean; upstream candidate |

**Retired**: `[sglang-miles] restore pause-aware flush_cache` — upstream fixed it
officially in #31962 (`is_fully_idle(ignore_waiting=self._engine_paused)`); verified by
content that the semantics fully cover our patch. The mirror stack shrinks by one.

Patch-shipped unit tests on the new tip: **54 passed + 13 subtests**
(scoring suffix ×3 files, pretokenized IDs, io_struct).

## Landing mechanics (FORCE case — upstream rebased the line)

This PR cannot merge via the UI (non-fast-forward vs the rebased base). Landing:

```bash
OLD=$(git ls-remote origin sglang-miles | awk '{print $1}')   # expect 27d5e97c3
git push origin "$OLD:refs/heads/sglang-miles-v0.5.13-final"  # archive branch
git tag sglang-miles-v0.5.13-20260724 "$OLD" && git push origin sglang-miles-v0.5.13-20260724
git push --force-with-lease=sglang-miles:"$OLD" origin 38d4bbef5:refs/heads/sglang-miles
git tag sglang-miles-v0.5.15-20260724 38d4bbef5 && git push origin sglang-miles-v0.5.15-20260724
```

The archive keeps every SHA-pinned gitlink in old miles-imp commits resolvable.
After the force-advance this PR auto-marks Merged.
