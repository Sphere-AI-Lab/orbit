# sglang-sync notes — combined with miles-sync-2026-07-24

Target: `sgl-project/sglang@sglang-miles` = `94949da73` = **v0.5.15-31** (rebased line;
old pin `27d5e97c3` on the v0.5.13 line is NOT an ancestor → FORCE path).
New pin: **`9dd80e5f8`** = target + 4 re-applied local patches + one test-only
CI-registration review follow-up.
torch unchanged (2.11.0) → ACTIVE bundle uses rolling `cu129-x86_64`.
Its Apex/FA2/FA3 SHA256 values match the job-28782 validated
`cu129-x86_64-v0.5.12` cache byte-for-byte after upstream retired that tag.

| rolling `cu129-x86_64` asset | SHA256 |
|---|---|
| `apex-0.1-cp312-cp312-linux_x86_64.whl` | `53b0a257f8099f7bb8472838331e5af79c9b16364ab950a18ad9b53bea66c45b` |
| `flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl` | `939d18fcef21db5c354390b353eb6f9f8815f1d681a57139d22b0b60e474c087` |
| `flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl` | `b0f4d97418aa129522cd4b4e65ce516ddf8af64815f4ce040cb38a6d94cef971` |

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
- Runtime unit tests on `38d4bbef5`: **54 passed + 13 subtests**
  (test_scoring_suffix{,_scheduler,_api}, test_pretokenized_input_ids,
  test_io_struct). Final pin `9dd80e5f8` only migrates the exact-suffix E2E to
  v0.5.15 CI registration metadata; `check_registered_tests.py` and the
  file-scoped SGLang pre-commit hooks pass.

## Publication and landing status

Completed:

- `sync-v0.5.15-20260724` is published on `impossible-inc/sglang`.
- The branch tip is `9dd80e5f8fc52e87dd84a43ebc0e125cd4f4c9d8`,
  exactly matching the Miles gitlink.
- The mirror review
  [impossible-inc/sglang#5](https://github.com/impossible-inc/sglang/pull/5)
  was approved and recorded as Merged on 2026-07-27.
- The old `27d5e97c3b26127d2282900823a4abd172a3b6d5` tip is retained by
  `sglang-miles-v0.5.13-final` and `sglang-miles-v0.5.13-20260727`.
- `sglang-miles` and `sglang-miles-v0.5.15-20260727` both resolve to
  `9dd80e5f8fc52e87dd84a43ebc0e125cd4f4c9d8`; the branch move used
  `--force-with-lease=sglang-miles:27d5e97c3b26127d2282900823a4abd172a3b6d5`.
- The consuming Miles sync is
  [impossible-inc/miles-imp#41](https://github.com/impossible-inc/miles-imp/pull/41).

Note: this clone's submodule origin is ssh (`git@github.com:impossible-inc/sglang.git`)
and this user pushed to it over ssh on 2026-07-23 — the skill's token-URL dance is for
clones where origin is anonymous https.
