## Summary

Sync with upstream `radixark/miles` — **38 upstream commits** merged.
Period: since `7a6cf48e6` (2026-05-19, *add Code of Conduct (#1145)*).

Branch contains two merge commits: the main `ab58adbdc` (37 commits from `7a6cf48e6..7deb4b744`), and a follow-up `05c0baf2b` that patches in upstream's one-commit pre-commit hotfix (`fe89c8d0e Fix pre-commit formatting #1263`) which landed after we drafted the initial sync. Each upstream SHA is preserved individually in `git log` so future `merge-base` detection works.

## Upstream PRs merged

See [`prs.md`](prs.md) for the full per-PR analysis (covers the first 37). Bucket summary:

- **⚠️ Watchlist (pin sources): 1** — #1164 Bump sglang to v0.5.12 (touches `docker/Dockerfile` and `miles/ray/rollout.py`).
- **Flagged (touch files we've modified): 8** — concentrated in `miles/utils/arguments.py` (#963 DeepSeek v3.2, #1216 hf_config refactor, #1219/#1222/#1223 Kimi 2.5 LoRA, #1243 precommit ban) and CI workflow refactor (#1149, #1173 — these caused the only structural conflict).
- **Other (routine): 28** — CI tweaks, code-owner additions, small fixes.
- **Hotfix (added after initial drafting): 1** — #1263 Fix pre-commit formatting. Auto-formats two files we got from this same sync window (`examples/eval/terminal_bench_via_agent_server/eval_tb_deepseek_v4_pro.py`, `scripts/run_qwen3_4b_npu.py`). Merged in cleanly with no conflicts.

Notable upstream features now in miles-imp:

- **DeepSeek v3.2 support** (#963, #1213): config, chat template encoder.
- **Kimi K2.5 VL full + LoRA** (#1219–#1223, 5-PR series): VL-aware quantization, shared-outer grouped-expert LoRA, LoRA base-weight CPU backup.
- **Loss refactor** (#753 / #1121 / #1132, 3-PR series): snapshot tests + file restructure; existing call sites unchanged.
- **CI refactor** (#1149): hardware-classified stages (`stage-a-cpu`, `stage-b-2-gpu-h200`, `stage-c-8-gpu-h100`, `stage-c-4-gpu-h200`, `stage-c-2-gpu-h200`) with label-based `--match-all-labels` filtering.
- **Terminal-Bench eval driver** (#1225, #1236): new agent-server-based eval/training examples.
- **NPU support for qwen3-4B** (#1125): new docker patches under `docker/npu_patch/`.

## Conflict resolutions

- `.gitignore`: **union** — kept our slurm-launcher / debug-notes / runs / wheels / scheduled_tasks.lock entries, added upstream's `.humanize/`.
- `.github/workflows/pr-test.yml`: **took upstream wholesale**, then re-applied our "no-GPU-runner gating" pattern on top. All 4 GPU stages (`stage-b-2-gpu-h200`, `stage-c-8-gpu-h100`, `stage-c-4-gpu-h200`, `stage-c-2-gpu-h200`) now require the `run-ci-gpu` label on the PR to run. CPU stages (`stage-a-cpu`, `stage-b-cpu`) run unconditionally.

Six other files auto-merged cleanly (no conflict markers): `.pre-commit-config.yaml`, `miles/ray/rollout.py`, `miles/rollout/sglang_rollout.py`, `miles/utils/arguments.py`, `tests/fast-gpu/test_mxfp8_quantizer.py`, `tests/fast-gpu/test_nvfp4_quantizer.py`.

## Pin / install script changes

- `scripts/slurm/setup/pins.env`: `MILES_WHEELS_TAG` `cu129-x86_64` → `cu129-x86_64-v0.5.12`. Regenerated via `python scripts/slurm/setup/extract_pins.py --write`.
- `scripts/slurm/setup/install_env.sh`: **no changes needed.** No new `RUN`/`ARG`/`ENV` lines in upstream Dockerfile beyond the WHEELS_TAG bump that pins.env already absorbed.
- `thirdparty/sglang/python/pyproject.toml`: **not touched by upstream** — `TORCH_VERSION` unchanged; no sglang-sync workflow needed yet.

## ⚠️ Attention items

- `miles/utils/arguments.py` had 5 upstream PRs touching it during this sync window. Auto-merge succeeded textually, but **please verify semantically** before merging — the new DeepSeek/Kimi/hf_config code paths may have subtle interactions with our local additions.
- `miles/ray/rollout.py` was touched by #1164 (sglang bump). Auto-merged cleanly but worth a once-over since #1164 was a watchlist PR.
- The new `run-ci-gpu` label gating means upstream's CI refactor (#1149) will silently skip ALL GPU stages on PRs without the label, even on impossible-inc/miles-imp. Remove the gate (or attach a self-hosted GPU runner pool to the org) when GPU CI becomes feasible.

## Divergence from upstream after sync

See [`divergence.stat`](divergence.stat) for the full file-level stat (68 files changed, +8088/-42). Top divergence buckets:

- `scripts/slurm/{setup,docs,lib}/` — slurm launcher + env setup (~3.5k lines)
- `examples/vagen/` — Sokoban + FrozenLake multi-turn RL example
- `scripts/experiments/` — recipe shell scripts
- `.claude/skills/{slurm-launch,rl-monitor-loop}/` — workflow skills (`miles-sync` and `miles-upstream-prs` are untracked locally, not on this branch yet)
- `miles/backends/megatron_utils/update_weight_from_distributed/broadcast.py` — bridge-mode VLM weight sync
- `tests/.../test_update_weight_from_distributed_bridge.py` — companion test

## Test plan

- [ ] Run `bash scripts/slurm/setup/install_env.sh` in a fresh GPU salloc (or check that an existing miles env still works after `extract_pins.py --check` passes).
- [ ] Run `python scripts/slurm/setup/verify_env.py` and confirm all checks pass.
- [ ] Sanity-launch a recipe (e.g. `bash scripts/slurm/submit.sh scripts/experiments/qwen3-4B-disagg-1node.sh`) and let it reach the first eval.
- [ ] Semantic review of the 6 auto-merged files (esp. `miles/utils/arguments.py`).

⚠️ **Merge mode**: this PR MUST be merged via "Create a merge commit". Squash or rebase will break future `merge-base` detection in `/miles-sync`.
