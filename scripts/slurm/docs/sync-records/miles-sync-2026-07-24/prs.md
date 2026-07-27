# miles upstream PRs report — 2026-07-24

**Period**: merge-base `5e97c865` (2026-06-29, tip of the miles-sync-2026-06-30 event) → `upstream/main` = `c94c2fa9` (2026-07-23)
**Total**: 187 upstream commits (~186 PRs, squash-merged 1:1)
**Watchlist hits (pin sources)**: 12
**Flagged (touch files we've modified)**: 72
**Clean (no overlap)**: 103
**Merge dry-run**: 20 conflicted files (throwaway-worktree experiment, aborted; see section below)

> Method note: `gh` CLI is unavailable on this host, so PR metadata comes from the
> squash-merge commit history (title/author/date are equivalent for this repo). The
> HEAD side of this analysis is `sync-upstream-20260723` = main + the OPD PRs (#33-#40).

---

## ⚠️ Watchlist hits (pin sources) — 12

Cumulative `docker/Dockerfile` + `requirements.txt` delta (merge-base → tip):

**`docker/Dockerfile`**
- `SGLANG_IMAGE_TAG`: `v0.5.13` → **`v0.5.15`** (#1587 →v0.5.14, #1655 →v0.5.15 — a TWO-version sglang jump).
- `SGLANG_BRANCH`: `sglang-miles-v0.5.15` → **`sglang-miles`** (#1701; interim pin during the bump, now back on the floating branch).
- `WHEELS_TAG_X86`: `cu130-x86_64-v0.5.12` → **`cu130-x86_64`** (suffix-less; #1655). Wheels releases are now named by CUDA+arch only — they are torch-ABI-bound, the sglang-version suffix is gone. `WHEELS_TAG_ARM64` likewise.
- **`RUN pip install "numpy<2"` REMOVED** (#1587) — upstream no longer forces numpy 1.x for Megatron. Our `install_env.sh` still pairs `scipy<1.16` with `numpy<2`; decide whether to follow.
- `torch_memory_saver`: pinned `@d64a639` → **unpinned tip, then `6d5bce48`** (#1773) + `TMS_CUDA_MAJOR` derived from torch for the source build (#1774).
- **+ `Emerging-Optimizers v0.1.0`** (#1577) — required by Megatron's Muon optimizer (`megatron/core/optimizer/muon.py`); NOT on PyPI (dependency-confusion stub), installed from git.
- fast-hadamard-transform / causal-conv1d / mamba-ssm now prefer prebuilt `/tmp/wheels/*.whl` with source-build fallback (#1779) — docker-layer concern, minor for bare-metal.

**`requirements.txt`**
- **+ `transformers<5.13`** (#1574) — 5.13.0 registers `qwen3_asr` natively and collides with sglang's `AutoConfig.register(exist_ok=False)`; lift once sglang registers with `exist_ok=True`. ⚠️ our env runs transformers 5.x — check the installed version against the cap at env-refresh time.
- **+ `blake3==1.0.9`, `xxhash==3.7.1`, `zstandard==0.25.0`** (#1235 disk-delta weight sync: checksum + codec).
- **+ `nvidia-resiliency-ext~=0.6.0`** (#1598, fault-tolerance series).
- **+ `psycopg[binary]`** (#1517, metric-history CI gate / Neon Postgres).
- `polars` → `polars==1.42.1` (#1631, versioned-requirements CI move).

Per-PR detail:

| PR | date | author | subject | watch files |
|---|---|---|---|---|
| #1574 | 2026-07-05 | Jiajun Li | fix(deps): cap transformers<5.13 — 5.13.0 breaks all CPU CI via qwen3_asr collision (#1574) | `requirements.txt` |
| #1587 | 2026-07-06 | Yueming Yuan | Bump sglang to v0.5.14 (#1587) | `docker/Dockerfile` |
| #1598 | 2026-07-09 | fzyzcjy | Add nvidia-resiliency-ext to requirements (#1598) | `requirements.txt` |
| #1235 | 2026-07-08 | Nan Jiang | feat: disk-delta weight sync for non-colocated rollout engines (#1235) | `requirements.txt` |
| #1631 | 2026-07-15 | Jiajun Li | fix(ci): move dependency setup into versioned requirements (#1631) | `docker/Dockerfile`, `requirements.txt` |
| #1655 | 2026-07-16 | Yueming Yuan | Bump sglang to v0.5.15 (#1655) | `docker/Dockerfile` |
| #1701 | 2026-07-16 | Yueming Yuan | Point SGLANG_BRANCH back to sglang-miles (#1701) | `docker/Dockerfile` |
| #1577 | 2026-07-18 | Zhichen Zeng | fix(docker): install emerging_optimizers for Muon optimizer support (#1577) | `docker/Dockerfile` |
| #1517 | 2026-07-20 | Jiajun Li | (3/4) feat(ci): wire metric-history gate into harness + psycopg NeonStore (M3) (#1517) | `requirements.txt` |
| #1773 | 2026-07-23 | Yuzhen Zhou | chore(docker): upgrade torch_memory_saver to 6d5bce48 (#1773) | `docker/Dockerfile` |
| #1774 | 2026-07-23 | Zhichen Zeng | fix(docker): set TMS_CUDA_MAJOR for torch_memory_saver source build (#1774) | `docker/Dockerfile` |
| #1779 | 2026-07-23 | Jiajun Li | chore(docker): install prebuilt aarch64 wheels for mamba/fast-hadamard kernels (#1779) | `docker/Dockerfile` |

---

## sglang gate pre-assessment (miles-sync Step 5d forecast)

`[sglang-sync pending]` WILL fire after the merge (upstream wants v0.5.15; ACTIVE is the v0.5.13-line bundle). Pre-checked the target line (read-only):

- `sgl-project/sglang@sglang-miles` tip = **`94949da73` = `v0.5.15-31`** — REBASED line (our pin `27d5e97c3` is NOT an ancestor → sglang-sync Step 3 FORCE path: archive old tip, re-apply local patches, force-advance the mirror).
- **torch stays `2.11.0`** — NO ABI jump. Same-torch bump ⇒ the bundle-may-lag rule applies; no fresh-env rebuild forced by ABI (still rerun install to pick up dep drift).
- miles-wheels now publishes suffix-less releases: **`cu129-x86_64`** exists (apex, FA2 2.7.4.post1, FA3 3.0.0b1, sglang_router 0.3.2, gateway, **+ transformer_engine 2.10.0 wheel — new asset**). At report time the old `cu129-x86_64-v0.5.12` still existed; upstream #1784 subsequently retired it and moved the cu12 consumer to the rolling tag.
- **Five local mirror patch candidates classified; four carried onto v0.5.15** (three were active at the 06-30 sync):
  1. mrope text-only gate (`rl_on_policy_target`) — geo3k VLM fix; verify content on the new base.
  2. cu12 dep flavors (mirror-only by design; cu13-linked PyPI defaults still can't load on the CUDA-12.8 driver).
  3. pause-aware `flush_cache` restore — **LIKELY SUBSUMED**: upstream's new tip commit is `[sglang-miles] Fix flush_cache() no-op after pause_generation in retract (#31962)`. Verify by CONTENT (grep `_engine_paused and self.running_batch.is_empty` semantics), then drop ours if covered.
  4. exact multimodal scoring suffix (mirror PR #3) — OPD teacher scoring; must carry.
  5. qwen-vl pretokenized-IDs retokenize fix (mirror PR #4) — multiturn exact-history; must carry. Check whether upstream v0.5.15 threaded `input_ids` in `legacy_load_mm_data` itself.
- **`extract_pins.py` work item**: `WHEELS_STACK` rows key on versioned tags and `[sglang-sync pending]` compares the *version component* of wheels tags — the new suffix-less naming has no version component, so the comparator and the WHEELS_STACK schema need a code adjustment during Step 5, not just new rows.

---

## Merge dry-run — 20 conflicted files

`git merge upstream/main` exercised in a throwaway worktree (aborted afterwards; nothing kept). Hunk counts are textual conflict regions; semantic risk noted per file. This sync has a much larger conflict surface than 06-30 (3 files) because main now carries the OPD series (#23-#40) and upstream landed the fault-tolerance megaseries into the same core files.

| file | hunks | upstream side | our side | expected resolution |
|---|---|---|---|---|
| `.github/workflows/pr-test.yml` | 4 | CI stages: H200 8-GPU (#1626,#1629), FT tag scopes (#1632), run_suite scopes (#1644) | GPU-label gates `run-ci-*-gpu` (644c9309) | take upstream, re-AND our GPU-hardware label gates (same as 06-30 conflict #1) |
| `miles/utils/arguments.py` | 3 | 29 PRs: FT flags, multi-lora, dashboard, dual-clip, sessions | OPD args + validators (#26,#29,#33,#35,#38) | weave both — expect OPD arg-group + validation regions |
| `miles/backends/megatron_utils/model.py` | 2 | FT series (15 PRs: in-mem ckpt, witness, indep-DP allreduce, ckpt scheduler #1627) | OPD trainer-side Top-K distillation (#25), VLM bringup | keep both; verify OPD loss path against FT refactors |
| `miles/backends/training_utils/data.py` | 2 | ⚠️ #1754 OPD response-signal CP alignment; FT CP-token helper (#1422,#1423); multi-lora batch routing (#1744) | OPD top-k data path (#25) | SEMANTIC: #1754 touches the OPD signals our top-k rides on — reconcile by hand |
| `miles/utils/data.py` | 1 | ⚠️ #1767 filter_long_prompt restructure (same bug class) | our #34 fix: reuse `sample.multimodal_inputs`, skip-paths return samples | take upstream structure, KEEP our stored-inputs reuse (upstream still re-runs process_vision_info on templated prompts) |
| `examples/fully_async/fully_async_rollout.py` | 1 | #1677 `--async-max-concurrent-samples` | John's fail-closed version query + `_active_tasks` (47fd02f8) | keep both: adopt the flag, keep fail-closed collector |
| `train.py` | 2 | FT control server (#1453), py-spy (#1461), sentinel ckpt (#1570), pre-train eval fix (#1579) | envpack/watchdog/heartbeat (#11,#15) | keep both |
| `train_async.py` | 1 | same FT/ckpt series | watchdog hardening (#15) | keep both |
| `miles/backends/megatron_utils/update_weight/.../broadcast.py` | 1 | #1608 contiguous tensors, #1395 always-reconnect, #1444 staleness, multi-lora #1745 | bridge-mode VLM export (#6) | keep both (same spot as 06-30 conflict #2) |
| `miles/backends/megatron_utils/model_provider.py` | 1 | #1483 args→bridge config, #1559 GLM-5 LoRA, #1405 witness params | freeze-vision block (#14) | keep both |
| `miles/ray/rollout/rollout_manager.py` | 1 | FT wiring (#1449,#1450), multi-lora buffers (#1747), dashboard (#1654) | OPD scoring transport (#24) | keep both |
| `miles/ray/rollout/router_manager.py` | 1 | session refactor (#1569), N-instance scale-out (#1659), /health drop (#1602) | 600s readiness timeout (84f65a8d) | keep both; verify timeout survives the refactor |
| `miles/ray/rollout/train_data_conversion.py` | 1 | #1397 DP-split delay, #1595, multi-lora #1747, dashboard | OPD fields (#25) | keep both |
| `miles/ray/rollout/metrics.py` | 1 | #1649 pass@k move, #1706 None-reward guard | OPD scoring telemetry (#23-#25,#35,#38) | keep both |
| `miles/backends/training_utils/log_utils.py` | 2 | FT log context, multi-lora per-adapter logs, #1649 | OPD metrics (#25,#36) | keep both |
| `miles/utils/misc.py` | 1 | #1386 FT tweaks, #1639 teardown hook | watchdog helpers (#15) | keep both |
| `miles/utils/tracking_utils/base.py` | 2 | metric-gate #1523, multi-lora #1742, dashboard #1654 | no-backend guard + monotonic W&B step (#3,#15) | keep both (same file as 06-30 conflict #3) |
| `miles/ray/actor_group.py` | 1 | FT series (8 PRs: factory, witness, indep-DP) | per-user triton JIT cache dirs (d477a5c6) | keep both |
| `tests/fast/utils/test_arguments.py` | 2 | FT/multi-lora/gate arg tests | OPD arg tests (#26-#38) | keep both |
| `tests/fast/ray/rollout/test_metrics.py` | 1 | #1649 pass@k tests | OPD telemetry tests (#23-#38) | keep both |

No `thirdparty/*` gitlink conflicts this time (upstream still doesn't track them).

---

## Flagged PRs (touch files we've modified) — 72, grouped

### fault-tolerance megaseries (fzyzcjy) — flag-gated, biggest core-file churn — 38 PRs

| PR | date | subject | overlap (ours) |
|---|---|---|---|
| #1386 | 2026-07-10 | Add fault-tolerance support tweaks to shared utilities (#1386) | `misc.py` |
| #1394 | 2026-07-10 | Add the fault-tolerance dependency, CI label, and logger-config setup (#1394) | `arguments.py` |
| #1395 | 2026-07-10 | Always reconnect rollout engines on weight-update setup (#1395) | `broadcast.py` |
| #1396 | 2026-07-10 | Add a fault-injection RPC to train actors (#1396) | `actor_group.py` |
| #1397 | 2026-07-10 | Delay splitting train data by DP until actor-side processing (#1397) | `actor_group.py`, `rollout_manager.py`, `train_data_conversion.py`, `arguments.py` +3 |
| #1398 | 2026-07-10 | Add a deterministic NCCL backend for order-stable collectives (#1398) | `arguments.py` |
| #1401 | 2026-07-10 | Add structured event logging keyed by per-process identity (#1401) | `rollout_manager.py`, `arguments.py`, `train.py`, `train_async.py` |
| #1402 | 2026-07-10 | Add event-log snapshot and restore checkpointing (#1402) | `arguments.py` |
| #1404 | 2026-07-10 | Add the witness id allocator (#1404) | `arguments.py` |
| #1405 | 2026-07-10 | Trace witness ids through the model via injected witness parameters (#1405) | `model_provider.py` |
| #1408 | 2026-07-10 | Add the event-log analyzer that applies analysis rules (#1408) | `arguments.py` |
| #1414 | 2026-07-10 | Add an in-memory (non-persistent) checkpoint manager (#1414) | `model.py` |
| #1416 | 2026-07-10 | Extract actor construction into a shared allocate_gpus_for_actor factory (#1416) | `actor_group.py` |
| #1417 | 2026-07-10 | Thread independent-DP / role / cell-index / rollout-manager context through the train group (#1417) | `actor_group.py`, `rollout_manager.py`, `train.py`, `train_async.py` |
| #1419 | 2026-07-10 | Add the cell-based independent-DP train group selected via experimental flag (#1419) | `actor_group.py`, `arguments.py` |
| #1420 | 2026-07-10 | Drop the legacy self.rollout_engines initialization from MegatronTrainRayActor (#1420) | `model.py` |
| #1422 | 2026-07-10 | Extract the CP-aware token-id transform into a shared helper (#1422) | `data.py` |
| #1423 | 2026-07-10 | Thread witness ids through the training data path (#1423) | `model.py`, `data.py`, `log_utils.py`, `actor_group.py` +1 |
| #1424 | 2026-07-10 | Make the tensor dumper fault-tolerance aware (#1424) | `model.py` |
| #1426 | 2026-07-10 | Support in-memory (non-persistent local) checkpoints (#1426) | `model.py` |
| #1427 | 2026-07-10 | Create independent-DP process groups at init (#1427) | `test_ulysses_cp_utils.py` |
| #1430 | 2026-07-10 | Route training collectives through the effective-DP group spanning replicas (#1430) | `model.py`, `data.py`, `log_utils.py` |
| #1431 | 2026-07-10 | Report the first-replica main rank for logging under independent DP (#1431) | `model.py` |
| #1432 | 2026-07-10 | Allreduce grads and losses across independent-DP replicas (#1432) | `model.py` |
| #1433 | 2026-07-10 | Reach intra-cell consensus on cross-replica allreduce and discard on disagreement (#1433) | `model.py` |
| #1434 | 2026-07-10 | Inject FT test actions to deterministically exercise fault tolerance (#1434) | `model.py`, `arguments.py` |
| #1435 | 2026-07-10 | Dump per-rank per-step local weight checksums during training (#1435) | `model.py`, `arguments.py` |
| #1442 | 2026-07-10 | Wire per-cell heartbeat health monitoring into the train group (#1442) | `arguments.py` |
| #1444 | 2026-07-10 | Track rollout-engine connection staleness on the weight updater (#1444) | `broadcast.py` |
| #1446 | 2026-07-10 | Inject witness ids into the Megatron forward and train step (#1446) | `model.py` |
| #1449 | 2026-07-10 | Add CI rollout-data injection with recorded-data metadata round-trip (#1449) | `rollout_manager.py`, `arguments.py` |
| #1450 | 2026-07-10 | Wire FT event logging and component gating into RolloutManager (#1450) | `rollout_manager.py`, `arguments.py`, `test_arguments.py` |
| #1453 | 2026-07-10 | Start HTTP control server and mini FT controller in train entrypoints (#1453) | `arguments.py`, `train.py`, `train_async.py` |
| #1454 | 2026-07-10 | Always save rollout debug data regardless of rollout_global_dataset (#1454) | `train.py`, `train_async.py` |
| #1455 | 2026-07-10 | Add debug-exit-after-rollout to train entrypoints (#1455) | `arguments.py`, `train.py`, `train_async.py` |
| #1461 | 2026-07-10 | Add opt-in periodic py-spy dumper for hang debugging (#1461) | `train.py`, `train_async.py` |
| #1595 | 2026-07-10 | Remove the dead pre-loop prompt assignment in split_train_data_by_dp_raw (#1595) | `train_data_conversion.py` |
| #1596 | 2026-07-10 | Move FT modules into topic folders (mechanical) (#1596) | `model.py`, `model_provider.py`, `data.py`, `log_utils.py` +7 |

### multi-LoRA series (Mathew Han, 7 parts) — 5 PRs

| PR | date | subject | overlap (ours) |
|---|---|---|---|
| #1742 | 2026-07-21 | [1/7][multi-lora]: utils foundation — sample/adapter types, adapter yaml config, shared helpers, CLI flags and validation (#1742) | `sample_utils.py`, `arguments.py`, `base.py`, `types.py` +1 |
| #1744 | 2026-07-21 | [3/7][multi-lora]: trainer core - per-slot optimizers, per-adapter LR schedules, slot lifecycle, batch routing in get_batch (#1744) | `model.py`, `data.py`, `log_utils.py` |
| #1745 | 2026-07-21 | [4/7][multi-lora]: weight sync and actor integration - per-adapter upsert push to engines, reconcile/train/save hooks (#1745) | `broadcast.py`, `actor_group.py` |
| #1746 | 2026-07-21 | [5/7][multi-lora]: rollout request routing - per sample lora_path/rid, per-adapter rewards, prefill logprob grouping (#1746) | `prefill_logprobs.py`, `sglang_rollout.py`, `test_prefill_logprobs.py` |
| #1747 | 2026-07-21 | [6/7][multi-lora]: async batch collection and data conversion - per-adapter buffers, round robin collection, batch metadata (#1747) | `rollout_manager.py`, `train_data_conversion.py` |

### session-server refactor / scale-out (Jiajun Li) — 3 PRs

| PR | date | subject | overlap (ours) |
|---|---|---|---|
| #1569 | 2026-07-03 | (1/7) refactor(session): drop the session_ prefix — rename errors/types/server modules (#1569) | `router_manager.py`, `test_openai_endpoint_utils.py` |
| #1602 | 2026-07-15 | (4/N) fix(rollout): drop the per-session /health probe that stalls session creation under load (#1602) | `router_manager.py`, `test_openai_endpoint_utils.py` |
| #1659 | 2026-07-15 | (2/N) feat(session): scale out as N session-server instances on a port range (#1659) | `router_manager.py`, `arguments.py`, `test_openai_endpoint_utils.py` |

### metric-history CI gate (Jiajun Li, 4 parts) — 2 PRs

| PR | date | subject | overlap (ours) |
|---|---|---|---|
| #1516 | 2026-07-20 | (2/4) feat(ci): metric-history gate — offline historical gate + register_ci_gate (M2) (#1516) | `arguments.py` |
| #1523 | 2026-07-20 | (1/4) feat(ci): metric-history gate foundation — storage contract + collection backend (M0+M1) (#1523) | `base.py` |

### CI-only — 4 PRs

| PR | date | subject | overlap (ours) |
|---|---|---|---|
| #1626 | 2026-07-10 | Add stage-c-8-gpu-h200 CI stage for full-node 8-GPU H200 runners (#1626) | `pr-test.yml` |
| #1629 | 2026-07-10 | fix(ci): partition 8-GPU H200 stage across two runners (#1629) | `pr-test.yml` |
| #1632 | 2026-07-12 | fix(ci): separate image and nightly FT tag scopes (#1632) | `pr-test.yml` |
| #1644 | 2026-07-14 | refactor(ci): resolve broad CI scopes in run_suite.py, not per-stage YAML (#1644) | `pr-test.yml` |

### individual flagged PRs — 20 PRs

| PR | date | subject | overlap (ours) |
|---|---|---|---|
| #1464 | 2026-07-01 | Enable observe training entropy without computing entropy loss (#1464) | `logit_processors.py`, `losses.py`, `math_utils.py`, `arguments.py` +1 |
| #1483 | 2026-06-29 | fix(megatron): propagate training-infra args to bridge model config (#1483) | `model_provider.py` |
| #1512 | 2026-06-30 | Megatron e2e: weight-check skip-list, Qwen3.5 MTP cases (#1512) | `rollout_manager.py`, `server_group.py`, `arguments.py`, `train.py` +1 |
| #1559 | 2026-07-07 | Add GLM-5/5.1/5.2 (744B MoE) LoRA RL support  (#1559) | `model_provider.py`, `broadcast.py`, `generate_endpoint_utils.py`, `prefill_logprobs.py` +2 |
| #1570 | 2026-07-20 | Support externally-triggered checkpoint save via sentinel file (#1570) | `arguments.py`, `train.py`, `train_async.py` |
| #1579 | 2026-07-07 | Fix pre-train eval trigger when start_rollout_id != 0 (#1579) | `train.py` |
| #1586 | 2026-07-10 | feat(megatron): add post-save checkpoint hook (#1586) | `arguments.py`, `test_arguments.py` |
| #1608 | 2026-07-20 | [fix] make tensors contiguous before broadcast weight sync (#1608) | `broadcast.py` |
| #1627 | 2026-07-15 | fix(megatron): avoid checkpoint scheduler double step (#1627) | `model.py` |
| #1639 | 2026-07-16 | rollout: decouple oversampling-abort teardown into a pluggable agent hook (#1639) | `inference_rollout_train.py`, `sglang_rollout.py`, `misc.py` |
| #1649 | 2026-07-21 | fix(metrics): move pass@k from trainer to rollout manager (#1649) | `log_utils.py`, `metrics.py`, `test_metrics.py` |
| #1654 | 2026-07-22 | feat: miles dashboard (#1654) | `rollout_manager.py`, `train_data_conversion.py`, `sample_utils.py`, `sglang_rollout.py` +4 |
| #1672 | 2026-07-15 | fix(rollout): stop merge_samples at routing-replay gap from aborted turns (#1672) | `sample_utils.py` |
| #1677 | 2026-07-23 | Add --async-max-concurrent-samples to decouple fully-async generation concurrency from batch size (#1677) | `fully_async_rollout.py`, `arguments.py` |
| #1703 | 2026-07-20 | rollout: consistent_hashing/manual routing for inference_rollout stack (#1703) | `generate_endpoint_utils.py`, `prefill_logprobs.py`, `sample_utils.py`, `inference_rollout_common.py` +2 |
| #1706 | 2026-07-18 | fix(eval): guard eval reward aggregation against None rewards (#1706) | `metrics.py` |
| #1727 | 2026-07-20 | [feat] --check-lora-weight-equal LoRA adapter weight-sync checker (#1727) | `arguments.py` |
| #1729 | 2026-07-20 | [algo] dual-clip PPO wiring and debug-disable-optimizer guard (#1729) | `losses.py`, `test_true_on_policy_loss_metrics.py` |
| #1754 | 2026-07-22 | fix(opd): align response signals with context parallelism (#1754) | `data.py` |
| #1767 | 2026-07-23 | fix: filter_long_prompt invalid return types and multimodal handling (#1767) | `data.py` |

---

## Other PRs (no overlap detected) — 103

| PR | date | author | subject |
|---|---|---|---|
| #1340 | 2026-07-22 | Zhihao Wang | [feat] Deepseek V4 Mxfp8 (#1340) |
| #1358 | 2026-07-07 | Jiajun Li | [CI] enable p2p rdma fallback ci (#1358) |
| #1384 | 2026-07-10 | fzyzcjy | Add a deterministic_random reward type (#1384) |
| #1385 | 2026-07-10 | fzyzcjy | Add an inplace_modify_args context manager (#1385) |
| #1387 | 2026-07-10 | fzyzcjy | Preserve the process-group backend across reload (#1387) |
| #1388 | 2026-07-10 | fzyzcjy | Add fault-tolerance foundation utilities (#1388) |
| #1389 | 2026-07-10 | fzyzcjy | Add structured logfmt logging helper (#1389) |
| #1390 | 2026-07-10 | fzyzcjy | Add a Clock abstraction with a fake clock for tests (#1390) |
| #1391 | 2026-07-10 | fzyzcjy | Add a fault injector test utility (#1391) |
| #1392 | 2026-07-10 | fzyzcjy | Add control-server data models (#1392) |
| #1393 | 2026-07-10 | fzyzcjy | Add a cell health checker and heartbeat utilities (#1393) |
| #1399 | 2026-07-10 | fzyzcjy | Add a per-process identity helper (#1399) |
| #1400 | 2026-07-10 | fzyzcjy | Add structured event models keyed by per-process identity (#1400) |
| #1403 | 2026-07-10 | fzyzcjy | Log training metrics as MetricEvents through the event logger (#1403) |
| #1406 | 2026-07-10 | fzyzcjy | Add event-log checksum-consistency analysis rules (#1406) |
| #1407 | 2026-07-10 | fzyzcjy | Add an event-log witness-tracing analysis rule (#1407) |
| #1409 | 2026-07-10 | fzyzcjy | Add dump and inference-engine-checksum comparison helpers for FT tests (#1409) |
| #1410 | 2026-07-10 | fzyzcjy | Add metric comparison helpers for FT tests (#1410) |
| #1411 | 2026-07-10 | fzyzcjy | Add reconfiguration assertions for fault-tolerance tests (#1411) |
| #1412 | 2026-07-10 | fzyzcjy | Relocate GroupInfo into shared process-group utilities (#1412) |
| #1413 | 2026-07-10 | fzyzcjy | Add _TensorViewCodec for storage-deduplicated tensor serialization (#1413) |
| #1415 | 2026-07-10 | fzyzcjy | Add peer checkpoint transfer for healing (#1415) |
| #1418 | 2026-07-10 | fzyzcjy | Add the RayTrainCell abstraction for independent-DP cells (#1418) |
| #1421 | 2026-07-10 | fzyzcjy | Skip the megatron dp_size hint under independent DP (#1421) |
| #1425 | 2026-07-10 | fzyzcjy | Scope an event-logger context around actor train (#1425) |
| #1428 | 2026-07-10 | fzyzcjy | Coordinate tensor dumps across independent-DP cells (#1428) |
| #1429 | 2026-07-10 | fzyzcjy | Pass single-cell independent-DP info into the offline convert_to_hf tool (#1429) |
| #1436 | 2026-07-10 | fzyzcjy | Tolerate per-cell failures by marking dead cells errored and skipping them (#1436) |
| #1437 | 2026-07-10 | fzyzcjy | Retry failed training, save, and weight-update attempts (#1437) |
| #1438 | 2026-07-10 | fzyzcjy | Support external stop/start (suspend/resume) of train cells (#1438) |
| #1439 | 2026-07-10 | fzyzcjy | Add cell- and actor-side cooperative-prepare and checkpoint-transfer primitives (#1439) |
| #1440 | 2026-07-10 | fzyzcjy | Heal cells from a peer checkpoint via cooperative reconfigure in the train group (#1440) |
| #1441 | 2026-07-10 | fzyzcjy | Track a per-actor heartbeat and expose it via RPC (#1441) |
| #1443 | 2026-07-10 | fzyzcjy | Kill failed cells immediately on execute failure (#1443) |
| #1445 | 2026-07-10 | fzyzcjy | Bracket Megatron actor methods with the with_logs decorator (#1445) |
| #1447 | 2026-07-10 | fzyzcjy | Log train-group step-end and analysis events (#1447) |
| #1448 | 2026-07-10 | fzyzcjy | Add FT test-action hooks to the train group (#1448) |
| #1451 | 2026-07-10 | fzyzcjy | Add mini FT controller (#1451) |
| #1452 | 2026-07-10 | fzyzcjy | Add HTTP control server for cell suspend/resume and fault injection (#1452) |
| #1456 | 2026-07-10 | fzyzcjy | Add FT e2e test framework (conftest_ft harness) (#1456) |
| #1457 | 2026-07-10 | fzyzcjy | Add FT no-failure e2e scenarios (#1457) |
| #1458 | 2026-07-10 | fzyzcjy | Add FT deterministic e2e scenarios (#1458) |
| #1459 | 2026-07-10 | fzyzcjy | Add FT with-failure e2e scenarios (#1459) |
| #1460 | 2026-07-10 | fzyzcjy | Add FT random and realistic-gsm8k e2e scenarios with periodic fault injection (#1460) |
| #1465 | 2026-07-15 | Zhiyao Jiang | [AMD] Add AMD MI350X/MI355X (gfx950) blockwise FP8 support for run_qwen3_30b_a3b (#1465) |
| #1487 | 2026-07-17 | Shi-Dong | openenv: TB2 agentic RL adapter + GLM-4.7-Flash launcher (#1487) |
| #1488 | 2026-07-20 | maocheng23 | [OPD] Add Qwen3.5-35B-A3B single-node self-distillation example (#1488) |
| #1490 | 2026-07-15 | Shi-Dong | docs: correct cli-reference argument defaults (#1490) |
| #1508 | 2026-06-29 | Yueming Yuan | [test] mv deepseek v4 to `/manual` (#1508) |
| #1510 | 2026-07-05 | Jiajun Li | (2/7) refactor: extract session core with direct HTTP responses (#1510) |
| #1513 | 2026-06-30 | Jiajun Li | chore: make megatron e2e CaseConfig topology explicit instead of inferred, and tune e2e test time (#1513) |
| #1518 | 2026-07-05 | Jiajun Li | (4/7) feat(session): multi-process session-server data plane (routing + IPC + worker + thin router) (#1518) |
| #1658 | 2026-07-15 | Jiajun Li | (1/N) Revert "(4/7) feat(session): multi-process session-server data plane (routing + IPC + worker + thin router) (#1518)" (#1658) |
| #1556 | 2026-07-09 | Zhiyao Jiang | [AMD] Enable R3 CI and add the rocm700-mi35x image variant (#1556) |
| #1563 | 2026-07-05 | Jiajun Li | (3/7) feat(session): strip R3 replay payloads from client chat responses (#1563) |
| #1580 | 2026-07-06 | Shi-Dong | Add Shi-Dong to megatron/sglang backends and rollout/session (#1580) |
| #1584 | 2026-07-21 | Zhihao Wang | fix: decide NVTE_FP8_BLOCK_SCALING_FP32_SCALES by hardware instead of hardcoding it (#1584) |
| #1593 | 2026-07-07 | Ethan (Yusheng) Su | [Lora] Fix bridge-LoRA path silently dropping recompute args (#1593) |
| #1594 | 2026-07-10 | fzyzcjy | Fix reloadable process group to preserve full new_group arguments across reloads (#1594) |
| #1597 | 2026-07-15 | Ethan (Yusheng) Su | [Lora] Add GDN (Qwen3.5/Qwen3-Next) LoRA target-module mapping (#1597) |
| #1599 | 2026-07-10 | Jiajun Li | fix(dpsk-v4): real-history TITO merge and parse-kwargs alignment for the {tool,user} surface (#1599) |
| #1600 | 2026-07-10 | Jiajun Li | refactor(chat-template): merge the DeepSeek V3.2/V4 bridges into one dispatch module (#1600) |
| #1604 | 2026-07-09 | Zhiyao Jiang | [AMD] Fix HF to torch_dist conversion segfault during checkpoint save (#1604) |
| #1607 | 2026-07-09 | Xinyu Jiang | [AMD] Enable DeepSeek-V4-Flash FP8 RL training on MI355X (#1607) |
| #1622 | 2026-07-13 | fzyzcjy | Re-enable ft-short FT e2e tests in CI (#1622) |
| #1630 | 2026-07-20 | Jiajun Li | (4/4) feat(ci): metric-history gate — GATE_DEFAULTS one-liners + standard-metric sweep (M4) (#1630) |
| #1638 | 2026-07-21 | Mathew Han | [lora] feat: multi lora async (full PR, see stacked diff PRs instead) (#1638) |
| #1642 | 2026-07-13 | Qijia Shen | fix(deepseek_v4): disable aggressive smem merge in sparse-MLA backward (NaN dQ/dKV) (#1642) |
| #1643 | 2026-07-13 | Jiajun Li | fix(weight-update): track disk-delta engine freshness (#1643) |
| #1645 | 2026-07-16 | lizamd | Add Qwen3 / Qwen3-Coder support to swe-agent-v2 example (ROCm-safe) (#1645) |
| #1650 | 2026-07-13 | JD | Add corresponding author to P2P Weight Transfer doc (#1650) |
| #1656 | 2026-07-14 | Xinyu Kang | [AMD][DeepSeek V4][1/N] Fit TP=1 sparse MLA kernels in MI35x LDS (#1656) |
| #1657 | 2026-07-13 | Yueming Yuan | [fix] fix session server `hash_consistent` mode (#1657) |
| #1660 | 2026-07-15 | Jiajun Li | (3/N) test(session): session-server overhead benchmark on the multi-instance topology (#1660) |
| #1661 | 2026-07-13 | Haoguang Cai | fix(docs): stop navbar logo from stretching, bump its size (#1661) |
| #1663 | 2026-07-15 | Shi-Dong | rollout: subtract agent-reported tool time from throughput accounting (#1663) |
| #1664 | 2026-07-14 | Xinyu Kang | [AMD][DeepSeek V4][2/N] Enable decode CUDA graphs on MI355X (#1664) |
| #1667 | 2026-07-14 | Jiajun Li | feat: add CI automation, stage sizing, and doc-dev skills (#1667) |
| #1668 | 2026-07-14 | Jiajun Li | fix(tests): defer tokenizer helper import during collection (#1668) |
| #1669 | 2026-07-14 | Jiajun Li | fix(ci): calibrate CUDA E2E estimates from run 29198133159 (#1669) |
| #1670 | 2026-07-14 | Jiajun Li | fix(ci): make Megatron model scripts consistently selectable (#1670) |
| #1671 | 2026-07-14 | Jiajun Li | fix(ci): align model-script jobs with GPU allocation (#1671) |
| #1676 | 2026-07-14 | Jiajun Li | fix(ci): exclude long tests from image scope (#1676) |
| #1678 | 2026-07-15 | Ethan (Yusheng) Su | [WIP] feat: lora GLM-5.2 FP8 rollout support (#1678) |
| #1680 | 2026-07-15 | Xinyu Jiang | [AMD] fix GPU memory release for TMS pause on ROCm 7.2 (#1680) |
| #1681 | 2026-07-16 | Shi-Dong | docs: fix incorrect GLM-5 conversion claim and FSDP2 backend path (#1681) |
| #1684 | 2026-07-15 | Zhichen Zeng | [tml] Inkling model doc (#1684) |
| #1685 | 2026-07-17 | Zhiyao Jiang | [tiny] Make NVTE_FP8_BLOCK_SCALING_FP32_SCALES overridable via env (#1685) |
| #1686 | 2026-07-15 | Xinyu Jiang | [AMD] use prebuilt wheels for flash-attn and sgl-model-gateway (#1686) |
| #1688 | 2026-07-15 | Xinyu Kang | [AMD] Restore Python 3.10 StrEnum compatibility (#1688) |
| #1695 | 2026-07-17 | Shi-Dong | docs: document the optional agent abort hook (#1695) |
| #1710 | 2026-07-22 | Tao Lin | openenv/tbench2: per-task sandbox image recipe + Daytona materialization (#1710) |
| #1711 | 2026-07-22 | Tao Lin | openenv/tbench2: orphan TTL + ownership labels for per-task sandboxes (#1711) |
| #1712 | 2026-07-23 | Jiajun Li | fix(tito): tokenize appended messages as one suffix (#1712) |
| #1713 | 2026-07-17 | Xinyu Jiang | [AMD] rocm720 MI355X image: Transformer Engine wheel + build fixes (#1713) |
| #1715 | 2026-07-19 | Jiajun Li | fix: preserve step-zero Dumper model dumps (#1715) |
| #1718 | 2026-07-19 | Zhichen Zeng | docs: add Thinking Machines model group with Inkling (#1718) |
| #1721 | 2026-07-20 | Ethan (Yusheng) Su | [fix] honor --accumulate-allreduce-grads-in-fp32 on the bridge LoRA DDP path (#1721) |
| #1722 | 2026-07-20 | Ethan (Yusheng) Su | [fix] gate LoRA checkpoint writers on dp and cp, not dp alone (#1722) |
| #1743 | 2026-07-21 | Mathew Han | [2/7][multi-lora]: adapter controller - registry state machine, backend, control-plane HTTP API, named Ray actor (#1743) |
| #1748 | 2026-07-21 | Mathew Han | [7/7][multi-lora]: driver and example - fully async driver, typer launcher, adapter configs, service smoke client (#1748) |
| #1750 | 2026-07-22 | Zhichen Zeng | [Fix] retract-mode flush_cache no-op crash (#1750) |
| #1751 | 2026-07-23 | Jiajun Li | refactor(tito): rename render_messages to apply_chat_template (#1751) |
