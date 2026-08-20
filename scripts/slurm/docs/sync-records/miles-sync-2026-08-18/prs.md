# miles upstream PRs report

**Period**: since merge-base `c94c2fa9` (2026-07-23, "Add --async-max-concurrent-samples ... (#1677)") → upstream tip `fc04f666` (2026-08-18)
**Total upstream commits / PRs**: 274 / 274
**Watchlist hits (pin-source files)**: 18
**Flagged (touch files we've modified)**: 95

Metadata source: upstream squash commits (`git diff-tree` file lists); `gh` was unauthenticated on this host, so PR bodies were not fetched — impact notes below come from reading the actual diffs.

---

## ⚠️ Watchlist hits (pin sources)

### [#1781](https://github.com/radixark/miles/pull/1781) — Bump TransformerEngine to 2.17.0
**Merged**: 2026-07-24 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: TE 2.12→**2.17.0** everywhere (cu13 dist, torch dist, `transformer_engine[pytorch]` for cu12). New TE patch `te_dequantized_backward_override.patch` (hot fix from NVIDIA/TransformerEngine#3141, drop after TE 2.18). ⇒ `extract_pins.py --write` moves `TE_VERSION`; **our install_env.sh compiles TE in place**, so 2.17 needs a rebuild-and-verify on cluster; check its new import-time deps (see #1800).

### [#1675](https://github.com/radixark/miles/pull/1675) — openenv/tbench2: Daytona sandbox mode — one cloud sandbox per episode, as its own agent function
**Merged**: 2026-07-24 | **Author**: Tao Lin | **Pin files**: `requirements.txt`

**Pin impact**: requirements: +`openai` (tool_call_utils + openenv adapter import it directly; was transitive via sglang). Keep in merge; our env already has openai via sglang.

### [#1789](https://github.com/radixark/miles/pull/1789) — docker: fix main image build — install TransformerEngine from PyPI, drop stale wheel path
**Merged**: 2026-07-24 | **Author**: Zhichen Zeng | **Pin files**: `docker/Dockerfile`

**Pin impact**: Docker-only cleanup of the TE wheel path (always build `transformer_engine_torch` from PyPI + `nvidia-mathdx==25.6.0`). Mirror-check: our install_env.sh TE section.

### [#1791](https://github.com/radixark/miles/pull/1791) — ci(docker): PR-side image build and run the GPU matrix inside the freshly built image
**Merged**: 2026-07-24 | **Author**: Zhichen Zeng | **Pin files**: `docker/Dockerfile`

**Pin impact**: TE_DIR lookup via importlib (works without importing TE). CI/docker-only; harmless to mirror if our scripts locate TE for patching.

### [#1784](https://github.com/radixark/miles/pull/1784) — feat(docker): overlay mooncake structured object store module pending release
**Merged**: 2026-07-24 | **Author**: Jiajun Li | **Pin files**: `docker/Dockerfile`

**Pin impact**: Mooncake structured-object-store wheel overlay for cu13 only; cu12 keeps base mooncake. We are cu12 ⇒ no action, but MOONCAKE_VERSION extraction unaffected.

### [#1796](https://github.com/radixark/miles/pull/1796) — fix(docker): install TransformerEngine from rolling wheels
**Merged**: 2026-07-25 | **Author**: Jiajun Li | **Pin files**: `docker/Dockerfile`

**Pin impact**: TE now installed from rolling wheels with a bind-mounted `docker/verify_transformer_engine.py` check. If we mirror the verify step, that script is new.

### [#1799](https://github.com/radixark/miles/pull/1799) — docker: use FA3 3.0.0 for transformer_engine 2.17
**Merged**: 2026-07-26 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: FA3 wheel → **3.0.0** for TE 2.17; the old `flash_attn_interface.py` curl-fetch is REMOVED (wheel now ships the interface + shim; overwriting double-registers torch custom ops). ⇒ our wheels bundle carries FA3: the rolling `cu129-x86_64` bundle must be re-validated for FA3 3.0.0 vs TE 2.17 — this is an sglang-sync/wheels item, and our install_env.sh must DROP any interface-fetch mirror.

### [#1800](https://github.com/radixark/miles/pull/1800) — docker: install transformer_engine_torch runtime deps
**Merged**: 2026-07-26 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: TE 2.17 `--no-deps` install needs explicit runtime deps: `einops onnx onnxscript pydantic nvdlfw-inspect` (TE imports onnxscript unconditionally). ⇒ add to install_env.sh next to TE, matches the requirements.txt onnxscript addition in #1469.

### [#1808](https://github.com/radixark/miles/pull/1808) — docker: raise cuDNN to 9.22.0 for transformer_engine 2.17
**Merged**: 2026-07-27 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: cuDNN 9.16.0.29 → **9.22.0.52** (cu12+cu13) for TE 2.17. ⇒ `CUDNN_CU12_VERSION` in pins.env moves via --write; install_env.sh consumes the pin.

### [#1575](https://github.com/radixark/miles/pull/1575) — feat(offload): support disk target for training-actor offload
**Merged**: 2026-07-27 | **Author**: Zhichen Zeng | **Pin files**: `docker/Dockerfile`

**Pin impact**: torch_memory_saver pinned to commit `74d68c5e` (was branch HEAD). Mirror into install_env.sh if we install it from git.

### [#1469](https://github.com/radixark/miles/pull/1469) — [fsdp] fsdp backend support fix
**Merged**: 2026-07-27 | **Author**: Zhichen Zeng | **Pin files**: `requirements.txt`

**Pin impact**: requirements: +`onnxscript` (TE 2.17 import-time), `transformers<5.13` → **`==5.12.1` exact** (HF-native weight conversion). Our env is already 5.12.1 ⇒ no runtime change, but requirements merge must keep the exact pin.

### [#1795](https://github.com/radixark/miles/pull/1795) — Bump sglang to v0.5.16
**Merged**: 2026-07-29 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: **sglang v0.5.15 → v0.5.16** (`SGLANG_IMAGE_TAG`) ⇒ post-merge `extract_pins.py --write` sets `UPSTREAM_SGLANG_IMAGE_TAG=v0.5.16` and prints `[sglang-sync pending]` ⇒ decide sync-together vs defer (needs sgl-project/sglang@sglang-miles rebased to v0.5.16 + our 6 mirror patches re-applied). Also: TE patch loop now fails loudly on non-applying patch; **Megatron-Bridge pinned to commit `7f0fb345`** (was `@bridge` branch; carries Bridge#24 TE-2.17 grouped-linear contract) ⇒ `MBRIDGE_COMMIT` extraction changes meaning from branch to SHA.

### [#1759](https://github.com/radixark/miles/pull/1759) — (2/2) refactor(session): assemble training samples on the session server; records never leave it
**Merged**: 2026-07-29 | **Author**: Jiajun Li | **Pin files**: `requirements.txt`

**Pin impact**: requirements: +`safetensors>=0.8.0` (session-server samples-reply wire codec). Part of the 2-part session refactor (records never leave the session server) — also an overlap PR for our envpack/session usage.

### [#1961](https://github.com/radixark/miles/pull/1961) — docker: keep the cu12 dependency markers after checking out sglang-miles
**Merged**: 2026-07-29 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: cu12 marker sed for sglang pyproject (cuda-python <13, flashinfer[cu12], plain cutlass-dsl) now applied in Docker after checkout -f. This is upstream adopting the SAME cu12 flavor rewrite our mirror carries as the `[sglang-miles cu129] bare-metal cu12 dep flavors` patch — on the v0.5.16 sglang-sync, compare and possibly retire part of our mirror patch #2.

### [#2404](https://github.com/radixark/miles/pull/2404) — Install the kubernetes toolchain the charts and the k8s provider need
**Merged**: 2026-08-12 | **Author**: fzyzcjy | **Pin files**: `docker/Dockerfile`, `requirements.txt`

**Pin impact**: k8s toolchain (kubectl/helm via install-kube-tools.sh) + `kubernetes_asyncio` requirement. Cluster-mode features we don't run; requirements merge keeps it (harmless).

### [#2538](https://github.com/radixark/miles/pull/2538) — release: miles version release workflow
**Merged**: 2026-08-14 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: Release workflow; new `ARG MEGATRON_COMMIT` (empty = branch HEAD, release freezes). extract_pins should keep reading the default branch behavior; verify extract_pins handles the new ARG without confusion.

### [#2567](https://github.com/radixark/miles/pull/2567) — Install tmux into the training image
**Merged**: 2026-08-16 | **Author**: fzyzcjy | **Pin files**: `docker/Dockerfile`

**Pin impact**: tmux in training image. apt-level; no conda-env equivalent needed (cluster hosts have tmux).

### [#2600](https://github.com/radixark/miles/pull/2600) — fix(docker): pin cutlass-dsl 4.6.2 and flashinfer 0.6.15.post1 over the sglang base
**Merged**: 2026-08-18 | **Author**: Yueming Yuan | **Pin files**: `docker/Dockerfile`

**Pin impact**: sglang v0.5.16 base ships cutlass-dsl 4.6.0 whose runtime **hangs FA4 CuTe backward on sm_103 (B300/GB300)**; pin all `nvidia-cutlass-dsl*==4.6.2` components + `flashinfer 0.6.15.post1` (new `ENV FLASHINFER_VERSION`). We run H200 (sm_90) ⇒ hang class doesn't bite us, but on the v0.5.16 sglang-sync our pyproject cu12-flavor patch must pick compatible cutlass/flashinfer pins.

---

## PRs that touch files we've modified

| PR | Merged | Title | Overlapping files |
|----|--------|-------|-------------------|
| [#1762](https://github.com/radixark/miles/pull/1762) | 2026-07-23 | feat(session): serve stream:true chat completions as fake streaming | `tests/fast/utils/chat_template_utils/test_template.py` |
| [#1783](https://github.com/radixark/miles/pull/1783) | 2026-07-23 | chore(ci): pin docker build to 2-GPU H200 runners | `.github/workflows/docker-build.yml` |
| [#1785](https://github.com/radixark/miles/pull/1785) | 2026-07-24 | Fix torch_memory_saver LD_PRELOAD path for CUDA-suffixed binaries | `miles/ray/train/actor_factory.py` |
| [#1758](https://github.com/radixark/miles/pull/1758) | 2026-07-24 | (1/3) refactor(session): move sample assembly into the session package | `tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py` |
| [#1261](https://github.com/radixark/miles/pull/1261) | 2026-07-24 | NVFP4 RL | `miles/utils/arguments.py`, `tests/fast-gpu/test_nvfp4_quantizer.py` |
| [#1798](https://github.com/radixark/miles/pull/1798) | 2026-07-26 | fix (docker): stop the docker gate from skipping the whole test suite | `.github/workflows/pr-test.yml` |
| [#1628](https://github.com/radixark/miles/pull/1628) | 2026-07-26 | feat(tito): add DeepSeek V3.2 speculative session verification | `.pre-commit-config.yaml`, `tests/fast/utils/test_arguments.py` |
| [#1720](https://github.com/radixark/miles/pull/1720) | 2026-07-27 | fix: inject PPO terminal rewards after CP gather | `miles/backends/training_utils/loss_hub/math_utils.py` |
| [#591](https://github.com/radixark/miles/pull/591) | 2026-07-28 | [feat]support rollout data transfer with mooncake | `miles/backends/megatron_utils/actor.py`, `miles/backends/training_utils/data.py`, `miles/ray/rollout/rollout_manager.py`, `miles/ray/rollout/train_data_conversion.py` +5 |
| [#1817](https://github.com/radixark/miles/pull/1817) | 2026-07-27 | refactor(tito): bind append roles to fixed templates | `miles/utils/arguments.py`, `tests/fast/utils/chat_template_utils/test_template.py` |
| [#1818](https://github.com/radixark/miles/pull/1818) | 2026-07-27 | refactor(tito): remove --tito-allowed-append-roles and derive append surfaces from fixed templa | `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py` |
| [#1835](https://github.com/radixark/miles/pull/1835) | 2026-07-28 | ci: compare PR merge commits with their current base | `.github/workflows/pr-test.yml` |
| [#1752](https://github.com/radixark/miles/pull/1752) | 2026-07-28 | fix: preserve BSHD layout in PPO CP advantages | `miles/backends/training_utils/loss_hub/math_utils.py` |
| [#1918](https://github.com/radixark/miles/pull/1918) | 2026-07-28 | Remove the experimental swe-agent examples | `.gitmodules` |
| [#1827](https://github.com/radixark/miles/pull/1827) | 2026-07-28 | fix: run PPO GAE over trainable tokens only | `miles/backends/training_utils/loss_hub/math_utils.py` |
| [#1829](https://github.com/radixark/miles/pull/1829) | 2026-07-28 | fix: require explicit off-policy correction for async PPO training | `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py`, `train_async.py` |
| [#1920](https://github.com/radixark/miles/pull/1920) | 2026-07-28 | fix(ci): stabilize SGLang and FSDP E2E and remove long label from nightly run | `.github/workflows/pr-test.yml` |
| [#1932](https://github.com/radixark/miles/pull/1932) | 2026-07-29 | docs: the miles dashboard design and usage | `docs/user-guide/monitoring.md` |
| [#1916](https://github.com/radixark/miles/pull/1916) | 2026-07-29 | (1/2) refactor(rollout): drop --generate-multi-samples and its per-turn sample semantics | `tests/fast/rollout/generate_hub/test_single_turn.py` |
| [#1953](https://github.com/radixark/miles/pull/1953) | 2026-07-29 | refactor(examples): group examples into infra_features/ and experimental/ | `examples/geo3k_vlm_multi_turn/env_geo3k.py`, `examples/geo3k_vlm_multi_turn/rollout.py`, `miles/utils/arguments.py`, `train_async.py` |
| [#1926](https://github.com/radixark/miles/pull/1926) | 2026-07-29 | Fix weight sync selector for frozen speculative drafts | `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py` |
| [#1794](https://github.com/radixark/miles/pull/1794) | 2026-07-29 | feat(multi-lora): enable and validate MoE expert adapters | `tests/fast/utils/test_arguments.py` |
| [#1965](https://github.com/radixark/miles/pull/1965) | 2026-07-29 | dashboard: fix phase visibility for manager events and idle processes | `miles/dashboard/store.py` |
| [#1735](https://github.com/radixark/miles/pull/1735) | 2026-07-30 | [PPO] Share Actor/Critic GPUs | `miles/backends/megatron_utils/actor.py`, `miles/backends/megatron_utils/model_provider.py`, `miles/backends/training_utils/data.py`, `miles/ray/actor_group.py` +5 |
| [#1716](https://github.com/radixark/miles/pull/1716) | 2026-07-31 | Move fully-async rollout from examples into miles/rollout | `examples/fully_async/README.md`, `examples/fully_async/fully_async_rollout.py`, `train_async.py` |
| [#2039](https://github.com/radixark/miles/pull/2039) | 2026-07-31 | [AMD] fix mooncake object store support | `docker/build.py` |
| [#2041](https://github.com/radixark/miles/pull/2041) | 2026-07-31 | Address engines by base URL when the router is dp-aware | `miles/rollout/inference_rollout/inference_rollout_train.py`, `miles/rollout/sglang_rollout.py` |
| [#1717](https://github.com/radixark/miles/pull/1717) | 2026-07-31 | Rewrite fully-async rollout as FullyAsyncRolloutFn on the class-based rollout API | `examples/fully_async/README.md`, `miles/utils/arguments.py`, `train.py` |
| [#1793](https://github.com/radixark/miles/pull/1793) | 2026-07-31 | feat(optimizer): NVMe optimizer-state streaming as a miles plugin | `miles/backends/megatron_utils/model.py`, `miles/ray/train/actor_factory.py`, `miles/utils/arguments.py` |
| [#1298](https://github.com/radixark/miles/pull/1298) | 2026-08-03 | [OPD] Per-position teacher scoring (sparse top-k) + kaixih's robustness fixes | `miles/rollout/on_policy_distillation.py`, `miles/rollout/sglang_rollout.py`, `miles/utils/arguments.py`, `tests/fast/rollout/test_on_policy_distillation.py` |
| [#2031](https://github.com/radixark/miles/pull/2031) | 2026-08-03 | feat(fsdp): add hybrid sharding | `miles/backends/training_utils/parallel.py`, `miles/utils/arguments.py` |
| [#1683](https://github.com/radixark/miles/pull/1683) | 2026-08-03 | [tml] Inkling model support | `miles/backends/megatron_utils/actor.py`, `miles/backends/megatron_utils/model.py`, `miles/backends/training_utils/data.py`, `miles/rollout/generate_utils/generate_endpoint_utils.py` +2 |
| [#2122](https://github.com/radixark/miles/pull/2122) | 2026-08-03 | [tml] native LoRA support for inkling model | `miles/backends/megatron_utils/actor.py`, `miles/backends/megatron_utils/model.py`, `miles/rollout/generate_utils/generate_endpoint_utils.py`, `miles/rollout/generate_utils/prefill_logprobs.py` +2 |
| [#2043](https://github.com/radixark/miles/pull/2043) | 2026-08-03 | Compact NVFP4 BF16 MoE exclusion metadata | `tests/fast-gpu/test_nvfp4_quantizer.py` |
| [#2132](https://github.com/radixark/miles/pull/2132) | 2026-08-03 | fix(docs): grad_norm is logged before clipping, not after | `miles/backends/training_utils/log_utils.py` |
| [#2131](https://github.com/radixark/miles/pull/2131) | 2026-08-03 | ci(docker): rebuild scheduled images at least once every 24h | `.github/workflows/docker-build.yml`, `docs/ci/02-docker-build.md` |
| [#1740](https://github.com/radixark/miles/pull/1740) | 2026-08-03 | feat (async): support evaluation for fully-async training (dedicated fleet / pause-the-world /  | `examples/fully_async/README.md`, `miles/backends/megatron_utils/actor.py`, `miles/backends/megatron_utils/model.py`, `miles/ray/actor_group.py` +7 |
| [#2077](https://github.com/radixark/miles/pull/2077) | 2026-08-04 | Reject --disable-weights-backuper for LoRA + colocate + offload-train | `miles/utils/arguments.py` |
| [#2203](https://github.com/radixark/miles/pull/2203) | 2026-08-04 | Revert "Reject --disable-weights-backuper for LoRA + colocate + offload-train" (#2077) | `miles/utils/arguments.py` |
| [#1673](https://github.com/radixark/miles/pull/1673) | 2026-08-04 | [feat] support `sample` rollout submission granularity to keep fully async concurrency | `miles/rollout/inference_rollout/inference_rollout_common.py`, `miles/rollout/inference_rollout/inference_rollout_train.py`, `miles/utils/arguments.py` |
| [#1927](https://github.com/radixark/miles/pull/1927) | 2026-08-04 | [feat] Support training with variable global batch size | `miles/backends/megatron_utils/actor.py`, `miles/backends/megatron_utils/model.py`, `miles/backends/training_utils/cp_utils.py`, `miles/backends/training_utils/data.py` +9 |
| [#1284](https://github.com/radixark/miles/pull/1284) | 2026-08-04 | Nemotron RL support | `miles/backends/megatron_utils/model.py` |
| [#2215](https://github.com/radixark/miles/pull/2215) | 2026-08-05 | fix(mtp): double-shift GPT-path MTP labels | `miles/backends/megatron_utils/model.py` |
| [#2126](https://github.com/radixark/miles/pull/2126) | 2026-08-05 | feat(session): serve trajectory trees behind session server v2 | `miles/utils/arguments.py`, `miles/utils/misc.py` |
| [#2128](https://github.com/radixark/miles/pull/2128) | 2026-08-05 | feat(rollout): bridge agentic generation to session server v2 | `miles/utils/arguments.py`, `tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py`, `tests/fast/utils/test_arguments.py` |
| [#2222](https://github.com/radixark/miles/pull/2222) | 2026-08-05 | [AMD] Remove PR#301 FileSystemWriterAsync swap — superseded by Megatron-LM#74 | `miles/backends/megatron_utils/model.py` |
| [#2221](https://github.com/radixark/miles/pull/2221) | 2026-08-06 | fix(ci): split CPU and GPU reusable workflows | `.github/workflows/pr-test.yml` |
| [#2231](https://github.com/radixark/miles/pull/2231) | 2026-08-06 | fix(ci): cancel PR tests after closure | `.github/workflows/pr-test.yml` |
| [#2224](https://github.com/radixark/miles/pull/2224) | 2026-08-06 | fix: derive --critic-save from --save so PPO critic checkpoints are not silently skipped | `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py` |
| [#2226](https://github.com/radixark/miles/pull/2226) | 2026-08-06 | fix: preserve routing-replay state around MTP spec creation | `miles/backends/megatron_utils/model_provider.py` |
| [#1572](https://github.com/radixark/miles/pull/1572) | 2026-08-06 | [optim]--rematerialize-param-from-master-weight: save the bf16 weight backup in colocate | `miles/backends/megatron_utils/actor.py`, `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py` |
| [#1968](https://github.com/radixark/miles/pull/1968) | 2026-08-07 | [Bug Fix] Reduce distributed min/max metrics as extrema | `miles/backends/training_utils/log_utils.py` |
| [#2218](https://github.com/radixark/miles/pull/2218) | 2026-08-07 | ci: version TITO metrics by session server | `miles/ray/rollout/metrics.py`, `tests/fast/ray/rollout/test_metrics.py` |
| [#2223](https://github.com/radixark/miles/pull/2223) | 2026-08-07 | Remove --disable-weights-backuper; default eligible colocate launchers to rematerialize | `miles/backends/megatron_utils/actor.py`, `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py` |
| [#2235](https://github.com/radixark/miles/pull/2235) | 2026-08-07 | feat(fsdp): support rollout routing replay (R3) for fsdp backend | `miles/utils/arguments.py` |
| [#2244](https://github.com/radixark/miles/pull/2244) | 2026-08-07 | pass the engine weight version from the trainer instead of polling the router | `miles/backends/megatron_utils/actor.py`, `miles/ray/rollout/rollout_manager.py` |
| [#2030](https://github.com/radixark/miles/pull/2030) | 2026-08-07 | [async] async data buffer: unified filters and better observability | `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py` |
| [#2272](https://github.com/radixark/miles/pull/2272) | 2026-08-07 | refactor: move the FSDP backend out of experimental | `miles/ray/train/actor_factory.py`, `miles/utils/arguments.py` |
| [#1967](https://github.com/radixark/miles/pull/1967) | 2026-08-08 | [Bug Fix] Always release rollout-engine broadcast lock on failure | `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py` |
| [#1903](https://github.com/radixark/miles/pull/1903) | 2026-08-09 | Rename exec_command by the resource its command needs | `miles/utils/misc.py` |
| [#1904](https://github.com/radixark/miles/pull/1904) | 2026-08-09 | Move the shell exec helpers next to their only consumers | `miles/utils/misc.py` |
| [#1905](https://github.com/radixark/miles/pull/1905) | 2026-08-09 | Remove non-reproducible file arguments by supporting inline base64 payloads | `miles/utils/arguments.py` |
| [#1966](https://github.com/radixark/miles/pull/1966) | 2026-08-09 | [Bug Fix][OPD] Stop gradients through old-policy and teacher scores | `miles/backends/training_utils/loss_hub/losses.py`, `miles/backends/training_utils/loss_hub/opd.py`, `tests/fast/backends/training_utils/loss/test_opd.py`, `tests/fast/backends/training_utils/test_true_on_policy_loss_metrics.py` |
| [#2216](https://github.com/radixark/miles/pull/2216) | 2026-08-10 | docs: add GLM-5.2 model page and update supported-models tables | `.gitignore` |
| [#2363](https://github.com/radixark/miles/pull/2363) | 2026-08-11 | ci(docker): run image builds on docker-build runners | `.github/workflows/docker-build.yml`, `.github/workflows/pr-test.yml`, `docs/ci/02-docker-build.md` |
| [#2382](https://github.com/radixark/miles/pull/2382) | 2026-08-11 | fix: drop duplicated rematerialize validation call | `miles/utils/arguments.py` |
| [#2400](https://github.com/radixark/miles/pull/2400) | 2026-08-11 | docs: fix SEO gaps across the docs site | `docs/ci/02-docker-build.md` |
| [#2390](https://github.com/radixark/miles/pull/2390) | 2026-08-11 | docs(args): correct the offload flags' help text | `miles/utils/arguments.py` |
| [#2024](https://github.com/radixark/miles/pull/2024) | 2026-08-11 | dashboard: read open phase markers regardless of age | `miles/dashboard/store.py` |
| [#2278](https://github.com/radixark/miles/pull/2278) | 2026-08-11 | feat(session): request and assemble additional R3 rows under in-place weight updates | `miles/rollout/generate_utils/generate_endpoint_utils.py`, `miles/utils/arguments.py`, `tests/fast/utils/test_arguments.py` |
| [#2369](https://github.com/radixark/miles/pull/2369) | 2026-08-12 | fix(rollout): normalize rewards per rollout | `miles/ray/rollout/train_data_conversion.py`, `tests/fast/ray/rollout/test_train_data_conversion.py` |
| [#2026](https://github.com/radixark/miles/pull/2026) | 2026-08-12 | dashboard: scrape engines directly by default | `tests/fast/dashboard/test_core_integration.py` |
| [#2353](https://github.com/radixark/miles/pull/2353) | 2026-08-12 | dashboard: report model FLOPs utilization | `miles/backends/training_utils/log_utils.py`, `miles/utils/arguments.py` |
| [#2022](https://github.com/radixark/miles/pull/2022) | 2026-08-12 | dashboard: cache full-window engine series queries | `miles/dashboard/store.py` |
| [#2023](https://github.com/radixark/miles/pull/2023) | 2026-08-12 | dashboard: dp-aware engine metrics | `miles/dashboard/store.py` |
| [#2027](https://github.com/radixark/miles/pull/2027) | 2026-08-12 | dashboard: advisory v2 — run-health alarms before config tuning | `miles/dashboard/store.py` |
| [#2481](https://github.com/radixark/miles/pull/2481) | 2026-08-12 | docs: mirror examples/ READMEs into the Examples tab | `.pre-commit-config.yaml` |
| [#2240](https://github.com/radixark/miles/pull/2240) | 2026-08-12 | feat(session): add configurable replay matching | `miles/utils/arguments.py`, `tests/fast/utils/chat_template_utils/test_template.py`, `tests/fast/utils/test_arguments.py` |
| [#2485](https://github.com/radixark/miles/pull/2485) | 2026-08-12 | clean up fully async example and mv to examples/infra_features | `examples/fully_async/README.md` |
| [#2498](https://github.com/radixark/miles/pull/2498) | 2026-08-12 | fix: require shared rewards within rollouts | `miles/ray/rollout/train_data_conversion.py`, `tests/fast/ray/rollout/test_train_data_conversion.py` |
| [#2477](https://github.com/radixark/miles/pull/2477) | 2026-08-13 | fix(docs): correct agentic rollout guidance | `miles/utils/arguments.py` |
| [#2499](https://github.com/radixark/miles/pull/2499) | 2026-08-13 | feat(ci): add weekly full-suite cadence | `.github/workflows/pr-test.yml` |
| [#2489](https://github.com/radixark/miles/pull/2489) | 2026-08-13 | docs: label-only sidebar groups, explicit Overview pages, scoped link color | `.pre-commit-config.yaml` |
| [#2219](https://github.com/radixark/miles/pull/2219) | 2026-08-13 | [feat] Add training log-prob reuse to skip the redundant forward-only pass | `miles/backends/megatron_utils/actor.py`, `miles/backends/training_utils/loss_hub/losses.py`, `miles/utils/arguments.py`, `tests/fast/backends/training_utils/loss/test_opd.py` +2 |
| [#2522](https://github.com/radixark/miles/pull/2522) | 2026-08-13 | Make the class-based rollout the default and convert legacy path to env var gated | `miles/ray/rollout/rollout_manager.py`, `miles/rollout/generate_utils/sample_utils.py`, `miles/rollout/inference_rollout/inference_rollout_common.py`, `miles/rollout/inference_rollout/inference_rollout_train.py` +3 |
| [#2543](https://github.com/radixark/miles/pull/2543) | 2026-08-14 | Allow --stream-optimizer-state-to-disk without trainer offload | `miles/utils/arguments.py` |
| [#2566](https://github.com/radixark/miles/pull/2566) | 2026-08-15 | fix: use wandb.sdk.lib.runid.generate_id for the wandb group suffix | `miles/utils/tracking_utils/wandb_utils.py` |
| [#2529](https://github.com/radixark/miles/pull/2529) | 2026-08-16 | refactor(ci): gate PR image builds on CPU tests | `.github/workflows/pr-test.yml`, `docs/ci/02-docker-build.md` |
| [#2530](https://github.com/radixark/miles/pull/2530) | 2026-08-16 | refactor(ci): select GPU stages from PR impact | `.github/workflows/pr-test.yml` |
| [#2548](https://github.com/radixark/miles/pull/2548) | 2026-08-16 | refactor(ci): require domain labels for GPU tests | `.github/workflows/pr-test.yml`, `tests/fast-gpu/test_mxfp8_quantizer.py`, `tests/fast-gpu/test_nvfp4_quantizer.py` |
| [#2584](https://github.com/radixark/miles/pull/2584) | 2026-08-17 | fix(ci): honor nightly fast-fail bypass | `.github/workflows/pr-test.yml` |
| [#2581](https://github.com/radixark/miles/pull/2581) | 2026-08-17 | dashboard: resolve CP/PP-sharded train dumps offline and fix a partition-reader race | `miles/backends/training_utils/cp_utils.py`, `miles/dashboard/store.py` |
| [#2576](https://github.com/radixark/miles/pull/2576) | 2026-08-18 | fix: skip rollout construction for debug replay | `miles/ray/actor_group.py`, `miles/ray/rollout/rollout_manager.py`, `tests/fast/ray/rollout/real_ray/test_rollout_manager.py` |
| [#2606](https://github.com/radixark/miles/pull/2606) | 2026-08-18 | update doc & readme | `docs/user-guide/monitoring.md` |
| [#2651](https://github.com/radixark/miles/pull/2651) | 2026-08-18 | docs: fix stale paths, flags, env vars and metric names | `docs/ci/02-docker-build.md`, `docs/user-guide/monitoring.md` |

---

## Other PRs (no overlap detected)

| PR | Merged | Title |
|----|--------|-------|
| [#1802](https://github.com/radixark/miles/pull/1802) | 2026-07-26 | docs: remove nonexistent --sglang-log-dir flag and /tmp/sglang log path |
| [#1698](https://github.com/radixark/miles/pull/1698) | 2026-07-26 | docs: fix inaccuracies and expand explanations in Quick Start |
| [#1815](https://github.com/radixark/miles/pull/1815) | 2026-07-27 | refactor(tito): rename tokenize_additional_non_assistant to tokenize_additional_messages |
| [#1816](https://github.com/radixark/miles/pull/1816) | 2026-07-27 | feat(chat-template): DSv3.2 encoder honors drop_thinking=False when preserving thinking |
| [#1821](https://github.com/radixark/miles/pull/1821) | 2026-07-27 | ci: run-ci* labels double as fork-PR CI approval |
| [#1834](https://github.com/radixark/miles/pull/1834) | 2026-07-28 | test: add offload_train_target to the actor-factory args stub |
| [#1828](https://github.com/radixark/miles/pull/1828) | 2026-07-27 | fix(session): drop upstream Server/Date so aiohttp clients can read replies |
| [#1819](https://github.com/radixark/miles/pull/1819) | 2026-07-28 | refactor(session): track generated checkpoints separately from client assistant history |
| [#1826](https://github.com/radixark/miles/pull/1826) | 2026-07-28 | fix: allow TITO session rollback to the empty checkpoint |
| [#1833](https://github.com/radixark/miles/pull/1833) | 2026-07-28 | docs: add an Environments section to the user guide |
| [#1831](https://github.com/radixark/miles/pull/1831) | 2026-07-28 | docs: add Claude general code style rule |
| [#1741](https://github.com/radixark/miles/pull/1741) | 2026-07-28 | Add examples/swe-agent: GLM-4.7-Flash agentic training with Harbor |
| [#1915](https://github.com/radixark/miles/pull/1915) | 2026-07-28 | fix(lora): honor --attention-backend in the bridge LoRA path |
| [#1768](https://github.com/radixark/miles/pull/1768) | 2026-07-28 | codeowners: add Zhichenzzz to /miles/ and a /miles/dashboard/ rule |
| [#1820](https://github.com/radixark/miles/pull/1820) | 2026-07-28 | feat(session): pass request chat_template_kwargs to apply_chat_template |
| [#1935](https://github.com/radixark/miles/pull/1935) | 2026-07-29 | Use TransformerEngine for MXFP8 quantization |
| [#1733](https://github.com/radixark/miles/pull/1733) | 2026-07-29 | [AMD] Drop inert DSv4 rollout knobs and add an MTP recipe |
| [#1921](https://github.com/radixark/miles/pull/1921) | 2026-07-29 | Add NeMo-Gym integration: mini_swe_agent_2 via the agent function |
| [#1928](https://github.com/radixark/miles/pull/1928) | 2026-07-29 | [fix] DSA indexer on Blackwell: send the DSA indexer wk unquantized |
| [#2009](https://github.com/radixark/miles/pull/2009) | 2026-07-30 | [docs] Add Inkling-Small model page |
| [#1790](https://github.com/radixark/miles/pull/1790) | 2026-07-30 | openenv/tbench2: score the shared-server leg natively; retire the adapter compensation |
| [#2014](https://github.com/radixark/miles/pull/2014) | 2026-07-30 | fix: quantize non-interleaved DSA indexer wk |
| [#2012](https://github.com/radixark/miles/pull/2012) | 2026-07-30 | router: enable dp-aware routing under dp-attention |
| [#2028](https://github.com/radixark/miles/pull/2028) | 2026-07-30 | session: collect speculative-decoding counters |
| [#2013](https://github.com/radixark/miles/pull/2013) | 2026-07-31 | [tito] Add the Inkling TITO family (Inkling / Inkling-Small) |
| [#2035](https://github.com/radixark/miles/pull/2035) | 2026-07-31 | tests: mirror sglang_speculative_algorithm into session-server test namespaces |
| [#2040](https://github.com/radixark/miles/pull/2040) | 2026-07-31 | [AMD] DeepSeek-V4: use the ROCm precision-parity norm path |
| [#2045](https://github.com/radixark/miles/pull/2045) | 2026-08-01 | codeowners: cover miles/backends/experimental/fsdp_utils |
| [#2047](https://github.com/radixark/miles/pull/2047) | 2026-08-02 | fix glm47-flash: use paged MLA prefill on B200 |
| [#2075](https://github.com/radixark/miles/pull/2075) | 2026-08-02 | session: apply the trained LoRA adapter to session-server rollouts |
| [#2015](https://github.com/radixark/miles/pull/2015) | 2026-08-03 | scripts, examples: stop forcing the deprecated Miles router in launchers |
| [#1571](https://github.com/radixark/miles/pull/1571) | 2026-08-03 | GLM-5.2 kernel fix and GB300 training config |
| [#2134](https://github.com/radixark/miles/pull/2134) | 2026-08-03 | fix: skip the --dump-details processor dump when it cannot serialise |
| [#2136](https://github.com/radixark/miles/pull/2136) | 2026-08-04 | openenv/tbench2: drop the daytona bake CLI, which nothing can consume |
| [#2034](https://github.com/radixark/miles/pull/2034) | 2026-08-04 | [fp8] Gate the ue8m0 weight quantizer on the backend scale format |
| [#2139](https://github.com/radixark/miles/pull/2139) | 2026-08-04 | [AMD] retune AMD 4-node dsv4 config |
| [#2202](https://github.com/radixark/miles/pull/2202) | 2026-08-04 | fix(tito): prevent DeepSeek V4 system-tail mismatch |
| [#2205](https://github.com/radixark/miles/pull/2205) | 2026-08-04 | ci: authenticate CPU Hugging Face downloads |
| [#2021](https://github.com/radixark/miles/pull/2021) | 2026-08-04 | feat: expose Claude skills to agents |
| [#1913](https://github.com/radixark/miles/pull/1913) | 2026-08-05 | E2B sandbox backend (E2B Cloud / self-hosted AgentENV) + dedicated AgentENV recipe |
| [#2213](https://github.com/radixark/miles/pull/2213) | 2026-08-05 | fix(fsdp): apply the GDN packing patch to dense qwen3_5 and fix patch |
| [#2123](https://github.com/radixark/miles/pull/2123) | 2026-08-05 | refactor(session): extract helpers shared by v1 and v2 |
| [#2124](https://github.com/radixark/miles/pull/2124) | 2026-08-05 | feat(session): add the trajectory-tree data model |
| [#2125](https://github.com/radixark/miles/pull/2125) | 2026-08-05 | feat(session): add always-branch session state |
| [#2127](https://github.com/radixark/miles/pull/2127) | 2026-08-05 | docs(tito): rename the guide for agentic rollout |
| [#2129](https://github.com/radixark/miles/pull/2129) | 2026-08-05 | test(session): validate H200 v1/v2 agentic parity |
| [#2130](https://github.com/radixark/miles/pull/2130) | 2026-08-05 | test(ci): run session servers on v2 |
| [#1606](https://github.com/radixark/miles/pull/1606) | 2026-08-05 | ci(rocm): add ROCm CI workflow for MI300X self-hosted runners |
| [#2230](https://github.com/radixark/miles/pull/2230) | 2026-08-06 | fix(ci): disable MI300X runner jobs |
| [#2217](https://github.com/radixark/miles/pull/2217) | 2026-08-06 | test(ci): right-size session model GPU coverage |
| [#1919](https://github.com/radixark/miles/pull/1919) | 2026-08-06 | examples: add swe-agent-harbor-daytona (Harbor sandboxes on Daytona) |
| [#2228](https://github.com/radixark/miles/pull/2228) | 2026-08-06 | examples: make the agent-server trial timeout configurable |
| [#2233](https://github.com/radixark/miles/pull/2233) | 2026-08-06 | examples: rename swe-agent to swe-agent-harbor-docker |
| [#1739](https://github.com/radixark/miles/pull/1739) | 2026-08-06 | feat: add Verifiers rollout integration |
| [#2257](https://github.com/radixark/miles/pull/2257) | 2026-08-07 | ci: deprecated some sglang ci |
| [#2237](https://github.com/radixark/miles/pull/2237) | 2026-08-07 | fix(openenv): build E2B task templates as root |
| [#2269](https://github.com/radixark/miles/pull/2269) | 2026-08-07 | [fix] drop --disable-weights-backuper from the nemotron CI test |
| [#2079](https://github.com/radixark/miles/pull/2079) | 2026-08-07 | Fix padded -1 index handling in GLM-5 sparse-MLA tilelang kernels |
| [#2290](https://github.com/radixark/miles/pull/2290) | 2026-08-08 | fix(ci): persist every step metric for historical gate |
| [#1895](https://github.com/radixark/miles/pull/1895) | 2026-08-09 | Fix typo environment variable and unbuffer python outputs |
| [#1896](https://github.com/radixark/miles/pull/1896) | 2026-08-09 | Add a shell launch script test harness for future protection |
| [#1897](https://github.com/radixark/miles/pull/1897) | 2026-08-09 | Fix various launch scripts errors about missing line concatenations or paths |
| [#1898](https://github.com/radixark/miles/pull/1898) | 2026-08-09 | Derive the miles checkout location instead of hardcoding it in launch scripts |
| [#1899](https://github.com/radixark/miles/pull/1899) | 2026-08-09 | Snapshot the external commands of every shell launch script |
| [#1900](https://github.com/radixark/miles/pull/1900) | 2026-08-09 | Read the slurm allocation when the train config is built |
| [#1901](https://github.com/radixark/miles/pull/1901) | 2026-08-09 | Snapshot the commands and generated configs of every python launch script |
| [#1902](https://github.com/radixark/miles/pull/1902) | 2026-08-09 | Cover the public surface of command_utils with unit tests |
| [#1906](https://github.com/radixark/miles/pull/1906) | 2026-08-09 | Snapshot the launchers that build their own command line |
| [#1907](https://github.com/radixark/miles/pull/1907) | 2026-08-09 | Fix p2p profile's rotary_base not reaching the model script it configures |
| [#1908](https://github.com/radixark/miles/pull/1908) | 2026-08-09 | Snapshot test the argv of all model scripts |
| [#1909](https://github.com/radixark/miles/pull/1909) | 2026-08-09 | Expand the model args in python before building the command |
| [#1910](https://github.com/radixark/miles/pull/1910) | 2026-08-09 | Replace the model config shell scripts with python |
| [#1911](https://github.com/radixark/miles/pull/1911) | 2026-08-09 | Quote the model args miles inlines into the launch command |
| [#2279](https://github.com/radixark/miles/pull/2279) | 2026-08-09 | Run the launch script snapshot tests by hand instead of in CI |
| [#2274](https://github.com/radixark/miles/pull/2274) | 2026-08-09 | refactor(openenv): move duplicated sandbox helpers into the TB2 recipe |
| [#2275](https://github.com/radixark/miles/pull/2275) | 2026-08-09 | refactor(openenv): give the sandbox backends one SandboxBackend to share |
| [#2276](https://github.com/radixark/miles/pull/2276) | 2026-08-09 | feat(openenv): add Modal as a third per-episode sandbox backend |
| [#2220](https://github.com/radixark/miles/pull/2220) | 2026-08-09 | Add the GLM-5.2 744B x terminal-bench-2 Daytona example |
| [#2300](https://github.com/radixark/miles/pull/2300) | 2026-08-10 | scripts: enable the Miles dashboard in the quick-start launcher |
| [#2264](https://github.com/radixark/miles/pull/2264) | 2026-08-10 | docs: update homepage Core features section per the v0.1 feature list |
| [#2298](https://github.com/radixark/miles/pull/2298) | 2026-08-10 | docs: polish the Quick Start page |
| [#2271](https://github.com/radixark/miles/pull/2271) | 2026-08-10 | docs: refresh the homepage supported-models table |
| [#2225](https://github.com/radixark/miles/pull/2225) | 2026-08-10 | docs: trim repo README to banner and nav links |
| [#2373](https://github.com/radixark/miles/pull/2373) | 2026-08-11 | docs: NeMo-Gym server no longer needs a fork branch |
| [#2366](https://github.com/radixark/miles/pull/2366) | 2026-08-11 | docs: rewrite the fully async page around schedule, data path, eval, and metrics |
| [#2359](https://github.com/radixark/miles/pull/2359) | 2026-08-11 | docs: dashboard advanced features and example virtualization  |
| [#2374](https://github.com/radixark/miles/pull/2374) | 2026-08-11 | docs: split FAQ out of Resources and link the blog to LMSYS |
| [#2379](https://github.com/radixark/miles/pull/2379) | 2026-08-11 | docs: point the TB2 agent-server eval at the public harbor branch |
| [#2380](https://github.com/radixark/miles/pull/2380) | 2026-08-11 | docs: use NVIDIA's "NeMo Gym" spelling instead of "NeMo-Gym" |
| [#2377](https://github.com/radixark/miles/pull/2377) | 2026-08-11 | fix(kimi): align YaRN parameters with checkpoints |
| [#2303](https://github.com/radixark/miles/pull/2303) | 2026-08-11 | feat(ci): add authorized Neon SQL workflow |
| [#2376](https://github.com/radixark/miles/pull/2376) | 2026-08-11 | docs(developer): rewrite the developer guide against the code |
| [#2386](https://github.com/radixark/miles/pull/2386) | 2026-08-11 | fix: drop context parallelism from the FSDP backend |
| [#2391](https://github.com/radixark/miles/pull/2391) | 2026-08-11 | docs: replace DeepSeek V3/R1 page with a DeepSeek-V3.2 recipe |
| [#2384](https://github.com/radixark/miles/pull/2384) | 2026-08-11 | fix(fsdp): stop store_true from shadowing bool defaults in FSDPArgs |
| [#2381](https://github.com/radixark/miles/pull/2381) | 2026-08-11 | docs: recipe pages for Kimi-K3, Nemotron-3-Ultra and Gemma-4 |
| [#2395](https://github.com/radixark/miles/pull/2395) | 2026-08-11 | docs: drop the Platforms section and refresh the hardware list |
| [#2360](https://github.com/radixark/miles/pull/2360) | 2026-08-11 | examples: select fully-async with --fully-async in the GLM-5.2 Daytona recipe |
| [#2398](https://github.com/radixark/miles/pull/2398) | 2026-08-11 | docs: download the DAPO jsonl mirror the launcher expects |
| [#2396](https://github.com/radixark/miles/pull/2396) | 2026-08-11 | docs: refresh MXFP8 and NVFP4 RL guide |
| [#2399](https://github.com/radixark/miles/pull/2399) | 2026-08-11 | docs: fix the gsm8k download on the reproducibility page |
| [#2402](https://github.com/radixark/miles/pull/2402) | 2026-08-11 | scripts: download the DAPO dataset the NPU recipe trains on |
| [#2265](https://github.com/radixark/miles/pull/2265) | 2026-08-11 | [AMD] Enable Qwen3 FSDP hybrid-shard CI on ROCm |
| [#2354](https://github.com/radixark/miles/pull/2354) | 2026-08-11 | Delete the glm4-9B, mimo-7B, moonlight-16B and deepseek-r1 launch scripts |
| [#2388](https://github.com/radixark/miles/pull/2388) | 2026-08-11 | fix(fsdp): make --config actually apply, and reject keys it does not know |
| [#2355](https://github.com/radixark/miles/pull/2355) | 2026-08-11 | [fix] fix the bugs/outdated commands in `.sh` scripts and the corresponding snapshots |
| [#2375](https://github.com/radixark/miles/pull/2375) | 2026-08-11 | docs: reorganize the Training Backends section, drop the experimental FSDP framing |
| [#2349](https://github.com/radixark/miles/pull/2349) | 2026-08-11 | docs: expand LoRA training and serving guide |
| [#2368](https://github.com/radixark/miles/pull/2368) | 2026-08-11 | fix(rollout): group session v2 leaf samples |
| [#2356](https://github.com/radixark/miles/pull/2356) | 2026-08-11 | Replace all the `.sh` launch scripts with `.py` launch script |
| [#2403](https://github.com/radixark/miles/pull/2403) | 2026-08-11 | fix(ci): unblock ROCm fork PRs at checkout |
| [#2411](https://github.com/radixark/miles/pull/2411) | 2026-08-12 | docs: drop the Latest updates section from the homepage |
| [#2383](https://github.com/radixark/miles/pull/2383) | 2026-08-12 | [readme]: rewrite the README |
| [#2476](https://github.com/radixark/miles/pull/2476) | 2026-08-12 | fix(dashboard): zero the trainer log-probs the loss masks out |
| [#2478](https://github.com/radixark/miles/pull/2478) | 2026-08-12 | docs: rewrite INT4 QAT guide |
| [#2394](https://github.com/radixark/miles/pull/2394) | 2026-08-12 | docs: let tables wrap instead of silently clipping at max-content |
| [#2280](https://github.com/radixark/miles/pull/2280) | 2026-08-12 | [example] GLM-5.2 744B-A40B LoRA agentic launcher (TB2 on Daytona) |
| [#2367](https://github.com/radixark/miles/pull/2367) | 2026-08-12 | Add a PPO example for the shared actor/critic setup |
| [#2482](https://github.com/radixark/miles/pull/2482) | 2026-08-12 | examples: fix READMEs — dead links, broken list structure, missing entries |
| [#2484](https://github.com/radixark/miles/pull/2484) | 2026-08-12 | docs: add disaggregated RL rollout guide |
| [#2490](https://github.com/radixark/miles/pull/2490) | 2026-08-12 | dashboard: keep sample status/reward chips visible on the Tokens tab |
| [#2492](https://github.com/radixark/miles/pull/2492) | 2026-08-12 | docs: list Kimi-K2.6 as a P2P-supported model |
| [#2479](https://github.com/radixark/miles/pull/2479) | 2026-08-12 | examples: computer-use RL on HUD v6 environments |
| [#2389](https://github.com/radixark/miles/pull/2389) | 2026-08-13 | fix: avoid Mooncake metrics port collisions |
| [#2491](https://github.com/radixark/miles/pull/2491) | 2026-08-13 | docs: nest Environments under User Guide, rename to Agentic Environments |
| [#2487](https://github.com/radixark/miles/pull/2487) | 2026-08-13 | docs: drop the Contact button from the docs navbar |
| [#2526](https://github.com/radixark/miles/pull/2526) | 2026-08-13 | fix(swe-agent example): stop logging unmeasured agent metrics as zero |
| [#2214](https://github.com/radixark/miles/pull/2214) | 2026-08-13 | fix(ci): calibrate lora E2E estimates from nightly runs and halve GLM5 lora matrices |
| [#2533](https://github.com/radixark/miles/pull/2533) | 2026-08-13 | docs: rebrand environments bullet as agentic, add HUD, sync README and docs index |
| [#2347](https://github.com/radixark/miles/pull/2347) | 2026-08-13 | [AMD] Enable amd pr ci |
| [#2536](https://github.com/radixark/miles/pull/2536) | 2026-08-14 | Carry the rollout id on both agentic paths |
| [#2531](https://github.com/radixark/miles/pull/2531) | 2026-08-13 | fix(flops): stop assuming every HF config has intermediate_size |
| [#2397](https://github.com/radixark/miles/pull/2397) | 2026-08-13 | fix(fsdp): move the AMD Triton attention bridge in-tree |
| [#2537](https://github.com/radixark/miles/pull/2537) | 2026-08-13 | docs: quick-start pre-flight checks, disk sizing, and honest timing |
| [#2544](https://github.com/radixark/miles/pull/2544) | 2026-08-14 | Do not kill the run when one sample's collect_samples loses its connection |
| [#2480](https://github.com/radixark/miles/pull/2480) | 2026-08-14 | [AMD CI] Enable verified ROCm tests |
| [#2545](https://github.com/radixark/miles/pull/2545) | 2026-08-14 | Reuse one Daytona client per process instead of one per create attempt |
| [#2564](https://github.com/radixark/miles/pull/2564) | 2026-08-14 | release: bump miles version to 0.1.0 |
| [#2523](https://github.com/radixark/miles/pull/2523) | 2026-08-14 | docs: delete the FAQ page |
| [#2565](https://github.com/radixark/miles/pull/2565) | 2026-08-15 | docs: fold Welcome into the User Guide tab and lead with it |
| [#2577](https://github.com/radixark/miles/pull/2577) | 2026-08-16 | fix: make the Verifiers E2E self-contained |
| [#2578](https://github.com/radixark/miles/pull/2578) | 2026-08-16 | fix: Improve Qwen3 FSDP long CI perf |
| [#2575](https://github.com/radixark/miles/pull/2575) | 2026-08-16 | feat: add the Qwen3.8-27B recipe |
| [#2558](https://github.com/radixark/miles/pull/2558) | 2026-08-17 | test(ci): move the agentic-env integration tests under tests/fast so CI runs them |
| [#2561](https://github.com/radixark/miles/pull/2561) | 2026-08-17 | test(ci): file the verifiers tests under the example they test |
| [#2585](https://github.com/radixark/miles/pull/2585) | 2026-08-17 | fix: add H200, B200 and B300 to NUM_GPUS_OF_HARDWARE |
| [#2587](https://github.com/radixark/miles/pull/2587) | 2026-08-17 | openenv: stop the episode at a length-truncated turn |
| [#2589](https://github.com/radixark/miles/pull/2589) | 2026-08-17 | Do not abort in-flight requests when bumping the engine weight version |
| [#2588](https://github.com/radixark/miles/pull/2588) | 2026-08-17 | glm52_tbench2: collapse levers and launch fixes from the GB300 16-node smoke runs |
| [#2594](https://github.com/radixark/miles/pull/2594) | 2026-08-18 | docs: name the dense variant in the Qwen3.8 model-table rows |
| [#2583](https://github.com/radixark/miles/pull/2583) | 2026-08-18 | docs(diffusion): miles diffusion doc |
| [#2599](https://github.com/radixark/miles/pull/2599) | 2026-08-18 | docs: point at Miles-diffusion for diffusion post-training |
| [#2603](https://github.com/radixark/miles/pull/2603) | 2026-08-18 | docs: add the acknowledgment logo wall to the README and docs landing page |
| [#2602](https://github.com/radixark/miles/pull/2602) | 2026-08-18 | glm52_tbench2: default the train set to the 69-task split |
| [#2607](https://github.com/radixark/miles/pull/2607) | 2026-08-18 | docs: fix diffusion docs |
| [#2650](https://github.com/radixark/miles/pull/2650) | 2026-08-18 | docs: flesh out the Qwen3.8-2.4T-A95B recipe on the Qwen3.8 page |
| [#2654](https://github.com/radixark/miles/pull/2654) | 2026-08-18 | docs: name the Slack channel #miles-rl |
| [#2655](https://github.com/radixark/miles/pull/2655) | 2026-08-18 | docs: improve Miles Diffusion discovery |
| [#2656](https://github.com/radixark/miles/pull/2656) | 2026-08-18 | docs: filter the blog link to Miles posts |
| [#2657](https://github.com/radixark/miles/pull/2657) | 2026-08-18 | docs: add v0.1 release news to README and docs landing |
| [#2658](https://github.com/radixark/miles/pull/2658) | 2026-08-18 | docs: link the Miles website from the README |
