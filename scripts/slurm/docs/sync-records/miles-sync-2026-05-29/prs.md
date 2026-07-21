# miles upstream PRs report

**Period**: since `7a6cf48e6` (2026-05-19, *add Code of Conduct (#1145)*)
**Generated**: 2026-05-29
**Total PRs**: 37
**Watchlist hits (pin sources)**: 1
**Flagged (touch files we've modified)**: 8
**Other**: 28

---

## ⚠️ Watchlist hits (pin sources)

These upstream PRs touch files that drive `scripts/slurm/setup/pins.env` and `install_env.sh`. After sync, run `extract_pins.py --check` (and probably `--write`), then verify install_env.sh mirrors any new RUN lines.

### PR #1164 — Bump sglang to v0.5.12
**Merged**: 2026-05-22 | **Author**: yueming-yuan | **+49/-12** | **Labels**: run-ci-image
**Pin sources touched**: `docker/Dockerfile`
**Also touches our modified files**: `miles/ray/rollout.py`

---

## PRs that touch files we've modified

### PR #963 — Support DeepSeek v3.2
**Merged**: 2026-05-28 | **Author**: zianglih | **+974/-31** | **Labels**: run-ci-model-scripts
**Overlap with OUR_TOUCHED**: `miles/utils/arguments.py`

### PR #1149 — [CI] Refactor CI workflow into stage classified by hardware and gpu num, fix buggy tests
**Merged**: 2026-05-21 | **Author**: guapisolo | **+2500/-1375** | **Labels**: run-ci-image
**Overlap with OUR_TOUCHED**: `.github/workflows/pr-test.yml`, `.gitignore`

### PR #1173 — ci: allow overriding PR test image tag
**Merged**: 2026-05-21 | **Author**: yueming-yuan | **+40/-8** | **Labels**: run-ci-short
**Overlap with OUR_TOUCHED**: `.github/workflows/pr-test.yml`, `.gitignore`

### PR #1216 — [refactor] hf_config: single entry point with alias registration and overrides
**Merged**: 2026-05-28 | **Author**: yueming-yuan | **+269/-19** | **Labels**: run-ci-megatron
**Overlap with OUR_TOUCHED**: `miles/utils/arguments.py`

### PR #1219 — 1/5 support kimi 2.5 full + lora: unify multimodal processor invocation
**Merged**: 2026-05-28 | **Author**: nanjiangwill | **+32/-5** | **Labels**: run-ci-short
**Overlap with OUR_TOUCHED**: `miles/rollout/sglang_rollout.py`

### PR #1222 — 4/5 support kimi 2.5 full + lora: shared-outer grouped-expert LoRA
**Merged**: 2026-05-28 | **Author**: nanjiangwill | **+312/-37** | **Labels**: run-ci-lora
**Overlap with OUR_TOUCHED**: `miles/utils/arguments.py`

### PR #1223 — 5/5 support kimi 2.5 full + lora: LoRA base-weight CPU backup
**Merged**: 2026-05-29 | **Author**: nanjiangwill | **+46/-8** | **Labels**: run-ci-megatron, run-ci-lora
**Overlap with OUR_TOUCHED**: `miles/utils/arguments.py`

### PR #1243 — [ci] precommit: ban bare AutoConfig/AutoTokenizer.from_pretrained
**Merged**: 2026-05-29 | **Author**: yueming-yuan | **+9/-2**
**Overlap with OUR_TOUCHED**: `.pre-commit-config.yaml`, `miles/utils/arguments.py`

---

## Other PRs (no overlap detected)

| PR | Title | Merged | +/- |
|----|-------|--------|-----|
| #753 | [loss refactor] [1] snapshot tests + file structure | 2026-05-22 | +1607/-924 |
| #984 | fix: add b300 qwen35 script and upd spec v2 | 2026-05-22 | +29/-14 |
| #1062 | Random fully async agent example | 2026-05-27 | +1000/-1 |
| #1069 | [fix] fix truncate function return routed expert | 2026-05-21 | +2/-16 |
| #1121 | [loss refactor] [2] code style improvements | 2026-05-22 | +94/-41 |
| #1125 | npu support qwen3-4b. | 2026-05-29 | +2431/-0 |
| #1132 | [loss refactor] [3] file restructure and rename | 2026-05-22 | +70/-65 |
| #1150 | fix: expert_overlap and pp issues in nemotron-3-super | 2026-05-20 | +272/-0 |
| #1151 | Fix assignment of train_scored_log_probs variable | 2026-05-19 | +1/-1 |
| #1161 | fix: skip actor CPU backup for disaggregated Megatron training | 2026-05-21 | +13/-2 |
| #1167 | Add @guapisolo as code owner for workflows | 2026-05-21 | +2/-2 |
| #1175 | fix: nemotron H megatron bridge bug, recover test_dumper.py & test_run_megatron.py | 2026-05-22 | +29/-25 |
| #1180 | ci: display improvement | 2026-05-23 | +644/-44 |
| #1185 | Fix CP slice parallel state access | 2026-05-24 | +2/-2 |
| #1189 | fix GLM-5 training script + dsv32/glm-5 config | 2026-05-25 | +82/-27 |
| #1190 | Guard zero rollout temperature logprob scaling | 2026-05-25 | +2/-1 |
| #1191 | [CI] Add GLM-5 4-layer CI e2e test + model-scripts test suite | 2026-05-26 | +54/-2 |
| #1193 | Add yueming-yuan as code owner for workflows | 2026-05-25 | +1/-1 |
| #1195 | Fix dump details debug import | 2026-05-26 | +1/-1 |
| #1213 | resolve chat template encoder for non-HF models (deepseek v32) | 2026-05-27 | +414/-17 |
| #1215 | hf_attention: drop outdated namespace fallback for qwen3.5 | 2026-05-26 | +8/-42 |
| #1220 | 2/5 support kimi 2.5 full + lora: VL-aware quantization/conversion tools | 2026-05-28 | +16/-6 |
| #1221 | 3/5 support kimi 2.5 full + lora: Kimi K2.5 VL full-parameter support | 2026-05-28 | +516/-2 |
| #1224 | ci: remove broken release-docs workflow | 2026-05-29 | +0/-53 |
| #1225 | Add Terminal-Bench eval driver targeting miles_agent_server (DeepSeek-V4-Pro example) | 2026-05-29 | +407/-0 |
| #1227 | refactor(ci): restructure stage-a cpu registration to location-based discovery, move gpu fast tests | 2026-05-29 | +773/-383 |
| #1236 | Add 2-node TB2 training example targeting the harbor-private agent server | 2026-05-29 | +220/-0 |
| #1237 | [refactor] hf_tokenizer: route AutoTokenizer call sites through load_tokenizer | 2026-05-28 | +30/-20 |
