# miles upstream PRs report

**Period**: since `7a6cf48e6` (`2026-05-19` — "add Code of Conduct (#1145)")
**Upstream tip**: `1e1679706` "Support atomic weight update groups (#1264)"
**Total commits**: 100 (~99 PRs)
**Flagged (touch files we've modified)**: 9 files
**Watchlist hits (touch pin-source files)**: 1 (sglang bump), +2 requirements.txt (test-infra)

> Note: per-PR `gh pr view` metadata was **not** fetched for all ~99 PRs (cost). This
> report is built from git topology — watchlist hits, the both-sides-changed file set
> (the conflict predictor), and thematic grouping. The flagged section is what matters
> for the merge.

---

## ⚠️ Watchlist hits (pin sources)

### PR #1164 — Bump sglang to v0.5.12 — `docker/Dockerfile`
**Commit**: `ae083c551`

**Pin impact** (the only ARG churn; no new RUN/ENV lines):
- `SGLANG_IMAGE_TAG`: `v0.5.10` → `v0.5.12-cu129`
- `WHEELS_TAG`: `cu129-x86_64` → `cu129-x86_64-v0.5.12`

⇒ This is the **sglang-sync trigger**. After the merge, `extract_pins.py --write`
will move `UPSTREAM_SGLANG_IMAGE_TAG`/`UPSTREAM_WHEELS_TAG` forward and print
`[sglang-sync pending]`. ACTIVE (`cu129-x86_64`, sglang v0.5.10, torch 2.9.1) stays
put until `/sglang-sync cu129-x86_64-v0.5.12` advances it (target: sglang v0.5.12,
torch 2.11.0, router 0.3.2). The target tag `cu129-x86_64-v0.5.12` is **already in
`WHEELS_STACK`**, so the line is ready → **sync together** (miles-sync default).

### PRs #1084, #1259 — `requirements.txt` (test-infra, low risk)
- #1084 "Add real-Ray integration tests for RolloutManager" — adds a test dep.
- #1259 "Fix stage-c GPU CI: test_dumper entrypoint" — CI fix.

`requirements.txt` is **not** consumed by `extract_pins.py` (pins come from
`docker/Dockerfile`), so no pins.env impact. Worth a glance for new runtime deps but
not a blocker. We have not modified `requirements.txt` → clean merge expected.

---

## 🚩 PRs that touch files we've modified (conflict predictor)

9 files changed on **both** sides of the merge-base. Listed by expected conflict severity.

### 🔴 `miles/ray/rollout.py` — MODIFY/DELETE
- **Upstream**: `68a8f11ca` "Refactor rollout.py by file decompositions (#899)" **DELETES**
  the file (1326 → 0 lines; logic moved into `miles/rollout/` modules). Also touched by
  `ae083c551` (#1164) before deletion.
- **Ours**: `+4 -4` patch to `_compute_zero_std_metrics` — change `zero_std/*` metric
  keys from rounded-string to float keys (`counts.get(0.0, ...)`, `{reward:g}` fmt).
- **Conflict type**: modify/delete. Git will report it as a conflict on a deleted file.
  Resolution is **not** ours/theirs — our `_compute_zero_std_metrics` tweak must be
  **ported forward** to wherever that function landed in the decomposition. NEEDS HUMAN
  DECISION.

### 🟠 `.github/workflows/pr-test.yml`
- **Upstream**: `+101 -129` — `cdb6f0032` (#1149) "Refactor CI workflow into stage
  classified by hardware/gpu" + `39958648e` (#1173) "allow overriding PR test image tag".
- **Ours**: `+6 -2` (last sync we took upstream's structure wholesale, then re-applied a
  small local tweak).
- **Conflict type**: content. Likely "take upstream's new structure wholesale, re-apply
  our small tweak" again — same call as last sync. CONFIRM WITH USER.

### 🟠 `miles/rollout/sglang_rollout.py`
- **Upstream**: `+12 -7` — `b39634df4` (#1252 indexer topk), `f1843b6fc` (#906 walrus),
  `5af8043da` (#1219 kimi multimodal processor).
- **Ours**: `+47 -3` — VLM `mm_token_type_ids` drop for Qwen3-VL/transformers>=5.0,
  `call_all_samples_process_fn` soft-call on train path, eval-path all-samples-process
  hook mirror.
- **Conflict type**: content. Our additions are substantial; #1219 touches the same
  multimodal-processor region. Likely conflict. NEEDS REVIEW.

### 🟡 `miles/utils/arguments.py`
- **Upstream**: `+55 -4` — touched by 7 PRs (#1253 replay, #1252 indexer, #1243 precommit,
  #1222/#1223 kimi lora, #963 dsv32, #1216 hf_config).
- **Ours**: `+5 -1` — a small arg addition.
- **Conflict type**: content, but our change is tiny; likely auto-merges or a small hunk.

### 🟡 `miles/backends/megatron_utils/update_weight/.../broadcast.py`
- **Upstream**: `+2 -4` — `f1843b6fc` (#906 walrus refactor).
- **Ours**: `+91 -21` — our disagg/bridge weight-update change (large local mod).
- **Conflict type**: content. Upstream's change is tiny (walrus operator); ours is large.
  Likely a small overlapping hunk. NEEDS REVIEW.

### 🟡 `miles/rollout/generate_utils/generate_endpoint_utils.py`
- **Upstream**: `+27 -5` — `b39634df4` (#1252 indexer topk).
- **Ours**: `+2 -1`. Small local change; possible small conflict.

### 🟢 `miles/backends/megatron_utils/model.py`
- **Upstream**: `+6 -6` — `f1843b6fc` (#906 walrus).
- **Ours**: `+10 -2`. Possible small conflict.

### 🟢 `.gitignore`
- **Upstream**: `+2 -0` (#1149, #1173). **Ours**: `+16 -0`.
- **Conflict type**: union (same call as last sync — keep both sides' ignore entries).

### 🟢 `.pre-commit-config.yaml`
- **Upstream**: `+7 -0` — `3c2cef5de` (#1243) bans bare `AutoConfig/AutoTokenizer.from_pretrained`.
- **Ours**: `+1 -1`. Likely small conflict or clean union.

---

## Thematic grouping of the ~99 PRs

- **Rollout refactor (huge)**: #899–#947, #1070–#1084 — decompose `miles/ray/rollout.py`
  into modules, make rollout management async, add MockSGLangEngine + extensive rollout
  tests. **This is why `miles/ray/rollout.py` is deleted.**
- **sglang v0.5.12 bump**: #1164 (watchlist — drives sglang-sync).
- **Loss refactor**: #753, #1121, #1132 (snapshot tests, file restructure/rename).
- **Model support**: DeepSeek v3.2 (#963, #1213, #1273, #1278 v4), Kimi 2.5 full+lora
  (#1219–#1223), GLM-5 (#1189, #1191), qwen3.5 (#1215), npu qwen3-4b (#1125), gemma.
- **Indexer / replay**: #1248–#1257 (replay registration, GLM5 indexer replay).
- **Atomic weight update groups**: #1264 (upstream tip).
- **fp8**: #1182 "Select SGLang FP8 block quant kernel to match inference".
- **CI**: #1149, #1173, #1191, #1227, #1259, #1270, #1274.
- **TITO session refactor**: #1142.
- **Misc fixes / code owners**: #1150, #1151, #1161, #1167, #1175, #1185, #1190, #1193,
  #1195, #1262, #1263, #1265, #1266, #1279, #1247.

---

## Recommendation

Proceed with the merge. Expect conflicts on the 9 both-changed files above — the
`miles/ray/rollout.py` modify/delete is the one that needs real thought (port our
`_compute_zero_std_metrics` tweak forward). Per HARD RULE 1 the skill STOPS at the
conflict and surfaces everything before resolving.

After the merge resolves, the sglang gate fires: `[sglang-sync pending]` → drive
`/sglang-sync cu129-x86_64-v0.5.12` in the same PR (line is ready).
