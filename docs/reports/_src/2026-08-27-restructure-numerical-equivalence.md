---
title: Restructure numerical equivalence — updated orbit-main heads vs the pre-restructure stack on H200
kind: investigation
profile: development-log
status: final
date: 2026-08-27
tags: numerics, restructure, megatron, bridge, sglang, oft, lora, verification, h200
old_stack: orbit 2f8015cb (worktree 60679af, package-identical), Megatron-LM 00eb75b0c, Megatron-Bridge ad26fc46, sglang a6fe249b3, env orbit_cu130_zeju
new_stack: orbit 197c9137 (pushed head be8692e, runtime-identical), Megatron-LM 83879096b, Megatron-Bridge 988d6426, sglang a6fe249b3, env orbit_cu130_omr
fixes: Megatron-Bridge orbit-main fast-forwarded ad26fc46 to 988d6426; orbit be8692e repins megatron-core 83879096b and sglang a6fe249b3
jobs: 2952 install, 2955 mlm-kernels, 2966 bridge (2956 first attempt), 2983/2984/2985 stage A, 2980/2981/2982 stage B, 2959/2960 smokes
evidence: remote-cluster-runs slurm/orbit/orbit-main/20260827T004939Z-r517330440 (cluster + local mirror)
cluster: slurm H200 (slinky nodes), partition all
---

<section class="report-summary" aria-label="Outcome">
  <p class="summary-label">Outcome</p>
  <p class="summary-title">The updated orbit-main branch heads — restructured Megatron-LM, the shimless orbit-namespace Megatron-Bridge, and the migrated orbit — are numerically equivalent to the pre-restructure stack: bit-identical on every deterministic H200 axis. Serving matches in all 12 cross-arm cells (fa3, triton, flashinfer, each with base, OFT b32, LoRA r16), the trainer step matches per-token logprobs and grad norms to the last digit for base, OFT, and LoRA, and the moved Megatron kernels match 15/15 captured tensors bitwise.</p>
  <p class="summary-detail">Caveats: the GRPO e2e smoke is non-bitwise by construction (sampled rollouts) and instead matched the qualified logprob-sync envelope; two vendored TE reference files under te_oft/ are unimportable in the new tree, but provably unimportable in the old tree too (pre-existing dead code, not a restructure effect). En route, two repo-state gaps were fixed and pushed: Megatron-Bridge orbit-main was fast-forwarded to the pinned commit, and orbit's stale megatron-core/sglang pins were bumped (be8692e).</p>
</section>

<div class="status-grid" role="list" aria-label="Program status">
  <div class="status-item" data-status="complete" role="listitem"><strong class="status-value">7</strong><span class="status-label">Battery legs green</span></div>
  <div class="status-item" data-status="complete" role="listitem"><strong class="status-value">18</strong><span class="status-label">Bit-identical comparison cells</span></div>
  <div class="status-item" data-status="complete" role="listitem"><strong class="status-value">2</strong><span class="status-label">Repo-state fixes pushed</span></div>
  <div class="status-item" data-status="open" role="listitem"><strong class="status-value">3</strong><span class="status-label">Optional follow-ups</span></div>
</div>

## Question and scope

The Sphere fork isolation restructure (GPU-qualified pre-merge on 2026-08-25) was pushed to
the Sphere-AI-Lab `orbit-main` branches on 2026-08-26, with two additional commits beyond the
qualified state: a Bridge-side sphere-to-orbit namespace rename plus removal of all legacy
import shims, and an orbit-side migration to the new namespace. The question: is the published
new code numerically the same as the old code?

| repo | old (baseline) | new (orbit-main head) | delta |
|:--|:--|:--|:--|
| orbit | `2f8015cb` (via worktree `60679af`; `orbit/` + `train.py` verified byte-identical) | `197c9137` | namespace migration `3306f84` (imports + 2 docstrings), OPD-only fix `18cc27f`, launchers/setup |
| Megatron-LM | `00eb75b0c` | `83879096b` | exactly the qualified restructure commit (kernels moved into `experimental_attention_variant/`) |
| Megatron-Bridge | `ad26fc46` | `988d6426` | qualified `69eba707` + rename `69a8e369` + shim removal `988d6426` — the two new commits were unqualified before this campaign |
| sglang | `a6fe249b3` | `a6fe249b3` | unchanged |

Both arms therefore share one sglang commit and differ only in the three restructured trees.
The comparison arms are complete installed environments built by the same cu130 installer:
`orbit_cu130_zeju` (old, pre-existing and previously verified) and `orbit_cu130_omr` (new,
built this campaign from the branch heads). Acceptance bar: bit-identity, justified because
both arms share the same megatron-core 0.18 line and the static delta is code motion only.

## Completed deliverables

- New-stack environment `orbit_cu130_omr` installed from the orbit-main heads and verified
  39/39 (job 2952); editable sources at `~/miles-orbit/sources/orbit_cu130_omr/`.
- Full H200 equivalence battery: static equivalence, kernel parity, Bridge contracts and unit
  tests, Stage A serving matrix, Stage B frozen-batch trainer, GRPO OFT e2e smoke pair.
- Megatron-Bridge `orbit-main` fast-forwarded `ad26fc46..988d6426` (7 commits; clean
  ancestry verified) so the branch equals the commit orbit pins.
- orbit `be8692e` (pushed as the new `orbit-main` head): repins megatron-core to `83879096b`
  and sglang to `a6fe249b3` across `pyproject.toml`, `uv.lock`, both `pins.env` files,
  `CUDA-13-install.md`, and the two pin-guard test suites. Before this fix a fresh install
  from orbit's own pins silently got the pre-restructure Megatron-LM. `be8692e` touches no
  runtime code, so this campaign's verdict (measured at `197c9137`) applies to it verbatim.
- Durable evidence in the run store
  `slurm/orbit/orbit-main/20260827T004939Z-r517330440` (cluster and local mirror), including
  `VERDICT.md`, per-leg provenance, raw probe captures, and all failed attempts.

## Static equivalence

Login-node analysis before any GPU was spent:

- **Bridge `69eba707..988d6426`** (the unqualified delta): 68/68 renamed files are
  byte-identical after the mechanical sphere-to-orbit token rewrite; all 31 deletions are
  compatibility shims (every header carries the shim marker; `kimi_vl/utils.py` is a shim with
  a long license header); additions are the renamed package inits plus
  `tests/unit_tests/test_orbit_namespace.py`. The single non-mechanical modification is the
  removal of the legacy `low_precision` re-export from `models/conversion/__init__.py`,
  consistent with shim removal. No `megatron.bridge.sphere` references remain.
- **orbit `2f8015cb..197c9137`, package scope**: only two commits touch `orbit/` or
  `train.py`. The namespace migration `3306f84` changes import lines plus two docstring
  mentions; `18cc27f` touches the OPD rollout path only, which this battery does not
  exercise. `orbit_plugins/model_args/` (sourced by every recipe) is byte-identical across
  arms; the five `orbit_plugins/megatron_bridge/patches/` files change import paths only.
- **Import contract**: all 17 distinct `megatron.bridge` module references in the orbit
  package resolve against the Bridge `988d6426` tree.
- **sglang pin move context** (for the be8692e pins fix): `51845dc4a..a6fe249b3` is three
  serving fixes plus a merge, all Python-side; `sgl-kernel/` is byte-identical, so kernel
  builds and prebuilt wheels are unaffected.

## Verification — environment and structure

- Install job 2952 (installer from the new head itself, sglang overridden to the orbit-main
  head): verify_env **39/39**, same count as the old env's original qualification.
- **Megatron-LM kernel parity** (job 2955, both captures on one H200): the restructure-moved
  kernels — `matmul_fixed_order`, `LinearFixedOrderFn` forward and backward,
  `BatchInvariantRMSNormFn` forward and backward in both zero-centered modes, and the TEGemm
  TN / dsv4 fast paths — are **bitwise-equal in all 15 captured tensors** between the old and
  new environments (`KERNEL_PARITY_BITWISE_EQUAL`).
- **Bridge contracts and unit tests** (job 2966): old arm passes the legacy-path contract and
  `test_adapter_export.py` **29/29**; new arm passes **70 tests**
  (adapter-export 29 + the new orbit-namespace suite), all 14 removed legacy paths are
  confirmed absent, and a full import walk of `megatron.bridge.orbit` covers 51 modules with
  **0 unexpected errors**.

<aside class="finding" data-tone="neutral">
  <p class="block-label">Finding</p>
  <p>The import walk surfaced exactly two failing modules: the vendored TE reference copies
  te_oft/ref_te_common.py and te_oft/ref_te_layernorm_linear.py. Git shows their imports
  (a relative cpp_extensions, a relative debug module) dangling at the old SHA as well, and a
  parity-of-brokenness probe in the old environment confirms both are equally unimportable
  there (OLD_BROKEN_PARITY_OK). Pre-existing dead reference code; not a restructure effect.
  The first bridge job (2956) failed only because the walk was stricter than the old tree's
  reality; it was re-run as 2966 with the parity assertion.</p>
</aside>

## Verification — serving (Stage A)

Method: per cell, 16 gsm8k-test chat-template prompts, 64 greedy tokens,
`enable_deterministic_inference`, radix cache disabled, TP1, pinned attention backend,
synthetic byte-identical adapters shared by both arms (canonical OFT block 32 on all 7 linear
targets; LoRA rank 16). Each boot also records `base_greedy`, giving a per-backend within-arm
control. Both arms report the same engine build (`sglang 0.0.0.dev1+ga6fe249b3`).

**All 12 cross-arm cells are bit-identical: 16/16 tokens identical and 0.00 logprob delta at
every position** — OFT, LoRA, and both base boots, on fa3, triton, and flashinfer.

| backend | oft_greedy | base (oft boot) | lora_greedy | base (lora boot) |
|:--|:--|:--|:--|:--|
| fa3 | 16/16, 0.00 | 16/16, 0.00 | 16/16, 0.00 | 16/16, 0.00 |
| triton | 16/16, 0.00 | 16/16, 0.00 | 16/16, 0.00 | 16/16, 0.00 |
| flashinfer | 16/16, 0.00 | 16/16, 0.00 | 16/16, 0.00 | 16/16, 0.00 |

The within-arm control (base_greedy across the two independent engine boots of the same arm)
reproduces the known cross-config effect of the OFT boot's prefill-graph disable — and the two
arms show **identical fingerprints to the digit** (max abs dlogprob 1.369e-01 on fa3,
1.194e-01 on triton, 1.108e-01 on flashinfer, with identical token-flip counts 11/10/13 of
16). The new stack does not merely match outputs; it reproduces the old stack's
config-sensitivity exactly. flashinfer, unmeasurable on the older B200 comparison, boots and
matches cleanly on H200.

## Verification — trainer (Stage B)

Method: the frozen 16-sequence batch from the 2026-08-24 port campaign
(`batch_orbit_0.pt`, reused byte-identically by both arms as `batch_frozen_0.pt`), the
byte-copied campaign recipe with the debug-replay override block appended (argparse
last-wins): 1 GPU, TP1/PP1/DP1, one GRPO step, `--load-debug-rollout-data` +
`--debug-train-only` (zero engines), eps-clip 1e9, both normalizations off,
`--ci-save-grad-norm`. Cells: {old, new} x {base, oft b32, lora r16} x 2 repeats; each
method's four cells ran on one node and GPU.

Both arms are exactly deterministic (repeat pairs bit-equal everywhere), and **every
cross-arm comparison is bit-identical**: tokens 16/16, advantages 16/16, per-token logprobs
16/16 `torch.equal` with max abs delta 0.00, and grad norms equal to the last printed digit.

| method | logprobs (old vs new) | grad norm old | grad norm new | equal |
|:--|:--|:--|:--|:--|
| base | 16/16 bit-equal, max 0.00 | 5.604307651519775 | 5.604307651519775 | yes |
| oft b32 | 16/16 bit-equal, max 0.00 | 1.0412166118621826 | 1.0412166118621826 | yes |
| lora r16 | 16/16 bit-equal, max 0.00 | 0.8442838191986084 | 0.8442838191986084 | yes |

## Verification — e2e smoke (GRPO OFT, 8x H200)

The 3-rollout colocate GRPO OFT smoke (qwen2.5-0.5B, gsm8k, OFT b32 canonical, per-rollout
adapter re-sync, checkpoint save) ran once per arm from the arm's own worktree and
environment (jobs 2959/2960). Both completed cleanly (exit 0, all adapter POSTs 200, PEFT
checkpoint saved). This axis samples rollouts, so it is non-bitwise by construction; the
comparison metric is the trainer-vs-engine logprob sync fidelity:

| round | old lp_absdiff | new lp_absdiff |
|:--|:--|:--|
| 0 | 0.00874 | 0.00873 |
| 1 | 0.00934 | 0.00912 |
| 2 | 0.00946 | 0.00934 |

Both sit inside the previously qualified envelope (0.0081–0.0099 across the vanilla and
pre-merge-restructure smokes). Raw rewards (32 samples per round) are at noise scale in both
arms: old {0.625, 0.344, 0.313}, new {0.719, 0.375, 0.5}, against the 0.31–0.75 range of
prior runs of this smoke.

## Campaign mechanics and incidents

The battery ran as chained Slurm jobs with a shared run store. After the serial chain proved
slower than needed, Stage A was split per backend and Stage B per method into six
node-exclusive 1-GPU jobs running in parallel: every compared pair still shares one GPU
(cross-arm pairs live inside one job), while node exclusivity removes the cross-job
`pkill`/`ray stop` interference that had forced serialization. Remaining wall time dropped
from about 3 hours to about 35 minutes.

Failed attempts, kept in provenance: job 2956 (bridge, first attempt) failed on the
import-walk strictness later resolved by the parity-of-brokenness assertion; jobs 2977–2979
died instantly on a shell script corrupted by an in-place `perl` edit (pipe characters inside
`s|||` delimiters) and were resubmitted as 2983–2985 after a clean rewrite; jobs 2957/2958
were the superseded serial stages, cancelled un-started except for Stage A prep, whose
prompt/adapter artifacts the parallel jobs then reused.

## Risks and limitations

- Scope is H200 only, per the campaign request. Cross-GPU and cross-backend behavior of the
  underlying lines is covered by the 2026-08-19 merged-stack report; nothing in this
  restructure touches arithmetic, and kernel bit-parity held, so no new GPU-dependence is
  expected.
- The e2e smoke axis is statistical, not bitwise; its per-round sync metric and reward scale
  matched the qualified envelope.
- The two dead `ref_te_*` reference files remain in the tree (equally broken in both arms).
  Deleting them upstream is an optional chore; nothing imports them.
- The pre-fix window (2026-08-26 to be8692e) exists in history: installs pinned from orbit
  `197c9137` or earlier get the pre-restructure Megatron-LM with the new Bridge. That
  combination is exactly what jobs 2952-era environments avoided only via explicit overrides;
  anyone who built from raw orbit-main pins in that window should rebuild.

## Actions

| action | owner | status | evidence or trigger |
|:--|:--|:--|:--|
| Adopt `orbit_cu130_omr` as the go-forward slurm env | zeju | Ready | This report; env verified 39/39 |
| Delete or keep Bridge `orbit-main-restructured` (now equal to `orbit-main`) | zeju | Open | Branch cleanup preference |
| Retire pre-merge qualification envs `orbit_cu130_{mlmr,mbr}` | zeju | Open | Disk pressure; superseded by this campaign |
| Publish this report to VigaHub | zeju/agent | Open | On request |

<details class="reproducibility">
<summary>Reproducibility</summary>

- Run store (cluster and local mirror):
  `~/.local/state/remote-cluster-runs/slurm/orbit/orbit-main/20260827T004939Z-r517330440/`
  with per-leg directories `install-orbit_cu130_omr`, `static-equivalence`, `compare-mlm`,
  `compare-bridge`, `stage-a`, `stage-b`, `smoke-old`, `smoke-new`, plus `provenance.json`
  and `VERDICT.md`.
- New env install (job 2952), run from the new orbit head's own installer with source
  overrides pinning the branch heads:
  `bash scripts/slurm/setup/cu130/install_env.sh --env-prefix .../envs/orbit_cu130_omr
  --source-root .../sources/orbit_cu130_omr --cache-dir .../cache/orbit_cu130_zeju` with
  `ORBIT_MEGATRON_COMMIT=83879096b...`, `ORBIT_MEGATRON_BRIDGE_COMMIT=988d6426...`,
  `ORBIT_SGLANG_COMMIT=a6fe249b3...` (after be8692e the megatron/bridge overrides are the
  defaults).
- Arm activation: `scripts/arm_env2.sh <old|new> <serving|trainer>` in the run store —
  conda env + worktree PYTHONPATH + the cluster-required cuDNN pins +
  `ORBIT_PEFT_ADAPTER_TRANSPORT=cpu_gather` + node-local `TRITON_CACHE_DIR`.
- Probes: `kernel_parity.py`/`compare_tensors.py` (compare-mlm), `bridge_checks*.py`
  (compare-bridge), `matrix_probe.py` + `stage_a_matrix_report.py` (stage-a),
  `stageb_cell_oldnew.sh` + `stageb_compare_oldnew.py` (stage-b) — all in the run store's
  `scripts/` and leg directories, derived from the 2026-08-24 port-campaign harness.
- Hardware: NVIDIA H200 nodes (slinky), partition `all`; smokes used 8 GPUs per arm
  (cpu=156, 50 min limit), all other legs 1 GPU.
- Model and data: Qwen2.5-0.5B-Instruct, gsm8k train/test parquets from
  `/data/home/zeju/{models,datasets}`.

</details>
