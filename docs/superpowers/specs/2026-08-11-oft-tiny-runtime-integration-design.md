# OFT Tiny-Block Runtime Integration Design

**Date:** 2026-08-11
**Status:** Approved design, pending implementation plan

## Objective

Integrate the completed SGLang tiny-block OFT work into the stable runtime used by Orbit, update Orbit to pin that exact runtime, and separately validate the Orbit first-adapter-update transport fix before it is admitted to `feat/lora-without-regret`. After both gates pass, run the eight-GPU Math OFT LR3 campaign.

The integration is deliberately staged so a proven SGLang kernel/dispatch change is not coupled to an Orbit transport change that still needs a real colocated training test.

## Current Source Identities

### SGLang

- Stable target branch: `orbit-sgl-v0.5.9`
- Stable pre-integration commit: `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`
- Completed feature branch: `codex/oft-bs4`
- Completed feature commit: `b52394d22fc4b686016943efc47cce6fb892cef2`
- Relationship: the stable commit is an exact ancestor of the feature commit, so the intended integration is a fast-forward.

The feature contains the complete tiny-block runtime surface, tests, benchmarks, and documentation. It is larger than a one-line kernel patch, but it does not change `sgl-kernel/`. The separately pinned `sgl-kernel` revision therefore remains unchanged.

### Orbit

- Experiment target branch: `feat/lora-without-regret`
- Local pre-spec baseline: `4505286b8dfd8771f5b5830e531d2be11ff9dce2`
- Published upstream tip before implementation: `a7a87ddaa6130d2cc522770dd640d17938e4d250`
- Completed development branch containing both the pin and transport work: `codex/oft-bs4`
- Completed development commit: `49e0d3fc50cdd1a24cf4ab922f817dbed267de87`
- First-adapter-update transport commit: `ccc351678d7ccfa8a41a48d57fb064dcb3be0e2e`

`feat/lora-without-regret`, rather than Orbit `main`, is the integration target because it owns the Math OFT campaign tooling. The full `codex/oft-bs4` branch must not be merged wholesale: it also contains development and smoke-test work outside the two approved integration units.

Implementation starts from the reviewed commit containing this specification, not directly from the pre-spec baseline above.

## Non-Goals

- Do not merge the SGLang feature into `orbit-main`; Orbit currently consumes the `orbit-sgl-v0.5.9` line.
- Do not change Megatron-Bridge source or its pin.
- Do not change the `sgl-kernel` pin, currently `9c83ae8be07cbb1eb6898ce608ae244e3be375b4`.
- Do not claim the first-adapter-update issue is fixed from unit tests or server startup alone.
- Do not launch the full Math OFT LR3 sweep until both the runtime-pin gate and the real transport gate pass.
- Do not reuse a dirty remote checkout or an uncommitted remote patch as experiment provenance.

## Stage 1: Stabilize SGLang

Run the focused OFT suite from the clean SGLang feature commit before changing the stable branch. Coverage must include tiny-block validation, default QKV dispatch, fused and unfused dense paths, grouped MoE, backward Cayley, CUDA-graph behavior, and streamed chunk limits.

Immediately before integration, fetch the relevant remote refs and confirm the remote stable branch still resolves to `89ea43812ec6fb161fe29902a6c6f1fbefb524dd`. If the remote moved, stop and re-audit its ancestry instead of overwriting it.

If the suite passes and the remote remains at the audited revision, update `orbit-sgl-v0.5.9` to `b52394d22fc4b686016943efc47cce6fb892cef2` with an `--ff-only` operation and push the stable branch. No merge commit or rewritten feature history is needed.

The acceptance gate is:

1. The focused OFT suite passes freshly at the feature tip.
2. The stable branch advances by fast-forward only.
3. The pushed stable branch resolves to the exact tested commit.
4. `sgl-kernel/` and the external `sgl-kernel` pin remain unchanged.

If any condition fails, stop before changing Orbit.

## Stage 2: Update Orbit's Runtime Pin

Create one fresh pin-only commit on `feat/lora-without-regret` after Stage 1. Do not cherry-pick the existing pin commit because it assumes intermediate commits that are not on the target branch.

The pin-only commit may change exactly:

- `pyproject.toml`
- `uv.lock`
- `tests/fast/utils/test_lora_regret_arms_coverage.py`

All SGLang runtime references must resolve to the pushed `orbit-sgl-v0.5.9` commit from Stage 1. The lockfile contract test must make removal or drift of that exact revision fail. The `sgl-kernel` and Megatron-Bridge references must remain byte-for-byte unchanged.

Run the pin contract test, regenerate the lock through the repository's supported dependency workflow, inspect every SGLang and `sgl-kernel` source entry, and run the focused LoRA-regret configuration/preflight suite. Push this pin-only Orbit commit only after those checks pass.

Before publishing it, fetch and confirm that the remote `feat/lora-without-regret` branch still has the audited relationship to the reviewed local spec tip. If it moved, reconcile and rerun the pin checks rather than force-pushing or silently dropping either history.

## Shared Cluster Environment Update

After the pin-only commit is published, synchronize the existing shared environment at `/fast/zqiu/orbit-iclr/orbit_env` from the exact locked Orbit state. Do not create an ad hoc per-worktree environment.

Before mutating the shared environment:

1. Confirm no active job or test is using it.
2. Record the Orbit commit, SGLang commit, host, and update command.
3. Use the repository-supported locked synchronization command rather than manually installing an arbitrary checkout.

Afterward, verify from the environment itself that importing SGLang resolves to the expected installed source and revision. A successful lock update without a matching runtime import is not sufficient.

## Stage 3: Validate the First-Adapter-Update Fix

Create an isolated Orbit validation branch and worktree based on the newly pinned `feat/lora-without-regret`. Carry only the three-file transport change represented by `ccc351678d7ccfa8a41a48d57fb064dcb3be0e2e`:

- `orbit/backends/megatron_utils/peft_transport/backends/ipc.py`
- `orbit/backends/sglang_utils/sglang_engine.py`
- `tests/test_peft_ipc_transport.py`

Although one path contains `megatron_utils`, this is Orbit-owned integration code. No Megatron-Bridge repository update is involved.

The transport design copies rank tensors to CPU, uses host-local `file_system` serialization in the Orbit-owned SGLang parent actor, rejects unsupported multi-host topology, and propagates load/version failures across all training ranks so peers cannot hang at a later collective.

First run the focused transport tests. Then run a short, colocated, two-GPU OFT BS8 training smoke from clean, exact source identities. BS8 is the smallest campaign arm and exercises both the newly stabilized SGLang path and the adapter transport that previously failed with `pidfd_getfd: Operation not permitted`.

The validation branch itself still carries only the three transport files. Drive the smoke with a validation-only wrapper stored in the durable run directory that invokes the unchanged production launcher with two-GPU, BS8, and short-run overrides. Record the wrapper and its hash as run evidence. Do not cherry-pick the development smoke-harness commits into the validation branch or include the wrapper in the eventual transport merge.

The real-training acceptance gate requires durable evidence of all of the following:

1. The first adapter update completes (`stage=update_weights_complete` or its exact current equivalent).
2. At least one generation phase completes after that update.
3. At least one actor training phase completes.
4. A subsequent adapter update completes, proving the path is reusable rather than startup-only.
5. The smoke exits cleanly and publishes a completion status.
6. Logs, commands, source revisions, scheduler identity, and timing evidence are stored in the durable remote-cluster run store.

Server initialization, adapter-manager construction, or reaching the first update call does not satisfy this gate.

## Stage 4: Admit or Reject the Orbit Transport Change

If Stage 3 is green, merge the isolated transport change into `feat/lora-without-regret`, rerun the focused Orbit checks on the integrated tip, and push it. The resulting feature branch contains two auditable units: the pin-only commit followed by the proven transport fix.

Then synchronize the shared environment once more from the final integrated Orbit tip and its unchanged lockfile. From the launch checkout and activated environment, verify that Orbit resolves from the final clean checkout and SGLang resolves to the exact stable revision tested in Stage 1. Record both import locations and source revisions before allowing Stage 5.

If Stage 3 fails, preserve its logs and completion status, leave `feat/lora-without-regret` with only the safe SGLang pin, and diagnose on the isolated branch. Do not merge the transport change merely because its unit tests pass.

## Stage 5: Run the Math OFT LR3 Campaign

Only after Stage 4 succeeds, launch:

```bash
bash scripts/lora_regret/run_e4_math_oft_lr3_8gpu.sh
```

The launcher runs three sequential eight-GPU Math arms at learning rate `3e-5`, seed `0`, and all target modules:

- `oftscout-b8-all-math-lr3e-05-s0`
- `oftscout-b128-all-math-lr3e-05-s0`
- `oftscout-b1024-all-math-lr3e-05-s0`

The campaign uses 150 rollouts, evaluates every 25 rollouts, disables checkpoints, and records offline W&B data. It must run from the clean, pushed integrated Orbit commit and the verified shared environment, not from either existing dirty remote checkout.

## Failure and Provenance Policy

Each stage is a hard gate for the next. A failure does not justify broad cleanup, branch rewriting, or substitution of a different source tree. Preserve the exact failing state and evidence, then repair only on the stage's isolated branch.

For every remote test or training run, record:

- local and remote repository paths;
- branch names and full commit SHAs;
- worktree cleanliness;
- environment identity and resolved import locations;
- scheduler job, execution host, GPU type/count, login host, and tmux session;
- exact command, timestamps, exit status, and verification status;
- durable log and result locations.

The existing eight-B200 allocation may be reused only if it is still running and its runtime state is revalidated at launch time. Allocation availability is not part of the correctness claim.

## Resulting Branch State

On success:

- SGLang `orbit-sgl-v0.5.9` points to the freshly tested tiny-OFT commit.
- Orbit `feat/lora-without-regret` contains an auditable pin-only commit and a separately proven first-adapter-update fix.
- The shared cluster environment resolves to those exact sources.
- The full Math OFT LR3 campaign has clean, reproducible source and runtime provenance.
