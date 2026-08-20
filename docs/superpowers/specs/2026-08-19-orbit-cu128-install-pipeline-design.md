# Orbit CUDA 12.8/H200 installation pipeline design

Date: 2026-08-19
Status: Approved for implementation planning
Target: Native Conda/uv installation on the Slurm H200 cluster

## Summary

Orbit needs a reproducible CUDA 12.8/H200 installation path alongside its existing CUDA 13.2 workflow. The new path will follow the established Miles pattern:

```text
Orbit and upstream manifests
        |
        v  extract_pins.py --write
pins.env
        |
        v  sourced by
install_env.sh
        |
        v
verify_env.py
```

The implementation lives under `scripts/slurm/setup/cu128/`. It does not replace or modify the CUDA 13.2 environment workflow.

## Context

The current Orbit checkout documents a CUDA 13.2 source build driven by `pyproject.toml`, `uv.lock`, and `env.sh`. A separate cluster experiment is building `/data/home/zeju/miles-orbit/envs/orbit_cu128` for CUDA 12.8 on H200.

The experiment has established the following evidence:

- Python 3.12 and Torch 2.11.0+cu128 install successfully.
- Torch reports CUDA 12.8.
- An allocated H200 reports compute capability 9.0.
- A BF16 matrix multiplication and NCCL query succeeded.
- Transformer Engine 2.14.0 from the pinned source commit is installed.

The environment is not yet a reproducible reference. Orbit, SGLang, Megatron, FlashAttention, SGLang kernels, and several other packages were still absent when inspected. The active environment and its Claude-owned Slurm build are user-owned and must not be modified by this work.

## Goals

- Reproduce the complete Orbit CUDA 12.8/H200 environment from committed source.
- Keep the CUDA 13.2 workflow intact.
- Make critical versions and source commits reviewable in a generated `pins.env`.
- Detect manifest drift before mutating an environment.
- Install Orbit and its patched backends editable from validated source directories.
- Fail closed on CUDA, Torch ABI, source commit, or environment-path mismatches.
- Provide a rerunnable metadata/import audit and a full H200 runtime verification mode.
- Support restart after an interrupted installation without deleting environments automatically.
- Record enough provenance to explain exactly which source and binary stack is active.

## Non-goals

- Building or running Docker or Enroot images.
- Supporting CUDA 13, Blackwell, AMD, CPU-only, or multi-platform environments in this profile.
- Replacing Orbit's existing `pyproject.toml`, `uv.lock`, `env.sh`, or CUDA 13.2 guides.
- Automatically updating backend repositories to remote branch heads.
- Running training or scientific acceptance workloads as part of installation.
- Treating the incomplete `orbit_cu128` environment as authoritative.

## File layout

```text
scripts/slurm/setup/cu128/
├── README.md
├── extract_pins.py
├── pins.env
├── install_env.sh
└── verify_env.py

tests/fast/scripts/slurm/setup/cu128/
├── test_extract_pins.py
└── test_verify_env.py
```

The README is the user entry point. The scripts remain small enough to inspect independently and expose explicit command-line modes rather than hidden behavior.

## Sources of truth

`extract_pins.py` reads three classes of input.

### Orbit-owned manifests

The root `pyproject.toml` supplies:

- Orbit's Python requirement and package version.
- Tested SGLang, Megatron-LM, and Megatron-Bridge source commits.
- Shared runtime dependency pins.
- Source commits for Transformer Engine, DeepEP, DeepGEMM, and other Git dependencies.

The extractor cross-checks duplicate declarations such as `tool.uv.sources` and `tool.orbit.release.backend-pins`. Disagreement is an error.

### Pinned upstream manifests

The SGLang Python manifest at the exact Orbit-tested SGLang commit supplies its Torch and tightly coupled runtime requirements. Backend source directories are read only after their current commits match the Orbit-tested refs.

### CUDA 12.8 profile mapping

Values not represented by an upstream manifest are deliberately maintained in one named mapping inside `extract_pins.py`. This includes:

- CUDA build label `cu128`.
- PyTorch and compatible wheel index URLs.
- CUDA toolkit policy.
- FlashInfer and CUDA Python overrides needed to resolve the SGLang stack.
- Prebuilt-wheel release coordinates when a source build is not selected.
- H200/SM90 architecture policy.

Each hand-owned value is labeled as such in generated output. This is equivalent to the hand-owned wheel-stack mapping in the Miles extractor.

## Generated pins contract

`extract_pins.py --write` writes `pins.env` atomically. `--check` regenerates in memory, compares with the committed file, prints the source of each difference, and exits nonzero without changing files.

Generation must be deterministic. The file header names every input manifest and states which fields are extracted or hand-owned. Shell values are safely quoted and contain no credentials or machine-specific paths.

At minimum, `pins.env` records:

- Python, CUDA profile, Torch, TorchVision, and TorchAudio versions.
- Torch, FlashInfer, and SGLang wheel indexes.
- Expected cuDNN and NCCL policy.
- Orbit, SGLang, Megatron-LM, Megatron-Bridge, and Transformer Engine refs.
- FlashAttention, SGLang kernel, Apex, DeepEP, DeepGEMM, router, and memory-saver versions or refs.
- Hashes of the relevant Orbit and SGLang manifests.
- The expected source layout relative to the workspace.

The extractor does not access the network. Updating a remote ref is a separate, explicit maintainer action.

## Source layout

The installer resolves these defaults:

```text
<workspace>/orbit
<workspace>/sglang
<workspace>/Megatron-LM
<workspace>/Megatron-Bridge
<workspace>/envs/orbit_cu128
```

The Orbit checkout is inferred from the installer's location. `ORBIT_WORKSPACE`, `SGLANG_SRC`, `MEGATRON_SRC`, `MEGATRON_BRIDGE_SRC`, and `ORBIT_ENV_PREFIX` may override the defaults.

A missing backend may be cloned from the generated source URL at the pinned commit. An existing backend is never pulled, reset, or checked out automatically. Dirty state, a different commit, or an unexpected remote causes a preflight failure with a corrective command for the user to review.

## Installer flow

`install_env.sh` uses `set -euo pipefail` and performs all non-mutating checks before creating or changing the environment.

1. Resolve the Orbit repository, workspace, backend sources, environment prefix, Conda root, and uv executable.
2. Source `pins.env` and run `extract_pins.py --check`.
3. Require a scheduled H200 allocation rather than a login-node build.
4. Check `nvidia-smi`, compute capability 9.0, CUDA 12.8 compatibility, `nvcc`, driver capability, disk space, Conda, uv, compiler, CMake, Ninja, Cargo, and required system tools.
5. Validate all source repositories and pinned commits.
6. Create or reuse the prefix-based Python 3.12 Conda environment.
7. Install Torch, TorchVision, and TorchAudio from the cu128 index before any Torch ABI-bound package.
8. Derive and install Torch's declared cuDNN package, then validate the CUDA build tag.
9. Resolve and install SGLang's dependency tree with generated CUDA 12.8 overrides while preventing replacement of the selected Torch wheel.
10. Install Megatron-LM, Megatron-Bridge, SGLang, and Orbit editable from the validated source directories.
11. Install compiled and ABI-sensitive packages in a fixed order: Transformer Engine, FlashAttention, SGLang kernels, Apex, DeepEP, DeepGEMM, and remaining runtime packages.
12. Install Orbit's Python requirements with profile overrides, then reassert ABI-sensitive pins.
13. Write any required source-root `.pth` files and environment activation hooks.
14. Run `verify_env.py` in full H200 mode.
15. Print the exact activation command, source paths, commits, environment prefix, and verification result.

Temporary constraint and override files are created in a private temporary directory and removed on exit. Installation logs are written only when the caller supplies a log path or redirects output.

## Idempotence and safety

The installer may reuse a compatible prefix and finish interrupted work. It does not claim transactionality across package installation, so verification remains mandatory.

The installer never:

- Deletes or renames an environment.
- Runs `git pull`, `git reset`, or `git submodule update --remote`.
- Changes an existing dirty backend checkout.
- Installs compute-heavy packages on a login node.
- Starts background package installations.
- Modifies the active `orbit_cu128` environment unless it is explicitly selected.

A fresh reproducibility run uses a distinct prefix such as `orbit_cu128_repro`. Destructive cleanup remains a manual user action.

## Verification design

`verify_env.py` supports two modes.

### Metadata and import mode

This mode is safe for rerunning without a GPU workload. It checks:

- Installed versions against `pins.env`.
- Torch's `+cu128` build tag.
- Direct URL metadata and exact editable source paths.
- Git commits for Orbit and all backend sources.
- Required source-root `.pth` files.
- Imports for Orbit, SGLang, Megatron Core, Megatron Bridge, Transformer Engine, FlashAttention, SGLang kernels, Apex, DeepEP, and DeepGEMM.
- That imports resolve to real files rather than empty namespace packages.
- cuDNN, NCCL, CUDA Python, FlashInfer, router, and memory-saver package metadata.

### Full H200 mode

Full mode adds:

- CUDA availability and device-name reporting.
- Compute capability exactly 9.0.
- Runtime CUDA 12.8 and expected cuDNN/NCCL checks.
- Import and symbol checks for compiled Torch extensions.
- A small BF16 matrix multiplication followed by synchronization.
- A minimal allocator and collective-library query that does not launch distributed training.

Verification prints one labeled result per check and a final pass/fail count. Any failed required check returns exit code 1. Environmental inability to perform full GPU checks is a failure in full mode, not a skip presented as success.

## Error reporting

Errors state:

- Which layer failed.
- Expected and observed values.
- Whether mutation had begun.
- The environment and source paths involved.
- A safe resume or corrective command.

ABI mismatches fail before importing large frameworks where possible. Examples include a non-cu128 Torch wheel, a SGLang requirement that would replace Torch, an incompatible compiled wheel, or a backend source commit that differs from `pins.env`.

Secrets, proxy credentials, and private environment values are never printed.

## Documentation

`README.md` documents:

- Required H200 Slurm allocation and toolchain.
- First installation into a new prefix.
- Restarting an interrupted installation.
- Activation and daily-use commands.
- Pin update and drift-check procedures.
- Source checkout expectations.
- Metadata-only and full verification commands.
- Common CUDA, ABI, source-drift, and incomplete-environment failures.
- The explicit relationship to the existing CUDA 13.2 workflow.

The root installation documentation gains only a short link to the new profile after the pipeline is qualified.

## Testing strategy

Fast tests cover pure logic without creating an environment:

- Deterministic extraction from representative manifests.
- Duplicate-pin disagreement.
- Missing and malformed manifest fields.
- Shell quoting and atomic generation.
- `--check` success and drift output.
- Version, direct URL, editable-path, commit, and namespace-package verification helpers.
- Clear failure messages for CUDA tag and source mismatches.

Static checks cover shell syntax and Python syntax. No test mocks a successful GPU runtime.

Qualification uses a new remote worktree and a new environment prefix. It runs the installer inside one H200 allocation, preserves durable logs and provenance, runs full verification, then reruns the installer to demonstrate idempotence. This expensive qualification requires explicit user authorization before submission.

## Acceptance criteria

The profile is ready only when:

- `extract_pins.py --write` is deterministic and `--check` passes.
- Fast tests and syntax checks pass.
- A fresh environment builds under a new prefix without modifying `orbit_cu128`.
- Package metadata and editable source paths match generated pins.
- Full verification passes on an H200.
- A second installer run introduces no dependency or source drift.
- The README reconstructs the successful build and activation commands.
- The committed provenance identifies the source revision and environment used for qualification.

## Implementation sequence

Implementation planning should order work as follows:

1. Extractor and generated pins, with fast tests.
2. Verification helpers and metadata mode, with fast tests.
3. Installer preflight and source validation.
4. Installer package layers and full H200 verification.
5. Documentation and root-guide link.
6. Fresh-prefix qualification and idempotence run.

This order keeps version logic testable before any expensive environment mutation and prevents incomplete experimental state from becoming the source of truth.
