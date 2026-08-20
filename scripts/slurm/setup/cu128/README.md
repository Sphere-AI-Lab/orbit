# Orbit native CUDA 12.8 / H200 environment

This directory provides a reproducible **native Conda/uv installation** for Orbit on
CUDA 12.8 and NVIDIA H200 GPUs. It does not build or start a Docker image, and it
lives alongside the existing CUDA 13 workflow without changing that workflow.

## Source-of-truth flow

```text
orbit/pyproject.toml + ../sglang/pyproject.toml
                  + CUDA 12.8/H200 profile in extract_pins.py
                              |
                              v
                  extract_pins.py --write
                              |
                              v
                         pins.env
                              |
                              v
                       install_env.sh
                              |
                              v
                        verify_env.py
```

The project metadata and backend references remain authoritative. `extract_pins.py`
normalizes those inputs into shell-safe, deterministic assignments in `pins.env`.
The generated file includes its input hashes so drift is reviewable.

Do not hand-edit `pins.env`. Change `pyproject.toml`, the SGLang metadata, or the
explicit CUDA 12.8 profile in `extract_pins.py`, then regenerate it.

## Files

- `extract_pins.py`: extracts and cross-checks package versions, source URLs, and commits.
- `pins.env`: generated contract consumed by the installer and verifier.
- `install_env.sh`: non-destructive twelve-stage Conda/uv installation.
- `verify_env.py`: re-runnable package, editable-source, revision, import, and H200 audit.

## Refresh or audit pins

From the Orbit repository root:

```bash
python scripts/slurm/setup/cu128/extract_pins.py --write
python scripts/slurm/setup/cu128/extract_pins.py --check
```

`--check` exits nonzero when regeneration would change `pins.env`.

## Inspect the installation plan

Dry-run mode is safe on a login node. It does not require a Slurm allocation and
does not create the environment, lock, or source checkout directories.

```bash
scripts/slurm/setup/cu128/install_env.sh --dry-run \
  --env-prefix /data/home/zeju/miles-orbit/envs/orbit_cu128_v1 \
  --source-root /data/home/zeju/miles-orbit/sources/cu128_v1
```

Use a new versioned prefix when another process or job may be using an existing
environment. The installer never deletes an unknown or existing directory.

## Preflight on an H200 node

Acquire an interactive Slurm allocation containing an H200, then expose the site
CUDA 12.8 toolkit so `nvcc --version` reports release 12.8. Conda and uv must be on
`PATH`, or supplied with `--conda-exe` and `--uv-exe`.

```bash
scripts/slurm/setup/cu128/install_env.sh --preflight-only \
  --env-prefix /data/home/zeju/miles-orbit/envs/orbit_cu128_v1 \
  --source-root /data/home/zeju/miles-orbit/sources/cu128_v1
```

Preflight rejects login-node execution, non-H200 allocations, the wrong CUDA
toolkit, stale pins, missing tools, and unsafe target paths.

## Install

Run the same command without `--preflight-only` inside the H200 allocation:

```bash
scripts/slurm/setup/cu128/install_env.sh \
  --env-prefix /data/home/zeju/miles-orbit/envs/orbit_cu128_v1 \
  --source-root /data/home/zeju/miles-orbit/sources/cu128_v1 \
  --jobs 8
```

The labeled stages are:

1. Preflight scheduler, GPU, toolkit, pins, and tools.
2. Create or resume the Conda Python 3.12 prefix.
3. Record Conda, uv, and Python tooling.
4. Install exact PyTorch CUDA 12.8 wheels.
5. Install pinned CUDA Python and FlashInfer wheels.
6. Clone immutable external sources at pinned commits.
7. Install non-controlled Orbit runtime dependencies.
8. Build Transformer Engine for Hopper.
9. Build Apex CUDA extensions.
10. Install SGLang, Megatron-LM, and Megatron-Bridge editable.
11. Install Orbit editable and reassert the torch wheel set.
12. Run metadata plus full H200 verification.

A sibling `<env-prefix>.install.lock` prevents concurrent installers from mutating
the same prefix. Re-running the command resumes a recognizable Conda prefix and
reuses clean source checkouts at their pinned revisions. Dirty external checkouts
are rejected rather than reset.

## Re-run verification

The default verifier is GPU-free and can audit metadata from a login node:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128_v1/bin/python \
  scripts/slurm/setup/cu128/verify_env.py \
  --source-root /data/home/zeju/miles-orbit/sources/cu128_v1
```

Inside an H200 allocation, add `--full-h200`:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128_v1/bin/python \
  scripts/slurm/setup/cu128/verify_env.py \
  --source-root /data/home/zeju/miles-orbit/sources/cu128_v1 \
  --full-h200
```

Full mode checks the H200 device name, CUDA 12.8 runtime, compute capability 9.0,
BF16 support and a finite CUDA matmul, plus visible cuDNN and NCCL runtimes.
Every check prints a labeled pass/fail result, and any failure produces a nonzero
exit status.
