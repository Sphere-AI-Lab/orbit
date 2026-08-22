# Orbit native CUDA 13 / H100 or B200 environment

This profile follows the Miles-IMP installation model. The RadixArk Miles
Dockerfile is a pinned dependency recipe, not a runtime container. Its prebuilt
CUDA 13 wheels are installed into a persistent Conda environment, then Orbit's
exact Sphere-Lab sources are installed as editable overlays.

## Source-of-truth flow

~~~text
radixark/miles docker/Dockerfile at a pinned commit
                 +
orbit/pyproject.toml
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
~~~

No Docker, Enroot, or Apptainer runtime is involved.

## What is prebuilt

The primary path does not compile CUDA extensions. It installs:

- PyTorch 2.11 and Triton 3.6 for CUDA 13
- official sglang-kernel 0.4.5+cu130
- official sgl-deep-gemm
- FlashInfer CUDA 13 components
- FlashAttention 2 and 3 from the Miles wheel release
- Transformer Engine and Apex from the Miles wheel release
- additional Miles release wheels when available

A required missing wheel is an error. The installer does not silently fall back
to a multi-hour CUDA build.

## What is editable

These repositories are checked out at exact commits under the source root:

- Sphere-Lab SGLang
- Sphere-Lab Megatron-LM
- Sphere-Lab Megatron-Bridge

They and the current Orbit checkout are installed editable with --no-deps. That
keeps Python and Triton changes live while preserving the prebuilt binary layer.

## Paths

~~~text
Environment:
/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1

Sources:
/fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1

Cache:
/fast/zqiu/orbit-iclr/orbit/cache/orbit-cu130-v1
~~~

These directories must be excluded from Git. The installer never resets an
existing source checkout or overwrites an unknown environment.

## Refresh or audit pins

From the Orbit repository root:

~~~bash
python scripts/slurm/setup/cu130/extract_pins.py --write
python scripts/slurm/setup/cu130/extract_pins.py --check
~~~

The RadixArk commit is embedded in extract_pins.py, so regeneration does not
silently follow a moving main branch.

## Inspect the installation plan

Dry-run mode performs no network access and creates no files:

~~~bash
scripts/slurm/setup/cu130/install_env.sh --dry-run
~~~

## Install inside an H100 or B200 allocation

~~~bash
scripts/slurm/setup/cu130/install_env.sh
~~~

The default command creates or resumes the paths above. Separate paths can be
provided with --env-prefix, --source-root, and --cache-dir.

## Activate

~~~bash
source /home/zqiu/anaconda3/etc/profile.d/conda.sh
conda activate /fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1
~~~

The editable source links are part of the environment. No extra PYTHONPATH is
normally needed.

## Why cuda-python 13 appears

The cuda-python requirement comes from SGLang's Python metadata. This profile
satisfies it with Orbit's exact CUDA Python pin, then installs Sphere-Lab SGLang
with --no-deps so dependency resolution cannot replace the controlled stack.

## Re-run verification

Inside an H100 or B200 allocation (FlashAttention 3 is sm_90a-only and is not exercised on B200):

~~~bash
/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1/bin/python \
  scripts/slurm/setup/cu130/verify_env.py \
  --source-root /fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1 \
  --full-h100
~~~
