---
title: Installation
description: Install Orbit on NVIDIA or AMD GPUs. Docker is the recommended path.
---
There are three ways to install Orbit. Docker is recommended because Orbit pins patched
versions of SGLang, Megatron-LM, and a few CUDA kernels.

## Method 1: Docker (recommended)

<Tabs>

  <Tab title="NVIDIA">

    ```bash
    docker pull radixark/miles:latest

    docker run --rm \
      --gpus all --ipc=host --shm-size=32g \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --network=host \
      -it radixark/miles:latest /bin/bash
    ```

  </Tab>
  <Tab title="AMD MI300X / MI350X">

    ```bash
    docker pull rlsys/orbit:MI350-355-latest    # or MI300-latest

    docker run --rm \
      --device /dev/dri --device /dev/kfd \
      --group-add video --ipc=host --shm-size=32g \
      --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
      --privileged \
      -it rlsys/orbit:MI350-355-latest /bin/bash
    ```

  </Tab>

</Tabs>

The image ships with:

- PyTorch (matching the container's CUDA / ROCm version)
- Megatron-LM, SGLang, FlashAttention-3, DeepGEMM, Apex
- Ray, uv, and Orbit installed editable at `/root/orbit`

See [Hardware requirements](#hardware-requirements) for per-GPU status.

## Method 2: From source

Clone and install in an existing environment.

```bash
git clone https://github.com/Sphere-AI-Lab/orbit.git
cd orbit
pip install -r requirements.txt
pip install -e . --no-deps
```

<Warning>

**Patched dependencies.** Orbit pins patched versions of SGLang and Megatron-LM. Installing them yourself at
the wrong commit is the most common source of bug reports — use Docker if you can.

</Warning>

## Method 3: Update an existing container

If you already run a Orbit image and want the latest code:

```bash
cd /root/orbit
git pull --rebase
pip install -e . --no-deps
ray stop && ray start --head --port=6379
```

## Verify

Confirm Orbit imports and the GPUs are visible:

```bash
python -c "import orbit; print('Orbit import OK')"
nvidia-smi
```

If either command fails, see [Debugging](/developer/debug).

## Hardware requirements

| Hardware | Status |
|---|---|
| NVIDIA GB300 / GB200 / B300 / B200 | Production |
| NVIDIA H200 / H100 | Production (CI guarded) |
| NVIDIA A100 | Supported — FP8 features disabled |
| AMD MI300X, MI325, MI350X, MI355X | Supported via ROCm |

For multi-node training you also need a high-bandwidth interconnect — InfiniBand, RoCEv2,
or Slingshot — and 200+ GB/s per node. Single-node jobs run fine over NVLink only.

## Next steps

- [Quick Start](/getting-started/quick-start) — run your first training job.
- [Core concepts](/user-guide/concepts) — the mental model behind Orbit.
- [Training backends](/user-guide/training-backend) — Megatron vs FSDP.
