# Install orbit + megatron + sglang (CUDA 13.2)

Torch 2.11, SGLang 0.5.9, Python 3.12, Ubuntu 22.04, cudnn 9.22.

This is the only supported public CUDA runtime path for Orbit launchers.

## Setup env from scratch

### 1. Load CUDA 13.2

```bash
export CUDA_HOME=<cuda-13.2-root>
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export ORBIT_BUILD_WHEELS=https://github.com/liulixinkerry/orbit-build-wheels/releases/download/cu132-torch211-ubuntu2204
```

### 2. Create uv env + Setup env path

```bash
cd <workspace>/orbit
uv python pin 3.12
uv venv
source .venv/bin/activate

export SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export CUDA_ROOT="${CUDA_HOME:?set CUDA_HOME to your CUDA 13.2 root}"
export NCCL_ROOT="${SITE_PACKAGES}/nvidia/nccl"
export CPATH="${NCCL_ROOT}/include:${NVSHMEM_ROOT}/include:${CUDA_ROOT}/targets/x86_64-linux/include/cccl:${CUDA_ROOT}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### 3. PyTorch + cuda-python + cudnn 9.22

```bash
# Torch 2.11 + CUDA 13
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
uv pip install cuda-python==13.2

echo "torch==2.11.0
cuda-python==13.2" > overrides.txt
uv pip install --override overrides.txt sglang==0.5.9
rm -rf overrides.txt
uv pip install nvidia-cudnn-cu13==9.22.0.52
```

### 4. Common libs

```bash
uv pip install ninja packaging psutil \
    accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict==0.10.0" torchdata \
    ray[default] codetiming hydra-core==1.3.2 pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard wheel

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

uv pip install nvidia-mathdx==25.6.0
```

### 5. TransformerEngine @ 71bbefbf153418f943640df0f7373625dc93fa46

```bash
uv pip install pybind11
# export NVTE_FRAMEWORK=pytorch
# MAX_JOBS=16 NVTE_BUILD_THREADS_PER_JOB=2 uv pip install --reinstall --no-cache --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@71bbefbf153418f943640df0f7373625dc93fa46
uv pip install "$ORBIT_BUILD_WHEELS/transformer_engine-2.14.0-cp312-cp312-linux_x86_64.whl"
```

### 6. ML libs (open-clip, trl, math_verify, nvidia-resiliency, tilelang)

```bash
uv pip install ftfy
uv pip install --no-deps open-clip-torch==3.3.0

uv pip install --no-deps trl
uv pip install nvtx matplotlib liger-kernel==0.8.0
uv pip install math-verify==0.9.0 latex2sympy2-extended==1.11.0
uv pip install git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@63154570cea17f8805a7fd15cc3b8cc2919ba575

uv pip install tilelang==0.1.9 tile-kernels
```

### 7. Apex

```bash
# APEX commit: f199212da7234bf9be2244cad5b9bfa2f5fe2675
uv pip install "$ORBIT_BUILD_WHEELS/apex-0.1-cp312-cp312-linux_x86_64.whl"
```

### 8. Fast-hadamard-transform

```bash
# fast-hadamard-transform: e7706faf8d1c3b9f241e36860640ad1dac644ede
uv pip install --no-build-isolation -v git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede
```

### 9. Flash Attention (FA2 + FA3)

```bash
# FLASH_ATTENTION_FORCE_BUILD forces a source build. Without it, flash-attn's setup.py
# auto-downloads a prebuilt wheel built against an older torch ABI, which then fails at
# import with `undefined symbol: ...c10_cuda_check_implementation...`.
FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --no-build-isolation flash_attn==2.8.3

# FA3: 28ef22c99a135b234fb54bc33cfb638078bacb65
uv pip install "$ORBIT_BUILD_WHEELS/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl" --no-deps
# copy the FA3 python interface into the installed package dir
python_path=$(python -c "import site; print(site.getsitepackages()[0])") && mkdir -p $python_path/flash_attn_3 && \
   curl -fSL https://raw.githubusercontent.com/Dao-AILab/flash-attention/28ef22c99a135b234fb54bc33cfb638078bacb65/hopper/flash_attn_interface.py -o $python_path/flash_attn_3/flash_attn_interface.py
```

### 10. Linear attention (causal-conv1d, mamba, FLA)

```bash
uv pip install --no-build-isolation causal-conv1d==1.6.2.post1
uv pip install --no-build-isolation mamba-ssm==2.3.1
uv pip install --no-build-isolation flash-linear-attention==0.5.0
```

### 11. Megatron-Bridge Dependency + HF stack + observability

```bash
uv pip install git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@fad15ab214efc77155282a057c0ca139cad1ebc9
uv pip uninstall sglang megatron-bridge
uv pip install huggingface-hub==0.36.2 flashinfer-python==0.6.3 timm==1.0.17 transformers==4.57.1
uv pip install opentelemetry-api==1.41.1 opentelemetry-sdk==1.41.1 opentelemetry-semantic-conventions==0.62b1
uv pip install linkify-it-py==2.1.0 mdit-py-plugins==0.5.0 memray==1.19.3 pytest-asyncio==1.3.0 textual==8.2.4 uc-micro-py==2.0.0 ring-flash-attn==0.1.8
uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --force-reinstall
```

### 12. SGLang router + custom sgl-kernel

```bash
uv pip uninstall sglang_router sgl-kernel
uv pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl

uv pip install scikit-build-core isort black wheel
uv pip install -U "cmake>=3.31"
uv pip install "$ORBIT_BUILD_WHEELS/sgl_kernel-0.3.21-cp310-abi3-linux_x86_64.whl" --no-deps
```

### 13. Pin NVIDIA CUDA 13.2 runtime libraries

```bash
uv pip install -U \
    "nvidia-cublas==13.4.1.1" \
    "nvidia-cuda-runtime==13.2.75" \
    "nvidia-cuda-cupti==13.2.75" \
    "nvidia-cuda-nvrtc==13.2.78" \
    "nvidia-cufft==12.2.0.46" \
    "nvidia-curand==10.4.2.55" \
    "nvidia-cusolver==12.2.0.1" \
    "nvidia-cusparse==12.7.10.1" \
    "nvidia-nvjitlink==13.2.78" \
    "nvidia-nvtx==13.2.75" \
    "nvidia-cudnn-cu13==9.22.0.52"
```

### 14. NCCL 2.30 + DeepEP V2 + DeepGEMM
```bash
uv pip install "nvidia-nccl-cu13==2.30.4" --no-deps
export SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

NVSHMEM_ROOT="${SITE_PACKAGES}/nvidia/nvshmem"
export EP_NCCL_ROOT_DIR="${NCCL_ROOT}"
export EP_NVSHMEM_ROOT_DIR="${NVSHMEM_ROOT}"
export TORCH_CUDA_ARCH_LIST='10.0'
export MAX_JOBS=32
export CUDA_ROOT="${CUDA_HOME:?set CUDA_HOME to your CUDA 13.2 root}"
export NCCL_ROOT="${SITE_PACKAGES}/nvidia/nccl"
export CPATH="${NCCL_ROOT}/include:${NVSHMEM_ROOT}/include:${CUDA_ROOT}/targets/x86_64-linux/include/cccl:${CUDA_ROOT}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

uv pip install --no-build-isolation --reinstall git+https://github.com/deepseek-ai/DeepEP.git@d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb

# deep_gemm_official
uv pip install --no-build-isolation git+https://github.com/liulixinkerry/DeepGEMM.git@18db30e9db9703b906e9ab3803fbbc1a64be1520
```

### 15. Orbit and backend sources

Orbit's uv manifest installs Megatron-LM, Megatron-Bridge, and SGLang from
immutable public Git refs recorded under `tool.orbit.release.backend-pins` in
`pyproject.toml`.

Run this after the CUDA/Torch layer above is installed. Orbit's uv manifest keeps
Torch, TorchVision, TorchAudio, and Triton out of the resolver so this CUDA 13.2
layer does not get replaced by an untested transitive dependency choice. This
also installs the locked DeepEP commit using the build paths exported in step 8.

```bash
cd <workspace>/orbit
uv sync --inexact \
    --no-install-package transformer-engine \
    --no-install-package sgl-kernel
```

Use `uv sync --inexact` for metadata refreshes so uv does not prune the
CUDA/Torch packages installed by this guide. The manifest pins `transformer-engine`
and `sgl-kernel` to git sources (for the one-command `--extra allinone` build), so
`--no-install-package` for both keeps the prebuilt TE (step 5) and local cu13
sgl-kernel (step 12) from being rebuilt from source here.

# Troubleshooting

## `uv sync` Cannot Resolve A Backend Repo

Orbit installs the backend forks from immutable public Git refs recorded in
`pyproject.toml`. If `uv sync` cannot resolve one of them, check that the refs are
reachable:

```bash
git ls-remote https://github.com/Sphere-AI-Lab/Megatron-Bridge.git 85c84cbc26d4c983a3d6e46c804f02e2a99af5a2
git ls-remote https://github.com/Sphere-AI-Lab/Megatron-LM.git 00eb75b0c803b0fc8e5413d736529d9d3b82b6bd
git ls-remote https://github.com/Sphere-AI-Lab/sglang.git 9c83ae8be07cbb1eb6898ce608ae244e3be375b4
```

If a command prints no commit, the release ref has not been published.

## CUDA Imports Fail

Verify that the environment is using Python 3.12 and the CUDA 13 stack from `docs/CUDA-13-install.md`:

```bash
uv run python - <<'PY'
import importlib.metadata as md
import torch
import cuda.bindings

print(torch.__version__, torch.version.cuda)
print(md.version("cuda-python"))
PY
```

## `megatron.bridge` Import Fails (`libz3.so.4.15`)

`megatron.bridge` imports `nvidia-modelopt`, which dlopens `libz3.so.4.15` by soname.
`env.sh` and `examples/load_cuda13_2_orbit_env.sh` add `site-packages/z3/lib` to
`LD_LIBRARY_PATH` for this; if you do not use either loader, add it yourself.

## `sgl-kernel` / `torch-memory-saver` Import Fails (ABI mismatch)

The `$ORBIT_BUILD_WHEELS` prebuilts must match this torch 2.11 / CUDA 13 layer (the
TransformerEngine wheel is correct). If `sgl-kernel` or `torch-memory-saver` fail to
import (`undefined symbol: ...c10_cuda_check_implementation...` or `libcudart.so.12`),
they were built against an older torch ABI / cu12 — rebuild them from source against
this layer.

## Launchers Fail Before Training

Public launchers require model, checkpoint, and data paths from the user. Set the required variables before running a launcher:

```bash
export HF_CKPT=/path/to/hf/checkpoint
export MEGATRON_LOAD=/path/to/megatron/torch_dist
export TRAIN_JSONL=/path/to/train.jsonl
export TEST_JSONL=/path/to/test.jsonl
export ENABLE_WANDB=0
```
