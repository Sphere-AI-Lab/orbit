# Install orbit + megatron + sglang (CUDA 13.2)

Torch 2.11, SGLang 0.5.9, Python 3.12.

This is the only supported public CUDA runtime path for Orbit launchers.

> Prerequisite: a user-provided CUDA 13 build workspace
> (referred to as `$CUDA13_BUILD_ROOT` below) containing any source trees or
> local wheels you choose to build outside `uv sync`.

## Setup env from scratch

### 1. Load CUDA 13.2 + cuDNN modules

```bash
export TMPDIR=<your_path>/tmp
module load cuda/13.2
module load nccl
unset LD_LIBRARY_PATH
export CUDA_HOME=<cuda-13.2-root>
export CUDNN_HOME=<cudnn-root>
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH

export CUDA13_BUILD_ROOT="<your_path>/software/cu13"
```

### 2. Create uv env

```bash
UV_LINK_MODE=symlink
echo "UV_LINK_MODE=$UV_LINK_MODE"
export UV_LINK_MODE=$UV_LINK_MODE
cd <workspace>/orbit
uv python pin 3.12
uv venv
source .venv/bin/activate
```

### 3. PyTorch + cuda-python

```bash
# Torch 2.11 + CUDA 13
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
uv pip install cuda-python==13.2
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

### 5. TransformerEngine

```bash
uv pip install pybind11
export NVTE_FRAMEWORK=pytorch
MAX_JOBS=16 NVTE_BUILD_THREADS_PER_JOB=2 uv pip install --reinstall --no-cache --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@71bbefbf153418f943640df0f7373625dc93fa46
```

### 6. ML libs (open-clip, trl, math_verify, nvidia-resiliency, tilelang)

```bash
uv pip install ftfy
uv pip install --no-deps open-clip-torch==3.2.0

uv pip install --no-deps trl
uv pip install nvtx matplotlib liger_kernel
uv pip install math_verify latex2sympy2_extended
uv pip install git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@63154570cea17f8805a7fd15cc3b8cc2919ba575

uv pip install tilelang tile-kernels
```

### 7. Apex (build from source, ~5 min)

> IMPORTANT: If you face CUDA compatibility issues when compiling APEX, comment out the version check in https://github.com/NVIDIA/apex/blob/f199212da7234bf9be2244cad5b9bfa2f5fe2675/setup.py#L218

```bash
# APEX commit: f199212da7234bf9be2244cad5b9bfa2f5fe2675
cd "$CUDA13_BUILD_ROOT/apex" && \
    NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install -v --no-build-isolation . && \
    cd -
```

### 8. DeepEP, fast-hadamard-transform, DeepGEMM

DeepEP is pinned in `pyproject.toml` / `uv.lock`. Export these build paths
before running `uv sync --inexact` in step 14.

```bash
SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
CUDA_ROOT="${CUDA_HOME:?set CUDA_HOME to your CUDA 13.2 root}"
NCCL_ROOT="${SITE_PACKAGES}/nvidia/nccl"
NVSHMEM_ROOT="${SITE_PACKAGES}/nvidia/nvshmem"
export EP_NCCL_ROOT_DIR="${NCCL_ROOT}"
export EP_NVSHMEM_ROOT_DIR="${NVSHMEM_ROOT}"
export CPATH="${NCCL_ROOT}/include:${NVSHMEM_ROOT}/include:${CUDA_ROOT}/targets/x86_64-linux/include/cccl:${CUDA_ROOT}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${NVSHMEM_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST='10.0'
export MAX_JOBS=8

# fast-hadamard-transform: e7706faf8d1c3b9f241e36860640ad1dac644ede
cd "$CUDA13_BUILD_ROOT/fast-hadamard-transform" && uv pip install --no-build-isolation -v . && cd -

# DeepGEMM: 891d57b4db1071624b5c8fa0d1e51cb317fa709f
cd "$CUDA13_BUILD_ROOT/DeepGEMM" && ./install.sh && cd -
```

### 9. Flash Attention (FA2 + FA3)

```bash
uv pip install --no-build-isolation flash_attn==2.8.3

# FA3: 28ef22c99a135b234fb54bc33cfb638078bacb65
cd "$CUDA13_BUILD_ROOT/flash-attention/hopper" && uv pip install -v --no-build-isolation . && cd -

# Optional: copy the FA3 python interface into the installed package dir
# python_path=$(python -c "import site; print(site.getsitepackages()[0])") && mkdir -p $python_path/flash_attn_3 && \
#     cp "$CUDA13_BUILD_ROOT/flash-attention/hopper/flash_attn_interface.py" $python_path/flash_attn_3/flash_attn_interface.py
```

### 10. Linear attention (causal-conv1d, mamba, FLA)

```bash
uv pip install --no-build-isolation causal-conv1d==1.6.1
uv pip install --no-build-isolation mamba-ssm==2.3.1
uv pip install --no-build-isolation flash-linear-attention==0.4.1
```

### 11. HF stack + observability

```bash
uv pip install huggingface-hub==0.36.2 flashinfer-python==0.6.3 timm==1.0.17 transformers==4.57.1
uv pip install opentelemetry-api==1.41.1 opentelemetry-sdk==1.41.1 opentelemetry-semantic-conventions==0.62b1
uv pip install linkify-it-py==2.1.0 mdit-py-plugins==0.5.0 memray==1.19.3 pytest-asyncio==1.3.0 textual==8.2.4 uc-micro-py==2.0.0 ring-flash-attn==0.1.8
uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --force-reinstall
```

### 12. SGLang router + custom sgl-kernel

```bash
uv pip uninstall sglang_router sgl-kernel
uv pip install sglang-router==0.3.2

uv pip install scikit-build-core isort black wheel
uv pip install -U "cmake>=3.31"
export TORCH_CUDA_ARCH_LIST="8.0 9.0a 10.0a"

# To rebuild sgl-kernel from source instead of using the prebuilt wheel:
# git clone https://github.com/sgl-project/DeepGEMM DeepGEMM-sgl
# cd DeepGEMM-sgl && git checkout ffe2b6b
# cd <sglang-checkout>/sgl-kernel
# TORCH_CUDA_ARCH_LIST="8.0 9.0a 10.0a" make build CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_REPO-DEEPGEMM=$CUDA13_BUILD_ROOT/DeepGEMM-sgl"

# To install a prebuilt local wheel:
# uv pip install "${SGL_KERNEL_WHEEL:?set SGL_KERNEL_WHEEL to a local sgl-kernel wheel}" --force-reinstall --no-deps
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
    "nvidia-nvtx==13.2.75"
```

### 14. Orbit and backend sources

Orbit's uv manifest installs Megatron-LM, Megatron-Bridge, and SGLang from
immutable public Git refs recorded under `tool.orbit.release.backend-pins` in
`pyproject.toml`.

Run this after the CUDA/Torch layer above is installed. Orbit's uv manifest keeps
Torch, TorchVision, TorchAudio, and Triton out of the resolver so this CUDA 13.2
layer does not get replaced by an untested transitive dependency choice. This
also installs the locked DeepEP commit using the build paths exported in step 8.

```bash
cd <workspace>/orbit
uv sync --inexact
```

Use `uv sync --inexact` for metadata refreshes so uv does not prune the
CUDA/Torch packages installed by this guide.
