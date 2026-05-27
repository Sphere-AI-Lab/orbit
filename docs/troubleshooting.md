# Troubleshooting

## `uv sync` Cannot Resolve A Backend Repo

Orbit installs the backend forks from immutable public Git refs recorded in
`pyproject.toml`. If `uv sync` cannot resolve one of them, check that the refs are
reachable:

```bash
git ls-remote https://github.com/Sphere-AI-Lab/Megatron-Bridge.git fb886993a94de1ccc5f3835a6c92855e14797fab
git ls-remote https://github.com/Sphere-AI-Lab/Megatron-LM.git 06fbee5c6b8337784f83a49666fc74d81185329d
git ls-remote https://github.com/Sphere-AI-Lab/sglang.git 59918f926c4ef6a57dad1eb52cab8988a4025443
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

## Launchers Fail Before Training

Public launchers require model, checkpoint, and data paths from the user. Set the required variables before running a launcher:

```bash
export HF_CKPT=/path/to/hf/checkpoint
export MEGATRON_LOAD=/path/to/megatron/torch_dist
export TRAIN_JSONL=/path/to/train.jsonl
export TEST_JSONL=/path/to/test.jsonl
export ENABLE_WANDB=0
```
