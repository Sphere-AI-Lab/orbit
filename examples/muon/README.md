# Muon GSM8K launchers

This directory provides Muon launchers for minimal Qwen2.5-0.5B-Instruct and
Qwen3-30B-A3B GSM8K RL workloads. The Qwen2.5 launcher performs three
rollout/optimizer steps on one node with two visible GPUs and no model
parallelism. These are functional compatibility smokes, not convergence,
quality, memory, or speed benchmarks.

The Qwen2.5 public wrapper delegates to a shared runner with `--optimizer
muon`. The Qwen3 wrappers select Megatron's `dist_muon` optimizer.

## Prerequisites

The run requires:

- a Conda environment named `orbit` by default (or an override via `ORBIT_CONDA_ENV`);
- NVIDIA NeMo Emerging Optimizers v0.1.0 in that environment for the Muon arm;
- one Slurm node with at least two visible GPUs;
- the downloaded Hugging Face Qwen2.5-0.5B-Instruct checkpoint;
- GSM8K `train.parquet` in Orbit' `messages`/`label` format;
- a converted Megatron distributed checkpoint; and
- initialized `Megatron-LM` and `sglang` submodules in this worktree.

Install Megatron's pinned Muon dependency once in the selected environment:

```bash
git clone https://github.com/Sphere-AI-Lab/orbit.git orbit
cd orbit
source /data/shared/conda/miniconda3/etc/profile.d/conda.sh
conda activate "${ORBIT_CONDA_ENV:-orbit}"
python -m pip install \
  "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.1.0"
```

The Muon runners check the exact imports they need before starting Ray and
report this command when the dependency is missing.

The runner defaults are:

| Variable | Default |
| --- | --- |
| `HF_CHECKPOINT` | `${HOME}/models/Qwen2.5-0.5B-Instruct` |
| `REF_LOAD` | `${HOME}/models/Qwen2.5-0.5B-Instruct_torch_dist` |
| `PROMPT_DATA` | `${HOME}/datasets/gsm8k/train.parquet` |
| `NUM_GPUS` | `2` |
| `NUM_ROLLOUT` | `3` |
| `OUTPUT_ROOT` | `${PWD}/orbit-runs/muon-smoke` |
| `RUN_ID` | `qwen2.5-0.5b-gsm8k-muon-<UTC timestamp>` |
| `ORBIT_CONDA_ENV` | `orbit` |
| `ORBIT_CONDA_SH` | `/data/shared/conda/miniconda3/etc/profile.d/conda.sh` |

Override any of them in the environment. A run refuses to overwrite an
existing `${OUTPUT_ROOT}/${RUN_ID}` directory.

## One-time checkpoint conversion

Skip this section when `REF_LOAD` already exists and is nonempty.

```bash
git clone https://github.com/Sphere-AI-Lab/orbit.git orbit
cd orbit
source /data/shared/conda/miniconda3/etc/profile.d/conda.sh
conda activate "${ORBIT_CONDA_ENV:-orbit}"
PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${PY_SITE}/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
source scripts/models/qwen2.5-0.5B.sh
mkdir -p "${HOME}/models"
PYTHONPATH="${PWD}/thirdparty/Megatron-LM:${PWD}/thirdparty/Megatron-Bridge/src:${PWD}/thirdparty/sglang/python${PYTHONPATH:+:${PYTHONPATH}}" \
python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT:-${HOME}/models/Qwen2.5-0.5B-Instruct}" \
  --save "${REF_LOAD:-${HOME}/models/Qwen2.5-0.5B-Instruct_torch_dist}"
```

## Inspect the Qwen2.5 command

Dry-run mode performs no path, GPU, Ray, or submodule checks and creates no
output directory:

```bash
DRY_RUN=1 RUN_ID=muon-dry bash examples/muon/run_qwen2_5_0_5b_gsm8k_muon.sh
```

The command prints a shell-escaped `TRAIN_COMMAND=` line.

## Run the Qwen2.5 smoke

From an allocation exposing two GPUs, run:

```bash
srun -N1 -n1 bash -lc 'cd "$(pwd)" && bash examples/muon/run_qwen2_5_0_5b_gsm8k_muon.sh'
```

The runner activates `ORBIT_CONDA_ENV` itself, starts a node-local Ray runtime,
submits the training command, and stops only that Ray runtime on exit. The Ray
job's nonzero status is propagated.

## Outputs and pass criteria

The run writes a durable log to `${OUTPUT_ROOT}/${RUN_ID}/train.log`. A smoke
passes only when all of the following hold:

- the intended optimizer is selected;
- optimizer steps 0, 1, and 2 each complete exactly once;
- emitted loss, reward, gradient norm, and learning-rate values are finite;
- the process exits zero; and
- parsed training metrics and rollout rewards are finite, and runtime diagnostics
  contain no traceback, OOM, or scheduler error.

Passing this smoke establishes only that Muon can execute the small workload.
It does not establish training quality or performance.

## Qwen3-30B-A3B model-sharding matrix

The scripts below extend the compatibility smoke to the full Qwen3-30B-A3B
MoE model on one node with eight GPUs. Every topology has a distributed-Muon
wrapper that selects Megatron's `dist_muon` optimizer so the full model uses
Muon's layer-wise distributed optimizer-state path.

Pipeline parallelism and context parallelism remain 1. Each neighboring row
changes one model-parallel dimension. Expert tensor parallelism remains fixed
at 1; configurations above 1 are outside the current support scope.

| ID | TP | EP | expert-TP | Muon wrapper |
| --- | ---: | ---: | ---: | --- |
| `tp1_ep8_etp1` | 1 | 8 | 1 | `run_qwen3_30b_a3b_gsm8k_tp1_ep8_etp1_muon.sh` |
| `tp2_ep8_etp1` | 2 | 8 | 1 | `run_qwen3_30b_a3b_gsm8k_tp2_ep8_etp1_muon.sh` |
| `tp2_ep4_etp1` | 2 | 4 | 1 | `run_qwen3_30b_a3b_gsm8k_tp2_ep4_etp1_muon.sh` |

The commands keep `--sequence-parallel` common across the matrix. Megatron
intentionally normalizes it off at runtime when TP is 1.

### Qwen3 prerequisites and defaults

In addition to the initialized submodules and Emerging Optimizers dependency
described above, the Qwen3 runs require one Slurm allocation exposing eight
GPUs and the full converted Megatron checkpoint.

| Variable | Default |
| --- | --- |
| `HF_CHECKPOINT` | `${HOME}/models/Qwen3-30B-A3B` |
| `REF_LOAD` | `${HOME}/models/Qwen3-30B-A3B_torch_dist` |
| `PROMPT_DATA` | `${HOME}/datasets/gsm8k/train.parquet` |
| `NUM_GPUS` | `8` (fixed by this matrix) |
| `NUM_ROLLOUT` | `2` |
| `OUTPUT_ROOT` | `${PWD}/orbit-runs/muon-sharding-smoke` |
| `RUN_ID` | `qwen3-30b-a3b-gsm8k-<topology>-muon-<UTC timestamp>` |
| `ORBIT_CONDA_ENV` | `orbit` |
| `ORBIT_CONDA_SH` | `/data/shared/conda/miniconda3/etc/profile.d/conda.sh` |

`NUM_GPUS` must remain 8. Other paths and run-control values can be overridden
through the environment. Existing run directories are never overwritten.

### One-time Qwen3 checkpoint conversion

Skip this step if `REF_LOAD` already exists and is nonempty. Run the conversion
inside one Slurm allocation exposing eight GPUs:

```bash
git clone https://github.com/Sphere-AI-Lab/orbit.git orbit
cd orbit
source /data/shared/conda/miniconda3/etc/profile.d/conda.sh
conda activate "${ORBIT_CONDA_ENV:-orbit}"
MODEL_ARGS_NUM_LAYERS=48
source scripts/models/qwen3-30B-A3B.sh
mkdir -p "${HOME}/models"
PYTHONPATH="${PWD}/thirdparty/Megatron-LM:${PWD}/thirdparty/Megatron-Bridge/src:${PWD}/thirdparty/sglang/python${PYTHONPATH:+:${PYTHONPATH}}" \
torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT:-${HOME}/models/Qwen3-30B-A3B}" \
  --save "${REF_LOAD:-${HOME}/models/Qwen3-30B-A3B_torch_dist}"
```

### Inspect all three commands

Dry-run mode creates no output, activates no Conda environment, and performs
no model, GPU, Ray, or submodule checks:

```bash
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_tp1_ep8_etp1_muon.sh
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_tp2_ep8_etp1_muon.sh
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_tp2_ep4_etp1_muon.sh
```

### Run a topology

Use an eight-GPU allocation. For example, launch the first topology with:

```bash
srun -N1 -n1 bash -lc 'cd "$(pwd)" && bash examples/muon/run_qwen3_30b_a3b_gsm8k_tp1_ep8_etp1_muon.sh'
```

Each run writes `${OUTPUT_ROOT}/${RUN_ID}/train.log`. A runtime smoke passes only
when it selects the intended optimizer and topology, completes steps 0 and 1,
exits zero, emits finite metrics, and contains no traceback, CUDA OOM, Ray
failure, or scheduler failure.

These are compatibility smokes. Creating the scripts or passing dry-run tests
does not establish GPU runtime success, convergence, speed, memory use, quality,
or optimizer performance. Validate one topology on the cluster before expanding
to the remaining topologies.

## Qwen3-30B-A3B two-node Muon matrix

These Muon-only compatibility smokes use two allocated nodes with eight GPUs
per node. They keep tensor parallelism at 2 and expert tensor parallelism at 1,
then isolate pipeline, context, and expert parallelism relative to the baseline.
Every wrapper selects `dist_muon`; expert tensor parallelism above 1 is outside
the supported matrix.

| ID | TP | PP | CP | EP | expert-TP | Wrapper |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 2 | 1 | 1 | 8 | 1 | `run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp1_ep8_etp1_muon.sh` |
| `pp` | 2 | 2 | 1 | 8 | 1 | `run_qwen3_30b_a3b_gsm8k_2node_tp2_pp2_cp1_ep8_etp1_muon.sh` |
| `cp` | 2 | 1 | 2 | 8 | 1 | `run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp2_ep8_etp1_muon.sh` |
| `ep` | 2 | 1 | 1 | 16 | 1 | `run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp1_ep16_etp1_muon.sh` |

Inspect the four resolved commands without requiring Slurm, GPUs, assets,
Conda, or initialized submodules:

```bash
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp1_ep8_etp1_muon.sh
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_2node_tp2_pp2_cp1_ep8_etp1_muon.sh
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp2_ep8_etp1_muon.sh
DRY_RUN=1 bash examples/muon/run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp1_ep16_etp1_muon.sh
```

For a runtime smoke, enter an existing allocation containing exactly two nodes
with eight visible GPUs on each node, then invoke one wrapper directly from the
clean remote worktree. Do not wrap the command in another allocation request:

```bash
cd "$(pwd)"
bash examples/muon/run_qwen3_30b_a3b_gsm8k_2node_tp2_pp1_cp1_ep8_etp1_muon.sh
```

The runner validates the allocation and both nodes before checking model
assets. It starts one owned `srun`/Ray process group per node, refuses to attach
to a Ray head already using its configured address, and cleans up only those
two owned groups. It never uses node-wide `pkill`, `ray stop`, or `scancel`.
Each run writes `train.log`, `ray-head.log`, `ray-worker.log`, and `cleanup.log`
under `${OUTPUT_ROOT}/${RUN_ID}`; the two-node `OUTPUT_ROOT` default is
`${PWD}/orbit-runs/muon-two-node-sharding-smoke`.

A row passes only when the resolved command selects `dist_muon` and its named
topology, Ray registers all 16 GPUs, optimizer steps 0 and 1 complete, the job
exits zero with finite metrics, and its logs contain no traceback, CUDA OOM,
Ray failure, or scheduler failure. These short runs establish execution
compatibility only; they do not establish convergence, throughput, memory use,
or quality.
