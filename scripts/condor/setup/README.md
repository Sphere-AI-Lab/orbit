# Orbit environment setup on HTCondor

This directory provides HTCondor entrypoints for the CUDA 13 environment
recipe in `scripts/slurm/setup/cu130/`. The installer, pins, and Python
verifier remain canonical there; the wrappers here contain no dependency or
CUDA-version logic of their own. The same recipe installs on the H200 Slurm
cluster (see the canonical README's "H200 Slurm cluster" section) — this
directory covers the MPI HTCondor cluster, whose paths are the recipe's
built-in defaults.

The checked-in submit description selects one H100 because FlashAttention 3 is
sm_90a-only. This pool advertises two H100 variants — `CUDADeviceName ==
"NVIDIA H100"` (96 GB nodes) and `"NVIDIA H100 80GB HBM3"` — the description
matches the first; adjust the expression if you want the other. The cu130
profile also accepts B200 nodes (FA3 is not exercised there); change the
requirement and drop `--full-h100` from verification accordingly.

## Interactive installation

Obtain an approved one-GPU interactive allocation. A request with the same
resource shape as the batch description is:

```bash
condor_submit_bid BID -i \
  -append request_cpus=16 \
  -append request_gpus=1 \
  -append request_disk=500G \
  -append request_memory=128000 \
  -append 'requirements=(CUDADeviceName=="NVIDIA H100")'
```

Choose and approve `BID` from current `condor_free` capacity before submitting.
Once the shell is running on the allocated execution node:

```bash
cd /path/to/orbit
bash scripts/condor/setup/install_env.sh
bash scripts/condor/setup/verify_env.sh --full-h100
```

The wrapper adds `~/.local/bin` (uv) to `PATH` and otherwise forwards all
installer arguments and environment variables — `ENV_PREFIX`, `SOURCE_ROOT`,
`CACHE_DIR`, `UV_CACHE_DIR`, `JOBS`, and the `ORBIT_*_REPO`/`ORBIT_*_COMMIT`
overrides — to the canonical installer, whose defaults already point at this
cluster's paths (`/fast/zqiu/orbit-iclr/orbit/{envs,sources,cache}/orbit-cu130-v1`,
conda at `/home/zqiu/anaconda3`). `--dry-run` prints the plan without touching
anything. The installer holds a lock at `$ENV_PREFIX.install.lock`; do not run
concurrent installers against the same prefix.

## uv cache placement

uv requires a lock-capable cache filesystem, so the canonical installer defaults
`UV_CACHE_DIR` to cluster home. Keep that default for ordinary installs: the
cache reaches about 19 GB, persists between installs, and a warm cache skips
both the downloads and the extraction step entirely. Home is NFS
(`sc-fb1:/cluster-home`) but reads back at roughly 326 MB/s, so `materialize_env.py`
copies out of a warm cache quickly. The condor wrappers set no cache policy of
their own, so installs here behave the same as on the Slurm cluster.

What matters is whether the cache is warm, not where it lives. Two clean-room
installs on 2026-08-24 measured this directly:

| job | cache | sustained fill rate |
| --- | --- | --- |
| 17476896 | home NFS | ~90–220 KB/s |
| 17477168 | node-local scratch | ~326 KB/s |

Sampling job 17477168 over five minutes showed the whole cache growing by 95 MB
while the unpacked `archive-v0` tree did not grow at all, and only ten packages
had finished downloading in the first hour. A cold install on this cluster is
bound by download throughput, not by the cache filesystem, so relocating the
cache buys almost nothing: expect a cold cache to take the better part of a day
either way, and do not use node-local scratch to try to avoid that.

The way to make an install fast is to not download at all. Point `UV_CACHE_DIR`
at a cache that already holds these pins — the environments under `envs/` each
left one behind — and the install skips both the downloads and the extraction.
For the same reason, prefer one cache reused across installs of this profile
over a fresh cache per install.

Share that cache with the Miles-IMP environment on this cluster. Both profiles
derive their binary pins from the same `radixark/miles` Dockerfile, and the
installed artifacts are identical, not merely similarly versioned. Verified on
2026-08-24 by comparing the two environments rather than their caches:

| file | Miles env | this profile |
| --- | --- | --- |
| `torch/lib/libtorch_cuda.so` | 456,142,457 B, md5 `05b737fb2408…` | identical |
| `nvidia/cudnn/lib/libcudnn_adv.so.9` | 109,169,104 B, md5 `8bc5a9fb4c38…` | identical |

Compare installed files, not cache entries, if you re-check this. A uv cache
accumulates every artifact it has ever seen — the global cache on this account
holds sixteen distinct torch builds from unrelated work — so picking an
arbitrary entry out of one proves nothing.

## Batch installation

The submit description assumes the repository, environment prefix, and run
directory are visible from both the login and execution nodes. Create the
durable log directory before submission; HTCondor opens these paths before the
job executes.

```bash
cd /path/to/orbit
run_dir=${XDG_STATE_HOME:-$HOME/.local/state}/remote-cluster-runs/mpi1/orbit/orbit-main/EXECUTION_ID/install-env
mkdir -p "$run_dir"

condor_submit_bid BID \
  repo_dir="$PWD" \
  env_prefix=/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1 \
  source_root=/fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1 \
  cache_dir=/fast/zqiu/orbit-iclr/orbit/cache/orbit-cu130-v1 \
  run_dir="$run_dir" \
  scripts/condor/setup/install_env.sub
```

Batch mode uses the canonical installer defaults for everything not named in
the submit variables. The submit file imports only the listed path and cache
variables from the submitting shell (`getenv`). Use the interactive workflow
for non-default installer knobs, or add each intended variable explicitly to
`getenv` before submission.

The submit file requests 1 H100, 16 CPUs, 128 GB of host memory, and 500 GB of
disk (the install writes to shared filesystems; scratch use is small). It uses
the shared filesystem directly (`should_transfer_files = NO`), writes the event
log, stdout, and stderr to `run_dir`, and never embeds a bid.

For a clean-room installation, pick a fresh shared suffix for `env_prefix`,
`source_root`, and `cache_dir` as described in the canonical README — do not
overwrite the validated `orbit-cu130-v1` environment.

Decide deliberately what to do about `UV_CACHE_DIR`. The canonical default
hardcodes the `orbit-cu130-v1` suffix regardless of the prefix in use, so
leaving it unset reuses that cache — which makes the run fast but means the
downloads are not part of what is being tested. Pointing it at a fresh path
tests the whole path from nothing, and costs the better part of a day at the
download rates above. Neither is wrong; state which one a given run intends.

## Verification and activation

After a batch installation completes, enter an allocated GPU shell and run:

```bash
bash scripts/condor/setup/verify_env.sh --full-h100
```

(`ENV_PREFIX`/`SOURCE_ROOT` override the defaults.) Activation and the runtime
loader (`ORBIT_VENV=... source examples/load_cuda13_2_orbit_env.sh`) are
documented in the canonical README's "Activate" section.
