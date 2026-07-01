"""nccl_probe — multi-node NCCL all_reduce smoke test over torch.distributed.

Bootstraps a process group from SLURM env vars and runs an all_reduce size
sweep — the exact IB transport path miles training uses. Exits 0 on healthy
fabric; nonzero if a collective errors/hangs, or (with a bandwidth floor set)
if NCCL silently falls back to sockets. Run by the launcher as the tier-nccl
preflight; see docs/launcher.md "Healthcheck" for rationale + ad-hoc salloc recipe.

Launch with one task per GPU:
    srun --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=8 python lib/nccl_probe.py

Tunables (env):
    NCCL_HEALTHCHECK_MIN_BYTES   smallest message       [1048576   = 1 MiB]
    NCCL_HEALTHCHECK_MAX_BYTES   largest message        [268435456 = 256 MiB]
    NCCL_HEALTHCHECK_ITERS       timed iters per size   [10]
    NCCL_HEALTHCHECK_TIMEOUT_S   process-group timeout  [120]
    NCCL_HEALTHCHECK_MIN_BUSBW_GB  min bus bw (GB/s) to PASS; 0 = off [0]
"""

import datetime
import os
import resource
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist


def _env_int(name: str, default: int, *, minimum=None) -> int:
    # Unset/empty -> default; a set-but-unparseable or out-of-range value fails
    # the probe rather than silently reverting (which could disable a safety knob
    # or wedge the sweep, e.g. MIN_BYTES=0 would loop forever).
    val = os.environ.get(name)
    if not val:
        return default
    try:
        n = int(val)
    except ValueError:
        raise ValueError(f"{name}={val!r} must be an integer") from None
    if minimum is not None and n < minimum:
        raise ValueError(f"{name}={n} must be >= {minimum}")
    return n


def _env_float(name: str, default: float, *, minimum=None) -> float:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        x = float(val)
    except ValueError:
        raise ValueError(f"{name}={val!r} must be a number") from None
    if minimum is not None and x < minimum:
        raise ValueError(f"{name}={x} must be >= {minimum}")
    return x


def parse_nodelist_head(nodelist: str) -> str:
    """Get first hostname from a SLURM_NODELIST like 'slinky-[23,20]'."""
    out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
    return out.decode().splitlines()[0]


def dump_node_env(rank: int, local_rank: int) -> None:
    """One-shot dump (first task per node) of HCA, memlock limit, NCCL env."""
    if local_rank != 0:
        return
    host = socket.gethostname()
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    nccl_keys = [
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_SOCKET_IFNAME",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_NET",
        "CUDA_VISIBLE_DEVICES",
    ]
    print(f"\n[env {host} rank={rank}] memlock soft={soft} hard={hard}", flush=True)
    # memlock must be effectively unlimited for RDMA/IB registration; a small cap
    # silently forces NCCL to fall back to TCP sockets (which the MIN_BUSBW_GB gate
    # then catches as a bandwidth floor miss). Surface it directly as the root cause.
    # "Unlimited" is NOT always the RLIM_INFINITY sentinel: this cluster caps memlock
    # at a literal ~2^63 bytes (9 EiB — unlimited in practice), so a sentinel-equality
    # check WARNs on every healthy node. Warn only below a threshold that no sane
    # config uses but every real misconfig (the 64 KiB / 64 MiB defaults) sits under.
    memlock_warn_below = 1 << 40  # 1 TiB
    if soft != resource.RLIM_INFINITY and soft < memlock_warn_below:
        print(
            f"[env {host}] WARN: RLIMIT_MEMLOCK soft={soft} (<1 TiB) — RDMA/IB "
            f"registration can fail or fall back to sockets (raise via `ulimit -l unlimited`)",
            flush=True,
        )
    for k in nccl_keys:
        print(f"[env {host}]   {k}={os.environ.get(k, '<unset>')}", flush=True)
    try:
        ibstat = subprocess.check_output(["ibstat"], stderr=subprocess.STDOUT, timeout=5)
        print(f"[env {host}] ibstat:\n{ibstat.decode()}", flush=True)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[env {host}] ibstat failed: {e}", flush=True)


def main() -> int:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    # Rendezvous master is rank 0's node = first node of THIS step, not the job
    # allocation: on a GOOD_NODES subset, SLURM_JOB_NODELIST[0] may be excluded.
    # Prefer injected MASTER_ADDR, then step nodelist, then job nodelist (standalone).
    master_addr = os.environ.get("MASTER_ADDR") or parse_nodelist_head(
        os.environ.get("SLURM_STEP_NODELIST") or os.environ["SLURM_JOB_NODELIST"]
    )
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # --gpus-per-task=1 exposes one device (index 0); if the node's 8 GPUs are
    # all visible instead (some srun --overlap layouts), fan out by local_rank.
    n_visible = torch.cuda.device_count()
    torch.cuda.set_device(0 if n_visible <= 1 else local_rank % n_visible)
    host = socket.gethostname()
    if rank == 0:
        print(f"[nccl-probe] world={world_size} master={master_addr}:{master_port}", flush=True)
    print(f"[rank {rank:>2}] host={host} local_rank={local_rank}", flush=True)

    dump_node_env(rank, local_rank)

    timeout_s = _env_int("NCCL_HEALTHCHECK_TIMEOUT_S", 120, minimum=1)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    if rank == 0:
        print(f"[nccl-probe] process group ready ({world_size} ranks)", flush=True)

    min_bytes = _env_int("NCCL_HEALTHCHECK_MIN_BYTES", 1 << 20, minimum=1)
    max_bytes = _env_int("NCCL_HEALTHCHECK_MAX_BYTES", 1 << 28, minimum=1)
    iters = _env_int("NCCL_HEALTHCHECK_ITERS", 10, minimum=1)
    min_busbw = _env_float("NCCL_HEALTHCHECK_MIN_BUSBW_GB", 0.0, minimum=0.0)
    if max_bytes < min_bytes:
        raise ValueError(
            f"NCCL_HEALTHCHECK_MAX_BYTES ({max_bytes}) must be >= " f"NCCL_HEALTHCHECK_MIN_BYTES ({min_bytes})"
        )
    warmup_iters = 3
    bytes_per_elem = 2  # float16

    sizes = []
    s = min_bytes
    while s <= max_bytes:
        sizes.append(s)
        s <<= 2  # 4x steps: 1, 4, 16, 64, 256 MiB, ...
    if sizes and sizes[-1] != max_bytes:
        sizes.append(max_bytes)  # 4x steps can skip the exact max (e.g. 8 GiB) — measure it

    if rank == 0:
        print(f"\n{'size':>12}  {'time_ms':>10}  {'algbw_GB/s':>12}  {'busbw_GB/s':>12}", flush=True)
        print("-" * 52, flush=True)

    max_busbw = 0.0
    for size in sizes:
        buf = torch.ones(size // bytes_per_elem, dtype=torch.float16, device="cuda")
        for _ in range(warmup_iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        # ring all_reduce bus bandwidth: 2 * (N-1)/N * size / time
        algbw = size / dt / 1e9
        busbw = algbw * 2 * (world_size - 1) / world_size
        max_busbw = max(max_busbw, busbw)

        if rank == 0:
            human = f"{size / 1024**3:.2f} GiB" if size >= 1 << 30 else f"{size / 1024**2:.0f} MiB"
            print(f"{human:>12}  {dt * 1e3:>10.2f}  {algbw:>12.2f}  {busbw:>12.2f}", flush=True)

    dist.destroy_process_group()

    # Bandwidth floor (opt-in): gate on max busbw (the large-message point, not
    # the latency-bound small ones) to fail a silent IB->socket fallback that
    # completes but at socket speed. Each rank checks its own; any slow one fails.
    if min_busbw > 0 and max_busbw < min_busbw:
        print(
            f"[nccl-probe] FAIL  rank={rank}  max_busbw={max_busbw:.1f}GB/s "
            f"< floor {min_busbw:.0f}GB/s (suspect socket/TCP fallback, not IB)",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if rank == 0:
        print(f"\n[nccl-probe] PASS  ranks={world_size}  max_busbw={max_busbw:.1f}GB/s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — any failure here is a probe FAIL
        print(
            f"[nccl-probe] FAIL  rank={os.environ.get('SLURM_PROCID', '?')}  "
            f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
