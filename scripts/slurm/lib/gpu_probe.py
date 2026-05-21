"""gpu_probe — fail fast if torch.cuda.set_device(i) is broken on this node.

Catches the failure mode where `nvidia-smi --query-gpu=count` reports 8 GPUs
but `torch.cuda.set_device(i)` raises `cudaErrorDevicesUnavailable` because a
GPU is stuck at the driver level. Cluster admin reboot is the only fix.

Run via the launcher's healthcheck section. Exits 0 iff every visible GPU
accepts set_device + a tiny allocation.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as e:
        print(f"FAIL: torch import failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    n = torch.cuda.device_count()
    if n == 0:
        print("FAIL: no CUDA devices visible", file=sys.stderr)
        return 1

    bad: list[tuple[int, str, str]] = []
    for i in range(n):
        try:
            torch.cuda.set_device(i)
            x = torch.zeros(1, device=f"cuda:{i}")
            del x
        except Exception as e:
            bad.append((i, type(e).__name__, str(e).splitlines()[0]))

    if bad:
        for i, etype, msg in bad:
            print(f"FAIL: gpu{i} {etype}: {msg}", file=sys.stderr)
        return 1

    print(f"OK: {n} gpus initializable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
