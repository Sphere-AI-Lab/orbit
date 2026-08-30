"""Torch allocator memory-history snapshots (``ORBIT_MEMORY_SNAPSHOT_DIR``).

Lifted verbatim out of ``miles/backends/megatron_utils/model.py`` (Phase-3
slice 3f, P1 lift-out). The train loop there keeps the stamped
record/dump/disable seam around one training step (including the OOM path) and
calls in here to write the pickle that ``torch.cuda.memory._dump_snapshot``
produces, one file per rank/rollout/step.

Kept apart from ``orbit.megatron.memory_attribution`` on purpose: that module
does semantic bucket attribution over model/optimizer tensors and pulls in the
weight-sync layer to do it, while this is a two-call wrapper on the CUDA
allocator's own history recorder that the hot training module imports.
"""

import os

import torch


def _dump_memory_history_snapshot(snapshot_dir: str, rollout_id: int, step_id: int, suffix: str = "") -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    os.makedirs(snapshot_dir, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    pkl_path = os.path.join(snapshot_dir, f"mem_rank{rank}_rollout{rollout_id}_step{step_id}{suffix_part}.pickle")
    torch.cuda.memory._dump_snapshot(pkl_path)
    if rank == 0:
        print(f"[mem] snapshot dumped to {pkl_path}", flush=True)
