import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    # ORBIT-SEAM: available_memory() enriched with allocator-fragmentation diagnostics (single
    # memory_stats() snapshot, inactive_split/active/segments/alloc_retries) for OOM debugging.
    # `.get(key, 0)` on every lookup below is not a guard against an empty
    # dict -- current_device() above already forced CUDA's lazy init, so by
    # this point memory_stats() always returns the full stats dict (zeros for
    # a device the allocator has never served, never {}). The real reason is
    # key-name drift across torch versions; this is the same defaulting
    # torch's own memory_allocated()/memory_reserved() use when they read
    # this dict.
    stats = torch.cuda.memory_stats(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        # ORBIT-SEAM: allocated_GB/reserved_GB now read off the single `stats` snapshot above
        # instead of separate torch.cuda.memory_allocated()/memory_reserved() calls
        # Single snapshot: torch.cuda.memory_allocated()/memory_reserved() each
        # rebuild this same stats dict under their own mutex acquisition, which
        # would take three separate instants for numbers this module reasons
        # about as one consistent snapshot. Read them off `stats` instead.
        "allocated_GB": _byte_to_gb(stats.get("allocated_bytes.all.current", 0)),
        "reserved_GB": _byte_to_gb(stats.get("reserved_bytes.all.current", 0)),
        # Torch calls this "Non-releasable memory": free bytes trapped inside a
        # segment that still holds a live block, which empty_cache() cannot
        # return. A large value here against a small allocated_GB is
        # fragmentation, not a leak. (H1, stated numerically.)
        "inactive_split_GB": _byte_to_gb(stats.get("inactive_split_bytes.all.current", 0)),
        # active_bytes counts blocks that are allocated OR still recorded as
        # in-use by a CUDA stream. active_GB - allocated_GB is bytes held only
        # because a stream hasn't released them yet -- H2 stated numerically,
        # the same way inactive_split_GB states H1.
        "active_GB": _byte_to_gb(stats.get("active_bytes.all.current", 0)),
        "segments": stats.get("segment.all.current", 0),
        # segment.all.current is instantaneous; num_alloc_retries is
        # cumulative since process start. The delta between two probes is the
        # meaningful reading for alloc_retries -- a steady non-zero value here
        # can be old news, not live distress.
        "alloc_retries": stats.get("num_alloc_retries", 0),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
