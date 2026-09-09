from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass


def overlap_oft_sync(args: Namespace) -> bool:
    """Stage outside the pause only for the fully-async native OFT path."""
    return (
        getattr(args, "fully_async", False)
        and getattr(args, "peft_method", "none") == "oft"
        and getattr(args, "adapter_double_buffer", False)
        and getattr(args, "peft_distributed_transport", "nccl") == "nccl"
        and getattr(args, "pause_generation_mode", None) == "retract"
        and getattr(args, "sglang_oft_impl", "sibling") == "sibling"
        and not getattr(args, "colocate", False)
    )


@dataclass(frozen=True)
class PeftRuntimeMode:
    peft_method: str
    use_distribute: bool
    distributed_transport: str
    adapter_versioning: bool
    adapter_double_buffer: bool

    @property
    def transport(self) -> str:
        return self.distributed_transport if self.use_distribute else "ipc"

    def log_line(self) -> str:
        return (
            f"adapter_runtime method={self.peft_method} transport={self.transport} "
            f"versioning={'true' if self.adapter_versioning else 'false'} "
            f"double_buffer={'true' if self.adapter_double_buffer else 'false'}"
        )


def resolve_peft_runtime_mode(args: Namespace, *, use_distribute: bool) -> PeftRuntimeMode:
    peft_method = getattr(args, "peft_method", "none")
    adapter_double_buffer = bool(getattr(args, "adapter_double_buffer", False))
    distributed_transport = getattr(args, "peft_distributed_transport", "nccl") or "nccl"

    if adapter_double_buffer and peft_method == "none":
        raise ValueError("--adapter-double-buffer requires --peft-method lora or oft")
    if adapter_double_buffer and not use_distribute:
        raise ValueError("--adapter-double-buffer requires distributed PEFT transport")
    if distributed_transport not in {"nccl", "ray"}:
        raise ValueError(
            "--peft-distributed-transport must be one of {'nccl', 'ray'}, " f"got {distributed_transport!r}"
        )
    if adapter_double_buffer and distributed_transport != "nccl":
        raise ValueError("--adapter-double-buffer requires --peft-distributed-transport nccl")
    if (
        peft_method == "oft"
        and adapter_double_buffer
        and getattr(args, "fully_async", False)
        and getattr(args, "pause_generation_mode", None) == "in_place"
    ):
        raise ValueError("Fully-async OFT adapter activation requires --pause-generation-mode retract")

    adapter_versioning = (peft_method != "none" and use_distribute) or adapter_double_buffer

    return PeftRuntimeMode(
        peft_method=peft_method,
        use_distribute=use_distribute,
        distributed_transport=distributed_transport,
        adapter_versioning=adapter_versioning,
        adapter_double_buffer=adapter_double_buffer,
    )
