from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

from orbit.megatron.peft_utils import build_peft_sync_spec

from orbit.transport.interface import PeftPayload, PeftSendResult, PeftWeightTransport
from orbit.transport.registry import PEFT_METHODS, PeftMethodSpec
from orbit.transport.runtime import PeftRuntimeMode, resolve_peft_runtime_mode

if TYPE_CHECKING:
    import torch.distributed as dist

__all__ = [
    "PeftPayload",
    "PeftRuntimeMode",
    "PeftSendResult",
    "PeftWeightTransport",
    "PEFT_METHODS",
    "PeftMethodSpec",
    "build_peft_transport",
]


def build_peft_transport(
    args: Namespace,
    *,
    use_distribute: bool,
    ipc_gather_group: "dist.ProcessGroup | None" = None,
    ipc_gather_src: int | None = None,
) -> PeftWeightTransport | None:
    """Construct the right PEFT transport for the run.

    Returns None when peft_method is "none" — caller-side full-FT branch
    remains explicit (no NullTransport, see spec §5)."""
    sync_spec = build_peft_sync_spec(args)
    if sync_spec is None:                    # peft_method == "none"
        return None
    runtime_mode = resolve_peft_runtime_mode(args, use_distribute=use_distribute)
    method_spec = PEFT_METHODS[sync_spec.method]
    common_kwargs = dict(
        args=args,
        method_spec=method_spec,
        sync_spec=sync_spec,
        runtime_mode=runtime_mode,
    )
    if use_distribute:
        if runtime_mode.transport == "ray":
            from orbit.transport.backends.ray_object import RayObjectBackend

            return RayObjectBackend(**common_kwargs)
        # Late import — NcclBackend imported lazily so colocate-only deployments
        # never need to touch torch.distributed init paths.
        from orbit.transport.backends.nccl import NcclBackend
        return NcclBackend(**common_kwargs)
    if ipc_gather_group is None or ipc_gather_src is None:
        raise ValueError(
            "IpcBackend requires ipc_gather_group and ipc_gather_src "
            "(set use_distribute=True for distributed engines)."
        )
    from orbit.transport.backends.ipc import IpcBackend
    return IpcBackend(
        **common_kwargs,
        ipc_gather_group=ipc_gather_group,
        ipc_gather_src=ipc_gather_src,
    )
