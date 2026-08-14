import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args

from miles.backends.megatron_utils.cp_contract import canonicalize_cp_comm_type
from miles.utils.ft_utils.process_group_utils import GroupInfo

from ..training_utils.parallel import ParallelState, get_parallel_state

logger = logging.getLogger(__name__)


def create_megatron_parallel_state(
    indep_dp: GroupInfo,
) -> ParallelState:
    vpp_size, microbatch_group_size_per_vp_stage = _compute_vpp_fields()
    args = get_args()
    cp_comm_type = getattr(args, "cp_comm_type_canonical", None)
    if cp_comm_type is None:
        cp_comm_type = canonicalize_cp_comm_type(getattr(args, "cp_comm_type", None))

    def _create_intra_dp(with_context_parallel: bool):
        return GroupInfo(
            rank=mpu.get_data_parallel_rank(with_context_parallel=with_context_parallel),
            size=mpu.get_data_parallel_world_size(with_context_parallel=with_context_parallel),
            group=mpu.get_data_parallel_group(with_context_parallel=with_context_parallel),
            gloo_group=mpu.get_data_parallel_group_gloo(with_context_parallel=with_context_parallel),
        )

    return ParallelState(
        intra_dp=_create_intra_dp(with_context_parallel=False),
        intra_dp_cp=_create_intra_dp(with_context_parallel=True),
        cp=GroupInfo(
            rank=mpu.get_context_parallel_rank(),
            size=mpu.get_context_parallel_world_size(),
            group=mpu.get_context_parallel_group(),
        ),
        cp_comm_type=cp_comm_type,
        tp=GroupInfo(
            rank=mpu.get_tensor_model_parallel_rank(),
            size=mpu.get_tensor_model_parallel_world_size(),
            group=mpu.get_tensor_model_parallel_group(),
        ),
        pp=GroupInfo(
            rank=mpu.get_pipeline_model_parallel_rank(),
            size=mpu.get_pipeline_model_parallel_world_size(),
            group=mpu.get_pipeline_model_parallel_group(),
        ),
        ep=GroupInfo(
            rank=mpu.get_expert_model_parallel_rank(),
            size=mpu.get_expert_model_parallel_world_size(),
            group=mpu.get_expert_model_parallel_group(),
        ),
        etp=GroupInfo(
            rank=mpu.get_expert_tensor_parallel_rank(),
            size=mpu.get_expert_tensor_parallel_world_size(),
            group=mpu.get_expert_tensor_parallel_group(),
        ),
        indep_dp=indep_dp,
        is_pp_last_stage=mpu.is_pipeline_last_stage(),
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
    )


def _compute_vpp_fields() -> tuple[int, int | None]:
    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size_value is None or vpp_size_value <= 1:
        return 1, None

    return vpp_size_value, get_args().pipeline_model_parallel_size


def verify_megatron_parallel_state(
    model: torch.nn.Module | Sequence[torch.nn.Module],
    args: Namespace | None = None,
) -> None:
    """Verify that ParallelState fields match what the model config produces."""
    parallel_state = get_parallel_state()
    args = args or get_args()
    chunks = list(model) if isinstance(model, Sequence) else [model]
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    for chunk_index, model_chunk in enumerate(chunks):
        config = get_model_config(model_chunk)

        if vpp_size_value is not None and vpp_size_value > 1:
            expected = config.microbatch_group_size_per_vp_stage
            actual = parallel_state.microbatch_group_size_per_vp_stage
            if actual != expected:
                raise ValueError(
                    "Megatron parallel-state mismatch: "
                    f"rank={rank}, chunk={chunk_index}, "
                    f"microbatch_group_size_per_vp_stage state={actual}, model={expected}"
                )

        if parallel_state.cp.size <= 1:
            continue

        requested_value = getattr(args, "cp_comm_type_canonical", None)
        if requested_value is None:
            requested_value = getattr(args, "cp_comm_type", parallel_state.cp_comm_type)
        requested = canonicalize_cp_comm_type(requested_value)
        state_transport = canonicalize_cp_comm_type(parallel_state.cp_comm_type)
        model_value = getattr(config, "cp_comm_type", None)
        try:
            model_transport = canonicalize_cp_comm_type(model_value)
        except ValueError as exc:
            raise ValueError(
                "CP contract mismatch: "
                f"rank={rank}, chunk={chunk_index}, cp_size={parallel_state.cp.size}, "
                f"requested={requested}, model={model_value!r}, "
                f"layout={getattr(args, 'cp_token_layout', None)}; {exc}"
            ) from exc

        model_cp_size = getattr(config, "context_parallel_size", None)
        requested_hierarchy = getattr(args, "hierarchical_context_parallel_sizes", None)
        model_hierarchy = getattr(config, "hierarchical_context_parallel_sizes", None)
        requested_hierarchy_value = list(requested_hierarchy) if requested_hierarchy is not None else None
        model_hierarchy_value = list(model_hierarchy) if model_hierarchy is not None else None
        hierarchy_mismatch = requested == "a2a+p2p" and (
            requested_hierarchy_value is None or model_hierarchy_value != requested_hierarchy_value
        )
        if (
            model_cp_size != parallel_state.cp.size
            or state_transport != requested
            or model_transport != requested
            or hierarchy_mismatch
        ):
            raise ValueError(
                "CP contract mismatch: "
                f"rank={rank}, chunk={chunk_index}, cp_size={parallel_state.cp.size}, "
                f"model_cp_size={model_cp_size}, requested={requested}, "
                f"state={state_transport}, model={model_transport}, "
                f"hierarchy={requested_hierarchy}, model_hierarchy={model_hierarchy}, "
                f"layout={getattr(args, 'cp_token_layout', None)}, "
                f"bridge_mode={getattr(args, 'megatron_to_hf_mode', None)}"
            )

    if rank == 0 and parallel_state.cp.size > 1:
        logger.info(
            "CP_CONTRACT_VERIFIED cp=%s transport=%s layout=%s chunks=%s",
            parallel_state.cp.size,
            getattr(args, "cp_comm_type_canonical", parallel_state.cp_comm_type),
            getattr(args, "cp_token_layout", "inactive"),
            len(chunks),
        )


def get_packed_seq_params(batch: dict[str, torch.Tensor], args: Namespace) -> PackedSeqParams:
    if args.qkv_format == "thd":
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            max_seqlen_q=batch["max_seqlen"],
            max_seqlen_kv=batch["max_seqlen"],
            qkv_format="thd",
        )
        batch["packed_seq_params"] = packed_seq_params
        return packed_seq_params
    else:
        return None
