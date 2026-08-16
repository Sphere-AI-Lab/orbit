import asyncio
import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from orbit.ray.utils import build_noset_visible_devices_env_vars


def _uses_dsv4_deepep(args) -> bool:
    return getattr(args, "dsv4_moe_dispatcher", None) == "deepep"


def _build_train_actor_env(args) -> dict[str, str]:
    env_vars = {
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        **build_noset_visible_devices_env_vars(),
        **args.train_env_vars,
    }

    if _uses_dsv4_deepep(args):
        # DeepEP V2 ElasticBuffer requires NCCL communicators created with
        # symmetric-memory support. Match SGLang's DSV4 DeepEP default.
        env_vars["NCCL_CUMEM_ENABLE"] = "1"
        env_vars["NCCL_NVLS_ENABLE"] = "1"

    # PEFT hands the GPU back to a colocated engine every rollout, and its train
    # step frees nearly everything: measured on 8xB200, rank 0 held
    # `allocated 0.09 GB` against `reserved 65.71 GB`, of which 65.62 GB -- the
    # entire gap -- was non-releasable split memory across just 17 segments. A
    # few MB of straggler blocks pin ~3.9 GB apiece, `empty_cache()` may only
    # return a wholly-free segment so it returns nothing, and the engine's
    # `cuMemCreate` then fails at resume (`func=resume`, fatal at rollout 2 on
    # an 80 GB H100). Expandable segments map physical pages on demand, so a
    # live block pins pages instead of its whole segment.
    #
    # Full fine-tuning is left alone deliberately: its pool is tight -- reserved
    # tracks allocated to within 0.07 GB on the same node -- because
    # `_finalize_train_offload_args` forces the grad-buffer and optimizer
    # offloads on for it, which return genuinely live state every step. It has
    # no gap to close.
    if getattr(args, "peft_method", "none") != "none":
        env_vars.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        )

    if source_patcher_config := args.dumper_source_patcher_config_train:
        env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

    if "PYTHONPATH" in os.environ and "PYTHONPATH" not in env_vars:
        env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]

    return env_vars


class RayTrainGroup:
    """
    A group of ray actors

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        *,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
        with_opd_teacher: bool = False,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher

        # Allocate the GPUs for actors w/o instantiating them
        self._actor_handles = self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = _build_train_actor_env(self.args)

        backend = self.args.train_backend
        assert backend == "megatron", f"Only the Megatron train backend is supported (got {backend})."
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        actor_impl = MegatronTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        actor_handles = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            actor_handles.append(actor)

        return actor_handles

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimizer, local ckpt, etc.
        """
        return await self._broadcast(
            "init", self.args, self.role, with_ref=self.with_ref, with_opd_teacher=self.with_opd_teacher
        )

    async def train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        await self._broadcast("train", rollout_id, rollout_data_ref)

    async def compute_eval_nll(self, rollout_id) -> dict:
        """Held-out NLL of the current actor weights, reduced across ranks.

        Unlike :meth:`train`, this returns its result -- the number is the whole
        point. ``_broadcast`` hands back one entry per actor across the full
        TP x PP x DP grid, but the actors deduplicate themselves: exactly one
        returns the DP-reduced statistics and every other returns ``None``.
        Averaging the per-actor values instead would double-count TP/PP
        replicas (which hold the same samples) and would mis-weight DP shards
        (which hold different token counts).
        """
        from orbit.utils.eval_nll import select_eval_nll_result

        return select_eval_nll_result(await self._broadcast("compute_eval_nll", rollout_id))

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        await self._broadcast("update_weights")

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def prefetch_train_state(self, rollout_id):
        """Async wake-up overlap: issue frozen-base H2D on each actor's
        dedicated wake stream so wake_up() can wait on the recorded event
        instead of paying the H2D cost synchronously. No-op on actors where
        offload_train_async is off (see MegatronTrainRayActor.prefetch_train_state).
        """
        await self._broadcast("prefetch_train_state", rollout_id)

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def connect(self, critic_group):
        if len(self._actor_handles) != len(critic_group._actor_handles):
            raise RuntimeError(
                "actor and critic groups must have equal worker counts; "
                f"actor={len(self._actor_handles)}, critic={len(critic_group._actor_handles)}"
            )
        refs = [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=True)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self, rollout_manager):
        await self._broadcast("set_rollout_manager", rollout_manager)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
