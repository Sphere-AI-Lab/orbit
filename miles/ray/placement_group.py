import copy
import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# ORBIT-SEAM: needs_opd_teacher/uses_rollout_engines/uses_separate_critic gate PG sizing for OPD teacher
# actors, debug_rollout_only-style configs, and rollout-only critic setups (used throughout this file)
from miles.utils.arguments import needs_opd_teacher, uses_rollout_engines, uses_separate_critic
from miles.utils.environ import enable_experimental_ft_trainer
from ..utils.ray_utils import compute_ray_pin_head_options
from .rollout.rollout_manager import RolloutManager

logger = logging.getLogger(__name__)


def _select_train_group_class():
    if enable_experimental_ft_trainer():
        from miles.ray.train.group import RayTrainGroup
    else:
        from miles.ray.actor_group import RayTrainGroup
    return RayTrainGroup


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus, is_rdt: bool = False):
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    if is_rdt:
        # Give the trainer the node PACK filled, so rollout bundles land where GPUs
        # are still free: RayEngine STRICT_PACKs its SchedulerActors onto the engine
        # actor's node and deadlocks if nothing there is unreserved.
        node_bundle_counts: dict = {}
        for _, node_identifier, _ in bundle_infos:
            node_bundle_counts[node_identifier] = node_bundle_counts.get(node_identifier, 0) + 1
        sorted_bundle_infos = sorted(bundle_infos, key=lambda info: (-node_bundle_counts[info[1]], *sort_key(info)))
    else:
        sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


# ORBIT-SEAM: extra placement-group GPU bundles reserved for a managed OPD teacher / teacher-pool
# (--opd-serve-teacher / --opd-teacher-pool), zero under --colocate since the teacher shares actor/rollout GPUs
def _opd_teacher_extra_gpus(args) -> int:
    """Extra PG bundles for a managed OPD teacher (--opd-serve-teacher) on its own GPUs.

    Zero under --colocate: the teacher shares the actor/rollout GPUs there (see
    start_rollout_servers' bundle-cursor reset), matching how rollout itself colocates.
    """
    if args.colocate:
        return 0
    total = 0
    if getattr(args, "opd_serve_teacher", False):
        total += args.opd_teacher_num_gpus
    pool_path = getattr(args, "opd_teacher_pool", None)
    if pool_path is not None:
        # ORBIT-SEAM re-anchor: upstream dbbab156 turned miles/ray/rollout.py into a package, so the
        # pool helper is imported from its orbit home rather than re-exported through that module.
        from orbit.opd.teacher_servers import _opd_teacher_pool

        total += _opd_teacher_pool(args).served_num_gpus
    return total


def _get_placement_group_layout(args) -> tuple[int, int, int]:
    """Returns (num_gpus, rollout_offset, critic_offset).

    ORBIT-SEAM: two orbit deltas on upstream's extracted layout helper --
    (a) a `uses_separate_critic()` critic slice with its own bundles, because orbit's full-mode PPO
        critic runs on dedicated GPUs while upstream dbbab156 colocates the critic on the actor PG
        (`result["critic"] = result["actor"]`); adapter/head critic modes live inside the actor
        workers and reserve nothing, which is exactly what `uses_separate_critic()` encodes;
    (b) `uses_rollout_engines()` / `_opd_teacher_extra_gpus()` in the rollout-bearing branches.
    """
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    critic_num_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node if uses_separate_critic(args) else 0
    # The critic slice always sits directly after the actor bundles.
    critic_offset = actor_num_gpus

    if args.debug_train_only or not uses_rollout_engines(args):
        return actor_num_gpus + critic_num_gpus, 0, critic_offset
    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0, critic_offset
        return actor_num_gpus + critic_num_gpus, actor_num_gpus + critic_num_gpus, critic_offset
    if args.debug_rollout_only:
        return args.rollout_num_gpus + _opd_teacher_extra_gpus(args), 0, critic_offset
    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus) + critic_num_gpus, 0, critic_offset
    return (
        actor_num_gpus
        + critic_num_gpus
        + args.rollout_num_gpus
        + args.eval_num_gpus
        + _opd_teacher_extra_gpus(args),
        actor_num_gpus + critic_num_gpus,
        critic_offset,
    )


def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    num_gpus, rollout_offset, critic_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(
        num_gpus, is_rdt=args.update_weight_transfer_mode == "rdt"
    )

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]
    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }
    # ORBIT-SEAM: orbit's full-mode PPO critic owns its own bundle slice (upstream dbbab156 hands the
    # critic the actor's own slice); adapter/head critic modes never reach here.
    if uses_separate_critic(args):
        result["critic"] = (
            pg,
            actor_pg_reordered_bundle_indices[critic_offset:],
            actor_pg_reordered_gpu_ids[critic_offset:],
        )
    else:
        result["critic"] = None
    return result


# ORBIT-SEAM: PEFT runs derive the reference policy by disabling adapters (no separate "ref" checkpoint
# tag), so the with_ref decision below routes through this helper instead of the raw KL-coef check
def _actor_needs_reference_weights(args) -> bool:
    """Whether the actor should load a separate reference checkpoint.

    True only for full-FT KL runs. PEFT runs derive the reference policy by
    disabling adapters on the actor model and never load a "ref" tag.
    """
    return (args.kl_coef != 0 or args.use_kl_loss) and getattr(args, "peft_method", "none") == "none"


def allocate_train_group(
    args, num_nodes, num_gpus_per_node, pg, role: str, with_ref: bool, rollout_manager, with_opd_teacher: bool = False
):
    # RDT pins one NIXL/NCCL rank per physical GPU, so it cannot time-share a device
    # with a colocated rollout the way the fractional reservation allows.
    num_gpus_per_actor = 1 if args.update_weight_transfer_mode == "rdt" else 0.4
    train_group_cls = _select_train_group_class()
    return train_group_cls(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
        role=role,
        with_ref=with_ref,
        rollout_manager=rollout_manager,
        with_opd_teacher=with_opd_teacher,
    )


# ORBIT-SEAM: shared start_rollout_id-consistency helper used below for both the actor and (if
# present) the critic group, replacing the inline assert
def _single_start_rollout_id(role: str, start_rollout_ids: list[int]) -> int:
    if not start_rollout_ids:
        raise RuntimeError(f"{role} initialization returned no rollout ids")
    unique_ids = set(start_rollout_ids)
    if len(unique_ids) != 1:
        raise RuntimeError(f"{role} ranks resumed at different rollout ids: {sorted(unique_ids)}")
    return start_rollout_ids[0]


async def create_training_models(args, pgs, rollout_manager):
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        role="actor",
        # ORBIT-SEAM: with_ref derived from _actor_needs_reference_weights (PEFT-aware, upstream uses the
        # raw KL-coef check); needs_opd_teacher() covers both OPD objective forms where upstream reads
        # args.use_opd only
        with_ref=_actor_needs_reference_weights(args),
        rollout_manager=rollout_manager,
        with_opd_teacher=needs_opd_teacher(args) and args.opd_type == "megatron",
    )
    actor_start_rollout_ids = await actor_model.init()

    # ORBIT-SEAM: gate critic-group creation on uses_separate_critic() (adapter/head critic modes run
    # inside the actor workers), not raw args.use_critic
    if uses_separate_critic(args):
        critic_args = copy.deepcopy(args)
        critic_args.kl_coef = 0
        critic_args.use_opd = False
        critic_args.disable_param_buffers_cpu_backup = False
        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
            with_ref=False,
            rollout_manager=None,
        )
        critic_start_rollout_ids = await critic_model.init()
    else:
        critic_model = None

    # ORBIT-SEAM: consistency-checked start_rollout_id resolution via _single_start_rollout_id (replaces
    # the bare set-length assert), extended to also cross-check actor vs critic resume points
    actor_start_rollout_id = _single_start_rollout_id("actor", actor_start_rollout_ids)
    if args.start_rollout_id is None:
        args.start_rollout_id = actor_start_rollout_id

    if uses_separate_critic(args):
        critic_start_rollout_id = _single_start_rollout_id("critic", critic_start_rollout_ids)
        if actor_start_rollout_id != critic_start_rollout_id:
            raise RuntimeError(
                "actor and critic checkpoints must resume at the same rollout id; "
                f"actor={actor_start_rollout_id}, critic={critic_start_rollout_id}"
            )
        await actor_model.connect(critic_model)

    await actor_model.set_rollout_manager()
    if args.rollout_global_dataset:
        await rollout_manager.load.remote(args.start_rollout_id - 1)

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    rollout_manager = RolloutManager.options(
        num_cpus=1, num_gpus=0, **(compute_ray_pin_head_options() if args.pin_rollout_manager_to_head else {})
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    # ORBIT-SEAM: skip the weight-update-equal check when no rollout engines are in play
    # (uses_rollout_engines), not just on the raw flag
    if uses_rollout_engines(args) and args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(
            rollout_manager.check_weights.remote(action="reset_tensors", skip_list=args.check_weight_update_skip_list)
        )

    if args.offload_rollout:
        # ORBIT-SEAM: explains why the existing no-tags offload call below is a no-op in async/disjoint
        # topologies (async/disjoint rollout topology introduced by orbit)
        # No-tags call: RolloutManager.offload(tags=None) routes through
        # ServerGroup.needs_offload, which start_rollout_servers only sets for
        # groups whose GPUs overlap the megatron training GPUs (colocate). In
        # async/disjoint topologies every group has needs_offload=False, so this
        # releases nothing and the engines stay resident -- which is why
        # train_async.py has no onload_weights/onload_kv dance (train.py does,
        # for the colocated case). Pinned by tests/fast/test_async_offload_noop.py.
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
