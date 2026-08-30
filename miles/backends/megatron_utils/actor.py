import atexit
import logging
import os
import random
import shutil
import socket
from argparse import Namespace
from contextlib import ExitStack, nullcontext
from typing import TYPE_CHECKING

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig

from miles.backends.megatron_utils.rematerialize_utils import build_main_cast_context
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
from miles.utils.argparse_utils import inplace_modify_args

# ORBIT-SEAM: orbit-added argument predicates (one-trunk vs separate critic, rollout-engine
# presence, OPD top-k vocab validation). LOAD-BEARING BEYOND THIS MODULE: `uses_separate_critic`
# and `uses_rollout_engines` are now named only by the home mixin's train_actor/update_weights,
# which re-read them off THIS module at call time (see orbit/megatron/actor_ext.py).
from miles.utils.arguments import (
    uses_one_trunk_critic,
    uses_rollout_engines,
    uses_separate_critic,
    validate_opd_topk_vocab_size,
)
from miles.utils.audit_utils.event_logger.logger import event_logger_context
from miles.utils.audit_utils.witness.allocator import WitnessInfo

# ORBIT-SEAM: (vp_stage, name) adapter keys for the OPD teacher/self-teacher state built in init
from orbit.utils.adapter_tensors import AdapterTensorKey
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.hf_config import load_hf_config
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers, routing_replay_manager
from miles.utils.test_utils.ft_test_actions import FTTestActionActorExecutor

# ORBIT-SEAM: OPD self-distillation support for train_actor/update_weights (the dump is env-gated
# and no-ops unless ORBIT_OPD_TEACHER_LOGPROB_DUMP is set). LOAD-BEARING BEYOND THIS MODULE: both
# methods now live on the home mixin, which re-reads these names off THIS module at call time.
from orbit.opd.opd_dump import maybe_dump_teacher_logprobs
from orbit.opd.opd_teacher_spec import should_promote_teacher

# ORBIT-SEAM: the EMA/lag self-teacher buffer init seeds below
from orbit.opd.self_teacher import SelfTeacherBuffer
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.structured_log import with_logs
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
# ORBIT-SEAM: orbit's model-state manager generalises base's TensorBackuper (adapter-only state,
# multiple tags), so base's `from ...utils.tensor_backper import TensorBackuper` is dropped here
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import (
    DataIterator,
    get_data_iterator,
    get_num_rollouts,
    get_rollout_data,
    sync_actor_critic_data,
)
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import (
    compute_advantages_and_returns,
    get_log_probs_and_entropy,
    get_values,
    log_train_advantage_computation_event,
)
from ..training_utils.parallel import get_parallel_state
from ..training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from .checkpoint import load_checkpoint
from .ft.checkpoint_transfer import recv_ckpt
from .ft.checkpoint_transfer import send_ckpt as _send_ckpt
from .ft.in_memory_checkpoint import InMemoryCheckpointManager
from .ft.indep_dp import reconfigure_indep_dp_group
from .initialize import init, is_first_replica_megatron_main_rank
from .lora_utils import is_lora_enabled, lora_rollout_enabled
from .model import TrainStepOutcome, forward_only, initialize_model_and_optimizer, save, train
from .parallel import verify_megatron_parallel_state

# ORBIT-SEAM: upstream retired get_register_replay_list_func in favour of assigning
# register_replay_list_moe onto routing_replay_manager.register_replay_list_func directly
from .replay_utils import register_replay_list_moe

# ORBIT-SEAM: named_adapter_params backs the adapter-only weight snapshots below
from .update_weight.common import named_adapter_params, named_params_and_buffers

# ORBIT-SEAM: the one-trunk (adapter) critic - build, save and the value-loss phase context - is an
# orbit addition to this actor; the critic itself lives in orbit.critic. LOAD-BEARING BEYOND THIS
# MODULE: `value_loss_phase` is named only by the home mixin's train_actor, which re-reads it off
# THIS module at call time.
from orbit.critic.critic_adapter import (
    _expected_critic_resume_iteration,
    build_critic_instance,
    save_critic_checkpoint,
    value_loss_phase,
)

# ORBIT-SEAM: orbit's model-state manager generalises base's TensorBackuper (adapter-only state,
# multiple tags); every base `self.weights_backuper.<op>(...)` call is the same op on
# self.model_state_manager. Base's remaining weights_backuper call sites lived in
# _switch_model/train_actor/update_weights, which orbit now owns outright in the home mixin.
from orbit.megatron.model_state_manager import create_model_state_manager

# ORBIT-SEAM: orbit's explicit PEFT offload replaces base's torch_memory_saver pause/resume (frozen
# base, adapter, grad buffers and optimizer move separately). LOAD-BEARING BEYOND THIS MODULE: the
# sleep/wake_up that use these now live on the home mixin, which re-reads every name here off THIS
# module at call time (see orbit/megatron/actor_ext.py) - none of them is dead.
from orbit.megatron.peft_offload import (
    _should_offload_frozen_base,
    load_megatron_adapter_to_gpu,
    load_megatron_frozen_base_to_gpu,
    load_megatron_grad_buffers,
    load_megatron_optimizer,
    offload_megatron_adapter_to_cpu,
    offload_megatron_frozen_base_to_cpu,
    offload_megatron_grad_buffers,
    offload_megatron_optimizer,
)

# ORBIT-SEAM: PEFT (LoRA + OFT) predicates for init. LOAD-BEARING BEYOND THIS MODULE: the home
# mixin re-reads `create_peft_instance` and `is_adapter_param_name` off THIS module at call time
# (see orbit/megatron/actor_ext.py), so neither is dead even where nothing below names it.
from orbit.megatron.peft_utils import (
    create_peft_instance,
    is_adapter_param_name,
    is_peft_enabled,
    load_adapter_tensors_for_teacher,
)

# ORBIT-SEAM: adapter-state mode decides whether the CPU snapshot holds adapters or full weights.
# Also re-read off THIS module at call time by the home mixin's train_actor/update_weights.
from orbit.megatron.state_mode import should_backup_actor_after_train, uses_adapter_state

# ORBIT-SEAM: bridge-export weight updater for models without a megatron_to_hf name map; selected
# by orbit.megatron.actor_helpers._select_update_weight_cls, which reads it back off this module
from orbit.megatron.update_weight_bridge import UpdateWeightFromDistributedBridge

# ORBIT-SEAM: the OPD teacher LM head is resident state sleep/wake_up must move (opd_jsd_loss).
# LOAD-BEARING BEYOND THIS MODULE: sleep/wake_up live on the home mixin now and re-read
# offload_teacher_lm_head/onload_teacher_lm_head off THIS module at call time.
from orbit.opd.teacher_lm_head import load_teacher_lm_head, offload_teacher_lm_head, onload_teacher_lm_head
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed

# ORBIT-SEAM: base imports p2p unconditionally; guard it so importing this module still works on
# sglang builds without ParallelismContext (the p2p transfer mode then fails at use, not at import)
try:
    from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
except ImportError:
    # Older sglang (e.g. 0.5.9) lacks ParallelismContext/RankParallelismConfig.
    # UpdateWeightP2P is only used in non-colocate + p2p transfer mode, so
    # we leave it undefined and fail only if that code path is actually taken.
    UpdateWeightP2P = None
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor
# ORBIT-SEAM: orbit's added AND overriding actor methods, plus the module helpers, live in the home
# layer (P2 mixin + P1 lift-out, Phase 3 slice 3g; mixin-override slice added sleep, wake_up,
# _switch_model, train_actor and update_weights); re-exported here because callers import them off
# this module. Those five are DELETED from the class body below, not left as overridden copies: a
# class's own __dict__ precedes every base in the MRO, so a retained body would shadow the mixin.
from orbit.megatron.actor_ext import OrbitTrainActorExtensions
from orbit.megatron.actor_helpers import (
    _get_weight_updater_kwargs,
    _select_update_weight_cls,
    _start_rollout_id_from_checkpoint,
    _validate_train_offload_role,
)

if TYPE_CHECKING:
    from miles.ray.rollout.rollout_manager import EnginesAndLock

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _setup_disk_offload_reclaim(disk_dir: str) -> None:
    """Wipe this rank's train disk-offload dir on startup and re-arm the atexit wipe.

    torch_memory_saver unlinks each backup file as its allocation is freed on a
    graceful teardown, but a SIGKILL'd run leaves stale files behind. The dir is
    per-rank (see actor_factory), so clearing it wholesale touches nobody else.
    """
    if not disk_dir:
        return
    shutil.rmtree(disk_dir, ignore_errors=True)
    os.makedirs(disk_dir, exist_ok=True)
    atexit.register(shutil.rmtree, disk_dir, ignore_errors=True)
    logger.info(f"Train disk-offload reclaim armed for {disk_dir} (startup wipe + atexit)")


# ORBIT-SEAM: the mixin carries orbit's added methods (ref/teacher log-probs, held-out eval NLL,
# self-teacher restore/promotion, adapter-param view, train-state prefetch) AND orbit's replacements
# for sleep, wake_up, _switch_model, train_actor and update_weights - orbit owned >= 50% of each, so
# it carries the whole body. Those five are ABSENT from this class body on purpose: Python checks a
# class's own __dict__ before any base, so keeping upstream's copy here (even "for reference") would
# shadow the mixin and silently revert orbit's behaviour on the GPU training path. Deleting them
# also makes an upstream rewrite conflict loudly at merge time instead of landing unnoticed.
class MegatronTrainRayActor(OrbitTrainActorExtensions, TrainRayActor):
    @with_logs
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        *,
        with_ref: bool = False,
        # ORBIT-SEAM: OPD adds a second optional companion model alongside base's `with_ref`;
        # threaded to TrainRayActor.init and consumed by the teacher block below
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
    ) -> int | None:
        # ORBIT-SEAM: refuse the unsupported critic + --offload-train combination up front
        _validate_train_offload_role(args, role)

        monkey_patch_torch_dist()

        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        for m in all_replay_managers:
            m.register_replay_list_func = register_replay_list_sequential
        routing_replay_manager.register_replay_list_func = register_replay_list_moe

        init(
            args,
            indep_dp_store_addr=self._indep_dp_store_addr,
            indep_dp_info=indep_dp_info,
        )

        self._ft_test_action_executor = FTTestActionActorExecutor.from_args(
            args,
            cell_index=indep_dp_info.cell_index,
            num_cells=indep_dp_info.num_cells,
            rank=self._rank,
        )

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_first_replica_megatron_main_rank = is_first_replica_megatron_main_rank()

        if self._is_first_replica_megatron_main_rank:
            init_tracking(args, primary=False)

        dashboard_hooks.register_train_actor(args)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = load_hf_config(args.hf_checkpoint)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = (
            {}
            if args.indep_dp
            else {
                "dp_size": get_parallel_state().intra_dp.size,
                "cp_size": get_parallel_state().cp.size,
                "vpp_size": get_parallel_state().vpp_size,
                "microbatch_group_size_per_vp_stage": get_parallel_state().microbatch_group_size_per_vp_stage,
            }
        )
        dist.barrier(group=get_gloo_group())

        # ORBIT-SEAM: orbit offloads the train state explicitly (orbit.megatron.peft_offload)
        # rather than through torch_memory_saver, so --train-memory-margin-bytes has nothing to
        # tune; upstream's disk-offload reclaim is kept.
        if args.offload_train:
            if args.offload_train_target == "disk":
                _setup_disk_offload_reclaim(os.environ.get("TMS_DISK_BACKUP_DIR"))

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay", False)
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        # ORBIT-SEAM: MTP-in-RL patches must be applied before the model is built (home layer)
        from orbit.megatron.mtp_rl_patches import apply_mtp_in_rl_patches

        apply_mtp_in_rl_patches(self.args)

        checkpointing_context = None
        if recv_ckpt_src_rank is not None:
            ckpt_manager = recv_ckpt(
                indep_dp=get_parallel_state().indep_dp,
                src_rank=recv_ckpt_src_rank,
            )
            checkpointing_context = {"local_checkpoint_manager": ckpt_manager}
        elif args.non_persistent_ckpt_type == "local":
            checkpointing_context = {"local_checkpoint_manager": InMemoryCheckpointManager()}

        heal_load_overrides: dict[str, object] = (
            dict(no_load_optim=False, no_load_rng=False, finetune=False) if recv_ckpt_src_rank is not None else {}
        )
        with inplace_modify_args(args, heal_load_overrides):
            self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
                args, role, checkpointing_context=checkpointing_context
            )

        # ORBIT-SEAM: wire rollout routing replay into the freshly built model chunks
        if role != "critic" and getattr(self.args, "use_rollout_routing_replay", False):
            from orbit.megatron.routing_replay import wire_routing_replay_to_models

            wire_routing_replay_to_models(self.model)

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        # ORBIT-SEAM: side stream + event for the async train-state prefetch (prefetch_train_state
        # in the home mixin records on it, wake_up waits on it)
        if self.args.offload_train and getattr(self.args, "offload_train_async", False):
            self._wake_up_stream = torch.cuda.Stream()
        else:
            self._wake_up_stream = None
        self._wake_up_event = None

        # ORBIT-SEAM: base resumes at loaded_rollout_id + 1 unconditionally; a model-only bridge/HF
        # load is initialization, not a resume, and must start at rollout 0 (helper in the home)
        start_rollout_id = _start_rollout_id_from_checkpoint(self.args, loaded_rollout_id)
        self._asleep = False

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            # ORBIT-SEAM: base returns None for the critic; a resumed critic needs its rollout id
            return start_rollout_id

        # ORBIT-SEAM: one-trunk (adapter) critic shares this actor's trunk, so it is built here
        # rather than in a separate critic actor
        self.critic_model = None
        self.critic_optimizer = None
        self.critic_opt_param_scheduler = None
        if uses_one_trunk_critic(self.args):
            (
                self.critic_model,
                self.critic_optimizer,
                self.critic_opt_param_scheduler,
            ) = build_critic_instance(
                self.args,
                self.model,
                expected_iteration=_expected_critic_resume_iteration(self.args, loaded_rollout_id),
            )

        # ORBIT-SEAM: base always snapshots the full parameter set; under PEFT only the adapter is
        # trainable, so the snapshot source becomes mode-dependent and moves behind
        # create_model_state_manager (base built a TensorBackuper directly)
        if uses_adapter_state(self.args):

            def state_source_getter():
                return named_adapter_params(self.model)

        else:

            def state_source_getter():
                # ORBIT-SEAM: upstream dropped named_params_and_buffers' translate_gpu_to_cpu
                # parameter (and _maybe_get_cpu_backup with it)
                return named_params_and_buffers(
                    self.args,
                    self.model,
                    convert_to_global_name=args.megatron_to_hf_mode == "raw",
                )

        # PHASE-4 NOTE: upstream threads a main_cast_ctx (build_main_cast_context, used when
        # --rematerialize-param-from-master-weight is set) into TensorBackuper.create here. Orbit's
        # create_model_state_manager has no such parameter yet, so the option is not honoured on
        # this path - see the phase-4 merge report flag.
        self.model_state_manager = create_model_state_manager(
            self.args,
            source_getter=state_source_getter,
            single_tag=None if getattr(args, "enable_weights_backuper", True) else "actor",  # ORBIT-SEAM: flag retired upstream; default True preserved
        )
        self._active_model_tag: str | None = "actor"
        # ORBIT-SEAM: base always takes the startup snapshot; skip it in the modes that never
        # restore or copy from it (nothing reads the tag, and the copy is not free)
        if should_backup_actor_after_train(self.args):
            self.model_state_manager.backup("actor")

        # ORBIT-SEAM: under PEFT the reference policy IS the base with adapters disabled, so no
        # second checkpoint is loaded (compute_ref_log_probs takes the disable_adapter path)
        if with_ref and not is_peft_enabled(self.args):
            self.load_other_checkpoint("ref", args.ref_load)

        # ORBIT-SEAM: OPD teacher setup - the teacher LM head for opd_jsd_loss, then the in-process
        # teacher (load:/adapter:/self:*) and the sglang-side self-teacher shadow, then the
        # checkpoint sidecar restore. All orbit; base has no distillation teacher.
        if self.args.loss_type == "opd_jsd_loss":
            # Eagerly load now (onto CPU) so the first train step doesn't stall on a
            # safetensors read; wake_up() moves it to GPU before use.
            load_teacher_lm_head(self.args)

        # In-process OPD teacher. Same-base specs (base/adapter:/self:*) need no
        # second model: the teacher is the resident base with adapters toggled.
        # Only the legacy load:<ckpt> spec loads a full second model like "ref".
        self._opd_teacher_spec = getattr(self.args, "opd_teacher_spec", None)
        self._opd_teacher_tensors: dict[AdapterTensorKey, torch.Tensor] | None = None
        self._self_teacher = None
        if with_opd_teacher:
            if self._opd_teacher_spec is None:
                from orbit.opd.opd_teacher_spec import parse_teacher_spec

                self._opd_teacher_spec = parse_teacher_spec(
                    getattr(self.args, "opd_teacher", None), self.args.opd_teacher_load
                )
            spec = self._opd_teacher_spec
            if spec.source == "load":
                if self.args.opd_teacher_ckpt_step is not None:
                    _saved_ckpt_step = self.args.ckpt_step
                    self.args.ckpt_step = self.args.opd_teacher_ckpt_step
                self.load_other_checkpoint("teacher", spec.path)
                if self.args.opd_teacher_ckpt_step is not None:
                    self.args.ckpt_step = _saved_ckpt_step
            elif spec.source == "adapter":
                self._opd_teacher_tensors = load_adapter_tensors_for_teacher(self.model, spec.path)
            elif spec.source in ("self_ema", "self_lag"):
                self._self_teacher = SelfTeacherBuffer(
                    self._adapter_named_params(),
                    mode="ema" if spec.source == "self_ema" else "lag",
                    decay=self.args.opd_ema_decay,
                    interval=self.args.opd_self_teacher_interval,
                )

        # sglang self:* teachers score on the rollout engine, so with_opd_teacher
        # is False (no in-process teacher model/tensors are allocated). The actor
        # still shadows the student adapter here so _promote_self_teacher can push
        # the EMA/lag buffer into the engine's orbit_teacher slot. Kept separate
        # from the with_opd_teacher block above to leave the megatron path
        # byte-identical. Actor-only: the critic shares args.opd_teacher_spec but
        # never produces or promotes teachers.
        if (
            role == "actor"
            and self._self_teacher is None
            and self._opd_teacher_spec is not None
            and self._opd_teacher_spec.source in ("self_ema", "self_lag")
            and getattr(self.args, "opd_type", None) == "sglang"
        ):
            self._self_teacher = SelfTeacherBuffer(
                self._adapter_named_params(),
                mode="ema" if self._opd_teacher_spec.source == "self_ema" else "lag",
                decay=self.args.opd_ema_decay,
                interval=self.args.opd_self_teacher_interval,
            )
        # Set True after the engine's orbit_teacher slot has been filled once
        # (startup promotion in update_weights); scoring an empty slot 404s.
        self._teacher_slot_startup_promoted = False
        self._restore_checkpoint_teacher_state()

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.model_state_manager.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size
        # ORBIT-SEAM: OPD top-k K depends on the vocab size the tokenizer just filled in
        # Argument validation commonly runs before the tokenizer has filled the
        # real vocab size. Recheck K here, before the first rollout is launched.
        validate_opd_topk_vocab_size(self.args)

        # ORBIT-SEAM: base picks the updater inline from (colocate, transfer mode); orbit adds the
        # PEFT and bridge-export cases, so the choice and its kwargs live in the home helper.
        # Upstream's newer rdt / disk-delta transfer modes are handled here in front of it, because
        # the home helper does not know them yet (see the phase-4 merge report flag).
        if self.args.update_weight_transfer_mode == "rdt":
            from .update_weight.update_weight_from_rdt import UpdateWeightFromRDT

            update_weight_cls = UpdateWeightFromRDT
        elif self.args.update_weight_transfer_mode == "disk-delta" and not self.args.colocate:
            # Lazy import: keeps the delta deps (numpy/zstandard/xxhash) off the other paths.
            from .update_weight.update_weight_from_distributed.delta import UpdateWeightFromDiskDelta

            update_weight_cls = UpdateWeightFromDiskDelta
        else:
            update_weight_cls = _select_update_weight_cls(self.args)
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.model_state_manager.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            # ORBIT-SEAM: base passes is_lora=; orbit's updaters take the resolved PEFT method
            # (and per-class extras) from the home helper
            **_get_weight_updater_kwargs(self.args, update_weight_cls),
        )

        # Adapters currently loaded into Megatron slots on this rank.
        self.loaded_adapters: dict[str, object] = {}
        # Adapters with stale engine-side weights (newly loaded or just trained);
        # consumed by the next update_weights. Identical on every rank.
        self._multi_lora_pending_push: set[str] = set()

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_data_postprocess = None
        if (x := self.args.rollout_data_postprocess_path) is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(x)

        self.prof.on_init_end()

        return start_rollout_id

    @property
    def _enable_weight_backup(self) -> bool:
        """Weight backup is only needed for CPU-side model switching or colocated tensor weight sync."""
        return self.with_ref or self.with_opd_teacher or self.args.keep_old_actor or self.args.colocate

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    # ORBIT-SEAM: upstream moved this method to miles/backends/training_utils/replay_data.py
    # (fill_replay_data, imported above and called below). Orbit's per-sample max_seq_len argument
    # to slice_with_cp needs re-applying there - see the phase-4 merge report flag.
    @with_logs
    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        rollout_id: int,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
                store_prefix=store_prefix,
            )

    @with_logs
    @event_logger_context(
        lambda _self, rollout_id, rollout_data_ref, witness_info=None, attempt=0, external_data=None: dict(
            rollout_id=rollout_id, attempt=attempt
        )
    )
    def train(
        self,
        rollout_id: int,
        rollout_data_ref: Box,
        witness_info: WitnessInfo | None = None,
        attempt: int = 0,
        external_data=None,
    ):
        self._heartbeat.bump()
        self._last_rollout_id = rollout_id
        if self.args.offload_train and self._asleep:
            self.wake_up()

        with ExitStack() as stack:
            with timer("data_preprocess"):
                rollout_data, store_get_result = get_rollout_data(
                    self.args, rollout_data_ref, witness_info=witness_info
                )
                stack.enter_context(store_get_result)
                if self.args.debug_rollout_only:
                    log_rollout_data(rollout_id, self.args, rollout_data)
                    return TrainStepOutcome.NORMAL

            if self.role == "critic":
                with timer("critic_train"):
                    result = self.train_critic(rollout_id, rollout_data)
            else:
                result = self.train_actor(
                    rollout_id,
                    rollout_data,
                    external_data=external_data,
                    witness_info=witness_info,
                    attempt=attempt,
                )

            return result

    @with_logs
    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> dict:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        # ORBIT-SEAM: role tells the advantage/return pipeline it is running for the critic
        compute_advantages_and_returns(self.args, rollout_data, role="critic")

        self.args.loss_type = "value_loss"
        train_step_outcome: TrainStepOutcome = train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            get_num_rollouts(self.args, rollout_data, len(num_microbatches)),
            witness_info=None,
            attempt=0,
        )

        self._heartbeat.bump()
        result = {"train_step_outcome": train_step_outcome}
        if get_parallel_state().is_pp_last_stage and "values" in rollout_data:
            # Ship by object reference
            result["values"] = Box(ray.put([value.detach().cpu() for value in rollout_data["values"]]))
        return result

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay", False)

    @with_logs
    @timer
    def reconcile_adapters(self) -> None:
        """Load adapters the controller wants served; retire deregistered ones, dropping their untrained tail."""
        if not is_multi_lora_enabled(self.args):
            return
        from miles.backends.megatron_utils.multi_lora_utils import cleanup_adapters as _cleanup_adapters
        from miles.backends.megatron_utils.multi_lora_utils import load_adapters as _load_adapters
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        broadcast_buffer = [None]
        if is_first_replica_megatron_main_rank():
            controller = get_multi_lora_controller()
            ray.get(controller.retire_adapters.remote())
            broadcast_buffer[0] = ray.get(controller.snapshot.remote())
        if dist.is_initialized():
            dist.broadcast_object_list(broadcast_buffer, src=0, group=get_gloo_group())
        snapshot = broadcast_buffer[0]
        should_be_loaded = {**snapshot["active"], **snapshot["pending"], **snapshot["retiring"]}
        cleanup_names = set(snapshot["cleanup"])

        loaded_names = set(self.loaded_adapters)
        # Sorted so per-adapter collectives (checkpoint export) run in the same
        # order on every rank; set iteration order is process-specific.
        adapters_to_load = sorted(
            (adapter for name, adapter in should_be_loaded.items() if name not in loaded_names),
            key=lambda adapter: adapter.name,
        )
        adapters_to_clean_up = sorted(
            (self.loaded_adapters[n] for n in loaded_names if n in cleanup_names or n not in should_be_loaded),
            key=lambda adapter: adapter.name,
        )
        if adapters_to_load:
            _load_adapters(self.args, self.model, self.optimizer, adapters_to_load)
            for adapter in adapters_to_load:
                self.loaded_adapters[adapter.name] = adapter
                self._multi_lora_pending_push.add(adapter.name)
            self.model_state_manager.backup("actor")
        if adapters_to_clean_up:
            _cleanup_adapters(self.args, self.model, self.optimizer, adapters_to_clean_up)
            for adapter in adapters_to_clean_up:
                self.loaded_adapters.pop(adapter.name, None)
                self._multi_lora_pending_push.discard(adapter.name)
            self.model_state_manager.backup("actor")

        # Deregistered before ever being loaded: nothing to save or clear.
        if is_first_replica_megatron_main_rank():
            for name in cleanup_names - loaded_names:
                ray.get(get_multi_lora_controller().free_slot.remote(name))

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        self._heartbeat.bump()
        if self.args.debug_rollout_only:
            return

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        # ORBIT-SEAM: the save chain carries the self-teacher so its checkpoint sidecar is written
        # beside the adapter (here and in the save_hf_model call below)
        # getattr: the separate-critic actor shares this method but never runs the
        # OPD teacher init that creates _self_teacher.
        if is_multi_lora_enabled(self.args):
            from miles.backends.megatron_utils.multi_lora_utils import save_due_adapter_checkpoints

            if not save_due_adapter_checkpoints(self.args, self.model):
                return
        else:
            save(
                rollout_id,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                self_teacher=getattr(self, "_self_teacher", None),
            )

        # ORBIT-SEAM: the one-trunk critic lives on this actor, so its checkpoint is written here
        if uses_one_trunk_critic(self.args) and self.args.critic_save:
            save_critic_checkpoint(
                self.args,
                rollout_id,
                self.critic_model,
                optimizer=self.critic_optimizer,
                opt_param_scheduler=self.critic_opt_param_scheduler,
            )

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.hf_export import save_hf_model

            save_hf_model(
                self.args,
                rollout_id,
                self.model,
                self_teacher=getattr(self, "_self_teacher", None),
            )

        if self.args.custom_megatron_post_save_hook_path is not None and dist.get_rank() == 0:
            if self.args.async_save:
                maybe_finalize_async_save(blocking=True)

            from megatron.training.checkpointing import get_checkpoint_name

            from miles.utils.misc import load_function

            checkpoint_dir = get_checkpoint_name(self.args.save, rollout_id, return_base_dir=True)
            hf_checkpoint_dir = (
                self.args.save_hf.format(rollout_id=rollout_id)
                if self.args.save_hf is not None and self.role == "actor"
                else None
            )
            post_save_hook = load_function(self.args.custom_megatron_post_save_hook_path)
            post_save_hook(self.args, rollout_id, checkpoint_dir, hf_checkpoint_dir)

    @with_logs
    @timer
    def export_hf(self, rollout_id: int, path: str) -> None:
        """Export current weights as an HF checkpoint to ``path`` (collective).

        Uses the direct megatron->HF converters (the weight updater's machinery), so
        export coverage matches weight-sync coverage. Unlike the periodic --save-hf
        path inside save_model, failures propagate to the caller so an eval snapshot
        that failed to export can be skipped loudly.
        """
        self._heartbeat.bump()
        from miles.backends.megatron_utils.hf_export import save_hf_model

        save_hf_model(self.args, rollout_id, self.model, path=path, raise_on_error=True)

    @with_logs
    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        # load_checkpoint reads self.args.ckpt_step to pick which iteration to load.
        # Temporarily override it for ref/teacher loads, then restore after the load below.
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.model_state_manager.backup(model_tag)
        self._active_model_tag = model_tag

    @with_logs
    def send_ckpt(self, dst_rank: int) -> None:
        # These states are not handled
        assert not self.args.keep_old_actor

        _send_ckpt(
            indep_dp=get_parallel_state().indep_dp,
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            iteration=self._last_rollout_id,
            dst_rank=dst_rank,
        )

    @with_logs
    def reconfigure_indep_dp(self, indep_dp_info: IndepDPInfo) -> None:
        reconfigure_indep_dp_group(
            parallel_state=get_parallel_state(),
            store_addr=self._indep_dp_store_addr,
            indep_dp_info=indep_dp_info,
            megatron_rank=dist.get_rank(),
            megatron_world_size=dist.get_world_size(),
        )
        self.weight_updater.mark_engine_connection_stale()
