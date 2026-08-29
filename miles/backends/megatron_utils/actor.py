import logging
import random
import socket
# ORBIT-SEAM: SimpleNamespace stand-ins for the OPD teacher-logprob dump in train_actor below
import types
from argparse import Namespace

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from transformers import AutoConfig

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
# ORBIT-SEAM: (vp_stage, name) adapter keys for the OPD teacher/self-teacher state built in init
from orbit.utils.adapter_tensors import AdapterTensorKey
# ORBIT-SEAM: orbit-added argument predicates (one-trunk vs separate critic, rollout-engine
# presence, OPD top-k vocab validation) consumed by init/train_actor/update_weights below
from miles.utils.arguments import (
    uses_one_trunk_critic,
    uses_rollout_engines,
    uses_separate_critic,
    validate_opd_topk_vocab_size,
)
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.memory_utils import clear_memory, print_memory
# ORBIT-SEAM: OPD self-distillation support used by train_actor/update_weights (the dump is
# env-gated and no-ops unless ORBIT_OPD_TEACHER_LOGPROB_DUMP is set)
from orbit.opd.opd_dump import maybe_dump_teacher_logprobs
from orbit.opd.opd_teacher_spec import should_promote_teacher
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers
# ORBIT-SEAM: the EMA/lag self-teacher buffer init seeds below
from orbit.opd.self_teacher import SelfTeacherBuffer
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from ..training_utils.parallel import get_parallel_state
# ORBIT-SEAM: the OPD teacher LM head is resident state sleep/wake_up must move (opd_jsd_loss)
from orbit.opd.teacher_lm_head import load_teacher_lm_head, offload_teacher_lm_head, onload_teacher_lm_head
from .checkpoint import load_checkpoint
# ORBIT-SEAM: the one-trunk (adapter) critic - build, save and the value-loss phase context - is an
# orbit addition to this actor; the critic itself lives in orbit.critic
from orbit.critic.critic_adapter import (
    _expected_critic_resume_iteration,
    build_critic_instance,
    save_critic_checkpoint,
    value_loss_phase,
)
from .initialize import init, is_megatron_main_rank
from .lora_utils import is_lora_enabled
from .model import forward_only, initialize_model_and_optimizer, save, train
# ORBIT-SEAM: orbit's model-state manager generalises base's TensorBackuper (adapter-only state,
# multiple tags); removed base's `from ...utils.tensor_backper import TensorBackuper` with it, and
# every `self.weights_backuper.<op>(...)` call below is the same op on self.model_state_manager
from orbit.megatron.model_state_manager import create_model_state_manager
from .parallel import verify_megatron_parallel_state
# ORBIT-SEAM: orbit's explicit PEFT offload replaces base's torch_memory_saver pause/resume in
# sleep/wake_up below (frozen base, adapter, grad buffers and optimizer move separately); removed
# base's `torch_memory_saver` and `contextlib.nullcontext` imports, which only served that path
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
from .replay_utils import get_register_replay_list_func
# ORBIT-SEAM: adapter-state mode decides whether the CPU snapshot holds adapters or full weights
from orbit.megatron.state_mode import should_backup_actor_after_train, uses_adapter_state
from .update_weight.common import named_adapter_params, named_params_and_buffers
# ORBIT-SEAM: bridge-export weight updater for models without a megatron_to_hf name map; selected
# by orbit.megatron.actor_helpers._select_update_weight_cls, which reads it back off this module
from orbit.megatron.update_weight_bridge import UpdateWeightFromDistributedBridge
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
# ORBIT-SEAM: orbit's added actor methods and module helpers live in the home layer (P2 mixin + P1
# lift-out, Phase 3 slice 3g); re-exported here because callers import them off this module
from orbit.megatron.actor_ext import OrbitTrainActorExtensions
from orbit.megatron.actor_helpers import (
    _get_weight_updater_kwargs,
    _select_update_weight_cls,
    _start_rollout_id_from_checkpoint,
    _validate_train_offload_role,
)

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ORBIT-SEAM: the mixin carries orbit's added methods (ref/teacher log-probs, held-out eval NLL,
# self-teacher restore/promotion, adapter-param view, train-state prefetch)
class MegatronTrainRayActor(OrbitTrainActorExtensions, TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        # ORBIT-SEAM: OPD adds a second optional companion model alongside base's `with_ref`;
        # threaded to TrainRayActor.init and consumed by the teacher block below
        with_opd_teacher: bool = False,
    ) -> int | None:
        # ORBIT-SEAM: refuse the unsupported critic + --offload-train combination up front
        _validate_train_offload_role(args, role)

        monkey_patch_torch_dist()

        super().init(args, role, with_ref, with_opd_teacher)

        init(args)

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_main_rank = is_megatron_main_rank()

        if self._is_main_rank:
            init_tracking(args, primary=False)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }
        dist.barrier(group=get_gloo_group())

        # ORBIT-SEAM: removed base's torch_memory_saver.memory_margin_bytes tuning here: orbit
        # offloads the train state explicitly (orbit.megatron.peft_offload) rather than through
        # torch_memory_saver, so --train-memory-margin-bytes has nothing to tune
        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay")
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        # ORBIT-SEAM: MTP-in-RL patches must be applied before the model is built (home layer)
        from orbit.megatron.mtp_rl_patches import apply_mtp_in_rl_patches

        apply_mtp_in_rl_patches(self.args)

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )
        # ORBIT-SEAM: base resumes at loaded_rollout_id + 1 unconditionally; a model-only bridge/HF
        # load is initialization, not a resume, and must start at rollout 0 (helper in the home)
        start_rollout_id = _start_rollout_id_from_checkpoint(self.args, loaded_rollout_id)

        # ORBIT-SEAM: wire rollout routing replay into the freshly built model chunks
        if role != "critic" and getattr(self.args, "use_rollout_routing_replay", False):
            from miles.backends.megatron_utils.replay_utils import wire_routing_replay_to_models

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
                return named_params_and_buffers(
                    self.args,
                    self.model,
                    convert_to_global_name=args.megatron_to_hf_mode == "raw",
                    translate_gpu_to_cpu=not self.args.enable_weights_backuper,
                )

        self.model_state_manager = create_model_state_manager(
            self.args,
            source_getter=state_source_getter,
            single_tag=None if args.enable_weights_backuper else "actor",
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
        # PEFT and bridge-export cases, so the choice and its kwargs live in the home helper
        update_weight_cls = _select_update_weight_cls(self.args)
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.model_state_manager.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            **_get_weight_updater_kwargs(self.args, update_weight_cls),
        )

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        # ORBIT-SEAM: base pauses the whole allocator with torch_memory_saver.pause(); orbit
        # offloads the train state piece by piece (grad buffers, optimizer, frozen base, and the
        # OPD teacher LM head) so PEFT runs can keep the adapter resident on GPU
        if self.args.offload_train_grad_buffers:
            offload_megatron_grad_buffers(self.model)
            print_memory("after offload grad_buffers")
        if self.args.offload_train_optimizer:
            offload_megatron_optimizer(self.optimizer)
            print_memory("after offload optimizer")
        if _should_offload_frozen_base(self.args):
            offload_megatron_frozen_base_to_cpu(self.model)
            print_memory("after offload frozen_base")

        if self.args.loss_type == "opd_jsd_loss":
            offload_teacher_lm_head(self.args.teacher_hf_checkpoint)
            print_memory("after offload teacher_lm_head")

        print_memory("after offload model")
        # Read by compute_eval_nll: it must know whether the training state is
        # resident before deciding to wake it, and must not leave it awake.
        self._train_state_awake = False

        if self._is_main_rank and hasattr(self, "_last_rollout_id"):
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        # ORBIT-SEAM: mirror image of sleep() above - base resumes the allocator with
        # torch_memory_saver.resume(); orbit either waits on the prefetch event
        # prefetch_train_state recorded (home mixin) or reloads each piece synchronously
        wake_up_event = getattr(self, "_wake_up_event", None)
        if wake_up_event is not None:
            torch.cuda.current_stream().wait_event(wake_up_event)
            self._wake_up_event = None
            print_memory("after wake_up train_state_prefetch")
        elif _should_offload_frozen_base(self.args):
            load_megatron_frozen_base_to_gpu(self.model)
            print_memory("after wake_up frozen_base")
            if self.args.offload_train_adapter:
                load_megatron_adapter_to_gpu(self.model)
                print_memory("after wake_up adapter")

        if self.args.offload_train_optimizer:
            load_megatron_optimizer(self.optimizer)
            print_memory("after wake_up optimizer")
        if self.args.offload_train_grad_buffers:
            load_megatron_grad_buffers(self.model)
            print_memory("after wake_up grad_buffers")
        if self.args.loss_type == "opd_jsd_loss":
            onload_teacher_lm_head(self.args.teacher_hf_checkpoint, torch.device("cuda", torch.cuda.current_device()))
            print_memory("after wake_up teacher_lm_head")
        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")
        self._train_state_awake = True

    def _switch_model(self, target_tag: str) -> None:
        # ORBIT-SEAM: skip the restore when the tag is already resident; adapter-state restores are
        # frequent enough (ref/teacher toggling per step) that the no-op case must be free
        if target_tag == self._active_model_tag:
            return
        if target_tag not in self.model_state_manager.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.model_state_manager.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    def _fill_replay_data(
        self,
        data_iterator,
        num_microbatches,
        rollout_data,
        data_key: str,
        replay_list: list,
        register_replay_list_func,
        if_sp_region=True,
    ):
        if data_key not in rollout_data:
            raise ValueError(f"{data_key} is required in rollout_data for replay.")

        for iterator in data_iterator:
            iterator.reset()

        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size
        qkv_format = self.args.qkv_format

        def pad_func(data, pad):
            _, num_layers, topk = data.shape
            pad_tensor = torch.full(
                (pad, num_layers, topk),
                fill_value=-1,
                device=data.device,
                dtype=data.dtype,
            )
            return torch.cat([data, pad_tensor], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
            replay_data = batch[data_key]
            tokens = batch["tokens"]
            assert len(replay_data) == len(tokens)
            for a, b in zip(replay_data, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # ORBIT-SEAM: repo-wide comment-style pass (TODO -> Follow-up), no functional change
            # Follow-up: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            # Follow-up: maybe extract a common process function for here and get_batch?
            # ORBIT-SEAM: slice_with_cp gained a per-sample max_seq_len argument; supply it in the
            # non-bshd branch below (None keeps base's behaviour when the batch has no lengths)
            max_seq_lens = batch.get("max_seq_lens")

            def sample_max_seq_len(index: int, max_seq_lens=max_seq_lens) -> int | None:
                return max_seq_lens[index] if max_seq_lens is not None else None

            if qkv_format == "bshd":
                max_seqlen = batch["max_seq_lens"][0]
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
                replay_data = torch.stack(replay_data, dim=0)
                batch_size, seqlen, num_layers, topk = replay_data.shape
                replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
            else:
                replay_data = [
                    slice_with_cp(r, pad_func, qkv_format, sample_max_seq_len(i)) for i, r in enumerate(replay_data)
                ]
                replay_data = torch.cat(replay_data, dim=0)
                pad_size = parallel_state.tp.size * self.args.data_pad_size_multiplier
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)

            if self.args.sequence_parallel and if_sp_region:
                seqlen = replay_data.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                replay_data = replay_data[start:end]

            register_replay_list_func(replay_list, replay_data, self.model)

        del rollout_data[data_key]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        self._last_rollout_id = rollout_id
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        # ORBIT-SEAM: role tells the advantage/return pipeline it is running for the critic
        compute_advantages_and_returns(self.args, rollout_data, role="critic")

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay")

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # ORBIT-SEAM: one-trunk-critic iterator/microbatch counts, built inside the advantages
        # block below and consumed by the critic train phase at the end of this method
        critic_data_iterator = None
        critic_num_microbatches = None
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                self._fill_replay_data(
                    data_iterator,
                    num_microbatches,
                    rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=get_register_replay_list_func(m),
                    if_sp_region=m.if_sp_region,
                )

        with inverse_timer("train_wait"), timer("train"):
            # ORBIT-SEAM: base does its ref forward inline inside the advantages block; orbit hoists
            # it into compute_ref_log_probs (home mixin) so PEFT can serve the reference by
            # disabling adapters, and so a direct loss can still request it
            # Outside the advantages block so loss types that skip the PPO
            # advantage/returns pipeline (opd_jsd_loss) can still opt into
            # --use-kl-loss; compute_ref_log_probs returns None when neither
            # kl_coef nor use_kl_loss asks for a ref forward.
            ref_data = self.compute_ref_log_probs(data_iterator, num_microbatches)
            if ref_data is not None:
                rollout_data.update(ref_data)
            if self.args.compute_advantages_and_returns:
                # ORBIT-SEAM: on-policy-distillation teacher forward (home mixin) plus the env-gated
                # correctness dump of what it produced; returns None when no in-process teacher runs
                teacher_data = self.compute_teacher_log_probs(data_iterator, num_microbatches, ref_data=ref_data)
                if teacher_data is not None:
                    rollout_data.update(teacher_data)
                    if self._is_main_rank:
                        # M1 correctness leg (I-5): env-gated dump of the in-process
                        # teacher_log_probs just computed. No Sample objects exist at
                        # this point (megatron computes teacher_log_probs directly onto
                        # the batch-level rollout_data/teacher_data dicts, never through
                        # Sample), so synthesize lightweight per-sample stand-ins from
                        # this rank's local shard: .tokens is the same full
                        # prompt+response ids as Sample.tokens; sample_index numbers
                        # this rank's local (seqlen-balanced) order, not the original
                        # global rollout order -- process_rollout_data discards that
                        # partition mapping before this point. The compare CLI's
                        # required tokens-equality check still makes any accidental
                        # cross-index comparison fail loudly rather than silently.
                        # No-op unless ORBIT_OPD_TEACHER_LOGPROB_DUMP is set.
                        maybe_dump_teacher_logprobs(
                            rollout_id,
                            [
                                types.SimpleNamespace(tokens=tokens.tolist(), teacher_log_probs=teacher_lp.tolist())
                                for tokens, teacher_lp in zip(
                                    rollout_data["tokens"], teacher_data["teacher_log_probs"], strict=True
                                )
                            ],
                        )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                # ORBIT-SEAM: base's single `use_critic` splits in two - the separate critic actor
                # still syncs over the actor/critic groups, while the one-trunk critic runs its
                # value forward right here on this actor's own trunk
                if uses_separate_critic(self.args):
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                elif uses_one_trunk_critic(self.args):
                    critic_data_iterator, critic_num_microbatches = get_data_iterator(
                        self.args, self.critic_model, rollout_data
                    )
                    rollout_data.update(
                        forward_only(
                            get_values,
                            self.args,
                            self.critic_model,
                            critic_data_iterator,
                            critic_num_microbatches,
                        )
                    )
                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            # ORBIT-SEAM: hoisted out of the advantages block (base restores inside it)
            # A full-FT reference forward switches the resident parameters to
            # the "ref" backup. Direct losses such as opd_jsd_loss skip the
            # advantages block above, so restoration must not live inside it.
            if self._active_model_tag != "actor":
                self._switch_model("actor")

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            self._set_replay_stage("replay_backward")
            # ORBIT-SEAM: base always runs the policy step; with a one-trunk critic the first
            # --num-critic-only-steps rollouts train the value head alone, so the policy step (and
            # with it the self-teacher EMA/lag update and its promotion) is gated
            run_policy_phase = not uses_one_trunk_critic(self.args) or rollout_id >= self.args.num_critic_only_steps
            with timer("actor_train"):
                if run_policy_phase:
                    train(
                        rollout_id,
                        self.model,
                        self.optimizer,
                        self.opt_param_scheduler,
                        data_iterator,
                        num_microbatches,
                    )
                    if self._self_teacher is not None:
                        # EMA/lag cadence is defined in actor optimizer steps;
                        # critic-only warmup rollouts must not age the teacher.
                        self._self_teacher.update(self._adapter_named_params())
                        actor_step = rollout_id
                        if uses_one_trunk_critic(self.args):
                            actor_step -= self.args.num_critic_only_steps
                        if should_promote_teacher(
                            self._opd_teacher_spec.source, self.args.opd_promote_interval, actor_step
                        ):
                            self._promote_self_teacher()

            # ORBIT-SEAM: second train phase for the one-trunk critic, under the value-loss context
            if uses_one_trunk_critic(self.args) and critic_data_iterator is not None:
                with timer("critic_train"), value_loss_phase(self.args):
                    train(
                        rollout_id,
                        self.critic_model,
                        self.critic_optimizer,
                        self.critic_opt_param_scheduler,
                        critic_data_iterator,
                        critic_num_microbatches,
                    )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        # ORBIT-SEAM: base backs up unconditionally; skip it in the modes nothing restores from
        # update the CPU actor snapshot when a later restore/copy path needs it
        if should_backup_actor_after_train(self.args):
            self.model_state_manager.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.model_state_manager.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.model_state_manager.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        # ORBIT-SEAM: the save chain carries the self-teacher so its checkpoint sidecar is written
        # beside the adapter (here and in the save_hf_model call below)
        # getattr: the separate-critic actor shares this method but never runs the
        # OPD teacher init that creates _self_teacher.
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
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(
                self.args,
                rollout_id,
                self.model,
                self_teacher=getattr(self, "_self_teacher", None),
            )

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        # ORBIT-SEAM: engine-free runs (SFT/eval-only) have nothing to sync to
        if self.args.debug_train_only or self.args.debug_rollout_only or not uses_rollout_engines(self.args):
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        # ORBIT-SEAM: removed base's torch_memory_saver resume()/disable() wrapper around this whole
        # block (and the LoRA-specific resume before it): orbit never pauses the allocator, so the
        # adapter/base params the export reads are already backed by real GPU memory
        print_memory("before update_weights")
        self.weight_updater.update_weights()
        print_memory("after update_weights")

        # ORBIT-SEAM: fill the engine's OPD teacher slot, then release the adapter to CPU if the run
        # offloads it - both must happen after the sync and before anything else touches the params
        # Startup fill of the engine's orbit_teacher slot for sglang self:*
        # teachers (Task 6 reserves it EMPTY; the local scoring stage fires
        # unconditionally during the FIRST generate, and scoring an empty slot
        # 404s). Both drivers call actor update_weights once before the first
        # generate — on fresh start AND resume — so promoting here guarantees
        # the slot is filled before any scoring. Re-promote whenever new or
        # restarted engines connect: their slot starts empty too. Must run
        # before offload_train_adapter (the export reads live GPU params). A
        # failure raises out of the launch — never start training with an
        # empty teacher slot.
        if (
            self._self_teacher is not None
            and should_promote_teacher(
                self._opd_teacher_spec.source, getattr(self.args, "opd_promote_interval", None), 0
            )
            and (not self._teacher_slot_startup_promoted or num_new_engines > 0)
        ):
            self._promote_self_teacher()
            self._teacher_slot_startup_promoted = True

        if self.args.offload_train_adapter:
            offload_megatron_adapter_to_cpu(self.model)
            print_memory("after update_weights adapter_offload")

        if self.args.ci_test and len(rollout_engines) > 0 and not is_lora_enabled(self.args):
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )

        if getattr(self.args, "keep_old_actor", False):
            if self.args.update_weights_interval == 1:
                logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                # First copy rollout_actor to old_actor
                self.model_state_manager.copy(src_tag="rollout_actor", dst_tag="old_actor")
                # Then copy current actor to rollout_actor
                # ORBIT-SEAM: in adapter-state mode the "actor" snapshot is already current (the
                # adapter is the whole state), so the queue advances by copy instead of re-backup
                if uses_adapter_state(self.args):
                    self.model_state_manager.copy(src_tag="actor", dst_tag="rollout_actor")
                else:
                    self.model_state_manager.backup("rollout_actor")
            else:
                if uses_adapter_state(self.args):
                    self.model_state_manager.copy(src_tag="actor", dst_tag="old_actor")
                else:
                    self.model_state_manager.backup("old_actor")

        if self.args.offload_train:
            # ORBIT-SEAM: base pauses torch_memory_saver here; orbit instead frees the export's
            # scratch allocations so the offloaded state actually stays offloaded
            destroy_process_groups()
            clear_memory()
            print_memory("after update_weights destroy_process_groups")

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

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

        self.model_state_manager.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
