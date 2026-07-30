import logging
import os
import random
import socket
from argparse import Namespace

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from transformers import AutoConfig

from orbit.ray.train_actor import TrainRayActor
from orbit.utils import train_dump_utils
from orbit.utils.adapter_swap import swap_adapter_tensors
from orbit.utils.arguments import uses_rollout_engines
from orbit.utils.arguments import uses_adapter_critic, uses_separate_critic
from orbit.utils.context_utils import with_defer
from orbit.utils.distributed_utils import get_gloo_group, init_process_group
from orbit.utils.memory_utils import clear_memory, print_memory
from orbit.utils.opd_teacher_spec import should_promote_teacher, teacher_forward_plan
from orbit.utils.processing_utils import load_tokenizer
from orbit.utils.ray_utils import Box
from orbit.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from orbit.utils.replay_base import all_replay_managers
from orbit.utils.self_teacher import SelfTeacherBuffer
from orbit.utils.timer import Timer, inverse_timer, timer
from orbit.utils.tracking_utils import init_tracking
from orbit.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from ..training_utils.parallel import get_parallel_state
from .checkpoint import load_checkpoint
from .critic_adapter import build_critic_instance, save_critic_checkpoint, value_loss_phase
from .initialize import init, is_megatron_main_rank
from .lora_utils import is_lora_enabled
from .model import forward_only, initialize_model_and_optimizer, save, train
from .model_state_manager import create_model_state_manager
from .parallel import verify_megatron_parallel_state
from .peft_offload import (
    load_megatron_adapter_to_gpu,
    load_megatron_frozen_base_to_gpu,
    load_megatron_grad_buffers,
    load_megatron_optimizer,
    offload_megatron_adapter_to_cpu,
    offload_megatron_frozen_base_to_cpu,
    offload_megatron_grad_buffers,
    offload_megatron_optimizer,
)
from .peft_utils import (
    create_peft_instance,
    get_peft_method,
    is_adapter_param_name,
    is_peft_enabled,
    load_adapter_tensors_for_teacher,
)
from .replay_utils import get_register_replay_list_func
from .state_mode import should_backup_actor_after_train, uses_adapter_state
from .update_weight.common import named_adapter_params, named_params_and_buffers
from .update_weight.update_weight_from_distributed.bridge import UpdateWeightFromDistributedBridge
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed
try:
    from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
except ImportError:
    # Older sglang (e.g. 0.5.9) lacks ParallelismContext/RankParallelismConfig.
    # UpdateWeightP2P is only used in non-colocate + p2p transfer mode, so
    # we leave it undefined and fail only if that code path is actually taken.
    UpdateWeightP2P = None
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _get_weight_updater_kwargs(args: Namespace, update_weight_cls: type) -> dict[str, object]:
    if update_weight_cls is UpdateWeightFromTensor:
        return {"peft_method": get_peft_method(args)}
    return {"is_lora": is_lora_enabled(args)}


def _select_update_weight_cls(args: Namespace) -> type:
    """Pick the weight updater for this run's (colocate, PEFT, transfer, to-hf) combo."""
    if args.colocate or get_peft_method(args) != "none":
        # PEFT (LoRA/OFT) routes through UpdateWeightFromTensor regardless
        # of colocate, so the unified PeftWeightTransport (IPC for colocate,
        # NCCL for async) is the only adapter sync path.
        return UpdateWeightFromTensor
    if args.update_weight_transfer_mode == "broadcast":
        # Bridge-loaded models (Nemotron-H, Gemma-4) have no megatron_to_hf
        # name mapping; their disaggregated sync streams the bridge export.
        if args.megatron_to_hf_mode == "bridge":
            return UpdateWeightFromDistributedBridge
        return UpdateWeightFromDistributed
    return UpdateWeightP2P


def _validate_train_offload_role(args: Namespace, role: str) -> None:
    if role == "critic" and getattr(args, "offload_train", False):
        raise NotImplementedError(
            "Megatron critic --offload-train is not supported. Critic models are full-model even when "
            "actor PEFT is enabled, so they cannot use the PEFT frozen-base offload path."
        )


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
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

        from orbit.backends.megatron_utils.mtp_rl_patches import apply_mtp_in_rl_patches
        apply_mtp_in_rl_patches(self.args)

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        if role != "critic" and getattr(self.args, "use_rollout_routing_replay", False):
            from orbit.backends.megatron_utils.replay_utils import wire_routing_replay_to_models
            wire_routing_replay_to_models(self.model)

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from orbit_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        if self.args.offload_train and getattr(self.args, "offload_train_async", False):
            self._wake_up_stream = torch.cuda.Stream()
        else:
            self._wake_up_stream = None
        self._wake_up_event = None

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        self.critic_model = None
        self.critic_optimizer = None
        self.critic_opt_param_scheduler = None
        if uses_adapter_critic(self.args):
            (
                self.critic_model,
                self.critic_optimizer,
                self.critic_opt_param_scheduler,
            ) = build_critic_instance(self.args, self.model, expected_iteration=loaded_rollout_id)

        start_rollout_id = loaded_rollout_id + 1

        if uses_adapter_state(self.args):
            state_source_getter = lambda: named_adapter_params(self.model)
        else:
            state_source_getter = lambda: named_params_and_buffers(
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
        if should_backup_actor_after_train(self.args):
            self.model_state_manager.backup("actor")

        if with_ref and not is_peft_enabled(self.args):
            self.load_other_checkpoint("ref", args.ref_load)

        # In-process OPD teacher. Same-base specs (base/adapter:/self:*) need no
        # second model: the teacher is the resident base with adapters toggled.
        # Only the legacy load:<ckpt> spec loads a full second model like "ref".
        self._opd_teacher_spec = getattr(self.args, "opd_teacher_spec", None)
        self._opd_teacher_tensors: dict[str, torch.Tensor] | None = None
        self._self_teacher = None
        if with_opd_teacher:
            if self._opd_teacher_spec is None:
                from orbit.utils.opd_teacher_spec import parse_teacher_spec

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

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.model_state_manager.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

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
            from orbit.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        if self.args.offload_train_grad_buffers:
            offload_megatron_grad_buffers(self.model)
            print_memory("after offload grad_buffers")
        if self.args.offload_train_optimizer:
            offload_megatron_optimizer(self.optimizer)
            print_memory("after offload optimizer")
        offload_megatron_frozen_base_to_cpu(self.model)
        print_memory("after offload frozen_base")

        print_memory("after offload model")

        if self._is_main_rank and hasattr(self, "_last_rollout_id"):
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    def prefetch_train_state(self, rollout_id: int) -> None:
        """Issue train-state H2D wake-up on a dedicated stream so the
        train kernels can wait on it instead of blocking on a synchronous load.

        Called from train.py between 'offload rollout done' and 'actor train start'
        when args.offload_train_async is on. No-op otherwise.
        """
        if not self.args.offload_train:
            return
        if not getattr(self.args, "offload_train_async", False):
            return
        if self._wake_up_stream is None:
            return
        load_megatron_frozen_base_to_gpu(self.model, stream=self._wake_up_stream)
        if self.args.offload_train_adapter:
            load_megatron_adapter_to_gpu(self.model, stream=self._wake_up_stream)
        event = torch.cuda.Event()
        event.record(self._wake_up_stream)
        self._wake_up_event = event

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        wake_up_event = getattr(self, "_wake_up_event", None)
        if wake_up_event is not None:
            torch.cuda.current_stream().wait_event(wake_up_event)
            self._wake_up_event = None
            print_memory("after wake_up train_state_prefetch")
        else:
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
        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
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
            # Follow-up: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            # Follow-up: maybe extract a common process function for here and get_batch?
            max_seq_lens = batch.get("max_seq_lens")

            def sample_max_seq_len(index: int) -> int | None:
                return max_seq_lens[index] if max_seq_lens is not None else None

            if qkv_format == "bshd":
                max_seqlen = batch["max_seq_lens"][0]
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
                replay_data = torch.stack(replay_data, dim=0)
                batch_size, seqlen, num_layers, topk = replay_data.shape
                replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
            else:
                replay_data = [
                    slice_with_cp(r, pad_func, qkv_format, sample_max_seq_len(i))
                    for i, r in enumerate(replay_data)
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

    def compute_ref_log_probs(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
    ) -> dict[str, list[torch.Tensor]] | None:
        """Compute reference log-probs for the KL term.

        Returns a dict ready to merge into rollout_data (with "ref_log_probs"),
        or None if no ref forward should run this cycle.

        Routing:
          * (kl_coef == 0 AND not use_kl_loss)  -> return None.
          * is_peft_enabled(args)               -> fallthrough replay; run forward
                                                  inside create_peft_instance(args).disable_adapter(self.model).
          * "ref" in model_state_manager        -> fallthrough replay; _switch_model("ref"); run forward.
          * otherwise                           -> return None.
        """
        if self.args.kl_coef == 0 and not self.args.use_kl_loss:
            return None

        if is_peft_enabled(self.args):
            peft = create_peft_instance(self.args)
            if peft is None:
                raise RuntimeError("PEFT reference log-probs requested but no PEFT instance could be created.")
            self._set_replay_stage("fallthrough")
            with peft.disable_adapter(self.model):
                return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")

        if "ref" not in self.model_state_manager.backup_tags:
            return None

        self._set_replay_stage("fallthrough")
        self._switch_model("ref")
        return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")

    def _adapter_named_params(self) -> dict[str, torch.Tensor]:
        params: dict[str, torch.Tensor] = {}
        for chunk in self.model:
            for name, param in chunk.named_parameters():
                if is_adapter_param_name(name):
                    params[name] = param
        return params

    def compute_teacher_log_probs(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        ref_data: dict[str, list[torch.Tensor]] | None = None,
    ) -> dict[str, list[torch.Tensor]] | None:
        """Compute in-process teacher log-probs for on-policy distillation.

        Teacher-forcing forward on the student's sampled tokens, producing
        "teacher_log_probs". Routing follows teacher_forward_plan: same-base
        specs toggle adapters on the resident model (no second model); base
        aliases the already-computed ref forward when available; load: keeps
        the legacy full second model. Under --opd-type sglang the teacher is
        scored on the rollout engine, so this returns None (plan "none").
        """
        plan = teacher_forward_plan(
            self._opd_teacher_spec,
            is_peft_enabled(self.args),
            ref_data is not None,
            opd_type=getattr(self.args, "opd_type", None),
        )
        if plan == "none":
            return None
        if plan == "alias_ref":
            return {"teacher_log_probs": ref_data["ref_log_probs"]}
        if plan == "adapter_off":
            peft = create_peft_instance(self.args)
            if peft is None:
                raise RuntimeError("OPD base teacher requested but no PEFT instance could be created.")
            self._set_replay_stage("fallthrough")
            with peft.disable_adapter(self.model):
                return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="teacher_")
        if plan == "adapter_swap":
            tensors = self._opd_teacher_tensors
            if tensors is None and self._self_teacher is not None:
                tensors = self._self_teacher.tensors
            if tensors is None:
                raise RuntimeError(f"OPD teacher {self._opd_teacher_spec.source} has no tensors loaded.")
            self._set_replay_stage("fallthrough")
            with swap_adapter_tensors(self.model, tensors, is_adapter_param_name):
                return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="teacher_")
        # switch_model (legacy load:)
        if "teacher" not in self.model_state_manager.backup_tags:
            return None
        self._set_replay_stage("fallthrough")
        self._switch_model("teacher")
        return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="teacher_")

    def _promote_self_teacher(self) -> None:
        """Push the EMA/lag buffer to the engine's orbit_teacher slot.

        Swap buffer tensors into the live adapter params so the existing
        megatron->HF conversion + transport applies unchanged, sync to the
        teacher slot name, restore. Failure keeps the previous teacher slot
        (never silently distill from a half-loaded adapter).
        """
        with swap_adapter_tensors(self.model, self._self_teacher.tensors, is_adapter_param_name):
            try:
                self.weight_updater.push_teacher_adapter()
            except Exception:
                logger.exception("OPD self-teacher promotion FAILED; engine keeps the previous teacher slot.")
                raise

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
            if self.args.compute_advantages_and_returns:
                ref_data = self.compute_ref_log_probs(data_iterator, num_microbatches)
                if ref_data is not None:
                    rollout_data.update(ref_data)
                teacher_data = self.compute_teacher_log_probs(data_iterator, num_microbatches, ref_data=ref_data)
                if teacher_data is not None:
                    rollout_data.update(teacher_data)
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

                if uses_separate_critic(self.args):
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                elif uses_adapter_critic(self.args):
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
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            self._set_replay_stage("replay_backward")
            run_policy_phase = (
                not uses_adapter_critic(self.args) or rollout_id >= self.args.num_critic_only_steps
            )
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
                    self._self_teacher.update(self._adapter_named_params())
                    if should_promote_teacher(
                        self._opd_teacher_spec.source, self.args.opd_promote_interval, rollout_id
                    ):
                        self._promote_self_teacher()

            if uses_adapter_critic(self.args) and critic_data_iterator is not None:
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

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if uses_adapter_critic(self.args) and self.args.critic_save:
            save_critic_checkpoint(self.args, rollout_id, self.critic_model, optimizer=self.critic_optimizer)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from orbit.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
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

        print_memory("before update_weights")
        self.weight_updater.update_weights()
        print_memory("after update_weights")

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
