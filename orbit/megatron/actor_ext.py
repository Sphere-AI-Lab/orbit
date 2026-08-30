"""Orbit's added and overridden ``MegatronTrainRayActor`` methods.

Home mixin for the methods lifted out of
miles/backends/megatron_utils/actor.py. Two groups live here:

* ADDED methods (Phase 3 isolation, slice 3g) -- the train-state prefetch, the
  reference/teacher log-prob forwards, the held-out eval-NLL subsystem, the
  self-teacher checkpoint restore/promotion, and the adapter-parameter view.
  Nothing upstream defines these.
* REPLACING methods (mixin-override slice) -- ``sleep``, ``wake_up``,
  ``_switch_model``, ``train_actor`` and ``update_weights``. Upstream defines
  each of these too; orbit owned >= 50% of every one of them, so orbit now
  carries the whole body here and the method is DELETED from
  ``MegatronTrainRayActor``'s class body.

  Deleted, not left behind as an "upstream reference copy": Python resolves a
  class's OWN ``__dict__`` before any base, so ``class
  MegatronTrainRayActor(OrbitTrainActorExtensions, TrainRayActor)`` has MRO
  ``[MegatronTrainRayActor, OrbitTrainActorExtensions, TrainRayActor]``. A body
  retained in the vendored class would shadow the mixin -- the opposite of the
  intent -- and would do so silently, on the GPU training path. The mixin only
  wins because the name is absent from the vendored class. tests/fast/
  test_shadow_drift.py's premise ("mixin listed first shadows the vendored
  method") does not hold; see the report for this slice.

``MegatronTrainRayActor`` in the miles file lists ``OrbitTrainActorExtensions``
as its first base; every method here runs with ``self`` bound to a live actor
and reaches base-class state/methods (``self.args``, ``self.model``,
``self.role``, ``self.model_state_manager``, ``self.compute_log_prob``,
``self._set_replay_stage``, ``self._switch_model``, ``self.sleep``,
``self.wake_up``, ``self.weight_updater``) the normal attribute-lookup way --
no re-imports needed for those. ``super().<name>(...)`` from here resolves to
``TrainRayActor``, whose ``sleep``/``wake_up``/``update_weights`` are
``@abc.abstractmethod`` stubs that raise ``NotImplementedError`` (and which do
not even take the same arguments), so there is no upstream behaviour to defer
to: each of the five carries its whole body.

Plain mixin: no ``__init__``, no state of its own. Method bodies are verbatim
moves; the only additions are the call-time imports at the top of the methods
that need a name the old module supplied.

Import direction and WHY the call-time imports read off the miles ACTOR module
rather than each name's true home: these bodies used to be module-level code in
miles/backends/megatron_utils/actor.py, so every free name in them resolved
through THAT module's globals -- late, and through whatever the fast suite had
rebound there. Re-importing them from the actor module at call time reproduces
that resolution exactly, which is what keeps the move behaviour-preserving:

  * ``create_peft_instance`` -- tests/fast/test_opd_teacher_equivalence.py
    patches it for ``compute_teacher_log_probs``;
  * ``is_adapter_param_name`` -- tests/test_opd_scoring_stage.py patches it for
    ``_adapter_named_params``;
  * ``get_gloo_group`` -- tests/fast/test_self_teacher_save_chain.py patches it
    for ``_restore_checkpoint_teacher_state``;
  * ``timer``/``inverse_timer``/``train``/``get_data_iterator``/
    ``all_replay_managers``/``fill_replay_data``/``log_*``/
    ``compute_advantages_and_returns``/``uses_one_trunk_critic``/
    ``uses_separate_critic``/``should_backup_actor_after_train`` --
    tests/fast/backends/megatron_utils/test_shared_ppo_lifecycle.py and
    tests/fast/test_actor_ref_restore.py patch these on the actor module while
    driving ``train_actor``;
  * the ``offload_*``/``load_*``/``clear_memory``/``print_memory``/
    ``destroy_process_groups``/``reload_process_groups``/
    ``is_first_replica_megatron_main_rank`` set -- the same lifecycle test
    patches these while driving ``sleep``/``wake_up``/``update_weights``.

Consequence for the vendored file: several of its orbit-added imports are now
unused BY IT but load-bearing for this mixin (the ORBIT-SEAM stamps there say
so). Do not "clean up" those imports.

``timer`` and ``with_logs`` are the exception that must bind at module level:
they are applied as decorators when this class body is evaluated, so a
call-time import is too late. That is also true of the pre-move code -- the
decorator froze the object at actor.py import time -- so nothing changes.
``miles.utils.timer`` and ``miles.utils.tracking_utils.structured_log`` are
leaf modules (stdlib + ``torch.distributed``) and import nothing back into the
backends package, so this introduces no cycle.
"""

from __future__ import annotations

import logging
import random
import types
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.utils.timer import timer
from miles.utils.tracking_utils.structured_log import with_logs

from orbit.megatron.peft_offload import (
    _should_offload_frozen_base,
    load_megatron_adapter_to_gpu,
    load_megatron_frozen_base_to_gpu,
)
from orbit.megatron.peft_utils import is_peft_enabled
from orbit.opd.opd_teacher_spec import teacher_forward_plan
from orbit.utils.adapter_swap import swap_adapter_tensors
from orbit.utils.adapter_tensors import AdapterTensorKey, adapter_named_parameters
from orbit.utils.eval_nll import (
    NllStats,
    accumulate_nll,
    build_eval_nll_batch,
    is_eval_nll_reporting_rank,
    plan_eval_nll_microbatches,
    plan_eval_nll_shards,
)

if TYPE_CHECKING:
    from miles.backends.megatron_utils.model import TrainStepOutcome
    from miles.backends.training_utils.data import DataIterator
    from miles.ray.rollout.rollout_manager import EnginesAndLock
    from miles.utils.audit_utils.witness.allocator import WitnessInfo
    from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


class OrbitTrainActorExtensions:
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
        if not _should_offload_frozen_base(self.args):
            # Nothing was offloaded, so there is nothing to prefetch. Recording
            # no event leaves wake_up() on its synchronous branch, which is also
            # a no-op here -- rather than waiting on an event for a copy that
            # never ran.
            return
        load_megatron_frozen_base_to_gpu(self.model, stream=self._wake_up_stream)
        if self.args.offload_train_adapter:
            load_megatron_adapter_to_gpu(self.model, stream=self._wake_up_stream)
        event = torch.cuda.Event()
        event.record(self._wake_up_stream)
        self._wake_up_event = event

    @with_logs
    @timer
    def sleep(self) -> None:
        """Offload the train state piece by piece.

        Overrides upstream, which pauses the whole allocator with
        ``torch_memory_saver.pause()``. Orbit moves grad buffers, the
        optimizer, the frozen base and the OPD teacher LM head separately so a
        PEFT run can keep the adapter resident on GPU.
        """
        # Late-bound off the miles actor module -- see the module docstring:
        # this body used to live there, and the fast suite rebinds these names
        # on that module while driving this very method.
        from miles.backends.megatron_utils.actor import (
            _should_offload_frozen_base,
            clear_memory,
            destroy_process_groups,
            is_first_replica_megatron_main_rank,
            log_cpu_memory,
            offload_megatron_frozen_base_to_cpu,
            offload_megatron_grad_buffers,
            offload_megatron_optimizer,
            offload_teacher_lm_head,
            print_memory,
        )

        assert self.args.offload_train
        if self._asleep:
            logger.info("sleep() called while already offloaded; skipping")
            return

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        should_log_cpu_memory = is_first_replica_megatron_main_rank() and hasattr(self, "_last_rollout_id")

        destroy_process_groups()

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

        self._asleep = True
        print_memory("after offload model")
        # Read by compute_eval_nll: it must know whether the training state is
        # resident before deciding to wake it, and must not leave it awake.
        self._train_state_awake = False

        if should_log_cpu_memory:
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @with_logs
    @timer
    def wake_up(self) -> None:
        """Mirror image of :meth:`sleep`.

        Overrides upstream's ``torch_memory_saver.resume()``: orbit either
        waits on the prefetch event :meth:`prefetch_train_state` recorded or
        reloads each offloaded piece synchronously.
        """
        # Late-bound off the miles actor module -- see the module docstring.
        from miles.backends.megatron_utils.actor import (
            _should_offload_frozen_base,
            clear_memory,
            load_megatron_adapter_to_gpu,
            load_megatron_frozen_base_to_gpu,
            load_megatron_grad_buffers,
            load_megatron_optimizer,
            onload_teacher_lm_head,
            print_memory,
            reload_process_groups,
        )

        assert self.args.offload_train
        if not self._asleep:
            logger.info("wake_up() called while already resident; ensuring process groups only")
            reload_process_groups()
            return
        print_memory("before wake_up model")

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
        self._asleep = False
        print_memory("after wake_up model")
        self._train_state_awake = True

    def _switch_model(self, target_tag: str) -> None:
        """Restore a backed-up model state, skipping the no-op case.

        Overrides upstream twice: orbit returns early when the tag is already
        resident (adapter-state restores toggle ref/teacher per step, so the
        no-op case must be free), and reads orbit's ``model_state_manager``
        rather than base's ``weights_backuper``.
        """
        if target_tag == self._active_model_tag:
            return
        if not self._enable_weight_backup:
            return
        if target_tag not in self.model_state_manager.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.model_state_manager.restore(target_tag)
        self._active_model_tag = target_tag

    def compute_ref_log_probs(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        rollout_id: int,
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
        # Late-bound: the fast suite rebinds create_peft_instance on the miles
        # actor module (see module docstring).
        from miles.backends.megatron_utils.actor import create_peft_instance

        if self.args.kl_coef == 0 and not self.args.use_kl_loss:
            return None

        if is_peft_enabled(self.args):
            peft = create_peft_instance(self.args)
            if peft is None:
                raise RuntimeError("PEFT reference log-probs requested but no PEFT instance could be created.")
            self._set_replay_stage("fallthrough")
            with peft.disable_adapter(self.model):
                return self.compute_log_prob(
                    data_iterator, num_microbatches, rollout_id=rollout_id, store_prefix="ref_"
                )

        if "ref" not in self.model_state_manager.backup_tags:
            return None

        self._set_replay_stage("fallthrough")
        self._switch_model("ref")
        return self.compute_log_prob(
            data_iterator, num_microbatches, rollout_id=rollout_id, store_prefix="ref_"
        )

    def _build_eval_nll_local_batch(
        self, rows: list[tuple[int, bool]], full_batch: dict
    ) -> RolloutBatch:
        """Materialise this rank's shard of the held-out set on the GPU.

        Mirrors ``get_rollout_data``: token/mask tensors on the current device,
        plus the ``bshd`` max-sequence-length padding. Deliberately does NOT go
        through ``process_rollout_data`` -- the held-out set is not a rollout
        and must not be resharded a second time.
        """
        from miles.backends.training_utils.parallel import get_parallel_state

        device = torch.cuda.current_device()
        indices = [index for index, _ in rows]
        rollout_data: RolloutBatch = {
            "tokens": [
                torch.tensor(full_batch["tokens"][i], dtype=torch.long, device=device) for i in indices
            ],
            "loss_masks": [
                torch.tensor(full_batch["loss_masks"][i], dtype=torch.int, device=device) for i in indices
            ],
            "total_lengths": [full_batch["total_lengths"][i] for i in indices],
            "response_lengths": [full_batch["response_lengths"][i] for i in indices],
        }
        if self.args.qkv_format == "bshd":
            pad_size = get_parallel_state().tp.size * self.args.data_pad_size_multiplier
            max_seq_len = max(rollout_data["total_lengths"])
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
            rollout_data["max_seq_lens"] = [max_seq_len] * len(indices)
        return rollout_data

    def compute_eval_nll(self, rollout_id: int) -> dict[str, float] | None:
        """Token-weighted held-out NLL of the current (adapted) model.

        Forward-only: no optimizer state is touched and no weights change. Runs
        through ``compute_log_prob`` so there is exactly one forward path in the
        actor, and reuses ``sft_rollout``'s masking so train and eval score the
        identical tokens.

        Two things this method exists to get right:

        * **Coverage.** It bypasses ``get_data_iterator``, whose
          ``num_local_samples // num_local_gbs`` floor division would silently
          drop the remainder of the held-out file and make the reported NLL a
          function of ``--global-batch-size``. The micro-batch schedule here
          keeps the short final group, and the sample count is asserted against
          the number of rows read.
        * **Cross-rank reduction.** ``RayTrainGroup._broadcast`` returns one
          value per actor across the whole TP x PP x DP grid. TP/PP replicas
          hold the SAME samples and DP shards hold DIFFERENT token counts, so
          neither averaging per-actor floats nor summing them is correct. This
          method sums the *accumulators* over the DP group and then returns a
          value from exactly one rank; every other rank returns ``None``.

        Returns:
            On the single reporting rank (DP 0, TP 0, CP 0, last PP stage), the
            dict from :meth:`NllStats.as_dict`. ``None`` everywhere else.
        """
        from miles.backends.training_utils.data import DataIterator
        from miles.backends.training_utils.parallel import get_parallel_state
        from miles.utils.timer import timer

        parallel_state = get_parallel_state()

        # Refusals, not silent approximations. Both would need CP-chunked mask
        # slicing / VPP-divisible micro-batch counts that cannot be validated
        # without a multi-GPU run, and a quiet wrong NLL is worse than a stop.
        assert parallel_state.cp.size == 1, (
            f"--eval-nll-data does not support context parallelism (cp_size={parallel_state.cp.size}); "
            "log-probs would be CP-chunked while the loss masks are not."
        )
        assert (parallel_state.vpp_size or 1) == 1, (
            f"--eval-nll-data does not support virtual pipeline parallelism (vpp_size={parallel_state.vpp_size})."
        )
        assert self.role != "critic", "--eval-nll-data is an actor-only eval"
        assert self.args.eval_nll_data, "compute_eval_nll called without --eval-nll-data"

        full_batch = getattr(self, "_eval_nll_batch", None)
        if full_batch is None:
            full_batch = build_eval_nll_batch(self.args, tokenizer=self.tokenizer)
            self._eval_nll_batch = full_batch
        num_rows = len(full_batch["total_lengths"])

        dp_size = parallel_state.intra_dp.size
        shards = plan_eval_nll_shards(num_rows, dp_size, pad_index=full_batch["shortest_row_index"])
        rows = shards[parallel_state.intra_dp.rank]
        is_padding = [padded for _, padded in rows]

        micro_batch_size = self.args.eval_nll_micro_batch_size or self.args.micro_batch_size or 1
        micro_batch_indices = plan_eval_nll_microbatches(len(rows), micro_batch_size)
        num_microbatches = [len(micro_batch_indices)]

        # Both DP collectives below MUST execute between wake_up() and sleep().
        # sleep() calls destroy_process_groups(), which sets
        # ReloadableProcessGroup.group = None; the monkeypatched dist.all_reduce
        # then unwraps `group=parallel_state.intra_dp.group` to `group=None`,
        # which torch reads as the default WORLD group. No exception is raised,
        # so a reduction placed outside this window silently reduces over the
        # wrong communicator (over-counting by tp_size once tp > 1).
        woke_here = False
        if self.args.offload_train and not getattr(self, "_train_state_awake", True):
            self.wake_up()
            woke_here = True

        # The held-out NLL is defined at temperature 1; args.rollout_temperature
        # is applied inside get_responses and would otherwise silently rescale
        # every logit on an RL run.
        previous_temperature = self.args.rollout_temperature
        try:
            if dp_size > 1:
                # The pipeline schedule is a collective: DP ranks disagreeing on
                # the micro-batch count would hang rather than fail.
                # plan_eval_nll_shards equalises shard sizes so this cannot
                # happen; assert it anyway.
                counts = torch.tensor(
                    [len(micro_batch_indices), -len(micro_batch_indices)],
                    dtype=torch.int64,
                    device=torch.cuda.current_device(),
                )
                dist.all_reduce(counts, op=dist.ReduceOp.MAX, group=parallel_state.intra_dp.group)
                assert counts[0].item() == -counts[1].item() == len(micro_batch_indices), (
                    f"DP ranks disagree on eval NLL micro-batch count: local={len(micro_batch_indices)}, "
                    f"max={counts[0].item()}, min={-counts[1].item()}"
                )

            self.args.rollout_temperature = 1.0
            rollout_data = self._build_eval_nll_local_batch(rows, full_batch)
            data_iterator = [
                DataIterator(rollout_data, None, micro_batch_indices) for _ in range(parallel_state.vpp_size or 1)
            ]
            self._set_replay_stage("fallthrough")
            with timer("eval_nll"):
                out = self.compute_log_prob(
                    data_iterator, num_microbatches, rollout_id=rollout_id, store_prefix="eval_"
                )

            log_probs = out.get("eval_log_probs")
            if log_probs is None:
                # Not the last pipeline stage: this rank ran the forward but
                # holds no logits. It still participates in the all-reduce below.
                local = NllStats.zero()
            else:
                assert len(log_probs) == len(rows), (
                    f"eval NLL scored {len(log_probs)} samples but was given {len(rows)}"
                )
                local = accumulate_nll(log_probs, rollout_data["loss_masks"], is_padding=is_padding)

            if dp_size > 1:
                values = torch.tensor(
                    local.to_values(), dtype=torch.float64, device=torch.cuda.current_device()
                )
                dist.all_reduce(values, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp.group)
                total = NllStats.from_values(values.tolist())
            else:
                total = local
        finally:
            # Still a finally: the model must go back to sleep even if the
            # forward or a collective raises.
            self.args.rollout_temperature = previous_temperature
            if woke_here:
                self.sleep()

        if not is_eval_nll_reporting_rank(parallel_state):
            return None

        assert total.num_samples == num_rows, (
            f"eval NLL scored {total.num_samples} samples but the held-out file "
            f"{self.args.eval_nll_data} has {num_rows} rows"
        )
        logger.info(
            "eval_nll rollout_id=%s nll=%.6f sample_mean_nll=%.6f tokens=%d samples=%d",
            rollout_id,
            total.mean_nll,
            total.sample_mean_nll,
            total.num_tokens,
            total.num_samples,
        )
        return total.as_dict()

    def _restore_checkpoint_teacher_state(self) -> None:
        """Resume the EMA/lag self-teacher from its checkpoint sidecar, if present.

        The sidecar is written next to the PEFT adapter checkpoint (model.save);
        without it a resumed run silently re-seeds the self-teacher from the
        resumed student, losing the teacher's lag. Absence is legal (checkpoints
        predating sidecars); corruption is not.
        """
        # Late-bound: the fast suite rebinds get_gloo_group on the miles actor
        # module (see module docstring).
        from miles.backends.megatron_utils.actor import get_gloo_group
        from orbit.opd.self_teacher_checkpoint import (
            TeacherCheckpointError,
            has_self_teacher_sidecar,
            load_self_teacher_sidecar,
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        adapter_dir = getattr(self.args, "_peft_resume_adapter_dir", None)

        # Adapter loading is shard-local.  A missing TP/PP shard can therefore
        # leave only some ranks with a resume directory.  Reach consensus before
        # any early return so those ranks cannot diverge at the sidecar
        # collectives below and hang the job.
        local_restore_state = (
            self._self_teacher is not None,
            str(adapter_dir) if adapter_dir is not None else None,
        )
        if world_size > 1:
            restore_states = [None] * world_size
            dist.all_gather_object(restore_states, local_restore_state, group=get_gloo_group())
        else:
            restore_states = [local_restore_state]

        teacher_enabled = [state[0] for state in restore_states]
        if any(teacher_enabled) and not all(teacher_enabled):
            raise TeacherCheckpointError("self-teacher initialization differs across distributed ranks")
        if not any(teacher_enabled):
            return

        adapter_dirs = [state[1] for state in restore_states]
        adapter_loaded = [path is not None for path in adapter_dirs]
        if any(adapter_loaded) and not all(adapter_loaded):
            missing = [str(missing_rank) for missing_rank, path in enumerate(adapter_dirs) if path is None]
            raise TeacherCheckpointError(
                "PEFT adapter checkpoint loaded on only some ranks; missing adapter shards on ranks "
                + ", ".join(missing)
            )
        if not any(adapter_loaded):
            return
        if len(set(adapter_dirs)) != 1:
            raise TeacherCheckpointError(
                "PEFT adapter resume directory differs across distributed ranks: "
                + ", ".join(f"rank {state_rank}: {path}" for state_rank, path in enumerate(adapter_dirs))
            )
        adapter_dir = adapter_dirs[0]

        local_present = has_self_teacher_sidecar(adapter_dir, rank=rank)
        if world_size > 1:
            presence = [None] * world_size
            dist.all_gather_object(presence, local_present, group=get_gloo_group())
        else:
            presence = [local_present]

        if any(presence) and not all(presence):
            missing = [str(missing_rank) for missing_rank, present in enumerate(presence) if not present]
            raise TeacherCheckpointError(
                "self-teacher checkpoint is only partially present; missing sidecars for ranks " + ", ".join(missing)
            )
        if not any(presence):
            logger.info(f"No self-teacher sidecar at {adapter_dir}; keeping freshly seeded teacher state.")
            return

        local_error = None
        try:
            load_self_teacher_sidecar(adapter_dir, self._self_teacher, rank=rank, world_size=world_size)
        except Exception as exc:  # surface a corrupt shard consistently on all ranks
            local_error = f"{type(exc).__name__}: {exc}"
        if world_size > 1:
            errors: list[str | None] = [None] * world_size
            dist.all_gather_object(errors, local_error, group=get_gloo_group())
        else:
            errors = [local_error]
        failures = [f"rank {failed_rank}: {error}" for failed_rank, error in enumerate(errors) if error]
        if failures:
            raise TeacherCheckpointError(
                "self-teacher checkpoint load failed on one or more ranks; " + "; ".join(failures)
            )
        logger.info(f"Restored self-teacher state from checkpoint sidecar at {adapter_dir}")

    def _adapter_named_params(self) -> dict[AdapterTensorKey, torch.nn.Parameter]:
        # Late-bound: the fast suite rebinds is_adapter_param_name on the miles
        # actor module (see module docstring).
        from miles.backends.megatron_utils.actor import is_adapter_param_name

        # (vp_stage, name) keys: plain names collide across virtual-pipeline chunks,
        # silently merging distinct adapter tensors in self-teacher/transport flows.
        return adapter_named_parameters(self.model, is_adapter_param_name)

    def compute_teacher_log_probs(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        ref_data: dict[str, list[torch.Tensor]] | None = None,
        rollout_id: int = 0,
    ) -> dict[str, list[torch.Tensor]] | None:
        """Compute in-process teacher log-probs for on-policy distillation.

        Teacher-forcing forward on the student's sampled tokens, producing
        "teacher_log_probs". Routing follows teacher_forward_plan: same-base
        specs toggle adapters on the resident model (no second model); base
        aliases the already-computed ref forward when available; load: keeps
        the legacy full second model. Under --opd-type sglang the teacher is
        scored on the rollout engine, so this returns None (plan "none").
        """
        # Late-bound: the fast suite rebinds both names on the miles actor
        # module (see module docstring).
        from miles.backends.megatron_utils.actor import create_peft_instance, is_adapter_param_name

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
                return self.compute_log_prob(
                    data_iterator, num_microbatches, rollout_id=rollout_id, store_prefix="teacher_"
                )
        if plan == "adapter_swap":
            tensors = self._opd_teacher_tensors
            if tensors is None and self._self_teacher is not None:
                tensors = self._self_teacher.tensors
            if tensors is None:
                raise RuntimeError(f"OPD teacher {self._opd_teacher_spec.source} has no tensors loaded.")
            self._set_replay_stage("fallthrough")
            with swap_adapter_tensors(self.model, tensors, is_adapter_param_name):
                return self.compute_log_prob(
                    data_iterator, num_microbatches, rollout_id=rollout_id, store_prefix="teacher_"
                )
        # switch_model (legacy load:)
        if "teacher" not in self.model_state_manager.backup_tags:
            return None
        self._set_replay_stage("fallthrough")
        self._switch_model("teacher")
        return self.compute_log_prob(
            data_iterator, num_microbatches, rollout_id=rollout_id, store_prefix="teacher_"
        )

    def _promote_self_teacher(self) -> None:
        """Push the EMA/lag buffer to the engine's orbit_teacher slot.

        Swap buffer tensors into the live adapter params so the existing
        megatron->HF conversion + transport applies unchanged, sync to the
        teacher slot name, restore. Failure keeps the previous teacher slot
        (never silently distill from a half-loaded adapter).
        """
        # Late-bound: the fast suite rebinds is_adapter_param_name on the miles
        # actor module (see module docstring).
        from miles.backends.megatron_utils.actor import is_adapter_param_name

        with swap_adapter_tensors(self.model, self._self_teacher.tensors, is_adapter_param_name):
            try:
                self.weight_updater.push_teacher_adapter()
            except Exception:
                logger.exception("OPD self-teacher promotion FAILED; engine keeps the previous teacher slot.")
                raise

    @with_logs
    def train_actor(
        self,
        rollout_id: int,
        rollout_data: RolloutBatch,
        external_data=None,
        *,
        witness_info: WitnessInfo | None,
        attempt: int,
    ) -> TrainStepOutcome:
        """The actor training step.

        Overrides upstream. Orbit's divergences, in order down the body: the
        reference forward is hoisted into ``compute_ref_log_probs`` (so PEFT
        can serve it by disabling adapters, and a direct loss can still ask for
        it); the OPD teacher forward runs through ``compute_teacher_log_probs``
        plus an env-gated correctness dump; base's single ``use_critic`` splits
        into the separate-critic sync and the one-trunk critic's value forward
        on this actor's own trunk; the "back to actor" restore is hoisted out
        of the advantages block; the policy step is gated by
        ``--num-critic-only-steps`` and carries the self-teacher EMA/lag update
        and its promotion; a second train phase runs the one-trunk critic under
        ``value_loss_phase``; and the post-train CPU snapshot is gated by
        ``should_backup_actor_after_train`` on orbit's ``model_state_manager``.
        """
        # Late-bound off the miles actor module -- see the module docstring.
        from miles.backends.megatron_utils.actor import (
            TrainStepOutcome,
            all_replay_managers,
            compute_advantages_and_returns,
            fill_replay_data,
            forward_only,
            get_data_iterator,
            get_num_rollouts,
            get_parallel_state,
            get_values,
            inverse_timer,
            is_first_replica_megatron_main_rank,
            is_multi_lora_enabled,
            log_perf_data,
            log_rollout_data,
            log_train_advantage_computation_event,
            maybe_dump_teacher_logprobs,
            ray,
            should_backup_actor_after_train,
            should_promote_teacher,
            sync_actor_critic_data,
            timer,
            train,
            train_dump_utils,
            uses_one_trunk_critic,
            uses_separate_critic,
            value_loss_phase,
        )

        # One-trunk-critic iterator/microbatch counts, built inside the advantages
        # block below and consumed by the critic train phase at the end of this method
        critic_data_iterator = None
        critic_num_microbatches = None
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        num_optimizer_steps = len(num_microbatches)
        skip_actor_forward_only = self.args.skip_actor_forward_only
        if skip_actor_forward_only:
            option = "--skip-actor-forward-only"
            assert num_optimizer_steps == 1, f"{option} requires 1 optimizer step, got {num_optimizer_steps}"
            assert rollout_data.get("log_probs") is None, f"{option} requires rollout data without actor log probs"

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                fill_replay_data(
                    args=self.args,
                    models=self.model,
                    data_iterator=data_iterator,
                    num_microbatches=num_microbatches,
                    rollout_data=rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=m.register_replay_list_func,
                    if_sp_region=m.if_sp_region,
                    indices_are_token_positions=m.replay_indices_are_token_positions,
                )

        with inverse_timer("train_wait"), timer("train"):
            # Outside the advantages block so loss types that skip the PPO
            # advantage/returns pipeline (opd_jsd_loss) can still opt into
            # --use-kl-loss; compute_ref_log_probs returns None when neither
            # kl_coef nor use_kl_loss asks for a ref forward.
            ref_data = self.compute_ref_log_probs(data_iterator, num_microbatches, rollout_id=rollout_id)
            if ref_data is not None:
                rollout_data.update(ref_data)
            if self.args.compute_advantages_and_returns:
                # On-policy-distillation teacher forward plus the env-gated correctness
                # dump of what it produced; returns None when no in-process teacher runs
                teacher_data = self.compute_teacher_log_probs(
                    data_iterator, num_microbatches, ref_data=ref_data, rollout_id=rollout_id
                )
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
                if not skip_actor_forward_only and (
                    not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
                ):
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
                            rollout_id=rollout_id,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                # Base's single `use_critic` splits in two - the separate critic actor
                # still syncs over the actor/critic groups (or takes upstream's shipped `values`),
                # while the one-trunk critic runs its value forward right here on this actor's trunk
                if uses_separate_critic(self.args):
                    if external_data is not None and get_parallel_state().is_pp_last_stage:
                        values_ref = external_data.get("values")
                        assert values_ref is not None, (
                            "actor and critic share the same parallel topology, so the critic rank "
                            "paired with a pp-last-stage actor rank must have shipped 'values'"
                        )
                        rollout_data["values"] = [
                            value.to(device=torch.cuda.current_device(), non_blocking=True)
                            for value in ray.get(values_ref.inner)
                        ]
                    else:
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
                            rollout_id=rollout_id,
                        )
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)
                log_train_advantage_computation_event(rollout_data)

            # Hoisted out of the advantages block (base restores inside it).
            # A full-FT reference forward switches the resident parameters to
            # the "ref" backup. Direct losses such as opd_jsd_loss skip the
            # advantages block above, so restoration must not live inside it.
            if self._active_model_tag != "actor":
                self._switch_model("actor")

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            num_rollouts = get_num_rollouts(self.args, rollout_data, num_optimizer_steps)
            self._set_replay_stage("replay_backward")
            # Base always runs the policy step; with a one-trunk critic the first
            # --num-critic-only-steps rollouts train the value head alone, so the policy step (and
            # with it the self-teacher EMA/lag update and its promotion) is gated
            run_policy_phase = not uses_one_trunk_critic(self.args) or rollout_id >= self.args.num_critic_only_steps
            with timer("actor_train"):
                train_step_outcome = TrainStepOutcome.NORMAL
                if run_policy_phase:
                    train_step_outcome = train(
                        rollout_id,
                        self.model,
                        self.optimizer,
                        self.opt_param_scheduler,
                        data_iterator,
                        num_microbatches,
                        num_rollouts,
                        witness_info=witness_info,
                        attempt=attempt,
                        ft_test_action_executor=self._ft_test_action_executor,
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

            # Second train phase for the one-trunk critic, under the value-loss context
            if uses_one_trunk_critic(self.args) and critic_data_iterator is not None:
                with timer("critic_train"), value_loss_phase(self.args):
                    train(
                        rollout_id,
                        self.critic_model,
                        self.critic_optimizer,
                        self.critic_opt_param_scheduler,
                        critic_data_iterator,
                        critic_num_microbatches,
                        num_rollouts,
                        witness_info=witness_info,
                        attempt=attempt,
                        ft_test_action_executor=self._ft_test_action_executor,
                    )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        if train_step_outcome == TrainStepOutcome.NORMAL:
            # Base backs up unconditionally; skip it in the modes nothing restores from.
            # Update the CPU actor snapshot when a later restore/copy path needs it.
            if should_backup_actor_after_train(self.args):
                self.model_state_manager.backup("actor")
            else:
                torch.cuda.synchronize()

            # Update ref model if needed
            if (
                self.args.ref_update_interval is not None
                and (rollout_id + 1) % self.args.ref_update_interval == 0
                and "ref" in self.model_state_manager.backup_tags
            ):
                with timer("ref_model_update"):
                    if is_first_replica_megatron_main_rank():
                        logger.info(f"Updating ref model at rollout_id {rollout_id}")
                    self.model_state_manager.backup("ref")

        if train_step_outcome == TrainStepOutcome.NORMAL and is_multi_lora_enabled(self.args):
            from miles.backends.megatron_utils.multi_lora_utils import commit_trained_batch

            commit_trained_batch(rollout_data, rollout_id, self._multi_lora_pending_push)

        log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())

        self._heartbeat.bump()
        return train_step_outcome

    @with_logs
    @timer
    def update_weights(self, info: EnginesAndLock) -> None:
        """Push trained weights to the rollout engines.

        Overrides upstream. Orbit's divergences: engine-free runs (SFT/eval-only)
        return early; base's ``torch_memory_saver.disable()`` wrapper around the
        export block is dropped (orbit never pauses the allocator, so the params
        the export reads are already real GPU memory); the engine's OPD teacher
        slot is filled and the adapter released to CPU after the sync; the
        keep-old-actor queue advances by copy in adapter-state mode; and the
        teardown also frees the export's scratch allocations whenever the run
        offloads at all, not only when the process groups were temporary.
        """
        # Late-bound off the miles actor module -- see the module docstring.
        from miles.backends.megatron_utils.actor import (
            clear_memory,
            destroy_process_groups,
            get_gloo_group,
            is_lora_enabled,
            is_multi_lora_enabled,
            offload_megatron_adapter_to_cpu,
            print_memory,
            ray,
            reload_process_groups,
            should_promote_teacher,
            torch_memory_saver,
            uses_adapter_state,
            uses_rollout_engines,
        )

        self._heartbeat.bump()
        # Engine-free runs (SFT/eval-only) have nothing to sync to
        if self.args.debug_train_only or self.args.debug_rollout_only or not uses_rollout_engines(self.args):
            return

        rollout_engines = info.rollout_engines
        rollout_engine_lock = info.rollout_engine_lock
        has_new_engines = info.has_new_engines
        engine_gpu_counts = info.engine_gpu_counts
        engine_gpu_offsets = info.engine_gpu_offsets
        del info

        process_groups_are_temporary = self.args.offload_train and self._asleep
        if process_groups_are_temporary:
            reload_process_groups()

        if has_new_engines or not self.weight_updater.is_rollout_engines_fresh():
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())

        if self.args.debug_skip_weight_update:
            if dist.get_rank() == 0:
                logger.warning("Skipping actor-to-rollout weight update because " "--debug-skip-weight-update is set.")
            if self.args.rematerialize_param_from_master_weight:
                torch_memory_saver.pause(tag="param_buffer")
            if process_groups_are_temporary:
                destroy_process_groups()
            return

        version_update_names: list[str] = []
        if is_multi_lora_enabled(self.args):
            from miles.backends.megatron_utils.multi_lora_utils import select_adapters_to_push

            self.weight_updater.multi_lora_adapters, version_update_names = select_adapters_to_push(
                self.loaded_adapters, self._multi_lora_pending_push, has_new_engines
            )

        # Removed base's torch_memory_saver.disable() wrapper around this whole block
        # (and the LoRA-specific resume before it): orbit never pauses the allocator, so the
        # adapter/base params the export reads are already backed by real GPU memory
        print_memory("before update_weights")
        self.weight_updater.update_weights()
        print_memory("after update_weights")
        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.set_weight_version.remote(self.weight_updater.weight_version))

        if is_multi_lora_enabled(self.args):
            from miles.backends.megatron_utils.multi_lora_utils import commit_weight_push

            self._multi_lora_pending_push.clear()
            commit_weight_push(version_update_names, self._is_first_replica_megatron_main_rank)

        # Fill the engine's OPD teacher slot, then release the adapter to CPU if the run
        # offloads it - both must happen after the sync and before anything else touches the params.
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
        # NOTE: the pre-move body read `num_new_engines > 0` here, a name no
        # longer bound anywhere in this method (upstream replaced the count with
        # the has_new_engines flag) -- a latent NameError that only fires on a
        # self:* teacher run. Reading the flag is what that condition meant.
        if (
            self._self_teacher is not None
            and should_promote_teacher(
                self._opd_teacher_spec.source, getattr(self.args, "opd_promote_interval", None), 0
            )
            and (not self._teacher_slot_startup_promoted or has_new_engines)
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
                # Then copy current actor to rollout_actor.
                # In adapter-state mode the "actor" snapshot is already current (the
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

        if self.args.rematerialize_param_from_master_weight:
            torch_memory_saver.pause(tag="param_buffer")
        # Base gates this on process_groups_are_temporary only; orbit also frees the
        # export's scratch allocations so the offloaded state actually stays offloaded
        if process_groups_are_temporary or self.args.offload_train:
            destroy_process_groups()
            clear_memory()
            print_memory("after update_weights destroy_process_groups")


__all__ = ["OrbitTrainActorExtensions"]
