"""Orbit's added ``MegatronTrainRayActor`` methods.

Home mixin for the methods lifted out of
miles/backends/megatron_utils/actor.py (Phase 3 isolation, slice 3g): the
train-state prefetch, the reference/teacher log-prob forwards, the held-out
eval-NLL subsystem, the self-teacher checkpoint restore/promotion, and the
adapter-parameter view. ``MegatronTrainRayActor`` in the miles file lists
``OrbitTrainActorExtensions`` as its first base; every method here runs with
``self`` bound to a live actor and reaches base-class state/methods
(``self.args``, ``self.model``, ``self.role``, ``self.model_state_manager``,
``self.compute_log_prob``, ``self._set_replay_stage``, ``self._switch_model``,
``self.sleep``, ``self.wake_up``, ``self.weight_updater``) the normal
attribute-lookup way -- no re-imports needed for those.

Plain mixin: no ``__init__``, no state of its own. Method bodies are verbatim
moves; the only additions are the call-time imports at the top of the methods
that need a name the old module supplied.

Import direction: no miles import at module level (the two miles types used in
signatures are ``TYPE_CHECKING``-only, which ``from __future__ import
annotations`` makes sufficient).

Three names are imported at CALL time from ``miles.backends.megatron_utils
.actor`` rather than from their true homes, deliberately: the fast suite
rebinds them on that module while driving these very methods, and importing
them here at module load would freeze the pre-patch object.

  * ``create_peft_instance`` -- tests/fast/test_opd_teacher_equivalence.py
    patches it for ``compute_teacher_log_probs``;
  * ``is_adapter_param_name`` -- tests/test_opd_scoring_stage.py patches it for
    ``_adapter_named_params``;
  * ``get_gloo_group`` -- tests/fast/test_self_teacher_save_chain.py patches it
    for ``_restore_checkpoint_teacher_state``.

Every other miles name (``DataIterator``, ``get_parallel_state``, ``timer``) is
imported call-time from its true home, because nothing rebinds it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

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
    from miles.backends.training_utils.data import DataIterator
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
                return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")

        if "ref" not in self.model_state_manager.backup_tags:
            return None

        self._set_replay_stage("fallthrough")
        self._switch_model("ref")
        return self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")

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
                out = self.compute_log_prob(data_iterator, num_microbatches, store_prefix="eval_")

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
        # Late-bound: the fast suite rebinds is_adapter_param_name on the miles
        # actor module (see module docstring).
        from miles.backends.megatron_utils.actor import is_adapter_param_name

        with swap_adapter_tensors(self.model, self._self_teacher.tensors, is_adapter_param_name):
            try:
                self.weight_updater.push_teacher_adapter()
            except Exception:
                logger.exception("OPD self-teacher promotion FAILED; engine keeps the previous teacher slot.")
                raise


__all__ = ["OrbitTrainActorExtensions"]
