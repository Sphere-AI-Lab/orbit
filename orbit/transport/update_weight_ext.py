"""Orbit's added ``UpdateWeightFromTensor`` methods.

Home mixin for the methods lifted out of
miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py
(Phase 3 isolation, slice 3g): the PEFT transport send path, the one-shot
teacher-slot push, and the sync-mode label the metrics/timeline markers carry.
``UpdateWeightFromTensor`` in the miles file lists
``OrbitUpdateWeightExtensions`` as its first base; every method here runs with
``self`` bound to a live updater and reaches its state (``self._peft_args``,
``self._peft_sync_spec``, ``self._peft_transport``, ``self._teacher_slot_loaded``,
``self.use_distribute``, ``self._is_distributed_src_rank``, ``self.peft_method``,
``self.weight_version``, ``self.weights_getter``, ``self._hf_weight_iterator``)
the normal attribute-lookup way -- no re-imports needed for those.

Plain mixin: no ``__init__``, no state of its own. Method bodies are verbatim
moves; the only addition is the call-time import of
``_check_weight_sync_results`` in :meth:`push_teacher_adapter`.

Import direction: no miles import at module level.
``_check_weight_sync_results`` is base miles code whose only home is the miles
updater module, so it is imported there at call time (which also keeps it
late-bound).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import ray
from ray import ObjectRef

from orbit.megatron.sync_metrics import _sync_type_label
from orbit.transport.slots import (
    MutationPurpose,
    authorize_adapter_destination,
)


class OrbitUpdateWeightExtensions:
    def _sync_mode_label(self) -> str:
        """Human-readable sync mode for metrics/timeline markers.

        "full" for base-model sync, "adapter_double_buffer" when the PEFT
        transport stages into an inactive slot, "adapter_single_slot"
        otherwise. Only consulted for rank-0 emission; on distributed non-src
        PEFT ranks (transport is None) the label falls back to single-slot.
        """
        if self._peft_sync_spec is None:
            return "full"
        runtime_mode = getattr(self._peft_transport, "runtime_mode", None)
        if getattr(runtime_mode, "adapter_double_buffer", False):
            return "adapter_double_buffer"
        return "adapter_single_slot"

    def _send_adapter_params(
        self,
        hf_named_tensors,
        adapter_name: str | None = None,
        *,
        purpose: MutationPurpose = MutationPurpose.STUDENT_SYNC,
    ) -> tuple[list[ObjectRef], Any, list[Any] | None]:
        authorized_name = authorize_adapter_destination(
            self._peft_args,
            requested_name=adapter_name,
            purpose=purpose,
        )
        if self.use_distribute and not self._is_distributed_src_rank:
            return [], [], []
        if self._peft_transport is None:
            raise RuntimeError("_send_adapter_params called without a PEFT transport")

        transport = self._peft_transport
        original_sync_spec = transport.sync_spec
        transport.sync_spec = dataclasses.replace(
            original_sync_spec,
            adapter_name=authorized_name,
        )
        # The IPC/Ray LoRA transport unloads the previous adapter before
        # reloading, gated on transport._peft_loaded — which tracks the STUDENT
        # slot. Point it at the teacher slot's own loaded-state so the first
        # promotion does not unload an orbit_teacher that is not there yet.
        # (NcclBackend has no _peft_loaded and stages/activates instead, so the
        # getattr returns None and this is a no-op there.)
        student_peft_loaded = (
            getattr(transport, "_peft_loaded", None)
            if purpose is MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION
            else None
        )
        if student_peft_loaded is not None:
            transport._peft_loaded = self._teacher_slot_loaded
        try:
            send_result = transport.send_adapter(
                hf_named_tensors,
                weight_version=self.weight_version,
            )
        finally:
            if student_peft_loaded is not None:
                self._teacher_slot_loaded = transport._peft_loaded
                transport._peft_loaded = student_peft_loaded
            transport.sync_spec = original_sync_spec
        return send_result.refs, [], send_result.results

    def push_teacher_adapter(self) -> None:
        """One-shot push of the CURRENT adapter params to the orbit_teacher slot.

        Mirrors ``update_weights``' gather/coalesce/send for a single sync but
        targets the reserved teacher slot instead of the student adapter, and
        skips the pause/flush/continue-generation lifecycle (this fills an
        inactive scoring slot, not the live generation adapter). The caller
        (``actor._promote_self_teacher``) has swapped the EMA/lag buffer into
        the live adapter params, so the existing Megatron->HF adapter export
        picks up the teacher tensors unchanged.
        """
        # Base miles code; the miles updater module is its only home.
        from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
            _check_weight_sync_results,
        )
        from orbit.opd.opd_teacher_spec import OPD_TEACHER_ADAPTER_NAME

        authorized_name = authorize_adapter_destination(
            self._peft_args,
            requested_name=OPD_TEACHER_ADAPTER_NAME,
            purpose=MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION,
        )
        if self._peft_sync_spec is None:
            raise RuntimeError("push_teacher_adapter requires a PEFT (LoRA/OFT) run.")

        megatron_local_weights = self.weights_getter()
        weight_chunks = self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights)
        if self._peft_sync_spec.method == "lora":
            from orbit.transport._gather import (
                coalesce_lora_hf_weight_chunks,
            )
            _source_chunk_count, weight_chunks = coalesce_lora_hf_weight_chunks(weight_chunks)
        elif self._peft_sync_spec.method == "oft":
            from orbit.transport._gather import (
                coalesce_oft_hf_weight_chunks,
            )
            _source_chunk_count, weight_chunks = coalesce_oft_hf_weight_chunks(weight_chunks)

        sync_chunk_count = 0
        for hf_named_tensors in weight_chunks:
            refs, _long_lived, completed_results = self._send_adapter_params(
                hf_named_tensors,
                adapter_name=authorized_name,
                purpose=MutationPurpose.LEGACY_SELF_TEACHER_PROMOTION,
            )
            results = completed_results if completed_results is not None else ray.get(refs)
            _check_weight_sync_results(results, sync_type=_sync_type_label(self.peft_method))
            sync_chunk_count += 1

        if sync_chunk_count == 0:
            raise RuntimeError(
                f"{self._peft_sync_spec.method.upper()} teacher promotion failed: the weight "
                "iterator produced zero chunks; the orbit_teacher slot was not filled."
            )


__all__ = ["OrbitUpdateWeightExtensions"]
