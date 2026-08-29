from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orbit.backends.megatron_utils import actor as actor_utils
from orbit.backends.megatron_utils import checkpoint as checkpoint_utils
from orbit.backends.megatron_utils import lora_utils
from orbit.backends.megatron_utils import model as model_utils
from orbit.peft.megatron import peft_utils
from orbit.peft.opd.self_teacher import SelfTeacherBuffer
from orbit.peft.opd.self_teacher_checkpoint import (
    TeacherCheckpointError,
    has_self_teacher_sidecar,
    load_self_teacher_sidecar,
)


_ADAPTER_KEY = (0, "adapter.weight")


def _teacher(value: float = 1.0) -> SelfTeacherBuffer:
    teacher = SelfTeacherBuffer({_ADAPTER_KEY: torch.full((2, 3), value)}, mode="ema", decay=0.9)
    teacher.update({_ADAPTER_KEY: torch.full((2, 3), value + 1.0)})
    return teacher


def _install_native_peft_save(
    monkeypatch,
    tmp_path: Path,
    *,
    actual_adapter_dir: Path,
) -> tuple[Namespace, list[dict]]:
    args = Namespace(
        ci_test=False,
        ci_save_model_hash=False,
        peft_method="lora",
        save=str(tmp_path / "native"),
    )
    calls = []

    def save_lora_checkpoint(model, passed_args, save_dir, **kwargs):
        calls.append(
            {
                "model": model,
                "args": passed_args,
                "save_dir": save_dir,
                **kwargs,
            }
        )
        actual_adapter_dir.mkdir(parents=True)
        return str(actual_adapter_dir)

    monkeypatch.setattr(model_utils, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_utils, "get_args", lambda: args)
    monkeypatch.setattr(model_utils, "is_peft_model", lambda model: True)
    monkeypatch.setattr(checkpoint_utils, "is_peft_model", lambda model: True)
    monkeypatch.setattr(model_utils, "should_disable_forward_pre_hook", lambda passed_args: False)
    monkeypatch.setattr(lora_utils, "save_lora_checkpoint", save_lora_checkpoint)
    return args, calls


def test_native_peft_save_threads_teacher_to_actual_adapter_checkpoint(monkeypatch, tmp_path: Path) -> None:
    requested_adapter_dir = tmp_path / "native" / "iter_0000007" / "adapter"
    actual_adapter_dir = requested_adapter_dir / "resolved"
    _, calls = _install_native_peft_save(
        monkeypatch,
        tmp_path,
        actual_adapter_dir=actual_adapter_dir,
    )
    teacher = _teacher()
    model = [object()]
    optimizer = object()
    scheduler = object()

    model_utils.save(7, model, optimizer, scheduler, self_teacher=teacher)

    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["save_dir"] == str(requested_adapter_dir)
    assert calls[0]["optimizer"] is optimizer
    assert calls[0]["opt_param_scheduler"] is scheduler
    assert calls[0]["iteration"] == 7
    assert has_self_teacher_sidecar(actual_adapter_dir, rank=0)
    assert not has_self_teacher_sidecar(requested_adapter_dir, rank=0)

    restored = _teacher(9.0)
    load_self_teacher_sidecar(actual_adapter_dir, restored, rank=0, world_size=1)
    assert restored._step == teacher._step
    torch.testing.assert_close(restored.tensors[_ADAPTER_KEY], teacher.tensors[_ADAPTER_KEY])


def test_native_peft_save_without_teacher_writes_no_sidecar(monkeypatch, tmp_path: Path) -> None:
    actual_adapter_dir = tmp_path / "critic-adapter"
    _install_native_peft_save(
        monkeypatch,
        tmp_path,
        actual_adapter_dir=actual_adapter_dir,
    )

    model_utils.save(3, [object()], object(), object(), self_teacher=None)

    assert actual_adapter_dir.is_dir()
    assert not has_self_teacher_sidecar(actual_adapter_dir, rank=0)


def test_hf_peft_save_threads_teacher_to_actual_adapter_checkpoint(monkeypatch, tmp_path: Path) -> None:
    from megatron import bridge as megatron_bridge
    from orbit.utils import megatron_bridge_utils

    calls = []
    args = Namespace(
        hf_checkpoint="dummy-hf-checkpoint",
        peft_method="lora",
        save_hf=str(tmp_path / "hf-{rollout_id}"),
    )
    requested_adapter_dir = tmp_path / "hf-11" / "adapter"
    actual_adapter_dir = requested_adapter_dir / "resolved"

    class DummyBridge:
        @classmethod
        def from_hf_pretrained(cls, checkpoint, *, trust_remote_code):
            calls.append(("bridge", checkpoint, trust_remote_code))
            return cls()

        def save_hf_pretrained(self, model, *, path):
            calls.append(("merged", model, path))
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def patch_megatron_model(model):
        calls.append(("patch", model))
        yield

    def save_lora_checkpoint(model, passed_args, save_dir, **kwargs):
        calls.append(("adapter", model, passed_args, save_dir, kwargs))
        actual_adapter_dir.mkdir(parents=True)
        return str(actual_adapter_dir)

    parallel_state = SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0))
    monkeypatch.setattr(model_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(model_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_utils, "is_peft_model", lambda model: True)
    monkeypatch.setattr(megatron_bridge, "AutoBridge", DummyBridge)
    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", patch_megatron_model)
    monkeypatch.setattr(lora_utils, "save_lora_checkpoint", save_lora_checkpoint)
    model = [object()]
    teacher = _teacher()

    model_utils.save_hf_model(args, 11, model, self_teacher=teacher)

    assert ("bridge", args.hf_checkpoint, True) in calls
    assert ("merged", model, tmp_path / "hf-11") in calls
    adapter_call = next(call for call in calls if call[0] == "adapter")
    assert adapter_call[3] == str(requested_adapter_dir)
    assert has_self_teacher_sidecar(actual_adapter_dir, rank=0)
    assert not has_self_teacher_sidecar(requested_adapter_dir, rank=0)


def test_hf_peft_save_does_not_suppress_adapter_or_sidecar_failure(monkeypatch, tmp_path: Path) -> None:
    from megatron import bridge as megatron_bridge
    from orbit.utils import megatron_bridge_utils

    class DummyBridge:
        @classmethod
        def from_hf_pretrained(cls, checkpoint, *, trust_remote_code):
            return cls()

        def save_hf_pretrained(self, model, *, path):
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def patch_megatron_model(model):
        yield

    args = Namespace(
        hf_checkpoint="dummy-hf-checkpoint",
        peft_method="lora",
        save_hf=str(tmp_path / "hf-{rollout_id}"),
    )
    parallel_state = SimpleNamespace(intra_dp_cp=SimpleNamespace(rank=0))
    monkeypatch.setattr(model_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(model_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_utils, "is_peft_model", lambda model: True)
    monkeypatch.setattr(megatron_bridge, "AutoBridge", DummyBridge)
    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", patch_megatron_model)

    def _fail_adapter(*args, **kwargs):
        raise TeacherCheckpointError("sidecar incomplete")

    monkeypatch.setattr(model_utils, "save_peft_checkpoint", _fail_adapter)

    with pytest.raises(TeacherCheckpointError, match="sidecar incomplete"):
        model_utils.save_hf_model(args, 12, [object()], self_teacher=_teacher())

    # Adapter-only HF export was historically best-effort.  Keep that behavior
    # when no exact self-teacher state was requested.
    model_utils.save_hf_model(args, 13, [object()], self_teacher=None)


def test_actor_save_forwards_teacher_and_separate_critic_stays_teacher_free(monkeypatch) -> None:
    teacher = _teacher()
    calls = []

    def save(iteration, model, optimizer, scheduler, *, self_teacher):
        calls.append(("native", iteration, self_teacher))

    def save_hf_model(args, rollout_id, model, *, self_teacher):
        calls.append(("hf", rollout_id, self_teacher))

    monkeypatch.setattr(actor_utils, "save", save)
    monkeypatch.setattr(actor_utils, "uses_one_trunk_critic", lambda args: False)
    monkeypatch.setattr(model_utils, "save_hf_model", save_hf_model)

    def actor_for(role: str, *, with_teacher: bool):
        actor = object.__new__(actor_utils.MegatronTrainRayActor)
        actor.args = Namespace(
            async_save=False,
            critic_save=False,
            debug_rollout_only=False,
            offload_train=False,
            save_hf="checkpoint-{rollout_id}",
        )
        actor.role = role
        actor.model = [object()]
        actor.optimizer = object()
        actor.opt_param_scheduler = object()
        if with_teacher:
            actor._self_teacher = teacher
        return actor

    actor_utils.MegatronTrainRayActor.save_model.__wrapped__(actor_for("actor", with_teacher=True), 5)
    assert calls == [("native", 5, teacher), ("hf", 5, teacher)]

    calls.clear()
    actor_utils.MegatronTrainRayActor.save_model.__wrapped__(actor_for("critic", with_teacher=False), 6)
    assert calls == [("native", 6, None)]


def test_peft_save_propagates_self_teacher_sidecar_failure(monkeypatch, tmp_path: Path) -> None:
    from orbit.peft.opd import self_teacher_checkpoint

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    args = Namespace(peft_method="lora")
    monkeypatch.setattr(lora_utils, "save_lora_checkpoint", lambda *args, **kwargs: str(adapter_dir))

    def _fail_sidecar(*args, **kwargs):
        raise TeacherCheckpointError("disk failure")

    monkeypatch.setattr(self_teacher_checkpoint, "save_self_teacher_sidecar", _fail_sidecar)

    with pytest.raises(TeacherCheckpointError, match="rank 0"):
        peft_utils.save_peft_checkpoint([object()], args, str(adapter_dir), self_teacher=_teacher())


def test_self_teacher_restore_rejects_partial_rank_set_on_every_rank(monkeypatch, tmp_path: Path) -> None:
    actor = object.__new__(actor_utils.MegatronTrainRayActor)
    actor.args = Namespace(_peft_resume_adapter_dir=str(tmp_path))
    actor._self_teacher = _teacher()

    monkeypatch.setattr(actor_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(actor_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(actor_utils, "get_gloo_group", lambda: object())
    monkeypatch.setattr(
        "orbit.peft.opd.self_teacher_checkpoint.has_self_teacher_sidecar",
        lambda adapter_dir, *, rank: True,
    )

    def _gather_presence(output, value, *, group):
        if isinstance(value, tuple):
            output[:] = [(True, str(tmp_path)), (True, str(tmp_path))]
        else:
            output[:] = [True, False]

    monkeypatch.setattr(actor_utils.dist, "all_gather_object", _gather_presence)

    with pytest.raises(TeacherCheckpointError, match="partially present"):
        actor._restore_checkpoint_teacher_state()


def test_self_teacher_restore_rejects_partial_adapter_load_before_early_return(monkeypatch, tmp_path: Path) -> None:
    actor = object.__new__(actor_utils.MegatronTrainRayActor)
    actor.args = Namespace()
    actor._self_teacher = _teacher()

    monkeypatch.setattr(actor_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(actor_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(actor_utils, "get_gloo_group", lambda: object())

    def _gather_restore_state(output, value, *, group):
        assert value == (True, None)
        output[:] = [(True, None), (True, str(tmp_path))]

    monkeypatch.setattr(actor_utils.dist, "all_gather_object", _gather_restore_state)

    with pytest.raises(TeacherCheckpointError, match="missing adapter shards on ranks 0"):
        actor._restore_checkpoint_teacher_state()


def test_self_teacher_save_synchronizes_remote_rank_failure(monkeypatch, tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    args = Namespace(peft_method="lora")
    monkeypatch.setattr(lora_utils, "save_lora_checkpoint", lambda *args, **kwargs: str(adapter_dir))
    monkeypatch.setattr(peft_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(peft_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(peft_utils.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(peft_utils.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr("orbit.utils.distributed_utils.get_gloo_group", lambda: object())

    def _gather_errors(output, value, *, group):
        if group is None:
            output[:] = [value, value]
            return
        output[:] = [None, "TeacherCheckpointError: remote disk failure"]

    monkeypatch.setattr(peft_utils.dist, "all_gather_object", _gather_errors)

    with pytest.raises(TeacherCheckpointError, match="rank 1"):
        peft_utils.save_peft_checkpoint([object()], args, str(adapter_dir), self_teacher=_teacher())
