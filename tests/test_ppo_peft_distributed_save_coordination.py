import argparse
import json
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import orbit.peft.megatron.peft_utils as peft_utils


class _InlineOptimizer:
    def __init__(self):
        self.state_dict_calls = 0

    def state_dict(self):
        self.state_dict_calls += 1
        return {
            "state": {0: {"moment": torch.zeros(1)}},
            "param_groups": [],
        }


class _ExternalOptimizer(_InlineOptimizer):
    is_stub_optimizer = False
    data_parallel_group = object()

    def __init__(self, *, drop_layout_after_state_dict: bool):
        super().__init__()
        self.drop_layout_after_state_dict = drop_layout_after_state_dict

    def state_dict(self):
        state = super().state_dict()
        if self.drop_layout_after_state_dict:
            self.data_parallel_group = None
        return state

    def get_parameter_state_dp_zero(self):
        raise AssertionError("layout consensus must precede parameter-state collection")

    def load_parameter_state_from_dp_zero(self, _state):
        raise AssertionError("not used by save")

    def save_parameter_state(self, _path):
        raise AssertionError("layout consensus must precede parameter-state save")


class _Scheduler:
    def state_dict(self):
        return {"num_steps": 0}


def _distributed_adapter_save_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_root: str,
    result_dir: str,
) -> None:
    outcome = "worker did not initialize"
    case_results = {}
    counts = {}
    failure_phase = "setup"
    original_torch_save = torch.save
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=10),
        )
        import megatron.bridge as bridge_module

        from orbit.utils import distributed_utils, megatron_bridge_utils

        distributed_utils.GLOO_GROUP = None
        peft_utils.get_parallel_state = lambda: SimpleNamespace(
            # Both CP replicas are rank zero when CP is excluded from DP.
            intra_dp=SimpleNamespace(rank=0, size=1),
            # The combined group has one writer for their shared TP/PP shard.
            intra_dp_cp=SimpleNamespace(rank=rank, size=world_size),
            cp=SimpleNamespace(rank=rank, size=world_size),
        )
        peft_utils.mpu.get_tensor_model_parallel_rank = lambda: 0
        peft_utils.mpu.get_pipeline_model_parallel_rank = lambda: 0
        peft_utils.native_adapter_state = lambda _model: {(0, "lora_A"): torch.ones(1)}
        megatron_bridge_utils.patch_megatron_model = lambda _model: nullcontext()

        class _Bridge:
            @classmethod
            def from_hf_pretrained(cls, *_args, **_kwargs):
                return cls()

            def export_adapter_weights(self, *_args, **_kwargs):
                counts["export"] += 1
                if failure_phase == "export" and rank == 1:
                    raise OSError("injected rank-local export failure")
                return ()

        bridge_module.AutoBridge = _Bridge

        def tracked_torch_save(value, path, *args, **kwargs):
            if Path(path).name.startswith("adapter_megatron_tp"):
                counts["native"] += 1
                if failure_phase == "native" and rank == 0:
                    raise OSError("injected native shard write failure")
            return original_torch_save(value, path, *args, **kwargs)

        peft_utils.torch.save = tracked_torch_save

        def save_hf_artifacts(*_args, **_kwargs):
            counts["hf"] += 1
            if failure_phase == "hf" and rank == 0:
                raise OSError("injected HF artifact write failure")

        peft_utils._save_peft_hf_artifacts = save_hf_artifacts
        peft_utils.prepare_distributed_optimizer_state_for_save = lambda _optimizer: None
        for failure_phase in (
            "mkdir",
            "native",
            "export",
            "hf",
            "path",
            "method",
            "iteration",
            "version",
            "mode",
            "optimizer",
            "stub",
            "scheduler",
            "hf_checkpoint",
            "layout",
            "dispatch_method",
            "teacher",
            "success",
        ):
            counts = {"native": 0, "export": 0, "hf": 0, "optimizer": 0}
            optimizer = _InlineOptimizer()
            optimizer_arg = optimizer
            scheduler_arg = _Scheduler()
            local_checkpoint_path = str(Path(checkpoint_root, failure_phase))
            local_method = "lora"
            local_iteration = 0
            local_version = None
            local_no_save_optim = False
            local_hf_checkpoint = "base"
            if rank == 1:
                if failure_phase == "path":
                    local_checkpoint_path += "-rank1"
                elif failure_phase == "method":
                    local_method = "oft"
                elif failure_phase == "iteration":
                    local_iteration = 1
                elif failure_phase == "version":
                    local_version = "1"
                elif failure_phase == "mode":
                    local_no_save_optim = True
                elif failure_phase == "optimizer":
                    optimizer_arg = None
                elif failure_phase == "stub":
                    optimizer.is_stub_optimizer = True
                elif failure_phase == "scheduler":
                    scheduler_arg = None
                elif failure_phase == "hf_checkpoint":
                    local_hf_checkpoint = "other-base"
            if failure_phase == "layout" and rank == 0:
                optimizer_arg = _ExternalOptimizer(drop_layout_after_state_dict=True)
            elif failure_phase == "layout":
                optimizer_arg = _ExternalOptimizer(drop_layout_after_state_dict=False)

            try:
                args = argparse.Namespace(
                    hf_checkpoint=local_hf_checkpoint,
                    no_save_optim=local_no_save_optim,
                    peft_method="oft" if failure_phase == "dispatch_method" and rank == 1 else "lora",
                )
                if failure_phase in ("dispatch_method", "teacher"):
                    peft_utils.save_peft_checkpoint(
                        [object()],
                        args,
                        local_checkpoint_path,
                        optimizer=optimizer_arg,
                        opt_param_scheduler=scheduler_arg,
                        iteration=local_iteration,
                        active_student_version=local_version,
                        self_teacher=object() if failure_phase == "teacher" and rank == 1 else None,
                    )
                else:
                    peft_utils.save_peft_adapter_checkpoint(
                        [object()],
                        args,
                        local_checkpoint_path,
                        method=local_method,
                        build_config=dict,
                        optimizer=optimizer_arg,
                        opt_param_scheduler=scheduler_arg,
                        iteration=local_iteration,
                        active_student_version=local_version,
                    )
            except Exception as exc:
                outcome = f"{type(exc).__name__}: {exc}"
            else:
                outcome = "success"
            counts["optimizer"] = optimizer_arg.state_dict_calls if optimizer_arg is not None else 0
            case_results[failure_phase] = {"counts": counts, "outcome": outcome}
            dist.barrier()
    except Exception as exc:
        case_results["worker_failure"] = {
            "counts": counts,
            "outcome": f"{failure_phase}: {type(exc).__name__}: {exc}",
        }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(case_results, sort_keys=True))


@pytest.fixture(scope="module")
def distributed_adapter_saves(tmp_path_factory):
    root = tmp_path_factory.mktemp("peft-distributed-save-coordination")
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "mkdir").write_text("directory creation must fail")
    result_dir = root / "results"
    result_dir.mkdir()
    mp.start_processes(
        _distributed_adapter_save_worker,
        args=(
            2,
            str(root / "gloo-init"),
            str(checkpoint_root),
            str(result_dir),
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    rank_results = [json.loads((result_dir / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert all("worker_failure" not in result for result in rank_results)
    return checkpoint_root, {
        case: [rank_results[rank][case] for rank in range(2)] for case in rank_results[0]
    }


@pytest.mark.parametrize(
    ("failure_phase", "expected_label"),
    [
        ("mkdir", "PEFT checkpoint directory creation failed"),
        ("native", "PEFT native adapter shard save failed"),
        ("export", "PEFT HF adapter export failed"),
        ("hf", "PEFT HF adapter artifact save failed"),
        ("path", "PEFT save request differs across ranks"),
        ("method", "PEFT save request differs across ranks"),
        ("iteration", "PEFT save request differs across ranks"),
        ("version", "PEFT save request differs across ranks"),
        ("mode", "PEFT save request differs across ranks"),
        ("optimizer", "PEFT save request differs across ranks"),
        ("stub", "PEFT save request differs across ranks"),
        ("scheduler", "PEFT save request differs across ranks"),
        ("hf_checkpoint", "PEFT save request differs across ranks"),
        ("layout", "PEFT external optimizer layout differs across ranks"),
        ("dispatch_method", "PEFT save dispatch differs across ranks"),
        ("teacher", "PEFT save dispatch differs across ranks"),
    ],
)
def test_rank_local_adapter_save_failure_is_reported_to_every_rank(
    distributed_adapter_saves,
    failure_phase,
    expected_label,
):
    _, results_by_case = distributed_adapter_saves
    results = results_by_case[failure_phase]

    outcomes = [result["outcome"] for result in results]
    assert len(set(outcomes)) == 1
    assert outcomes[0].startswith(f"RuntimeError: {expected_label}")
    expected_optimizer_calls = [1, 1] if failure_phase == "layout" else [0, 0]
    assert [result["counts"]["optimizer"] for result in results] == expected_optimizer_calls
    if expected_label in (
        "PEFT save request differs across ranks",
        "PEFT save dispatch differs across ranks",
    ):
        assert [result["counts"]["native"] for result in results] == [0, 0]
        assert [result["counts"]["export"] for result in results] == [0, 0]
        assert [result["counts"]["hf"] for result in results] == [0, 0]


def test_context_parallel_replicas_have_one_native_writer_but_all_save_training_state(
    distributed_adapter_saves,
):
    checkpoint_root, results_by_case = distributed_adapter_saves
    results = results_by_case["success"]

    assert [result["outcome"] for result in results] == ["success", "success"]
    assert [result["counts"]["native"] for result in results] == [1, 0]
    assert [result["counts"]["export"] for result in results] == [1, 1]
    assert [result["counts"]["hf"] for result in results] == [1, 0]
    assert [result["counts"]["optimizer"] for result in results] == [1, 1]
    checkpoint_path = checkpoint_root / "success"
    assert (checkpoint_path / "adapter_megatron_tp0_pp0.pt").is_file()
    assert (checkpoint_path / "training_state_rank0.pt").is_file()
    assert (checkpoint_path / "training_state_rank1.pt").is_file()
