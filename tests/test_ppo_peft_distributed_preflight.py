import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import orbit.backends.megatron_utils.peft_utils as peft_utils


class _AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(1))


class _CountingOptimizer:
    def __init__(self):
        self.load_state_calls = 0

    def load_state_dict(self, _state):
        self.load_state_calls += 1


def _native_state() -> dict[tuple[int, str], torch.Tensor]:
    return {(0, "lora_A"): torch.ones(1)}


def _training_state() -> dict:
    return {
        "iteration": 3,
        "active_student_version": None,
        "optimizer": {"param_groups": []},
        "optimizer_parameter_state": False,
        "opt_param_scheduler": None,
    }


def _distributed_load_worker(
    rank: int,
    world_size: int,
    init_file: str,
    adapter_dirs: dict[str, str],
    result_dir: str,
) -> None:
    outcomes = {}
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=10),
        )
        from orbit.utils import distributed_utils

        distributed_utils.GLOO_GROUP = None
        peft_utils.mpu.get_tensor_model_parallel_rank = lambda: rank
        peft_utils.mpu.get_pipeline_model_parallel_rank = lambda: 0
        for case in ("mixed_native", "mixed_sidecar", "corrupt_sidecar"):
            adapter_dir = adapter_dirs[case]
            optimizer = _CountingOptimizer()
            try:
                peft_utils.load_peft_adapter(
                    [_AdapterModel()],
                    SimpleNamespace(peft_method="lora"),
                    adapter_dir,
                    optimizer=optimizer,
                )
            except Exception as exc:
                outcome = f"{type(exc).__name__}: {exc}"
            else:
                outcome = "unexpected success"
            outcomes[case] = f"optimizer_loads={optimizer.load_state_calls}|{outcome}"
            dist.barrier()

        try:
            peft_utils.preflight_peft_adapter_checkpoint(adapter_dirs[f"divergent_path_rank{rank}"])
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        else:
            outcome = "unexpected success"
        outcomes["divergent_path"] = f"optimizer_loads=0|{outcome}"
        dist.barrier()

        common_dir = adapter_dirs["divergent_preflight"]
        divergent_preflight = peft_utils.PeftCheckpointPreflight(
            adapter_dir=common_dir,
            native_shards_present=rank == 0,
            training_state_present=False,
        )
        try:
            peft_utils._validate_preflight_adapter_dir(common_dir, divergent_preflight)
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        else:
            outcome = "unexpected success"
        outcomes["divergent_preflight"] = f"optimizer_loads=0|{outcome}"
        dist.barrier()
    except Exception as exc:
        outcomes["worker_failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(outcomes, sort_keys=True))


def _save_native_shards(adapter_dir: Path, ranks: tuple[int, ...]) -> None:
    adapter_dir.mkdir()
    for rank in ranks:
        torch.save(_native_state(), adapter_dir / f"adapter_megatron_tp{rank}_pp0.pt")


@pytest.fixture(scope="module")
def two_process_outcomes(tmp_path_factory) -> dict[str, list[str]]:
    root = tmp_path_factory.mktemp("peft-distributed-preflight")
    adapter_dirs = {
        "mixed_native": root / "mixed-native",
        "mixed_sidecar": root / "mixed-sidecar",
        "corrupt_sidecar": root / "corrupt-sidecar",
    }

    _save_native_shards(adapter_dirs["mixed_native"], (0,))

    mixed_sidecar_dir = adapter_dirs["mixed_sidecar"]
    _save_native_shards(mixed_sidecar_dir, (0, 1))
    torch.save(_training_state(), mixed_sidecar_dir / "training_state_rank0.pt")

    corrupt_sidecar_dir = adapter_dirs["corrupt_sidecar"]
    _save_native_shards(corrupt_sidecar_dir, (0, 1))
    torch.save(_training_state(), corrupt_sidecar_dir / "training_state_rank0.pt")
    (corrupt_sidecar_dir / "training_state_rank1.pt").write_bytes(b"not a torch checkpoint")

    for rank in range(2):
        divergent_dir = root / f"divergent-path-rank{rank}"
        _save_native_shards(divergent_dir, (rank,))
        torch.save(_training_state(), divergent_dir / f"training_state_rank{rank}.pt")
        adapter_dirs[f"divergent_path_rank{rank}"] = divergent_dir
    divergent_preflight_dir = root / "divergent-preflight"
    divergent_preflight_dir.mkdir()
    adapter_dirs["divergent_preflight"] = divergent_preflight_dir

    result_dir = root / "results"
    result_dir.mkdir()
    mp.start_processes(
        _distributed_load_worker,
        args=(
            2,
            str(root / "gloo-init"),
            {case: str(path) for case, path in adapter_dirs.items()},
            str(result_dir),
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    rank_outcomes = [json.loads((result_dir / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert all("worker_failure" not in outcomes for outcomes in rank_outcomes)
    cases = ("mixed_native", "mixed_sidecar", "corrupt_sidecar", "divergent_path", "divergent_preflight")
    return {case: [outcomes[case] for outcomes in rank_outcomes] for case in cases}


def _assert_coordinated_failure(outcomes: list[str], expected_error: str) -> None:
    assert len(set(outcomes)) == 1
    assert all(outcome.startswith("optimizer_loads=0|RuntimeError:") for outcome in outcomes)
    assert all(expected_error in outcome for outcome in outcomes)


def test_two_process_mixed_native_shards_fail_together(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["mixed_native"], "native adapter shards")


def test_two_process_mixed_training_sidecars_fail_together(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["mixed_sidecar"], "training-state sidecars")


def test_two_process_sidecar_parse_failure_precedes_optimizer_load(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["corrupt_sidecar"], "training-state parse/validation")


def test_two_process_rank_divergent_adapter_paths_fail_together(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["divergent_path"], "adapter paths differ across ranks")


def test_two_process_rank_divergent_preflight_flags_fail_together(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["divergent_preflight"], "preflight binding differs across ranks")
