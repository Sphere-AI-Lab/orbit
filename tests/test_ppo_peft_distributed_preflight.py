import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import miles.orbit.megatron.peft_utils as peft_utils


class _AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(1))


class _CountingOptimizer:
    def __init__(self):
        self.load_state_calls = 0

    def load_state_dict(self, _state):
        self.load_state_calls += 1


class _DistributedLeaf:
    def __init__(self):
        self.is_stub_optimizer = False
        self.data_parallel_group = dist.group.WORLD
        self.data_parallel_group_gloo = dist.group.WORLD
        self.model_param = torch.nn.Parameter(torch.zeros(1))
        self.main_param = torch.nn.Parameter(torch.zeros(1))
        self.optimizer = SimpleNamespace(
            param_groups=[{"params": [self.main_param]}],
            state={
                self.main_param: {
                    "exp_avg": torch.zeros(1),
                    "exp_avg_sq": torch.zeros(1),
                }
            },
        )
        self.model_param_group_index_map = {self.model_param: (0, 0)}
        local_range = SimpleNamespace(start=0, end=1)
        self.gbuf_ranges = [
            {
                torch.float32: [
                    {
                        "param_map": {
                            self.model_param: {"gbuf_local": local_range},
                        }
                    }
                ]
            }
        ]
        self.buffers = [
            SimpleNamespace(
                numel_unpadded=2,
                buckets=[SimpleNamespace(grad_data=torch.zeros(2), numel_unpadded=2)],
            )
        ]
        self.load_state_calls = 0
        self.external_loads = []

    def load_state_dict(self, _state):
        self.load_state_calls += 1

    def get_parameter_state_dp_zero(self):
        return _external_parameter_state()

    def load_parameter_state(self, _filename):
        raise AssertionError("bound cached dispatch must bypass the filename loader")

    def load_parameter_state_from_dp_zero(self, state, *, update_legacy_format=False):
        assert update_legacy_format is False
        self.external_loads.append(state)

    def split_state_dict_if_needed(self, _state):
        return None


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


def _external_parameter_state() -> dict:
    return {
        "buckets_coalesced": True,
        0: {
            torch.float32: {
                "numel_unpadded": 2,
                "param": torch.ones(2),
                "exp_avg": torch.full((2,), 2.0),
                "exp_avg_sq": torch.full((2,), 3.0),
            }
        },
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
        from miles.utils import distributed_utils

        distributed_utils.GLOO_GROUP = None
        peft_utils.mpu.get_tensor_model_parallel_rank = lambda: rank
        peft_utils.mpu.get_pipeline_model_parallel_rank = lambda: 0
        for case in ("mixed_native", "mixed_sidecar", "corrupt_sidecar", "embedded_param_state"):
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
            native_shard_binding=None,
            training_state_binding=None,
            optimizer_parameter_state_binding=None,
        )
        try:
            peft_utils._validate_preflight_adapter_dir(common_dir, divergent_preflight)
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        else:
            outcome = "unexpected success"
        outcomes["divergent_preflight"] = f"optimizer_loads=0|{outcome}"
        dist.barrier()

        asymmetric_preflight = peft_utils.preflight_peft_adapter_checkpoint(
            adapter_dirs["rank_local_external"]
        )
        asymmetric_optimizer = _DistributedLeaf()
        restored_iteration = peft_utils.load_training_state(
            Path(adapter_dirs["rank_local_external"]),
            asymmetric_optimizer,
            None,
            checkpoint_preflight=asymmetric_preflight,
        )
        presence = "present" if asymmetric_preflight.optimizer_parameter_state_binding is not None else "absent"
        outcomes["rank_local_external"] = (
            f"{presence}|iteration={restored_iteration}|optimizer_loads={asymmetric_optimizer.load_state_calls}"
            f"|external_loads={len(asymmetric_optimizer.external_loads)}"
        )
        dist.barrier()

        replacement_dir = Path(adapter_dirs["rank_divergent_replacement"])
        replacement_preflight = peft_utils.preflight_peft_adapter_checkpoint(replacement_dir)
        dist.barrier()
        if rank == 0:
            replacement_path = replacement_dir / "training_state_rank0.replacement.pt"
            torch.save(_training_state(), replacement_path)
            replacement_path.replace(replacement_dir / "training_state_rank0.pt")
        dist.barrier()
        optimizer = _CountingOptimizer()
        try:
            peft_utils.load_training_state(
                replacement_dir,
                optimizer,
                None,
                checkpoint_preflight=replacement_preflight,
            )
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        else:
            outcome = "unexpected success"
        outcomes["rank_divergent_replacement"] = f"optimizer_loads={optimizer.load_state_calls}|{outcome}"
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
        "embedded_param_state": root / "embedded-param-state",
        "rank_local_external": root / "rank-local-external",
        "rank_divergent_replacement": root / "rank-divergent-replacement",
    }

    _save_native_shards(adapter_dirs["mixed_native"], (0,))

    mixed_sidecar_dir = adapter_dirs["mixed_sidecar"]
    _save_native_shards(mixed_sidecar_dir, (0, 1))
    torch.save(_training_state(), mixed_sidecar_dir / "training_state_rank0.pt")

    corrupt_sidecar_dir = adapter_dirs["corrupt_sidecar"]
    _save_native_shards(corrupt_sidecar_dir, (0, 1))
    torch.save(_training_state(), corrupt_sidecar_dir / "training_state_rank0.pt")
    (corrupt_sidecar_dir / "training_state_rank1.pt").write_bytes(b"not a torch checkpoint")

    embedded_param_state_dir = adapter_dirs["embedded_param_state"]
    _save_native_shards(embedded_param_state_dir, (0, 1))
    torch.save(_training_state(), embedded_param_state_dir / "training_state_rank0.pt")
    foreign_state = _training_state()
    foreign_state["optimizer"] = {
        "optimizer": {"param_groups": []},
        "nested": [{"param_state": {"rank": 1}, "param_state_sharding_type": "dp_zero_gather_scatter"}],
    }
    torch.save(foreign_state, embedded_param_state_dir / "training_state_rank1.pt")

    for rank in range(2):
        divergent_dir = root / f"divergent-path-rank{rank}"
        _save_native_shards(divergent_dir, (rank,))
        torch.save(_training_state(), divergent_dir / f"training_state_rank{rank}.pt")
        adapter_dirs[f"divergent_path_rank{rank}"] = divergent_dir
    divergent_preflight_dir = root / "divergent-preflight"
    divergent_preflight_dir.mkdir()
    adapter_dirs["divergent_preflight"] = divergent_preflight_dir

    rank_local_external_dir = adapter_dirs["rank_local_external"]
    rank_local_external_dir.mkdir()
    for rank in range(2):
        state = _training_state()
        state["optimizer_parameter_state"] = True
        torch.save(state, rank_local_external_dir / f"training_state_rank{rank}.pt")
    torch.save(_external_parameter_state(), rank_local_external_dir / "optimizer_parameter_state_rank0.pt")

    rank_divergent_replacement_dir = adapter_dirs["rank_divergent_replacement"]
    rank_divergent_replacement_dir.mkdir()
    for rank in range(2):
        torch.save(_training_state(), rank_divergent_replacement_dir / f"training_state_rank{rank}.pt")

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
    cases = (
        "mixed_native",
        "mixed_sidecar",
        "corrupt_sidecar",
        "embedded_param_state",
        "divergent_path",
        "divergent_preflight",
        "rank_local_external",
        "rank_divergent_replacement",
    )
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


def test_two_process_embedded_parameter_state_fails_before_optimizer_mutation(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["embedded_param_state"], "embedded distributed parameter state")


def test_two_process_rank_divergent_adapter_paths_fail_together(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["divergent_path"], "adapter paths differ across ranks")


def test_two_process_rank_divergent_preflight_flags_fail_together(two_process_outcomes):
    _assert_coordinated_failure(two_process_outcomes["divergent_preflight"], "preflight binding differs across ranks")


def test_two_process_rank_local_external_presence_is_not_required_equal(two_process_outcomes):
    assert two_process_outcomes["rank_local_external"] == [
        "present|iteration=3|optimizer_loads=1|external_loads=1",
        "absent|iteration=3|optimizer_loads=1|external_loads=1",
    ]


def test_two_process_rank_divergent_replacement_fails_together_before_mutation(two_process_outcomes):
    _assert_coordinated_failure(
        two_process_outcomes["rank_divergent_replacement"],
        "checkpoint file changed after preflight",
    )
