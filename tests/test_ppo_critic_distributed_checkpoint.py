import argparse
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from orbit.peft.critic.critic_adapter import (
    load_critic_checkpoint,
    save_critic_checkpoint,
)


class _Chunk(torch.nn.Module):
    def __init__(self, width: int = 2):
        super().__init__()
        self.trunk = torch.nn.Parameter(torch.zeros(width), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.zeros(width))


class _Scheduler:
    def __init__(self):
        self.load_calls = 0

    def state_dict(self):
        return {"num_steps": 3}

    def load_state_dict(self, _state):
        self.load_calls += 1


class _DistributedLeaf:
    """Pinned-Megatron-shaped optimizer with inspectable save/load counters."""

    def __init__(self, rank: int, world_size: int, *, malformed_source: bool = False):
        width = 2
        self.is_stub_optimizer = False
        self.data_parallel_group = dist.group.WORLD
        self.data_parallel_group_gloo = dist.group.WORLD
        self.model_param = torch.nn.Parameter(torch.zeros(width))
        self.main_param = torch.nn.Parameter(torch.ones(width))
        exp_avg_width = width + 1 if malformed_source and rank == 1 else width
        self.optimizer = SimpleNamespace(
            param_groups=[{"params": [self.main_param]}],
            state={
                self.main_param: {
                    "exp_avg": torch.zeros(exp_avg_width),
                    "exp_avg_sq": torch.zeros(width),
                }
            },
        )
        self.config = SimpleNamespace()
        self.init_state_fn = lambda _optimizer, _config: None
        self.model_param_group_index_map = {self.model_param: (0, 0)}
        self.gbuf_ranges = [
            {
                torch.float32: [
                    {
                        "param_map": {
                            self.model_param: {
                                "gbuf_local": SimpleNamespace(start=0, end=width),
                            }
                        }
                    }
                ]
            }
        ]
        padded_width = width * world_size
        self.buffers = [
            SimpleNamespace(
                numel_unpadded=padded_width,
                buckets=[
                    SimpleNamespace(
                        grad_data=torch.zeros(padded_width),
                        numel_unpadded=padded_width,
                    )
                ],
            )
        ]
        self.state_dict_calls = 0
        self.reload_calls = 0
        self.load_state_calls = 0
        self.save_parameter_state_calls = 0
        self.filename_load_calls = 0
        self.dispatch_calls = 0

    def state_dict(self):
        self.state_dict_calls += 1
        return {"optimizer": {"param_groups": [{"step": 3}]}}

    def load_state_dict(self, _state):
        self.load_state_calls += 1

    def reload_model_params(self):
        self.reload_calls += 1

    def _get_main_param_and_optimizer_states(self, _model_param):
        state = self.optimizer.state[self.main_param]
        return {
            "param": self.main_param,
            "exp_avg": state["exp_avg"],
            "exp_avg_sq": state["exp_avg_sq"],
        }

    def get_parameter_state_dp_zero(self):
        raise AssertionError("save source validation must run before parameter-state collection")

    def save_parameter_state(self, _filename):
        self.save_parameter_state_calls += 1
        raise AssertionError("save source validation must run before parameter-state collection")

    def load_parameter_state(self, _filename):
        self.filename_load_calls += 1
        raise AssertionError("pinned distributed resume must dispatch cached state directly")

    def load_parameter_state_from_dp_zero(self, _state, *, update_legacy_format=False):
        assert update_legacy_format is False
        self.dispatch_calls += 1

    def split_state_dict_if_needed(self, _state):
        return None


def _valid_external_state(world_size: int) -> dict:
    width = 2 * world_size
    return {
        "buckets_coalesced": True,
        0: {
            torch.float32: {
                "numel_unpadded": width,
                "param": torch.ones(width),
                "exp_avg": torch.full((width,), 2.0),
                "exp_avg_sq": torch.full((width,), 3.0),
            }
        },
    }


def _critic_payload(*, embedded_parameter_state: bool = False) -> dict:
    optimizer_state = {"optimizer": {"param_groups": [{"step": 3}]}}
    if embedded_parameter_state:
        optimizer_state["nested"] = [
            {
                "param_state": {"rank": 1},
                "param_state_sharding_type": "dp_zero_gather_scatter",
            }
        ]
    return {
        "tensors": {"0:adapter": torch.ones(2)},
        "optimizer": optimizer_state,
        "optimizer_parameter_state": True,
        "opt_param_scheduler": {"num_steps": 3},
        "iteration": 3,
    }


def _prepare_load_case(root: Path, case: str, world_size: int) -> None:
    checkpoint_dir = root / case / "iter_0000003"
    checkpoint_dir.mkdir(parents=True)
    (root / case / "latest_checkpointed_iteration.txt").write_text("3")
    for rank in range(world_size):
        torch.save(
            _critic_payload(embedded_parameter_state=case == "embedded" and rank == 1),
            checkpoint_dir / f"critic_rank{rank}.pt",
        )
    parameter_state_path = checkpoint_dir / "optimizer_parameter_state_rank0.pt"
    if case == "corrupt":
        parameter_state_path.write_bytes(b"not a torch checkpoint")
    elif case in ("embedded", "valid"):
        torch.save(_valid_external_state(world_size), parameter_state_path)


def _worker(rank: int, world_size: int, init_file: str, root: str, result_dir: str) -> None:
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

        save_optimizer = _DistributedLeaf(rank, world_size, malformed_source=True)
        try:
            save_critic_checkpoint(
                argparse.Namespace(
                    critic_save=str(Path(root) / "save"),
                    no_save_optim=False,
                ),
                0,
                [_Chunk()],
                optimizer=save_optimizer,
                opt_param_scheduler=_Scheduler(),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = "unexpected success"
        outcomes["save"] = {
            "error": error,
            "parameter_state_calls": save_optimizer.save_parameter_state_calls,
        }
        dist.barrier()

        for case in ("missing", "corrupt", "embedded", "valid"):
            model = [_Chunk()]
            before = model[0].adapter.detach().clone()
            optimizer = _DistributedLeaf(rank, world_size)
            scheduler = _Scheduler()
            try:
                loaded_iteration = load_critic_checkpoint(
                    argparse.Namespace(critic_load=str(Path(root) / case)),
                    model,
                    optimizer=optimizer,
                    opt_param_scheduler=scheduler,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                error = "success"
            outcomes[case] = {
                "error": error,
                "loaded_iteration": loaded_iteration if error == "success" else None,
                "model_unchanged": torch.equal(model[0].adapter, before),
                "reload_calls": optimizer.reload_calls,
                "optimizer_load_calls": optimizer.load_state_calls,
                "filename_load_calls": optimizer.filename_load_calls,
                "dispatch_calls": optimizer.dispatch_calls,
                "scheduler_load_calls": scheduler.load_calls,
            }
            dist.barrier()
    except Exception as exc:
        outcomes["worker_failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(outcomes, sort_keys=True))


def test_two_process_critic_checkpoint_preflights_fail_together_before_mutation(tmp_path):
    world_size = 2
    for case in ("missing", "corrupt", "embedded", "valid"):
        _prepare_load_case(tmp_path, case, world_size)
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    mp.start_processes(
        _worker,
        args=(
            world_size,
            str(tmp_path / "gloo-init"),
            str(tmp_path),
            str(result_dir),
        ),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )

    results = [json.loads((result_dir / f"rank{rank}.json").read_text()) for rank in range(world_size)]
    assert all("worker_failure" not in result for result in results)

    save_outcomes = [result["save"] for result in results]
    assert [outcome["parameter_state_calls"] for outcome in save_outcomes] == [0, 0]
    assert len({outcome["error"] for outcome in save_outcomes}) == 1
    assert "adapter critic distributed optimizer source validation failed" in save_outcomes[0]["error"]
    assert "rank 1" in save_outcomes[0]["error"]
    assert not (tmp_path / "save" / "latest_checkpointed_iteration.txt").exists()
    assert not list((tmp_path / "save").glob("**/critic_rank*.pt"))

    expected_errors = {
        "missing": "critic optimizer parameter state is missing",
        "corrupt": "optimizer parameter-state preflight failed",
        "embedded": "embedded distributed parameter state",
    }
    for case, expected_error in expected_errors.items():
        outcomes = [result[case] for result in results]
        assert len({outcome["error"] for outcome in outcomes}) == 1
        assert expected_error in outcomes[0]["error"]
        for outcome in outcomes:
            assert outcome["model_unchanged"] is True
            assert outcome["reload_calls"] == 0
            assert outcome["optimizer_load_calls"] == 0
            assert outcome["filename_load_calls"] == 0
            assert outcome["dispatch_calls"] == 0
            assert outcome["scheduler_load_calls"] == 0

    valid_outcomes = [result["valid"] for result in results]
    assert [outcome["error"] for outcome in valid_outcomes] == ["success", "success"]
    for outcome in valid_outcomes:
        assert outcome["loaded_iteration"] == 3
        assert outcome["model_unchanged"] is False
        assert outcome["reload_calls"] == 1
        assert outcome["optimizer_load_calls"] == 1
        assert outcome["filename_load_calls"] == 0
        assert outcome["dispatch_calls"] == 1
        assert outcome["scheduler_load_calls"] == 1
    valid_checkpoint_dir = tmp_path / "valid" / "iter_0000003"
    assert (valid_checkpoint_dir / "optimizer_parameter_state_rank0.pt").is_file()
    assert not (valid_checkpoint_dir / "optimizer_parameter_state_rank1.pt").exists()
