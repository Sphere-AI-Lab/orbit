import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import orbit.backends.megatron_utils.peft_utils as peft_utils


class _Scheduler:
    def state_dict(self):
        return {"num_steps": 0}


class _DistributedSaveLeaf:
    def __init__(self, rank: int, world_size: int):
        width = 2
        self.is_stub_optimizer = False
        self.data_parallel_group = dist.group.WORLD
        self.data_parallel_group_gloo = dist.group.WORLD
        self.model_param = torch.nn.Parameter(torch.zeros(width))
        self.main_param = torch.nn.Parameter(torch.ones(width))
        exp_avg_width = width if rank == 0 else width + 1
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
        self.save_parameter_state_calls = 0

    def state_dict(self):
        return {"optimizer": {"param_groups": [{"step": 0}]}}

    def _get_main_param_and_optimizer_states(self, _model_param):
        state = self.optimizer.state[self.main_param]
        return {
            "param": self.main_param,
            "exp_avg": state["exp_avg"],
            "exp_avg_sq": state["exp_avg_sq"],
        }

    def get_parameter_state_dp_zero(self):
        raise AssertionError("source validation must precede parameter-state collection")

    def load_parameter_state_from_dp_zero(self, _state, *, update_legacy_format=False):
        raise AssertionError("not used by save preflight")

    def save_parameter_state(self, _filename):
        self.save_parameter_state_calls += 1
        raise AssertionError("source validation must precede save_parameter_state")


def _distributed_save_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    result_dir: str,
) -> None:
    outcome = "unexpected success"
    save_calls = -1
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
        optimizer = _DistributedSaveLeaf(rank, world_size)
        try:
            peft_utils.save_training_state(
                Path(checkpoint_dir),
                optimizer,
                _Scheduler(),
                iteration=0,
            )
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        save_calls = optimizer.save_parameter_state_calls
    except Exception as exc:
        outcome = f"worker failure: {type(exc).__name__}: {exc}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps({"outcome": outcome, "save_calls": save_calls}, sort_keys=True)
        )


def test_two_process_rank_divergent_source_fails_together_before_parameter_state_save(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.start_processes(
        _distributed_save_worker,
        args=(
            2,
            str(tmp_path / "gloo-init"),
            str(checkpoint_dir),
            str(result_dir),
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    results = [json.loads((result_dir / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert [result["save_calls"] for result in results] == [0, 0]
    outcomes = [result["outcome"] for result in results]
    assert len(set(outcomes)) == 1
    assert outcomes[0].startswith("RuntimeError: PEFT distributed optimizer source validation failed")
    assert "rank 1" in outcomes[0]
    assert "source 'exp_avg' is incompatible" in outcomes[0]
    assert not list(checkpoint_dir.glob("optimizer_parameter_state_rank*.pt"))
