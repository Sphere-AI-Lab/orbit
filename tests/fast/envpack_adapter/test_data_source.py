from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from miles_plugins.envpack_adapter.config import EnvpackConfigError
    from miles_plugins.envpack_adapter.data_source import EnvpackDataSource


class EnvpackDataSourceTest(unittest.TestCase):
    def make_args(
        self,
        prompt_data: str,
        pool_env_config: dict | None = None,
        curriculum: dict | None = None,
    ):
        return SimpleNamespace(
            prompt_data=prompt_data,
            envpack={
                "api": "in_process",
                "pools": [
                    {
                        "env": "sokoban",
                        "profile": "vision_free_think_local",
                        "pool_id": "sokoban-vision",
                        "env_config": pool_env_config or {"sokoban_render_style": "sprite"},
                    }
                ],
                **({"curriculum": curriculum} if curriculum is not None else {}),
            },
            n_samples_per_prompt=2,
            rollout_shuffle=False,
            rollout_seed=0,
            seed=0,
            save="/tmp/envpack-adapter-test",
            load=None,
        )

    def test_envspec_yaml_materializes_envpack_metadata_and_groups(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sokoban.yaml"
            path.write_text(
                json.dumps(
                    {
                        "envs": [
                            {
                                "name": "Sokoban",
                                "n_envs": 2,
                                "seed": [1, 10, 1],
                                "pool_id": "sokoban-vision",
                                "config": {"render_mode": "vision", "prompt_format": "free_think"},
                            }
                        ]
                    }
                )
            )
            data_source = EnvpackDataSource(self.make_args(str(path)))
            groups = data_source.get_samples(2)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        meta = groups[0][0].metadata["envpack"]
        self.assertEqual(meta["env_name"], "sokoban")
        self.assertEqual(meta["pool_id"], "sokoban-vision")
        self.assertEqual(meta["env_config"]["render_mode"], "vision")
        self.assertEqual(meta["env_config"]["prompt_format"], "free_think")
        self.assertNotIn("max_turns", meta)
        self.assertNotIn("response_length_per_turn", meta)
        self.assertEqual(groups[0][0].group_index, 0)
        self.assertEqual(groups[0][1].group_index, 0)
        self.assertEqual(groups[1][0].group_index, 1)

    def test_envspec_yaml_rejects_structural_pool_env_config_without_baked_uuid(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sokoban.yaml"
            path.write_text(
                json.dumps(
                    {
                        "envs": [
                            {
                                "name": "Sokoban",
                                "n_envs": 1,
                                "seed": [1],
                                "pool_id": "sokoban-vision",
                                "config": {"prompt_format": "free_think"},
                            }
                        ]
                    }
                )
            )
            args = self.make_args(str(path), pool_env_config={"dim_room": [7, 7]})

            with self.assertRaisesRegex(EnvpackConfigError, "structural pool.env_config"):
                EnvpackDataSource(args)

    def test_envspec_yaml_rejects_render_mode_pool_env_config_without_baked_uuid(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sokoban.yaml"
            path.write_text(
                json.dumps(
                    {
                        "envs": [
                            {
                                "name": "Sokoban",
                                "n_envs": 1,
                                "seed": [1],
                                "pool_id": "sokoban-vision",
                                "config": {"render_mode": "vision", "prompt_format": "free_think"},
                            }
                        ]
                    }
                )
            )
            args = self.make_args(str(path), pool_env_config={"render_mode": "text"})

            with self.assertRaisesRegex(EnvpackConfigError, "render_mode"):
                EnvpackDataSource(args)

    def test_runtime_guard_runs_during_data_source_init(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        args = self.make_args("/tmp/does-not-need-to-exist.jsonl")
        args.use_rollout_routing_replay = True
        with self.assertRaisesRegex(EnvpackConfigError, "R3"):
            EnvpackDataSource(args)

    def test_load_reapplies_shuffle_for_restored_epoch(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "samples.jsonl"
            rows = [
                {
                    "input": "envpack_placeholder",
                    "images": [],
                    "metadata": {
                        "envpack": {
                            "env_name": "sokoban",
                            "seed": seed,
                            "env_config": {"render_mode": "text"},
                        }
                    },
                }
                for seed in range(6)
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            args = self.make_args(str(data_path))
            args.rollout_shuffle = True
            args.n_samples_per_prompt = 1
            args.save = tmp
            args.load = tmp
            state_dir = Path(tmp) / "rollout"
            state_dir.mkdir()
            torch.save(
                {
                    "sample_offset": 0,
                    "epoch_id": 1,
                    "sample_group_index": 0,
                    "sample_index": 0,
                },
                state_dir / "envpack_data_source_state_7.pt",
            )

            expected = EnvpackDataSource(args)
            expected._shuffle_for_epoch(1)
            expected_order = [sample.metadata["envpack"]["seed"] for sample in expected._prompt_samples]

            actual = EnvpackDataSource(args)
            actual.load(7)
            actual_order = [sample.metadata["envpack"]["seed"] for sample in actual._prompt_samples]

        self.assertEqual(actual_order, expected_order)

    def test_curriculum_selects_samples_by_solve_step_stage(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "samples.jsonl"
            rows = [
                _jsonl_row(seed=30, solve_step=3),
                _jsonl_row(seed=31, solve_step=3),
                _jsonl_row(seed=50, solve_step=5),
                _jsonl_row(seed=70, solve_step=7),
                _jsonl_row(seed=80, solve_step=8),
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = self.make_args(
                str(data_path),
                curriculum={
                    "enabled": True,
                    "stages": [
                        {"until": 2, "solve_steps": [3]},
                        {"until": 3, "solve_steps": [5]},
                        {"until": None, "solve_steps": [7, 8]},
                    ],
                },
            )
            args.n_samples_per_prompt = 1

            data_source = EnvpackDataSource(args)
            data_source.set_rollout_step(0)
            first = data_source.get_samples(2)
            second = data_source.get_samples(2)
            data_source.set_rollout_step(2)
            third = data_source.get_samples(1)
            data_source.set_rollout_step(3)
            fourth = data_source.get_samples(2)

        self.assertEqual(_group_solve_steps(first), [3, 3])
        self.assertEqual(_group_solve_steps(second), [3, 3])
        self.assertEqual(_group_solve_steps(third), [5])
        self.assertEqual(_group_solve_steps(fourth), [7, 8])

    def test_curriculum_does_not_advance_on_dapo_refill_draws(self) -> None:
        if torch is None:
            self.skipTest("requires torch because EnvpackDataSource creates Miles Sample objects")
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "samples.jsonl"
            rows = [
                _jsonl_row(seed=30, solve_step=3),
                _jsonl_row(seed=31, solve_step=3),
                _jsonl_row(seed=50, solve_step=5),
                _jsonl_row(seed=51, solve_step=5),
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = self.make_args(
                str(data_path),
                curriculum={
                    "enabled": True,
                    "stages": [
                        {"until": 1, "solve_steps": [3]},
                        {"until": None, "solve_steps": [5]},
                    ],
                },
            )
            args.n_samples_per_prompt = 1

            data_source = EnvpackDataSource(args)
            data_source.set_rollout_step(0)
            first_refill = data_source.get_samples(1)
            second_refill = data_source.get_samples(1)
            data_source.set_rollout_step(1)
            next_rollout = data_source.get_samples(2)

        self.assertEqual(_group_solve_steps(first_refill), [3])
        self.assertEqual(_group_solve_steps(second_refill), [3])
        self.assertEqual(_group_solve_steps(next_rollout), [5, 5])


def _jsonl_row(*, seed: int, solve_step: int):
    return {
        "input": "envpack_placeholder",
        "images": [],
        "metadata": {
            "envpack": {
                "env_name": "sokoban",
                "seed": seed,
                "env_config": {"render_mode": "text"},
                "solver_metrics": {"min_solve_steps": solve_step, "bucket_name": f"6x6_b1_solve_{solve_step}"},
                "bucket_name": f"6x6_b1_solve_{solve_step}",
            }
        },
    }


def _group_solve_steps(groups):
    return [int(group[0].metadata["envpack"]["solver_metrics"]["min_solve_steps"]) for group in groups]


if __name__ == "__main__":
    unittest.main()
