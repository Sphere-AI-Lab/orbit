from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")

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
    def make_args(self, prompt_data: str):
        return SimpleNamespace(
            prompt_data=prompt_data,
            envpack={
                "api": "in_process",
                "pools": [
                    {
                        "env": "sokoban",
                        "profile": "vision_free_think_local",
                        "pool_id": "sokoban-vision",
                        "env_config": {"render_mode": "vision"},
                    }
                ],
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
                                "config": {"prompt_format": "free_think"},
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


if __name__ == "__main__":
    unittest.main()
