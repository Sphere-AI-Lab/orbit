from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from miles_plugins.envpack_adapter.config import EnvpackConfigError, load_envpack_config, validate_runtime_args
from miles_plugins.envpack_adapter.runtime import (
    build_in_process_client,
    build_session_client,
    get_client_bundle,
    resolve_miles_active_episode_capacity,
    resolve_pool_runtime_demand,
)


class EnvpackConfigTest(unittest.TestCase):
    def test_single_env_config(self) -> None:
        args = SimpleNamespace(
            envpack_adapter={
                "api": "in_process",
                "env": "Sokoban",
                "profile": "vision_free_think_local",
                "pool_id": "sokoban-vision",
            }
        )
        config = load_envpack_config(args)
        self.assertEqual(config.api, "in_process")
        self.assertEqual(config.pool_for_env("sokoban").resolved_pool_id, "sokoban-vision")

    def test_legacy_envpack_config_is_still_accepted(self) -> None:
        args = SimpleNamespace(envpack={"api": "in_process", "env": "Sokoban"})

        config = load_envpack_config(args)

        self.assertEqual(config.api, "in_process")
        self.assertEqual(config.pool_for_env("sokoban").env, "sokoban")

    def test_rejects_both_adapter_and_legacy_config_keys(self) -> None:
        args = SimpleNamespace(
            envpack_adapter={"api": "in_process", "env": "sokoban"},
            envpack={"api": "in_process", "env": "sokoban"},
        )

        with self.assertRaisesRegex(EnvpackConfigError, "either `envpack_adapter` or legacy `envpack`"):
            load_envpack_config(args)

    def test_rejects_unknown_keys(self) -> None:
        args = SimpleNamespace(envpack={"api": "in_process", "env": "sokoban", "bad": True})
        with self.assertRaisesRegex(EnvpackConfigError, "unknown keys"):
            load_envpack_config(args)

    def test_session_mode_requires_server(self) -> None:
        args = SimpleNamespace(envpack={"api": "session", "env": "sokoban"})
        with self.assertRaisesRegex(EnvpackConfigError, "server is required"):
            load_envpack_config(args)

    def test_session_mode_loads_server_config(self) -> None:
        args = SimpleNamespace(envpack={"api": "session", "server": "http://env-node:18081", "env": "sokoban"})
        config = load_envpack_config(args)
        self.assertEqual(config.api, "session")
        self.assertEqual(config.server, "http://env-node:18081")
        self.assertEqual(config.http.timeout_s, 60.0)
        self.assertEqual(config.http.max_retries, 3)
        self.assertEqual(config.http.auth_token_env, "ENVPACK_AUTH_TOKEN")

    def test_session_mode_loads_http_retry_config(self) -> None:
        args = SimpleNamespace(
            envpack={
                "api": "session",
                "server": "http://env-node:18081",
                "env": "sokoban",
                "http": {
                    "timeout_s": 120,
                    "max_retries": 5,
                    "retry_backoff_s": 0.1,
                    "auth_token_env": "MY_ENVPACK_TOKEN",
                },
            }
        )
        config = load_envpack_config(args)
        self.assertEqual(config.http.timeout_s, 120.0)
        self.assertEqual(config.http.max_retries, 5)
        self.assertEqual(config.http.retry_backoff_s, 0.1)
        self.assertEqual(config.http.auth_token_env, "MY_ENVPACK_TOKEN")

    def test_loads_refill_config(self) -> None:
        args = SimpleNamespace(
            envpack={
                "api": "session",
                "server": "http://env-node:18081",
                "env": "sokoban",
                "refill": {
                    "max_attempts": 4,
                    "backoff_s": 0.2,
                },
            }
        )
        config = load_envpack_config(args)
        self.assertEqual(config.refill.max_attempts, 4)
        self.assertEqual(config.refill.backoff_s, 0.2)

    def test_loads_curriculum_config(self) -> None:
        args = SimpleNamespace(
            envpack={
                "api": "in_process",
                "env": "sokoban",
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {"until": 50, "solve_steps": [3, 4]},
                        {"until": 150, "solve_steps": [3, 4, 5, 6]},
                        {"until": None, "solve_steps": [3, 4, 5, 6, 7, 8, 9, 10]},
                    ],
                },
            }
        )

        config = load_envpack_config(args)

        self.assertTrue(config.curriculum.enabled)
        self.assertEqual(config.curriculum.stages[0].until, 50)
        self.assertEqual(config.curriculum.stages[0].solve_steps, (3, 4))
        self.assertIsNone(config.curriculum.stages[-1].until)

    def test_runtime_guards_reject_r3_until_generate_preserves_it(self) -> None:
        args = SimpleNamespace(
            envpack={"api": "in_process", "env": "sokoban"},
            partial_rollout=False,
            group_rm=False,
            rm_type=None,
            custom_rm_path=None,
            rollout_external=False,
            use_rollout_routing_replay=True,
        )
        config = load_envpack_config(args)
        with self.assertRaisesRegex(EnvpackConfigError, "R3"):
            validate_runtime_args(args, config)

    def test_runtime_guards_reject_radix_tree_middleware(self) -> None:
        args = SimpleNamespace(
            envpack={"api": "in_process", "env": "sokoban"},
            partial_rollout=False,
            group_rm=False,
            rm_type=None,
            custom_rm_path=None,
            rollout_external=False,
            use_rollout_routing_replay=False,
            use_miles_router=True,
            miles_router_middleware_paths=["miles.router.middleware_hub.radix_tree_middleware.RadixTreeMiddleware"],
        )
        config = load_envpack_config(args)
        with self.assertRaisesRegex(EnvpackConfigError, "RadixTreeMiddleware"):
            validate_runtime_args(args, config)

    def test_runtime_builder_fails_loudly_when_envpack_is_not_installed(self) -> None:
        args = SimpleNamespace(envpack={"api": "in_process", "env": "sokoban"})
        config = load_envpack_config(args)
        try:
            import envpack  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaisesRegex(EnvpackConfigError, "envpack is not importable"):
                build_in_process_client(config)
        else:
            self.skipTest("envpack is importable in this environment")

    def test_auto_runtime_demand_uses_miles_generate_slots(self) -> None:
        args = SimpleNamespace(
            envpack={"api": "in_process", "env": "sokoban"},
            sglang_server_concurrency=32,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=2,
        )
        config = load_envpack_config(args)
        pool = config.pool_for_env("sokoban")

        self.assertEqual(resolve_miles_active_episode_capacity(args), 128)
        self.assertEqual(
            resolve_pool_runtime_demand(args, pool),
            {"desired_concurrency": 128},
        )

    def test_explicit_runtime_capacity_is_preserved(self) -> None:
        args = SimpleNamespace(
            envpack={
                "api": "in_process",
                "pools": [
                    {
                        "env": "sokoban",
                        "runtime_config": {"num_instances": 4, "max_active_episodes_per_instance": 1},
                    }
                ],
            },
            sglang_server_concurrency=32,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=2,
        )
        config = load_envpack_config(args)
        pool = config.pool_for_env("sokoban")

        self.assertEqual(
            resolve_pool_runtime_demand(args, pool),
            {"num_instances": 4, "max_active_episodes_per_instance": 1},
        )

    def test_session_client_bundle_uses_remote_client_and_profile_env_config(self) -> None:
        args = SimpleNamespace(
            envpack={
                "api": "session",
                "server": "http://env-node:18081",
                "pools": [
                    {
                        "env": "sokoban",
                        "profile": "vision_free_think_local",
                        "pool_id": "sokoban-vision",
                        "env_config": {"render_mode": "vision"},
                    }
                ],
            }
        )
        config = load_envpack_config(args)
        try:
            with patch.dict("os.environ", {"ENVPACK_AUTH_TOKEN": "secret"}):
                bundle = build_session_client(config)
        except EnvpackConfigError as exc:
            if "envpack" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(bundle.client.base_url, "http://env-node:18081")
        self.assertEqual(bundle.client.timeout_s, 60.0)
        self.assertEqual(bundle.client.max_retries, 3)
        self.assertEqual(bundle.client.headers["Authorization"], "Bearer secret")
        self.assertEqual(bundle.env_config("sokoban-vision")["render_mode"], "vision")

    def test_get_client_bundle_dispatches_by_api(self) -> None:
        args = SimpleNamespace(envpack={"api": "session", "server": "http://env-node:18081", "env": "sokoban"})
        config = load_envpack_config(args)
        try:
            bundle = get_client_bundle(config)
        except EnvpackConfigError as exc:
            if "envpack" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertEqual(bundle.client.base_url, "http://env-node:18081")


if __name__ == "__main__":
    unittest.main()
