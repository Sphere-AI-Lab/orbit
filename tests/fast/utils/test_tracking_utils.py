from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from miles.utils.tracking_utils.base import WandbBackend


class WandbBackendTest(unittest.TestCase):
    def test_log_uses_monotonic_row_step_separate_from_logical_step(self) -> None:
        calls = []
        fake_wandb = types.SimpleNamespace(log=lambda metrics, **kwargs: calls.append((metrics, kwargs)))
        previous = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_wandb
        try:
            backend = WandbBackend()
            with patch("miles.utils.tracking_utils.base.time.time_ns", side_effect=[1000, 1000]):
                backend.log({"rollout/step": 3, "rollout/pre_filter_solve_rate": 0.25}, step=3)
                backend.log({"eval/step": 1, "eval/solve_rate": 0.5}, step=1)
        finally:
            if previous is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = previous

        self.assertEqual(calls[0], ({"rollout/step": 3, "rollout/pre_filter_solve_rate": 0.25}, {"step": 0}))
        self.assertEqual(calls[1], ({"eval/step": 1, "eval/solve_rate": 0.5}, {"step": 1}))


class WandbInitTest(unittest.TestCase):
    def test_primary_exports_run_id_for_secondary_actors(self) -> None:
        fake_wandb = _fake_wandb(run_id="run-123")
        previous_wandb = sys.modules.get("wandb")
        previous_module = sys.modules.pop("miles.utils.tracking_utils.wandb_utils", None)
        previous_env = os.environ.pop("WANDB_RUN_ID", None)
        sys.modules["wandb"] = fake_wandb
        try:
            wandb_utils = importlib.import_module("miles.utils.tracking_utils.wandb_utils")
            args = _wandb_args(wandb_run_id=None)

            self.assertTrue(wandb_utils.init_wandb_primary(args))

            self.assertEqual(args.wandb_run_id, "run-123")
            self.assertEqual(os.environ["WANDB_RUN_ID"], "run-123")
        finally:
            if previous_env is None:
                os.environ.pop("WANDB_RUN_ID", None)
            else:
                os.environ["WANDB_RUN_ID"] = previous_env
            _restore_module("miles.utils.tracking_utils.wandb_utils", previous_module)
            _restore_module("wandb", previous_wandb)

    def test_secondary_uses_run_id_from_environment(self) -> None:
        fake_wandb = _fake_wandb(run_id="ignored")
        previous_wandb = sys.modules.get("wandb")
        previous_module = sys.modules.pop("miles.utils.tracking_utils.wandb_utils", None)
        previous_env = os.environ.get("WANDB_RUN_ID")
        os.environ["WANDB_RUN_ID"] = "run-from-env"
        sys.modules["wandb"] = fake_wandb
        try:
            wandb_utils = importlib.import_module("miles.utils.tracking_utils.wandb_utils")
            args = _wandb_args(wandb_run_id=None)

            self.assertTrue(wandb_utils.init_wandb_secondary(args))

            self.assertEqual(args.wandb_run_id, "run-from-env")
            self.assertEqual(fake_wandb.init_calls[-1]["id"], "run-from-env")
            self.assertEqual(fake_wandb.init_calls[-1]["resume"], "allow")
        finally:
            if previous_env is None:
                os.environ.pop("WANDB_RUN_ID", None)
            else:
                os.environ["WANDB_RUN_ID"] = previous_env
            _restore_module("miles.utils.tracking_utils.wandb_utils", previous_module)
            _restore_module("wandb", previous_wandb)


def _wandb_args(*, wandb_run_id):
    return SimpleNamespace(
        use_wandb=True,
        wandb_mode=None,
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="test-run",
        rank=0,
        wandb_team=None,
        wandb_project="test-project",
        wandb_dir=None,
        wandb_run_id=wandb_run_id,
        env_report=None,
        sglang_enable_metrics=False,
    )


def _fake_wandb(*, run_id: str):
    init_calls = []

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def init(**kwargs):
        init_calls.append(kwargs)

    def define_metric(*args, **kwargs):
        return None

    return types.SimpleNamespace(
        init=init,
        init_calls=init_calls,
        login=lambda **kwargs: None,
        Settings=Settings,
        run=SimpleNamespace(id=run_id),
        util=SimpleNamespace(generate_id=lambda: "abcdef"),
        define_metric=define_metric,
    )


def _restore_module(name: str, module) -> None:
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
