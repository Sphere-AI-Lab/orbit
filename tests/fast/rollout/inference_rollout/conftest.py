from unittest.mock import patch

import pytest
from tests.fast.fixtures.generation_fixtures import megatron_shape_argv

from miles.utils.arguments import parse_args

MODEL_NAME = "Qwen/Qwen3-0.6B"


def _build_mock_args(extra_argv: list[str] | None = None):
    argv = [
        "pytest",
        # orbit: upstream uses the FSDP backend here; orbit deletes it and narrows
        # --train-backend to choices=["megatron"], so the megatron shape flags that
        # hf_validate_args checks must be supplied too (see megatron_shape_argv).
        "--train-backend",
        "megatron",
        "--ci-test",
        "--rollout-batch-size",
        "2",
        "--n-samples-per-prompt",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "4",
        "--rollout-num-gpus-per-engine",
        "2",
        "--hf-checkpoint",
        MODEL_NAME,
        "--prompt-data",
        "/dev/null",
        "--input-key",
        "input",
        "--label-key",
        "label",
        "--rm-type",
        "math",
        # orbit: renamed --use-miles-router -> --use-orbit-router (see rollout_fixtures.py).
        "--use-orbit-router",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        "30000",
    ] + megatron_shape_argv(MODEL_NAME) + (extra_argv or [])
    with patch("sys.argv", argv):
        return parse_args()


@pytest.fixture
def mock_args():
    return _build_mock_args()
