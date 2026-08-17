"""Pins the --true-on-policy-mode direct-flag fallback in validate_args.

orbit/true_on_policy/config.py::apply_true_on_policy_parse_defaults forces
sglang_enable_deterministic_inference at parse time, but that expansion only
runs through the --true-on-policy entry point. --true-on-policy-mode is also
an independently settable CLI flag (orbit/utils/arguments.py), and setting it
directly bypasses that expansion. validate_args must force determinism too,
as a fallback net (miles parity: backends/sglang_utils/arguments.py:146-147).
"""

import importlib
import sys
import types
from types import SimpleNamespace


def _import_validate_args(monkeypatch):
    """Import the real validate_args, stubbing sglang for this test only.

    orbit.backends.sglang_utils.arguments imports the real sglang package at
    module level, which isn't installed in this CPU test environment; stub
    the one symbol it needs (validate_args itself never touches ServerArgs).
    Everything is done via monkeypatch so sys.modules is restored after the
    test — an unconditional stub would leak a fake sglang/sglang.srt into
    the rest of the pytest process and break later tests that need the real
    (absent) sglang to hit their normal ImportError fallback path.
    """
    monkeypatch.delitem(sys.modules, "orbit.backends.sglang_utils.arguments", raising=False)
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    stub_server_args_module = types.ModuleType("sglang.srt.server_args")
    stub_server_args_module.ServerArgs = object
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", stub_server_args_module)

    module = importlib.import_module("orbit.backends.sglang_utils.arguments")
    return module.validate_args


def _args(**overrides):
    values = dict(
        rollout_num_gpus_per_engine=1,
        # sglang v0.5.14+ ServerArgs fields are dp_size/pp_size/ep_size; the
        # *_parallel_size spellings are CLI aliases the parser mirrors onto these
        # names. validate_args is called here directly on a namespace, so it
        # never sees that mirror and the mirrored names must be supplied.
        sglang_data_parallel_size=1,
        sglang_pipeline_parallel_size=1,
        sglang_expert_parallel_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        sglang_enable_dp_attention=False,
        sglang_router_policy=None,
        true_on_policy_mode=True,
        sglang_enable_deterministic_inference=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_true_on_policy_mode_forces_deterministic_inference_when_set_directly(monkeypatch):
    validate_args = _import_validate_args(monkeypatch)
    args = _args(true_on_policy_mode=True, sglang_enable_deterministic_inference=False)
    validate_args(args)
    assert args.sglang_enable_deterministic_inference is True


def test_deterministic_inference_untouched_without_true_on_policy_mode(monkeypatch):
    validate_args = _import_validate_args(monkeypatch)
    args = _args(true_on_policy_mode=False, sglang_enable_deterministic_inference=False)
    validate_args(args)
    assert args.sglang_enable_deterministic_inference is False
