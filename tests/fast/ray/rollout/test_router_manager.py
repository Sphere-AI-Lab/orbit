from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

import orbit.ray.rollout.router_manager as router_manager
from orbit.ray.rollout.router_manager import _resolve_session_server_ports, start_router, start_session_server


class TestStartRouter:
    def test_returns_existing_when_already_configured(self):
        """Happy path: ``sglang_router_ip`` and ``sglang_router_port`` are
        already set and ``force_new=False`` → skip subprocess launch entirely
        and return the existing tuple."""
        args = make_args(
            use_orbit_router=False,
            sglang_router_ip="10.1.2.3",
            sglang_router_port=4567,
        )
        # No mocks needed — the function returns before touching anything.
        ip, port = start_router(args, force_new=False)
        assert (ip, port) == ("10.1.2.3", 4567)

    def test_oft_rejects_unmarked_existing_router(self):
        args = make_args(
            peft_method="oft",
            use_orbit_router=False,
            sglang_router_ip="10.1.2.3",
            sglang_router_port=4567,
        )

        with pytest.raises(ValueError, match="--use-orbit-router"):
            start_router(args)

        assert args.use_orbit_router is False

    def test_oft_reuses_existing_router_marked_as_orbit_passthrough(self):
        args = make_args(
            peft_method="oft",
            use_orbit_router=True,
            sglang_router_ip="10.1.2.3",
            sglang_router_port=4567,
        )

        assert start_router(args) == ("10.1.2.3", 4567)

    def test_oft_without_existing_router_auto_launches_orbit_passthrough(self):
        started_processes = []

        class RecordingProcess:
            def __init__(self, *, target, args):
                self.daemon = False
                self.target = target
                self.args = args
                self.started = False
                started_processes.append(self)

            def start(self):
                self.started = True

        args = make_args(
            peft_method="oft",
            use_orbit_router=False,
            sglang_router_ip=None,
            sglang_router_port=None,
        )
        with patch("orbit.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "orbit.ray.rollout.router_manager.find_available_port", return_value=20000
        ), patch("orbit.ray.rollout.router_manager.is_port_available", return_value=True), patch(
            "orbit.ray.rollout.router_manager.multiprocessing.get_context",
            return_value=SimpleNamespace(Process=RecordingProcess),
        ), patch(
            "orbit.ray.rollout.router_manager.wait_for_server_ready"
        ):
            result = start_router(args)

        assert result == ("127.0.0.1", 20000)
        assert args.use_orbit_router is True
        assert len(started_processes) == 1
        assert started_processes[0].target is router_manager.run_orbit_router
        assert started_processes[0].started is True

    def test_oft_force_new_ignores_unmarked_existing_router_and_launches_orbit_passthrough(self):
        started_processes = []

        class RecordingProcess:
            def __init__(self, *, target, args):
                self.daemon = False
                self.target = target
                self.args = args
                self.started = False
                started_processes.append(self)

            def start(self):
                self.started = True

        args = make_args(
            peft_method="oft",
            use_orbit_router=False,
            sglang_router_ip="10.1.2.3",
            sglang_router_port=4567,
        )
        with patch("orbit.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "orbit.ray.rollout.router_manager.find_available_port", return_value=20000
        ), patch("orbit.ray.rollout.router_manager.is_port_available", return_value=True), patch(
            "orbit.ray.rollout.router_manager.multiprocessing.get_context",
            return_value=SimpleNamespace(Process=RecordingProcess),
        ), patch(
            "orbit.ray.rollout.router_manager.wait_for_server_ready"
        ):
            result = start_router(args, force_new=True)

        assert result == ("127.0.0.1", 20000)
        assert args.use_orbit_router is True
        assert len(started_processes) == 1
        assert started_processes[0].target is router_manager.run_orbit_router
        assert started_processes[0].started is True

    def test_pd_disagg_with_orbit_router_asserts(self):
        args = make_args(use_orbit_router=True, sglang_router_ip=None, sglang_router_port=None)
        with patch("orbit.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "orbit.ray.rollout.router_manager.find_available_port", return_value=20000
        ):
            with pytest.raises(AssertionError, match="orbit router does not support PD"):
                start_router(args, has_pd_disaggregation=True, force_new=False)

    def test_port_conflict_raises_runtime_error(self):
        args = make_args(use_orbit_router=False, sglang_router_ip=None, sglang_router_port=None)
        with patch("orbit.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "orbit.ray.rollout.router_manager.find_available_port", return_value=20000
        ), patch("orbit.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_router(args)


class TestStartSessionServer:
    def test_disabled_returns_silently(self):
        """Happy no-op: ``use_session_server=False`` → return without raising,
        without touching any other config."""
        args = make_args(use_session_server=False)
        start_session_server(args)

    def test_enabled_without_hf_checkpoint_raises(self):
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            start_session_server(args)

    def test_enabled_port_conflict_raises_runtime_error(self):
        """When a configured ``session_server_port`` is already bound, fail
        loud rather than silently re-using the stale process."""
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=20000,
            session_server_ip="127.0.0.1",
            session_server_port=[20001],
        )
        with patch("orbit.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_session_server(args)


class TestResolveSessionServerPorts:
    def test_none_auto_allocates_one_port(self):
        with patch("orbit.ray.rollout.router_manager.find_available_port", return_value=20002):
            assert _resolve_session_server_ports(None) == [20002]

    def test_single_value_is_a_single_server(self):
        assert _resolve_session_server_ports([30000]) == [30000]

    def test_two_values_expand_to_half_open_range(self):
        assert _resolve_session_server_ports([30000, 30004]) == [30000, 30001, 30002, 30003]

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _resolve_session_server_ports([30004, 30000])

    def test_more_than_two_values_raises(self):
        with pytest.raises(ValueError, match="one port or a start/end range"):
            _resolve_session_server_ports([30000, 30001, 30002])
