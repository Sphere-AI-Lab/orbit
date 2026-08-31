"""``MilesRouter``'s orbit methods after the move into orbit/rollout/router_ext.py.

Three methods live in the mixin: the added ``/remove_worker`` and ``/workers``
endpoints, and a FULL-REPLACE ``do_proxy`` (orbit's PEFT-request diagnostic and
its upstream-error -> 502 translation are, in lines, about the size of
upstream's own body).

``do_proxy`` is the one with teeth. It MUST NOT be left in ``MilesRouter``'s own
body: Python resolves a class's ``__dict__`` before any base, so a retained
upstream copy would shadow the mixin silently -- the 502 translation would stop
happening and an ``httpx`` transport error would escape as an unattributed 500,
while every test that only checks the happy path stayed green.

The routes themselves are registered by the vendored ``_setup_routes`` (left as
a stamped seam at 30% orbit ownership), so this also proves the vendored file
and the mixin still agree on the handler names.

Behaviour here overlaps tests/test_orbit_router_workers.py on purpose: that file
is not in the fast gate, and the shadowing failure above is exactly the kind
that must be caught in the CPU PR gate.
"""

import asyncio
from argparse import Namespace

import httpx
import pytest
from fastapi.testclient import TestClient

from miles.router.router import MilesRouter
from orbit.rollout.router_ext import OrbitRouterExtensions

WORKER = "http://127.0.0.1:10090"


def _router() -> MilesRouter:
    return MilesRouter(
        Namespace(
            miles_router_health_check_failure_threshold=3,
            miles_router_max_connections=4,
            miles_router_middleware_paths=[],
            miles_router_timeout=5,
            rollout_health_check_interval=3600,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            sglang_server_concurrency=1,
        )
    )


@pytest.mark.parametrize("name", ("do_proxy", "remove_worker", "workers"))
def test_moved_methods_resolve_to_the_mixin_and_are_not_shadowed(name):
    assert name not in MilesRouter.__dict__, (
        f"MilesRouter defines {name} in its own body; that copy beats the mixin in "
        f"attribute lookup and silently disables orbit's override"
    )
    assert getattr(MilesRouter, name).__qualname__ == f"{OrbitRouterExtensions.__name__}.{name}"


def test_mixin_precedes_the_vendored_class_in_the_mro():
    assert MilesRouter.__mro__[:2] == (MilesRouter, OrbitRouterExtensions)


def test_upstream_methods_stayed_in_the_vendored_class():
    """Only the three orbit-owned methods moved; the low-ownership seams
    (__init__, _setup_routes, _health_check_loop) stay upstream's."""
    for name in ("__init__", "_setup_routes", "_health_check_loop", "proxy", "_use_url"):
        assert name in MilesRouter.__dict__, f"{name} should not have moved out of miles"


def test_debug_counter_default_comes_from_the_mixin_not_the_vendored_init():
    """The counter is a mixin CLASS attribute so the vendored __init__ can stay
    byte-pristine; the first increment must shadow it per instance."""
    assert "_debug_peft_request_count" not in MilesRouter.__init__.__code__.co_names
    router = _router()
    assert router._debug_peft_request_count == 0
    assert "_debug_peft_request_count" not in vars(router)
    router._debug_peft_request_count += 1
    assert vars(router)["_debug_peft_request_count"] == 1
    assert OrbitRouterExtensions._debug_peft_request_count == 0, "class default must not be mutated"


def test_worker_endpoints_add_list_and_remove():
    client = TestClient(_router().app)

    assert client.post(f"/add_worker?url={WORKER}").status_code == 200
    assert client.get("/list_workers").json() == {"urls": [WORKER]}
    assert client.get("/workers").json() == {
        "workers": [{"id": "0", "url": WORKER, "worker_type": "regular"}],
        "urls": [WORKER],
    }

    assert client.post(f"/remove_worker?url={WORKER}").status_code == 200
    assert client.get("/list_workers").json() == {"urls": []}


def test_remove_worker_clears_the_dead_quarantine_too():
    """A worker retired while quarantined must not resurrect as 'dead' if the
    same URL is added again."""
    router = _router()
    client = TestClient(router.app)
    client.post(f"/add_worker?url={WORKER}")
    router.dead_workers.add(WORKER)
    router.worker_failure_counts[WORKER] = 3

    client.post("/remove_worker", json={"url": WORKER})

    assert router.dead_workers == set()
    assert WORKER not in router.worker_failure_counts
    assert WORKER not in router.worker_request_counts


def test_remove_worker_rejects_a_request_with_no_url():
    response = TestClient(_router().app).post("/remove_worker", json={})
    assert response.status_code == 400
    assert "worker_url is required" in response.json()["error"]


def test_upstream_transport_error_becomes_502_and_releases_the_slot():
    class FailingClient:
        async def request(self, *args, **kwargs):
            raise httpx.ReadError("backend disconnected")

    router = _router()
    client = TestClient(router.app)
    client.post(f"/add_worker?url={WORKER}")
    router.client = FailingClient()

    response = client.post("/generate", json={"input_ids": [1, 2, 3]})

    assert response.status_code == 502
    assert response.json() == {"error": "upstream request failed: ReadError"}
    assert router.worker_request_counts[WORKER] == 0, (
        "the finally must still release the active-request slot on the error path"
    )


def test_peft_request_diagnostic_is_off_unless_the_env_var_is_set(monkeypatch, caplog):
    monkeypatch.delenv("ORBIT_DEBUG_PEFT_REQUEST", raising=False)
    router = _router()
    client = TestClient(router.app)
    client.post(f"/add_worker?url={WORKER}")

    class Client:
        async def request(self, *args, **kwargs):
            raise httpx.ReadError("x")

    router.client = Client()
    with caplog.at_level("INFO"):
        client.post("/generate", json={"lora_path": "a"})
    assert not [r for r in caplog.records if "generate payload" in r.message]
    assert router._debug_peft_request_count == 0


def test_peft_request_diagnostic_logs_and_rate_limits(monkeypatch, caplog):
    monkeypatch.setenv("ORBIT_DEBUG_PEFT_REQUEST", "1")
    monkeypatch.setenv("ORBIT_DEBUG_PEFT_REQUEST_LIMIT", "2")
    router = _router()
    client = TestClient(router.app)
    client.post(f"/add_worker?url={WORKER}")

    class Client:
        async def request(self, *args, **kwargs):
            raise httpx.ReadError("x")

    router.client = Client()
    with caplog.at_level("INFO"):
        for _ in range(4):
            client.post("/generate", json={"oft_path": "a"})
        client.post("/health", json={})

    logged = [r for r in caplog.records if "generate payload" in r.message]
    assert len(logged) == 2, "the limit must cap the diagnostic, and only /generate is logged"
    assert router._debug_peft_request_count == 2
    assert logged[0].args[0] is False and logged[0].args[1] is True, "lora_path absent, oft_path present"


@pytest.mark.asyncio
async def test_health_check_only_probes_idle_workers(monkeypatch):
    """The _health_check_loop seam stayed in the vendored file; assert the
    behaviour it carries still holds after the class gained the mixin base."""
    router = _router()
    router.worker_request_counts[WORKER] = 32
    checked = []
    sleeps = 0

    async def fake_sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    async def failed(url):
        checked.append(url)
        return url, False

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(router, "_check_worker_health", failed)

    with pytest.raises(asyncio.CancelledError):
        await router._health_check_loop()
    assert checked == [], "a busy worker must not be health-probed"
