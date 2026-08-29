import asyncio
from argparse import Namespace

import httpx
from fastapi.testclient import TestClient
import pytest

from miles.router.router import MilesRouter


def _router_args():
    return Namespace(
        miles_router_health_check_failure_threshold=3,
        miles_router_max_connections=4,
        miles_router_middleware_paths=[],
        miles_router_timeout=5,
        rollout_health_check_interval=3600,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=1,
    )


def test_orbit_router_worker_compat_endpoints():
    router = MilesRouter(_router_args())
    client = TestClient(router.app)

    worker_url = "http://127.0.0.1:10090"
    response = client.post(f"/add_worker?url={worker_url}")
    assert response.status_code == 200

    assert client.get("/list_workers").json() == {"urls": [worker_url]}
    assert client.get("/workers").json() == {
        "workers": [{"id": "0", "url": worker_url, "worker_type": "regular"}],
        "urls": [worker_url],
    }

    response = client.post(f"/remove_worker?url={worker_url}")
    assert response.status_code == 200
    assert client.get("/list_workers").json() == {"urls": []}


def test_orbit_router_proxy_returns_502_for_upstream_transport_error():
    class FailingClient:
        async def request(self, *args, **kwargs):
            raise httpx.ReadError("backend disconnected")

    router = MilesRouter(_router_args())
    client = TestClient(router.app)

    worker_url = "http://127.0.0.1:10090"
    assert client.post(f"/add_worker?url={worker_url}").status_code == 200

    router.client = FailingClient()
    response = client.post("/generate", json={"input_ids": [1, 2, 3]})

    assert response.status_code == 502
    assert response.json() == {"error": "upstream request failed: ReadError"}
    assert router.worker_request_counts[worker_url] == 0


@pytest.mark.parametrize(
    ("active_requests", "expected_checks", "expected_failures", "expected_dead"),
    ((32, 0, 2, False), (0, 1, 3, True)),
)
@pytest.mark.asyncio
async def test_orbit_router_health_evicts_only_idle_workers(
    monkeypatch,
    active_requests,
    expected_checks,
    expected_failures,
    expected_dead,
):
    router = MilesRouter(_router_args())
    worker_url = "http://127.0.0.1:10090"
    router.worker_request_counts[worker_url] = active_requests
    router.worker_failure_counts[worker_url] = 2
    checked_urls = []
    sleep_calls = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    async def failed_health_check(url):
        checked_urls.append(url)
        return url, False

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(router, "_check_worker_health", failed_health_check)

    with pytest.raises(asyncio.CancelledError):
        await router._health_check_loop()

    assert checked_urls == [worker_url] * expected_checks
    assert router.worker_failure_counts[worker_url] == expected_failures
    assert (worker_url in router.dead_workers) is expected_dead
