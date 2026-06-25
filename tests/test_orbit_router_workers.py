from argparse import Namespace

import httpx
from fastapi.testclient import TestClient

from orbit.router.router import OrbitRouter


def _router_args():
    return Namespace(
        orbit_router_health_check_failure_threshold=3,
        orbit_router_max_connections=4,
        orbit_router_middleware_paths=[],
        orbit_router_timeout=5,
        rollout_health_check_interval=3600,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=1,
    )


def test_orbit_router_worker_compat_endpoints():
    router = OrbitRouter(_router_args())
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

    router = OrbitRouter(_router_args())
    client = TestClient(router.app)

    worker_url = "http://127.0.0.1:10090"
    assert client.post(f"/add_worker?url={worker_url}").status_code == 200

    router.client = FailingClient()
    response = client.post("/generate", json={"input_ids": [1, 2, 3]})

    assert response.status_code == 502
    assert response.json() == {"error": "upstream request failed: ReadError"}
    assert router.worker_request_counts[worker_url] == 0
