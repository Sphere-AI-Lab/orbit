"""``MockSGLangServer._setup_routes`` after the move into orbit's home mixin.

The route table is 84% orbit (four upstream routes, four added PEFT
adapter-version endpoints) so it moved WHOLE, to
orbit/sglang/mock_server_ext.py. Upstream's four registrations are reproduced
there verbatim and are asserted here alongside orbit's, because a
FULL-REPLACE override is exactly where an upstream route quietly goes missing.

The mock exists so the adapter-transport tests can exercise the real
staged-then-activated protocol without a GPU. Its rejections are the point: a
double-buffer bug that activates a version nobody staged, or that lets
``adapter_version`` and ``weight_version`` drift apart, must surface as a 400
here rather than as a stale adapter on a real engine.

Shadowing note: ``_setup_routes`` must NOT be left in ``MockSGLangServer``'s own
body. A class's ``__dict__`` beats every base, so a retained upstream copy would
register upstream's four routes only and orbit's four would 404 -- which reads
from a failing transport test as a transport bug, not a wiring bug.

The server is built with ``__new__``: the real ``__init__`` downloads a
tokenizer, and none of the endpoints under test touch it.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer
from orbit.sglang.mock_server_ext import OrbitMockSGLangServerExtensions

UPSTREAM_ROUTES = {
    ("/generate", "POST"),
    ("/v1/chat/completions", "POST"),
    ("/health", "GET"),
    ("/abort_request", "POST"),
}
ORBIT_ROUTES = {
    ("/model_info", "GET"),
    ("/update_adapter_from_distributed", "POST"),
    ("/update_weight_version", "POST"),
    ("/activate_adapter_version", "POST"),
}


@pytest.fixture
def server():
    """A mock server without ``__init__`` -- it would download a tokenizer."""
    mock = MockSGLangServer.__new__(MockSGLangServer)
    mock.app = FastAPI()
    mock.request_log = []
    mock._setup_routes()
    return mock


@pytest.fixture
def client(server):
    return TestClient(server.app)


def test_setup_routes_resolves_to_the_mixin_and_is_not_shadowed():
    assert "_setup_routes" not in MockSGLangServer.__dict__, (
        "MockSGLangServer defines _setup_routes in its own body; that copy beats "
        "the mixin and orbit's PEFT endpoints would silently 404"
    )
    assert MockSGLangServer._setup_routes.__qualname__ == "OrbitMockSGLangServerExtensions._setup_routes"
    assert MockSGLangServer.__mro__[:2] == (MockSGLangServer, OrbitMockSGLangServerExtensions)


def test_low_ownership_methods_stayed_in_the_vendored_class():
    for name in ("__init__", "_compute_generate_response", "_compute_chat_completions_response"):
        assert name in MockSGLangServer.__dict__, f"{name} should not have moved out of miles"


def test_every_upstream_route_survived_the_full_replace(server):
    registered = {(r.path, m) for r in server.app.routes for m in getattr(r, "methods", ())}
    assert UPSTREAM_ROUTES <= registered, "the mixin dropped one of upstream's own routes"
    assert ORBIT_ROUTES <= registered


def test_adapter_state_is_initialised_per_instance_not_on_the_class(server):
    """The vendored __init__ is 24% orbit and stays pristine, so the mixin owns
    this state. staged_adapter_versions is MUTABLE: a class-level dict would
    leak staged versions between two mock servers in one process."""
    assert (server.weight_version, server.adapter_version) == ("0", "0")
    assert server.staged_adapter_versions == {}
    assert not hasattr(OrbitMockSGLangServerExtensions, "staged_adapter_versions")

    other = MockSGLangServer.__new__(MockSGLangServer)
    other.app = FastAPI()
    other.request_log = []
    other._setup_routes()
    server.staged_adapter_versions["a"] = "7"
    assert other.staged_adapter_versions == {}


def test_model_info_reports_the_current_versions(client, server):
    server.weight_version, server.adapter_version = "4", "5"
    assert client.get("/model_info").json() == {
        "model_info": {"weight_version": "4", "adapter_version": "5"}
    }


def test_non_double_buffer_update_activates_immediately(client, server):
    body = client.post(
        "/update_adapter_from_distributed", json={"adapter_version": "3", "weight_version": "3"}
    ).json()
    assert body["success"] is True
    assert body["active_adapter_version"] == "3"
    assert (server.adapter_version, server.weight_version) == ("3", "3")
    assert server.staged_adapter_versions == {}


def test_double_buffer_update_stages_without_activating(client, server):
    body = client.post(
        "/update_adapter_from_distributed",
        json={"adapter_version": "3", "weight_version": "3", "double_buffer": True, "adapter_name": "a"},
    ).json()
    assert body["staged_adapter_version"] == "3"
    assert body["active_adapter_version"] == "0", "staging must not flip the live version"
    assert server.adapter_version == "0"
    assert server.staged_adapter_versions == {"a": "3"}


def test_update_rejects_disagreeing_adapter_and_weight_versions(client, server):
    response = client.post(
        "/update_adapter_from_distributed", json={"adapter_version": "3", "weight_version": "4"}
    )
    assert response.status_code == 400
    assert server.adapter_version == "0"


def test_activate_promotes_only_a_staged_version(client, server):
    client.post(
        "/update_adapter_from_distributed",
        json={"adapter_version": "3", "weight_version": "3", "double_buffer": True, "adapter_name": "a"},
    )
    body = client.post("/activate_adapter_version", json={"adapter_version": "3", "adapter_name": "a"}).json()
    assert body["active_adapter_version"] == "3"
    assert (server.adapter_version, server.weight_version) == ("3", "3")
    assert server.staged_adapter_versions == {}, "the staged entry must be consumed"


def test_activate_rejects_a_version_that_was_never_staged(client, server):
    response = client.post("/activate_adapter_version", json={"adapter_version": "9", "adapter_name": "a"})
    assert response.status_code == 400
    assert "No staged adapter version" in response.json()["message"]
    assert server.adapter_version == "0", "a rejected activation must not move the live version"


def test_update_weight_version_moves_both_versions_together(client, server):
    assert client.post("/update_weight_version", json={"weight_version": "8"}).json() == {
        "success": True,
        "weight_version": "8",
        "adapter_version": "8",
    }
    assert (server.weight_version, server.adapter_version) == ("8", "8")


def test_every_peft_endpoint_is_recorded_in_the_request_log(client, server):
    client.post("/update_adapter_from_distributed", json={"adapter_version": "1", "weight_version": "1"})
    client.post("/update_weight_version", json={"weight_version": "2"})
    client.post("/activate_adapter_version", json={"adapter_version": "9"})
    assert [entry["endpoint"] for entry in server.request_log] == [
        "update_adapter_from_distributed",
        "update_weight_version",
        "activate_adapter_version",
    ]


def test_health_and_abort_still_answer(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/abort_request", json={}).json() == {"status": "ok"}
