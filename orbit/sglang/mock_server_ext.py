"""Orbit's ``MockSGLangServer`` route table.

Home mixin for ``_setup_routes``, lifted out of
miles/utils/test_utils/mock_sglang_server.py. ``MockSGLangServer`` in the miles
file lists this mixin as its base:

    class MockSGLangServer(OrbitMockSGLangServerExtensions):

Why the WHOLE method moved rather than just orbit's additions: at 84% orbit
lines it is orbit's method with four upstream routes still in it, not the other
way round. Upstream's ``/generate``, ``/v1/chat/completions``, ``/health`` and
``/abort_request`` registrations are reproduced verbatim below and must stay
that way; they dispatch back into the vendored
``_handle_generate_like_request`` / ``_compute_generate_response`` /
``_compute_chat_completions_response`` through ``self``.

Orbit's four added endpoints mock the PEFT weight/adapter-version protocol the
real engine speaks, which is what lets the adapter-transport tests run without a
GPU:

  ``GET  /model_info``                     -- current weight/adapter versions
  ``POST /update_adapter_from_distributed``-- stage (``double_buffer=True``) or
                                              activate an adapter version
  ``POST /update_weight_version``          -- full-FT path: bump both versions
  ``POST /activate_adapter_version``       -- promote a STAGED version, and
                                              reject one that was never staged

The rejections are the point of the mock. A double-buffer bug that activates an
unstaged version, or one that lets ``adapter_version`` and ``weight_version``
drift apart, shows up here as a 400 instead of as a silently stale adapter on a
GPU run.

State: ``weight_version`` / ``adapter_version`` / ``staged_adapter_versions``
belong to those endpoints, so they are initialised HERE rather than by a seam in
the vendored ``__init__`` (which is 24% orbit and stays pristine).
``_setup_routes`` is called exactly once, from ``__init__``, so this runs exactly
when the vendored assignments used to. They are per-INSTANCE, deliberately not
mixin class attributes: ``staged_adapter_versions`` is mutable, and a shared
class-level dict would leak staged versions between two mock servers in one test
process.

MRO note: ``_setup_routes`` must NOT be left in ``MockSGLangServer``'s own body.
A class's ``__dict__`` beats every base, so a retained upstream body would
shadow this one and the four PEFT endpoints would silently 404 -- which reads,
from a test, as a transport bug rather than a wiring bug.

Plain mixin: no ``__init__``, no ``super()`` call (from here ``super()`` is
``object``, which has no ``_setup_routes``).
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class OrbitMockSGLangServerExtensions:
    def _orbit_reset_adapter_version_state(self) -> None:
        """Per-instance state for the PEFT version endpoints registered below."""
        self.weight_version = "0"
        self.adapter_version = "0"
        self.staged_adapter_versions = {}

    def _setup_routes(self):
        self._orbit_reset_adapter_version_state()

        @self.app.post("/generate")
        async def generate(request: Request):
            return await self._handle_generate_like_request(request, self._compute_generate_response)

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._handle_generate_like_request(request, self._compute_chat_completions_response)

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/abort_request")
        async def abort_request(_request: Request):
            return JSONResponse(content={"status": "ok"})

        @self.app.get("/model_info")
        async def model_info():
            return {
                "model_info": {
                    "weight_version": self.weight_version,
                    "adapter_version": self.adapter_version,
                }
            }

        @self.app.post("/update_adapter_from_distributed")
        async def update_adapter_from_distributed(request: Request):
            payload = await request.json()
            self.request_log.append(
                {"endpoint": "update_adapter_from_distributed", "payload": payload}
            )
            adapter_version = str(payload.get("adapter_version", payload.get("weight_version", "0")))
            weight_version = str(payload.get("weight_version", adapter_version))
            if adapter_version != weight_version:
                return JSONResponse(
                    {
                        "success": False,
                        "message": "adapter_version and weight_version disagree",
                        "adapter_version": adapter_version,
                        "weight_version": weight_version,
                    },
                    status_code=400,
                )
            if payload.get("double_buffer", False):
                self.staged_adapter_versions[payload.get("adapter_name", "adapter")] = adapter_version
                active_version = self.adapter_version
            else:
                self.adapter_version = adapter_version
                self.weight_version = weight_version
                active_version = adapter_version
            return {
                "success": True,
                "message": "ok",
                "adapter_version": adapter_version,
                "weight_version": weight_version,
                "staged_adapter_version": adapter_version,
                "active_adapter_version": active_version,
            }

        @self.app.post("/update_weight_version")
        async def update_weight_version(request: Request):
            payload = await request.json()
            self.request_log.append({"endpoint": "update_weight_version", "payload": payload})
            version = str(payload.get("weight_version", payload.get("new_version", "0")))
            self.weight_version = version
            self.adapter_version = version
            return {"success": True, "weight_version": version, "adapter_version": version}

        @self.app.post("/activate_adapter_version")
        async def activate_adapter_version(request: Request):
            payload = await request.json()
            self.request_log.append({"endpoint": "activate_adapter_version", "payload": payload})
            adapter_name = payload.get("adapter_name", "adapter")
            adapter_version = str(payload.get("adapter_version", payload.get("weight_version", "0")))
            weight_version = str(payload.get("weight_version", adapter_version))
            staged_version = self.staged_adapter_versions.get(adapter_name)
            if staged_version != adapter_version:
                return JSONResponse(
                    {
                        "success": False,
                        "message": f"No staged adapter version for {adapter_name}={adapter_version}",
                        "adapter_version": adapter_version,
                        "weight_version": weight_version,
                        "active_adapter_version": self.adapter_version,
                    },
                    status_code=400,
                )
            self.adapter_version = adapter_version
            self.weight_version = weight_version
            del self.staged_adapter_versions[adapter_name]
            return {
                "success": True,
                "message": "ok",
                "adapter_version": adapter_version,
                "weight_version": weight_version,
                "active_adapter_version": adapter_version,
            }


__all__ = ["OrbitMockSGLangServerExtensions"]
