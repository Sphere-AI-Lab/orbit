import asyncio
import re
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@dataclass(frozen=True)
class ProcessResultMetaInfo:
    weight_version: str | None = None
    routed_experts: str | None = None
    spec_accept_token_num: int | None = None
    spec_draft_token_num: int | None = None
    spec_verify_ct: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ProcessResult:
    text: str
    finish_reason: str = "stop"
    cached_tokens: int = 0
    meta_info: ProcessResultMetaInfo = ProcessResultMetaInfo()


ProcessFn = Callable[[str], ProcessResult]


class MockSGLangServer:
    def __init__(
        self,
        model_name: str,
        process_fn: ProcessFn,
        host: str,
        port: int,
        latency: float = 0.0,
        chat_template_path: str | None = None,
    ):
        self.tokenizer = load_tokenizer(model_name, chat_template_path=chat_template_path, trust_remote_code=True)
        self.process_fn = process_fn
        self.host = host
        self.port = port or find_available_port(30000)
        self.latency = latency

        self.app = FastAPI()
        self._server: UvicornThreadServer | None = None

        self.request_log: list[dict] = []
        self._concurrency = Counter()

        # ORBIT-SEAM: mock adapter/weight-version state for the PEFT double-buffer update endpoints
        # registered below (update_adapter_from_distributed / update_weight_version /
        # activate_adapter_version), mirroring the real SGLang engine's staged-then-activated adapter model
        self.weight_version = "0"
        self.adapter_version = "0"
        self.staged_adapter_versions = {}

        self._setup_routes()

    @property
    def max_concurrent(self) -> int:
        return self._concurrency.max_value

    def reset_stats(self):
        self.request_log.clear()
        self._concurrency.reset()

    def start(self):
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._server.start()

    def stop(self):
        if self._server is not None:
            self._server.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _setup_routes(self):
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

        # ORBIT-SEAM: mock endpoints for orbit's PEFT weight/adapter-version protocol -- model_info
        # exposes the current versions; update_adapter_from_distributed stages (double_buffer=True) or
        # immediately activates an adapter version; update_weight_version bumps both versions together
        # (full-FT path); activate_adapter_version promotes a staged version, rejecting an unstaged one
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

    async def _handle_generate_like_request(self, request: Request, compute_fn: Callable[[dict], dict]):
        payload = await request.json()
        self.request_log.append(payload)
        with self._concurrency.track():
            if self.latency > 0:
                await asyncio.sleep(self.latency)
            response = compute_fn(payload)
        return JSONResponse(content=response)

    # ORBIT-SEAM: removed base's `assert payload.get("return_logprob", True) is True` -- orbit callers
    # (e.g. compute_request_payload's return_logprob param) legitimately request return_logprob=False,
    # which this mock now honors below instead of rejecting
    def _compute_generate_response(self, payload: dict) -> dict:
        input_ids = payload.get("input_ids", [])

        prompt_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        prompt_tokens = len(input_ids)
        completion_tokens = len(output_ids)

        finish_reason_dict = {"type": process_result.finish_reason}
        if process_result.finish_reason == "length":
            finish_reason_dict["length"] = completion_tokens

        output_token_logprobs = [(-1 / 128 * i, token_id) for i, token_id in enumerate(output_ids)]

        meta_info = {
            "finish_reason": finish_reason_dict,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": process_result.cached_tokens,
            "completion_tokens": completion_tokens,
            # ORBIT-SEAM: output_token_logprobs moved out of the unconditional dict below -- only
            # included when the request actually set return_logprob (mirrors real SGLang and the
            # no-assert relaxation above)
            **process_result.meta_info.to_dict(),
        }
        if payload.get("return_logprob", False):
            meta_info["output_token_logprobs"] = output_token_logprobs

        # ORBIT-SEAM: output_ids now returned at the top level (generate_endpoint_utils prefers
        # output_ids over deriving tokens from output_token_logprobs when logprobs weren't requested)
        return {"text": process_result.text, "output_ids": output_ids, "meta_info": meta_info}

    def _compute_chat_completions_response(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        tools = payload.get("tools")

        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        input_ids = payload.get("input_ids")
        if input_ids is not None:
            prompt_ids = list(input_ids)
        else:
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        process_result = self.process_fn(prompt_str)
        output_ids = self.tokenizer.encode(process_result.text, add_special_tokens=False)

        logprobs_content = [
            {
                "token": self.tokenizer.convert_ids_to_tokens(tid),
                "token_id": tid,
                "logprob": -1 / 128 * i,
            }
            for i, tid in enumerate(output_ids)
        ]

        finish_reason = process_result.finish_reason
        tool_calls = None
        if tools and finish_reason == "stop":
            parser = FunctionCallParser(
                tools=TypeAdapter(list[Tool]).validate_python(tools),
                tool_call_parser="qwen25",
            )
            message_content, parsed_calls = parser.parse_non_stream(process_result.text)
            if parsed_calls:
                finish_reason = "tool_calls"
                tool_calls = [
                    {
                        "id": f"call{i:05d}",
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.parameters or "{}"},
                    }
                    for i, call in enumerate(parsed_calls)
                ]
        else:
            message_content = process_result.text

        output_token_logprobs = [(-1 / 128 * i, tid) for i, tid in enumerate(output_ids)]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                        "tool_calls": tool_calls,
                    },
                    "logprobs": {"content": logprobs_content},
                    "prompt_token_ids": prompt_ids,
                    "finish_reason": finish_reason,
                    "meta_info": {
                        "output_token_logprobs": output_token_logprobs,
                        "completion_tokens": len(output_ids),
                        **process_result.meta_info.to_dict(),
                    },
                }
            ],
        }


class Counter:
    def __init__(self):
        self._current = 0
        self._max = 0

    @property
    def max_value(self) -> int:
        return self._max

    def reset(self):
        self._current = 0
        self._max = 0

    @contextmanager
    def track(self):
        self._current += 1
        self._max = max(self._max, self._current)
        try:
            yield
        finally:
            self._current -= 1


def default_process_fn(prompt: str) -> ProcessResult:
    match = re.search(r"What is 1\+(\d+)\?", prompt)
    if match:
        num = int(match.group(1))
        ans = 1 + num
        return ProcessResult(text=f"\\boxed{{{ans}}}", finish_reason="stop")
    return ProcessResult(text="I don't understand.", finish_reason="stop")


@contextmanager
def with_mock_server(
    model_name: str = "Qwen/Qwen3-0.6B",
    process_fn: ProcessFn = default_process_fn,
    host: str = "127.0.0.1",
    port: int | None = None,
    latency: float = 0.0,
):
    server = MockSGLangServer(
        model_name=model_name,
        process_fn=process_fn,
        host=host,
        port=port,
        latency=latency,
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()
