"""
Fixtures to test custom-generate-function
"""

import copy
import uuid
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.session.server import SessionServer
from miles.utils.async_utils import run
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

MODEL_NAME = "Qwen/Qwen3-0.6B"
RESPONSE_TEXT = "\\boxed{8}"


def megatron_shape_argv(model_name: str) -> list[str]:
    """Megatron model-shape flags that agree with ``model_name``'s HF config.

    orbit: upstream builds these fixtures on ``--train-backend fsdp``, but orbit
    deletes the experimental FSDP backend and narrows ``--train-backend`` to
    ``choices=["megatron"]`` (orbit/arguments.py; ORBIT-SEAM in
    ``miles/utils/arguments.py::parse_args``). The megatron branch runs
    ``hf_validate_args``, which requires these flags to match the HF config, so
    derive them from the checkpoint rather than pinning one model's numbers.
    """
    from miles.utils.hf_config import load_hf_config

    cfg = load_hf_config(model_name)
    cfg = getattr(cfg, "text_config", cfg)

    rope_theta = getattr(cfg, "rope_theta", None)
    rope_parameters = getattr(cfg, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        rope_theta = rope_parameters["rope_theta"]

    argv: list[str] = []
    for flag, value in (
        ("--num-layers", getattr(cfg, "num_hidden_layers", None)),
        ("--hidden-size", getattr(cfg, "hidden_size", None)),
        ("--num-attention-heads", getattr(cfg, "num_attention_heads", None)),
        ("--ffn-hidden-size", getattr(cfg, "intermediate_size", None)),
        ("--moe-ffn-hidden-size", getattr(cfg, "moe_intermediate_size", None)),
        ("--norm-epsilon", getattr(cfg, "rms_norm_eps", None)),
        # --rotary-base is type=int; HF may store rope_theta as a float
        ("--rotary-base", None if rope_theta is None else int(rope_theta)),
    ):
        if value is not None:
            argv.extend([flag, str(value)])
    if getattr(cfg, "tie_word_embeddings", True) is False:
        argv.append("--untie-embeddings-and-output-weights")
    return argv


DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}

VARIANT_TO_GENERATE_FN_PATH = {
    "old_sglang_rollout": "miles.rollout.sglang_rollout.generate",
    "single_turn": "miles.rollout.generate_hub.single_turn.generate",
    "multi_turn": "miles.rollout.generate_hub.multi_turn.generate",
    "agentic_tool_call": "miles.rollout.generate_hub.agentic_tool_call.generate",
}


def extra_argv_for_variant(
    variant: str,
    *,
    custom_generate_function_path: str | None = None,
    generate_max_turns: int = 16,
    generate_tool_specs_path: str = "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    generate_tool_call_parser: str = "qwen25",
    generate_execute_tool_function_path: str = "miles.utils.test_utils.mock_tools.execute_tool_call",
    custom_agent_function_path: str = "miles.utils.test_utils.mock_tools.run_agentic_tool_call",
) -> list[str]:
    argv = [
        "--custom-generate-function-path",
        custom_generate_function_path or VARIANT_TO_GENERATE_FN_PATH[variant],
    ]

    if variant == "multi_turn":
        argv += [
            "--generate-max-turns",
            str(generate_max_turns),
            "--generate-tool-specs-path",
            generate_tool_specs_path,
            "--generate-execute-tool-function-path",
            generate_execute_tool_function_path,
        ]
        argv += ["--generate-tool-call-parser", generate_tool_call_parser]
    elif variant == "agentic_tool_call":
        argv += ["--custom-agent-function-path", custom_agent_function_path]
        argv += ["--use-session-server", "v2", "--tito-model", "qwen3"]

    return argv


def listify(x):
    return x if isinstance(x, list) else [x]


def make_sample(
    *,
    prompt: str | list[dict] = "What is 1+7?",
    tokens: list[int] | None = None,
    response: str = "",
    response_length: int = 0,
    status: Sample.Status = Sample.Status.PENDING,
    multimodal_inputs: dict | None = None,
) -> Sample:
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
    )


@dataclass
class GenerateEnv:
    args: Namespace
    mock_server: Any


@dataclass
class GenerateResult:
    sample: Sample | list[Sample]
    requests: list[dict]


def run_generate(
    env: GenerateEnv,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    *,
    variant: str = "single_turn",
) -> GenerateResult:
    env.mock_server.request_log.clear()
    result_sample = run(
        _call_generate(
            env.args,
            sample,
            sampling_params or DEFAULT_SAMPLING_PARAMS,
            variant=variant,
        )
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


async def _call_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    variant: str = "single_turn",
) -> Sample:
    generate_fn = load_generate_function(VARIANT_TO_GENERATE_FN_PATH[variant])
    state = GenerateState(args)
    input = GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params.copy(), evaluation=False)
    output = await generate_fn(input)
    return output.samples


def make_args(
    *,
    variant: str,
    router_port: int,
    use_rollout_routing_replay: bool = False,
    sglang_speculative_algorithm: str | None = None,
    model_name: str = MODEL_NAME,
    extra_argv: list[str] | None = None,
    custom_generate_function_path: str | None = None,
    generate_max_turns: int = 16,
    generate_tool_specs_path: str = "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    generate_tool_call_parser: str = "qwen25",
    generate_execute_tool_function_path: str = "miles.utils.test_utils.mock_tools.execute_tool_call",
    rollout_max_context_len: int | None = None,
    chat_template_path: str | None = None,
    num_layers: int | None = None,
    moe_router_topk: int | None = None,
) -> Namespace:
    argv = [
        "pytest",
        # orbit: upstream uses the FSDP backend here; orbit deletes it and narrows
        # --train-backend to choices=["megatron"], so the megatron shape flags that
        # hf_validate_args checks must be supplied too (see megatron_shape_argv).
        "--train-backend",
        "megatron",
        "--ci-test",
        "--rollout-batch-size",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--hf-checkpoint",
        model_name,
        "--prompt-data",
        "/dev/null",
        "--rm-type",
        "math",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--rollout-max-response-len",
        "16",
    ] + megatron_shape_argv(model_name)
    if chat_template_path:
        argv.extend(["--chat-template-path", chat_template_path])
    if use_rollout_routing_replay:
        argv.append("--use-rollout-routing-replay")
    if sglang_speculative_algorithm:
        argv.extend(["--sglang-speculative-algorithm", sglang_speculative_algorithm])
    if rollout_max_context_len is not None:
        argv.extend(["--rollout-max-context-len", str(rollout_max_context_len)])

    argv.extend(
        extra_argv_for_variant(
            variant,
            custom_generate_function_path=custom_generate_function_path,
            generate_max_turns=generate_max_turns,
            generate_tool_specs_path=generate_tool_specs_path,
            generate_tool_call_parser=generate_tool_call_parser,
            generate_execute_tool_function_path=generate_execute_tool_function_path,
        )
    )

    if extra_argv:
        argv.extend(extra_argv)

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()

    # R3 decode shape overrides — not CLI flags (derived from the model config
    # in production). Applied here, before with_session_server copies args into
    # the worker namespace, because sample assembly runs inside the worker.
    if num_layers is not None:
        args.num_layers = num_layers
    if moe_router_topk is not None:
        args.moe_router_topk = moe_router_topk

    init_http_client(args)
    return args


@contextmanager
def _noop_port(port: int):
    """No-op context manager that just yields the given port."""
    yield port


@contextmanager
def with_session_server(
    backend_url: str,
    args: Namespace,
    *,
    port: int,
):
    # Mirror start_session_server (router_manager.py): the id is minted into the
    # caller's per-port map, where OpenAIEndpointTracer.create reads it from.
    instance_id = uuid.uuid4().hex
    args.session_server_instance_ids = {port: instance_id}
    server_args = copy.deepcopy(args)
    server_args.miles_router_timeout = 30
    server_args.session_server_instance_id = instance_id
    session_server = SessionServer(server_args, backend_url=backend_url)

    server = UvicornThreadServer(session_server.app, host="127.0.0.1", port=port)
    server.start()

    try:
        yield port
    finally:
        server.stop()


@pytest.fixture
def generation_env(request, variant):
    # tests/conftest.py imports this fixture for every test; load the tokenizer-backed helper only when it is used.
    from miles.utils.test_utils import mock_tools

    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})
    args_kwargs = params.get("args_kwargs", {})
    model_name = args_kwargs.get("model_name", MODEL_NAME)
    custom_generate_function_path = VARIANT_TO_GENERATE_FN_PATH[variant]

    def process_fn(_):
        x = params.get("process_fn_kwargs", {})
        return ProcessResult(
            text=x.get("response_text", RESPONSE_TEXT),
            finish_reason=x.get("finish_reason", "stop"),
            cached_tokens=x.get("cached_tokens", 0),
            meta_info=ProcessResultMetaInfo(
                weight_version=x.get("weight_version"),
                routed_experts=x.get("routed_experts"),
                spec_accept_token_num=x.get("spec_accept_token_num"),
                spec_draft_token_num=x.get("spec_draft_token_num"),
                spec_verify_ct=x.get("spec_verify_ct"),
            ),
        )

    is_agentic = variant.startswith("agentic_tool_call")

    with with_mock_server(model_name=model_name, process_fn=process_fn) as mock_server:
        server_port = find_available_port(31000) if is_agentic else mock_server.port
        _FIXTURE_ONLY_KEYS = {"model_name", "agentic_return_metadata"}
        other_args_kwargs = {k: v for k, v in args_kwargs.items() if k not in _FIXTURE_ONLY_KEYS}
        args = make_args(
            variant=variant,
            router_port=server_port,
            model_name=model_name,
            custom_generate_function_path=custom_generate_function_path,
            **other_args_kwargs,
        )

        # Agentic variants need a SessionServer for TITO session tracking;
        # non-agentic variants talk directly to the mock sglang server.
        cm = with_session_server(mock_server.url, args, port=server_port) if is_agentic else _noop_port(server_port)

        with cm:
            if is_agentic:
                # Point session server address to the SessionServer we just started,
                # mirroring the driver-side contract set by start_session_server.
                args.session_server_ip = "127.0.0.1"
                args.session_server_ports = [server_port]
                mock_tools.AGENTIC_MAX_TURNS = args_kwargs.get("generate_max_turns")
                mock_tools.AGENTIC_RETURN_METADATA = args_kwargs.get("agentic_return_metadata")
            yield GenerateEnv(args=args, mock_server=mock_server)

    mock_tools.AGENTIC_MAX_TURNS = None
    mock_tools.AGENTIC_RETURN_METADATA = None
    SingletonMeta.clear_all_instances()
