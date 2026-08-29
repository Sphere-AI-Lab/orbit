"""Local SGLang tool-call parser adapter."""

from typing import Any


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25") -> dict[str, Any]:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.managers.io_struct import Function, Tool

    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]
    tool_parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser)
    normal_text, calls = tool_parser.parse_non_stream(response)
    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() for call in calls],
    }

