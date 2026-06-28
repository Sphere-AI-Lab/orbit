"""OpenAI-compatible wrapper around SGLang tool-call parsing."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from orbit_plugins.tau_bench.sglang_tool_parser import parse_tools

logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall:
    id: str
    type: str = "function"
    function: dict[str, Any] | None = None


@dataclass
class OpenAIAssistantMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    def __init__(self, tools_info: list[dict[str, Any]], parser_type: str = "qwen25"):
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> dict[str, Any]:
        try:
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]
            return {
                "openai_message": self._convert_to_openai_message(normal_text, calls),
                "parsed_result": parsed,
                "success": True,
            }
        except Exception as exc:
            logger.warning("Tau-bench tool parsing failed: %s", exc)
            return {
                "openai_message": None,
                "parsed_result": None,
                "success": False,
                "error": str(exc),
            }

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        if not calls:
            return OpenAIAssistantMessage(content=normal_text, tool_calls=None)

        tool_calls = []
        for idx, call in enumerate(calls):
            tool_calls.append(
                OpenAIToolCall(
                    id=f"call_{idx}_{call.get('name', 'unknown')}",
                    function={
                        "name": call.get("name", ""),
                        "arguments": call.get("parameters", "{}"),
                    },
                )
            )
        return OpenAIAssistantMessage(content=normal_text if normal_text.strip() else None, tool_calls=tool_calls)

    def call_to_action(self, calls: list[dict[str, Any]], text_response: str):
        from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
        from tau_bench.types import Action

        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
        if not calls:
            return action

        if len(calls) > 1:
            logger.debug("Multiple tool calls identified; using the first one.")

        tool_call = calls[0]
        params = json.loads(tool_call["parameters"])
        if not isinstance(params, dict):
            logger.warning("Tool call parameters are not a dict: %r", params)
            return action
        return Action(name=tool_call["name"], kwargs=params)

    def get_openai_tools_format(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            for tool in self.tools_info
        ]


def create_openai_adapter(
    tools_info: list[dict[str, Any]], parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
