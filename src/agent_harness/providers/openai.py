"""OpenAI adapters for Chat Completions and the Responses API (``agent-harness[openai]``)."""

from __future__ import annotations

import json
import os
from typing import Any

from agent_harness.providers.base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderError,
    classify_sdk_error,
)
from agent_harness.providers.pricing import pricing_for
from agent_harness.types import Message, ModelPricing, ToolCall, ToolSpec, Usage

DEFAULT_MODEL = "gpt-4.1-mini"


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Decode tool-call arguments; malformed JSON becomes a field validation will reject."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparseable_arguments": raw}
    return value if isinstance(value, dict) else {"_unparseable_arguments": raw}


# Chat Completions -----------------------------------------------------------------------


def to_chat_tools(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def to_chat_messages(system: str, messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "tool":
            out.append(
                {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
            )
        elif message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in message.tool_calls
                ]
            out.append(entry)
        else:
            out.append({"role": "user", "content": message.content})
    return out


def from_chat_response(response: Any) -> CompletionResponse:
    choice = response.choices[0]
    calls = tuple(
        ToolCall(id=c.id, name=c.function.name, arguments=parse_arguments(c.function.arguments))
        for c in (choice.message.tool_calls or [])
    )
    usage = Usage(response.usage.prompt_tokens, response.usage.completion_tokens)
    message = Message.assistant(choice.message.content or "", calls)
    return CompletionResponse(message, usage, response.model, choice.finish_reason or "stop")


# Responses API --------------------------------------------------------------------------


def to_responses_tools(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


def to_responses_input(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        elif message.role == "assistant":
            if message.content:
                items.append({"role": "assistant", "content": message.content})
            items.extend(
                {
                    "type": "function_call",
                    "call_id": c.id,
                    "name": c.name,
                    "arguments": json.dumps(c.arguments),
                }
                for c in message.tool_calls
            )
        else:
            items.append({"role": "user", "content": message.content})
    return items


def from_responses_response(response: Any) -> CompletionResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for item in response.output:
        if item.type == "message":
            texts.extend(part.text for part in item.content if part.type == "output_text")
        elif item.type == "function_call":
            calls.append(ToolCall(item.call_id, item.name, parse_arguments(item.arguments)))
    usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
    stop = "tool_use" if calls else (response.status or "completed")
    return CompletionResponse(
        Message.assistant("".join(texts), tuple(calls)), usage, response.model, stop
    )


# Providers ------------------------------------------------------------------------------


def _client(api_key: str | None) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ProviderError("install the 'openai' extra to use the OpenAI providers") from exc
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set")
    # Retries are owned by the agent loop so they show up in traces and budgets.
    return openai.OpenAI(api_key=key, max_retries=0)


class OpenAIChatProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        self.model = model
        self.pricing = pricing or pricing_for(model)
        self._client = client if client is not None else _client(api_key)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_chat_messages(request.system, request.messages),
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = to_chat_tools(request.tools)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise classify_sdk_error(exc) from exc
        return from_chat_response(response)


class OpenAIResponsesProvider(Provider):
    name = "openai-responses"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        self.model = model
        self.pricing = pricing or pricing_for(model)
        self._client = client if client is not None else _client(api_key)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": request.system,
            "input": to_responses_input(request.messages),
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = to_responses_tools(request.tools)
        try:
            response = self._client.responses.create(**kwargs)
        except Exception as exc:
            raise classify_sdk_error(exc) from exc
        return from_responses_response(response)
