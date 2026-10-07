"""Anthropic Messages API adapter (``pip install guarded-agent[anthropic]``)."""

from __future__ import annotations

import os
from typing import Any

from guarded_agent.providers.base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderError,
    classify_sdk_error,
)
from guarded_agent.providers.pricing import pricing_for
from guarded_agent.types import Message, ModelPricing, ToolCall, ToolSpec, Usage

DEFAULT_MODEL = "claude-sonnet-4-5"


def to_anthropic_tools(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
    ]


def to_anthropic_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """Translate to Anthropic's format: tool results ride in user turns as tool_result blocks.

    Consecutive tool results are merged into one user message, as the API requires.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
                "is_error": message.is_error,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in message.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": message.content})
    return out


def from_anthropic_response(response: Any) -> CompletionResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
    message = Message.assistant("".join(texts), tuple(calls))
    return CompletionResponse(message, usage, response.model, response.stop_reason or "end_turn")


class AnthropicProvider(Provider):
    name = "anthropic"

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
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(
                    "install the 'anthropic' extra to use AnthropicProvider"
                ) from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ProviderError("ANTHROPIC_API_KEY is not set")
            # Retries are owned by the agent loop so they show up in traces and budgets.
            client = anthropic.Anthropic(api_key=key, max_retries=0)
        self._client = client

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": to_anthropic_messages(request.messages),
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.tools:
            kwargs["tools"] = to_anthropic_tools(request.tools)
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise classify_sdk_error(exc) from exc
        return from_anthropic_response(response)
