"""Deterministic offline provider driven by a script of turns or by a rule policy."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from guarded_agent.providers.base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderError,
)
from guarded_agent.providers.pricing import SIMULATED_PRICING
from guarded_agent.types import Message, ModelPricing, ToolCall, Usage


@dataclass(frozen=True)
class MockTurn:
    """One scripted model turn: final text, tool calls, or both."""

    text: str = ""
    calls: tuple[tuple[str, dict[str, Any]], ...] = field(default=())

    @classmethod
    def call(cls, name: str, **arguments: Any) -> MockTurn:
        return cls(calls=((name, arguments),))

    @classmethod
    def call_many(cls, *calls: tuple[str, dict[str, Any]]) -> MockTurn:
        return cls(calls=tuple(calls))

    @classmethod
    def final(cls, text: str) -> MockTurn:
        return cls(text=text)

    @classmethod
    def final_json(cls, payload: dict[str, Any]) -> MockTurn:
        return cls(text=json.dumps(payload))


Policy = Callable[[CompletionRequest], MockTurn]


def estimate_tokens(text: str) -> int:
    """Rough token count (about four characters per token), stable across runs."""
    return math.ceil(len(text) / 4) if text else 0


def _request_text(request: CompletionRequest) -> str:
    parts = [request.system]
    parts.extend(json.dumps([t.name, t.description, t.parameters]) for t in request.tools)
    for message in request.messages:
        parts.append(message.content)
        parts.extend(json.dumps([c.name, c.arguments]) for c in message.tool_calls)
    return "\n".join(parts)


class MockProvider(Provider):
    """Offline provider for tests, demos, and CI.

    Turns come from ``script`` (consumed in order) or, when no script is given, from ``policy``,
    a function of the request. ``faults`` are exceptions raised on the first calls, before any
    turn is produced, which is how retry behaviour is exercised.
    """

    name = "mock"

    def __init__(
        self,
        script: Iterable[MockTurn | Exception] | None = None,
        *,
        policy: Policy | None = None,
        faults: Iterable[Exception] = (),
        model: str = "mock-1",
        pricing: ModelPricing = SIMULATED_PRICING,
    ) -> None:
        if script is None and policy is None:
            raise ValueError("MockProvider needs a script or a policy")
        self.model = model
        self.pricing = pricing
        self._script: deque[MockTurn | Exception] | None = (
            deque(script) if script is not None else None
        )
        self._policy = policy
        self._faults = deque(faults)
        self._call_counter = 0
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self._faults:
            raise self._faults.popleft()
        turn = self._next_turn(request)
        tool_calls = tuple(self._make_call(name, args) for name, args in turn.calls)
        message = Message.assistant(turn.text, tool_calls)
        output_text = turn.text + "".join(json.dumps(c.arguments) for c in tool_calls)
        usage = Usage(
            input_tokens=estimate_tokens(_request_text(request)),
            output_tokens=max(1, estimate_tokens(output_text)),
        )
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return CompletionResponse(message, usage, self.model, stop_reason)

    def _next_turn(self, request: CompletionRequest) -> MockTurn:
        if self._script is None:
            assert self._policy is not None
            return self._policy(request)
        if not self._script:
            raise ProviderError("mock script exhausted")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def _make_call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        self._call_counter += 1
        return ToolCall(id=f"call_{self._call_counter}", name=name, arguments=dict(arguments))
