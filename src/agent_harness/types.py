"""Conversation and accounting primitives shared by providers, tools, and the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]
ToolCallStatus = Literal["ok", "error", "invalid_args", "blocked", "denied"]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One provider-neutral conversation turn.

    Assistant messages may carry tool calls; tool messages carry the result of exactly one call.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, call_id: str, name: str, content: str, *, is_error: bool = False) -> Message:
        return cls(role="tool", content=content, tool_call_id=call_id, name=name, is_error=is_error)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class ToolSpec:
    """What a provider needs to advertise a tool to the model."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCallRecord:
    """The outcome of one attempted tool call, as seen by guardrails and evals.

    ``executed`` is True when the tool body ran, i.e. any side effect may have happened, even if
    it then raised. ``withheld`` means a guardrail kept the output away from the model.
    """

    name: str
    arguments: dict[str, Any]
    status: ToolCallStatus
    executed: bool = False
    output: str | None = None
    error: str | None = None
    withheld: bool = False
