"""Adapter translation tests against fake SDK clients; no network and no SDK install needed."""

from types import SimpleNamespace as NS
from typing import Any

import pytest

from guarded_agent.providers import (
    CompletionRequest,
    ProviderError,
    TransientProviderError,
    create_provider,
)
from guarded_agent.providers.anthropic import AnthropicProvider, to_anthropic_messages
from guarded_agent.providers.openai import (
    OpenAIChatProvider,
    OpenAIResponsesProvider,
    parse_arguments,
    to_chat_messages,
    to_responses_input,
)
from guarded_agent.types import Message, ToolCall, ToolSpec

TOOL = ToolSpec("lookup", "Look up an order.", {"type": "object", "properties": {}})
CONVERSATION = (
    Message.user("refund ORD-1"),
    Message.assistant(
        "Checking.",
        (ToolCall("c1", "lookup", {"id": "ORD-1"}), ToolCall("c2", "policy", {"id": "ORD-1"})),
    ),
    Message.tool("c1", "lookup", '{"total": 10}'),
    Message.tool("c2", "policy", "not found", is_error=True),
)
REQUEST = CompletionRequest(system="sys", messages=CONVERSATION, tools=(TOOL,), max_tokens=256)


class Recorder:
    """Stands in for an SDK resource: records kwargs and returns or raises a canned value."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_anthropic_message_translation_merges_tool_results() -> None:
    translated = to_anthropic_messages(CONVERSATION)
    assert [m["role"] for m in translated] == ["user", "assistant", "user"]
    assert translated[1]["content"][0] == {"type": "text", "text": "Checking."}
    assert translated[1]["content"][1]["type"] == "tool_use"
    results = translated[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]
    assert results[1]["is_error"] is True


def test_anthropic_provider_round_trip() -> None:
    response = NS(
        content=[
            NS(type="text", text="Let me check."),
            NS(type="tool_use", id="tu_1", name="lookup", input={"id": "ORD-2"}),
        ],
        usage=NS(input_tokens=120, output_tokens=30),
        model="claude-sonnet-4-5-20250929",
        stop_reason="tool_use",
    )
    messages = Recorder(response)
    provider = AnthropicProvider(client=NS(messages=messages))

    result = provider.complete(REQUEST)

    assert messages.kwargs["system"] == "sys"
    assert messages.kwargs["tools"][0]["input_schema"] == TOOL.parameters
    assert messages.kwargs["max_tokens"] == 256
    assert result.message.content == "Let me check."
    assert result.message.tool_calls == (ToolCall("tu_1", "lookup", {"id": "ORD-2"}),)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert provider.pricing.input_per_mtok == 3.0


def test_openai_chat_translation() -> None:
    translated = to_chat_messages("sys", CONVERSATION)
    assert translated[0] == {"role": "system", "content": "sys"}
    call = translated[2]["tool_calls"][0]
    assert call["function"] == {"name": "lookup", "arguments": '{"id": "ORD-1"}'}
    assert translated[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"total": 10}'}


def test_openai_chat_provider_round_trip() -> None:
    response = NS(
        choices=[
            NS(
                message=NS(
                    content=None,
                    tool_calls=[
                        NS(id="call_9", function=NS(name="lookup", arguments='{"id": "ORD-3"}'))
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=NS(prompt_tokens=50, completion_tokens=10),
        model="gpt-4.1-mini",
    )
    completions = Recorder(response)
    provider = OpenAIChatProvider(client=NS(chat=NS(completions=completions)))

    result = provider.complete(REQUEST)

    assert completions.kwargs["tools"][0]["function"]["name"] == "lookup"
    assert completions.kwargs["max_completion_tokens"] == 256
    assert result.message.tool_calls[0].arguments == {"id": "ORD-3"}
    assert result.message.content == ""
    assert result.usage.total_tokens == 60


def test_openai_responses_translation_and_round_trip() -> None:
    items = to_responses_input(CONVERSATION)
    assert [i.get("type", i.get("role")) for i in items] == [
        "user",
        "assistant",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]

    response = NS(
        output=[
            NS(type="reasoning"),
            NS(type="message", content=[NS(type="output_text", text='{"status": "ok"}')]),
        ],
        usage=NS(input_tokens=80, output_tokens=12),
        model="gpt-4.1-mini",
        status="completed",
    )
    responses = Recorder(response)
    provider = OpenAIResponsesProvider(client=NS(responses=responses))
    result = provider.complete(REQUEST)

    assert responses.kwargs["instructions"] == "sys"
    assert responses.kwargs["tools"][0]["name"] == "lookup"
    assert result.message.content == '{"status": "ok"}'
    assert result.stop_reason == "completed"


class RateLimitError(Exception):
    status_code = 429


class BadRequestError(Exception):
    status_code = 400


@pytest.mark.parametrize(
    ("error", "expected"),
    [(RateLimitError("slow"), TransientProviderError), (BadRequestError("no"), ProviderError)],
)
def test_sdk_errors_are_classified(error: Exception, expected: type[Exception]) -> None:
    provider = AnthropicProvider(client=NS(messages=Recorder(error)))
    with pytest.raises(expected) as info:
        provider.complete(REQUEST)
    assert type(info.value) is expected


def test_parse_arguments_handles_malformed_json() -> None:
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("") == {}
    assert parse_arguments("{oops") == {"_unparseable_arguments": "{oops"}
    assert parse_arguments("[1]") == {"_unparseable_arguments": "[1]"}


def test_real_providers_require_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in ("anthropic", "openai", "openai-responses"):
        with pytest.raises(ProviderError):
            create_provider(name)


def test_create_provider_validates_names() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        create_provider("cohere")
    with pytest.raises(ValueError, match="policy"):
        create_provider("mock")
