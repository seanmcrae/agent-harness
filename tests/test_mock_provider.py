import pytest

from guarded_agent.providers import (
    CompletionRequest,
    MockProvider,
    MockTurn,
    ProviderError,
    TransientProviderError,
)
from guarded_agent.providers.base import classify_sdk_error
from guarded_agent.providers.pricing import SIMULATED_PRICING, UNKNOWN_PRICING, pricing_for
from guarded_agent.types import Message, ModelPricing, Usage


def _request(text: str = "hello") -> CompletionRequest:
    return CompletionRequest(system="be brief", messages=(Message.user(text),))


def test_script_is_consumed_in_order_with_stable_call_ids() -> None:
    provider = MockProvider([MockTurn.call("lookup", id="1"), MockTurn.final("done")])

    first = provider.complete(_request())
    second = provider.complete(_request())

    assert first.message.tool_calls[0].id == "call_1"
    assert first.message.tool_calls[0].arguments == {"id": "1"}
    assert first.stop_reason == "tool_use"
    assert second.message.content == "done"
    assert second.stop_reason == "end_turn"


def test_exhausted_script_raises_provider_error() -> None:
    provider = MockProvider([MockTurn.final("only")])
    provider.complete(_request())
    with pytest.raises(ProviderError, match="exhausted"):
        provider.complete(_request())


def test_faults_are_raised_before_turns() -> None:
    provider = MockProvider([MockTurn.final("ok")], faults=[TransientProviderError("overloaded")])
    with pytest.raises(TransientProviderError):
        provider.complete(_request())
    assert provider.complete(_request()).message.content == "ok"


def test_policy_sees_the_request() -> None:
    provider = MockProvider(policy=lambda req: MockTurn.final(req.messages[-1].content.upper()))
    assert provider.complete(_request("ping")).message.content == "PING"


def test_usage_grows_with_prompt_length() -> None:
    provider = MockProvider(policy=lambda _: MockTurn.final("x"))
    short = provider.complete(_request("a")).usage
    long = provider.complete(_request("a" * 400)).usage
    assert long.input_tokens - short.input_tokens == 100


def test_requires_script_or_policy() -> None:
    with pytest.raises(ValueError, match="script or a policy"):
        MockProvider()


def test_pricing_math_and_lookup() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert ModelPricing(3.0, 15.0).cost(usage) == pytest.approx(4.5)
    assert pricing_for("claude-sonnet-4-5-20250929").input_per_mtok == 3.0
    assert pricing_for("gpt-4o-mini-2024-07-18").input_per_mtok == 0.15
    assert pricing_for("some-unknown-model") == UNKNOWN_PRICING
    assert SIMULATED_PRICING.output_per_mtok == 15.0


class _FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class RateLimitError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (_FakeStatusError(429), True),
        (_FakeStatusError(529), True),
        (_FakeStatusError(400), False),
        (RateLimitError("slow down"), True),
        (ValueError("bad"), False),
    ],
)
def test_sdk_error_classification(exc: Exception, transient: bool) -> None:
    assert isinstance(classify_sdk_error(exc), TransientProviderError) is transient
