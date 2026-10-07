"""Model providers: a common interface, real API adapters, and a deterministic mock."""

from guarded_agent.providers.base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderError,
    TransientProviderError,
)
from guarded_agent.providers.mock import MockProvider, MockTurn, Policy

__all__ = [
    "PROVIDER_NAMES",
    "CompletionRequest",
    "CompletionResponse",
    "MockProvider",
    "MockTurn",
    "Policy",
    "Provider",
    "ProviderError",
    "TransientProviderError",
    "create_provider",
]


PROVIDER_NAMES = ("mock", "anthropic", "openai", "openai-responses")


def create_provider(
    name: str, model: str | None = None, *, policy: Policy | None = None
) -> Provider:
    """Build a provider by name. Real adapters import their SDK lazily and read keys from env."""
    if name == "mock":
        if policy is None:
            raise ValueError("the mock provider needs a policy")
        return MockProvider(policy=policy)
    if name == "anthropic":
        from guarded_agent.providers.anthropic import DEFAULT_MODEL, AnthropicProvider

        return AnthropicProvider(model or DEFAULT_MODEL)
    if name in ("openai", "openai-responses"):
        from guarded_agent.providers.openai import (
            DEFAULT_MODEL as OPENAI_DEFAULT,
        )
        from guarded_agent.providers.openai import (
            OpenAIChatProvider,
            OpenAIResponsesProvider,
        )

        cls = OpenAIChatProvider if name == "openai" else OpenAIResponsesProvider
        return cls(model or OPENAI_DEFAULT)
    raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDER_NAMES)}")
