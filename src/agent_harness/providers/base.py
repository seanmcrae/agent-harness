"""Provider-neutral completion interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent_harness.types import Message, ModelPricing, ToolSpec, Usage


@dataclass(frozen=True)
class CompletionRequest:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int = 1024
    temperature: float | None = None
    """None leaves the vendor default; some models reject an explicit temperature."""


@dataclass(frozen=True)
class CompletionResponse:
    message: Message
    usage: Usage
    model: str
    stop_reason: str = "end_turn"


class ProviderError(Exception):
    """A provider failure that retrying will not fix (bad request, auth, malformed response)."""


class TransientProviderError(ProviderError):
    """A provider failure worth retrying with backoff (rate limit, overload, network)."""


class Provider(ABC):
    name: str
    model: str
    pricing: ModelPricing

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one model turn. Raise TransientProviderError for retryable failures."""


TRANSIENT_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
TRANSIENT_ERROR_NAMES = frozenset(
    {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
)


def classify_sdk_error(exc: Exception) -> ProviderError:
    """Map an SDK exception to our error taxonomy without importing the SDK.

    Both the Anthropic and OpenAI SDKs expose ``status_code`` on HTTP errors and share the
    exception class names used here, so duck typing keeps the adapters import-light.
    """
    status = getattr(exc, "status_code", None)
    if type(exc).__name__ in TRANSIENT_ERROR_NAMES or status in TRANSIENT_STATUS_CODES:
        return TransientProviderError(f"{type(exc).__name__}: {exc}")
    return ProviderError(f"{type(exc).__name__}: {exc}")
