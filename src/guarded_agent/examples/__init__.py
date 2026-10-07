"""Example agents with demo tools over SYNTHETIC data, plus their offline mock policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from guarded_agent.agent import AgentSpec
from guarded_agent.providers import Policy

from .refund import build_refund_spec, refund_policy
from .research import build_research_spec, research_policy


@dataclass(frozen=True)
class ExampleAgent:
    name: str
    description: str
    build_spec: Callable[..., AgentSpec]
    """Called as ``build_spec(guardrails=bool)``; returns a spec with fresh SYNTHETIC state."""
    mock_policy: Policy


EXAMPLES: dict[str, ExampleAgent] = {
    "refund": ExampleAgent(
        "refund",
        "Support refunds: look up order, check policy, issue refund (needs approval).",
        build_refund_spec,
        refund_policy,
    ),
    "research": ExampleAgent(
        "research",
        "Answer questions from a local knowledge base with citations.",
        build_research_spec,
        research_policy,
    ),
}


def get_example(name: str) -> ExampleAgent:
    try:
        return EXAMPLES[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {', '.join(EXAMPLES)}") from None
