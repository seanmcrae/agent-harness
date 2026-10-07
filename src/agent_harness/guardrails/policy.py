"""Configuration-driven guardrails: per-agent tool allowlists and output content policies."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from agent_harness.guardrails.base import Decision, Guardrail, GuardrailContext, Stage


@dataclass
class ToolAllowlist(Guardrail):
    """Block calls to tools outside the agent's allowlist, including hallucinated tool names."""

    allowed: frozenset[str]
    name: str = "tool_allowlist"
    stages: frozenset[Stage] = field(default_factory=lambda: frozenset({Stage.TOOL_CALL}))

    @classmethod
    def of(cls, names: Collection[str]) -> ToolAllowlist:
        return cls(allowed=frozenset(names))

    def check(self, text: str, ctx: GuardrailContext) -> Decision:
        if ctx.tool_name in self.allowed:
            return self.allow(ctx)
        return self.block(ctx, f"tool {ctx.tool_name!r} is not allowed for agent {ctx.agent!r}")


@dataclass
class OutputPolicy(Guardrail):
    """Block final answers matching any named forbidden pattern."""

    forbidden: Mapping[str, re.Pattern[str]]
    name: str = "output_policy"
    stages: frozenset[Stage] = field(default_factory=lambda: frozenset({Stage.OUTPUT}))

    @classmethod
    def from_patterns(cls, patterns: Mapping[str, str]) -> OutputPolicy:
        return cls({key: re.compile(value, re.IGNORECASE) for key, value in patterns.items()})

    def check(self, text: str, ctx: GuardrailContext) -> Decision:
        hits = [key for key, pattern in self.forbidden.items() if pattern.search(text)]
        if hits:
            return self.block(ctx, "final answer violates output policy", hits)
        return self.allow(ctx)
