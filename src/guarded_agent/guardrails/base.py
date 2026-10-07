"""Guardrail contract: every check returns a typed decision that the agent records in the trace."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from guarded_agent.types import ToolCallRecord


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class Stage(StrEnum):
    INPUT = "input"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    OUTPUT = "output"
    BUDGET = "budget"


@dataclass(frozen=True)
class Decision:
    guardrail: str
    stage: Stage
    action: Action
    reason: str = ""
    findings: tuple[str, ...] = ()
    content: str | None = None
    """Replacement text when ``action`` is REDACT."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "guardrail": self.guardrail,
            "stage": self.stage.value,
            "action": self.action.value,
            "reason": self.reason,
            "findings": list(self.findings),
        }


@dataclass(frozen=True)
class GuardrailContext:
    stage: Stage
    agent: str
    tool_name: str | None = None
    tool_history: tuple[ToolCallRecord, ...] = ()


class Guardrail(ABC):
    name: str
    stages: frozenset[Stage]

    @abstractmethod
    def check(self, text: str, ctx: GuardrailContext) -> Decision:
        """Inspect ``text`` at ``ctx.stage`` and decide."""

    def allow(self, ctx: GuardrailContext) -> Decision:
        return Decision(self.name, ctx.stage, Action.ALLOW)

    def block(self, ctx: GuardrailContext, reason: str, findings: Sequence[str] = ()) -> Decision:
        return Decision(self.name, ctx.stage, Action.BLOCK, reason, tuple(findings))

    def redact(
        self, ctx: GuardrailContext, content: str, reason: str, findings: Sequence[str] = ()
    ) -> Decision:
        return Decision(self.name, ctx.stage, Action.REDACT, reason, tuple(findings), content)


@dataclass(frozen=True)
class GuardrailOutcome:
    text: str
    decisions: tuple[Decision, ...] = field(default=())

    @property
    def blocked(self) -> Decision | None:
        return next((d for d in self.decisions if d.action is Action.BLOCK), None)


class GuardrailSet:
    """Ordered guardrails. Redactions feed forward; the first block stops evaluation."""

    def __init__(self, guardrails: Sequence[Guardrail] = ()) -> None:
        self.guardrails = tuple(guardrails)

    def with_guardrail(self, guardrail: Guardrail) -> GuardrailSet:
        return GuardrailSet((*self.guardrails, guardrail))

    def apply(self, text: str, ctx: GuardrailContext) -> GuardrailOutcome:
        decisions: list[Decision] = []
        for guardrail in self.guardrails:
            if ctx.stage not in guardrail.stages:
                continue
            decision = guardrail.check(text, ctx)
            decisions.append(decision)
            if decision.action is Action.REDACT and decision.content is not None:
                text = decision.content
            elif decision.action is Action.BLOCK:
                break
        return GuardrailOutcome(text, tuple(decisions))
