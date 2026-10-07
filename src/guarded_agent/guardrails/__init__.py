"""Guardrails: typed allow / redact / block decisions at each stage of a run."""

from guarded_agent.guardrails.base import (
    Action,
    Decision,
    Guardrail,
    GuardrailContext,
    GuardrailOutcome,
    GuardrailSet,
    Stage,
)
from guarded_agent.guardrails.injection import InjectionDetector
from guarded_agent.guardrails.pii import PIIRedactor, luhn_valid
from guarded_agent.guardrails.policy import OutputPolicy, ToolAllowlist


def default_guardrails() -> GuardrailSet:
    """PII redaction on every text boundary plus injection screening of tool outputs."""
    return GuardrailSet([PIIRedactor(), InjectionDetector()])


__all__ = [
    "Action",
    "Decision",
    "Guardrail",
    "GuardrailContext",
    "GuardrailOutcome",
    "GuardrailSet",
    "InjectionDetector",
    "OutputPolicy",
    "PIIRedactor",
    "Stage",
    "ToolAllowlist",
    "default_guardrails",
    "luhn_valid",
]
