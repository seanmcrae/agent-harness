"""Guardrails: typed allow / redact / block decisions at each stage of a run."""

from agent_harness.guardrails.base import (
    Action,
    Decision,
    Guardrail,
    GuardrailContext,
    GuardrailOutcome,
    GuardrailSet,
    Stage,
)
from agent_harness.guardrails.injection import InjectionDetector
from agent_harness.guardrails.pii import PIIRedactor, luhn_valid
from agent_harness.guardrails.policy import OutputPolicy, ToolAllowlist


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
