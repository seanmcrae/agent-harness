"""Heuristic prompt-injection detection for untrusted text such as tool outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from guarded_agent.guardrails.base import Decision, Guardrail, GuardrailContext, Stage


@dataclass(frozen=True)
class InjectionRule:
    name: str
    pattern: re.Pattern[str]
    weight: float


def _rule(name: str, pattern: str, weight: float) -> InjectionRule:
    return InjectionRule(name, re.compile(pattern, re.IGNORECASE | re.MULTILINE), weight)


DEFAULT_RULES: tuple[InjectionRule, ...] = (
    _rule(
        "override_instructions",
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
        r"\b(previous|prior|above|earlier|all|any|your|system|original)\b[^.\n]{0,20}?"
        r"\b(instructions?|rules|prompts?|directions|guidelines|policy|policies)\b",
        1.0,
    ),
    _rule(
        "role_reassignment",
        r"\byou are now\b|\bfrom now on,? you\b|\bnew (system )?instructions?\s*:",
        0.8,
    ),
    _rule(
        "prompt_exfiltration",
        r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,30}?"
        r"\b(system prompt|hidden prompt|your instructions|api keys?|credentials|passwords?)\b",
        0.8,
    ),
    _rule(
        "credential_phishing",
        r"\b(send|email|share|provide|give)\b[^.\n]{0,30}?\b(their|your)\s+"
        r"(password|passcode|one[- ]time code|2fa code|pin)\b",
        0.8,
    ),
    _rule(
        "addressed_to_model",
        r"\b(note|message|instructions?|attention)\s+(to|for)\s+(the\s+)?"
        r"(ai|assistant|llm|language model|agent|chatbot)s?\b",
        0.6,
    ),
    _rule(
        "fake_role_marker",
        r"^\s*(system|assistant|developer)\s*:|<\|?(im_start|system)\|?>|\[/?(system|inst)\]",
        0.5,
    ),
    _rule(
        "concealment",
        r"\b(do not|don't|never)\s+(tell|inform|mention|reveal|let)\b[^.\n]{0,20}?"
        r"\b(the\s+)?(user|customer|human|operator)\b",
        0.6,
    ),
)


@dataclass
class InjectionDetector(Guardrail):
    """Score text against weighted rules and block at or above ``threshold``.

    Single weak signals (a log line starting with "System:") stay below the threshold; an explicit
    instruction override, or two corroborating weaker signals, crosses it.
    """

    threshold: float = 1.0
    rules: tuple[InjectionRule, ...] = DEFAULT_RULES
    name: str = "prompt_injection"
    stages: frozenset[Stage] = field(default_factory=lambda: frozenset({Stage.TOOL_OUTPUT}))

    def score(self, text: str) -> tuple[float, list[str]]:
        matched = [rule for rule in self.rules if rule.pattern.search(text)]
        return sum(rule.weight for rule in matched), [rule.name for rule in matched]

    def check(self, text: str, ctx: GuardrailContext) -> Decision:
        score, matched = self.score(text)
        if score >= self.threshold:
            return self.block(ctx, f"injection score {score:.1f} >= {self.threshold:.1f}", matched)
        return self.allow(ctx)
