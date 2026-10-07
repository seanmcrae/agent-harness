"""Regex PII redaction for emails, phone numbers, and payment card numbers (Luhn-checked)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from guarded_agent.guardrails.base import Decision, Guardrail, GuardrailContext, Stage

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
# 13-19 digits, optionally grouped by single spaces or hyphens.
CARD_CANDIDATE_RE = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
# Starts with an optional +country code or an opening paren, then digit groups.
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?\(?\d[\d .()-]{7,}\d(?![\w-])")
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
PHONE_MIN_DIGITS, PHONE_MAX_DIGITS = 10, 15


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


@dataclass
class PIIRedactor(Guardrail):
    """Replace PII with typed placeholders.

    Card candidates are only redacted when they pass the Luhn checksum, which keeps order numbers
    and tracking ids readable. Phone candidates need 10-15 digits, must not contain an ISO date,
    and, when written as a bare digit run, must have a North American length.
    """

    emails: bool = True
    phones: bool = True
    cards: bool = True
    name: str = "pii_redaction"
    stages: frozenset[Stage] = field(
        default_factory=lambda: frozenset({Stage.INPUT, Stage.TOOL_OUTPUT, Stage.OUTPUT})
    )

    def scan(self, text: str) -> tuple[str, list[str]]:
        """Return the redacted text and one finding label per redacted span."""
        findings: list[str] = []
        if self.emails:
            text = _substitute(EMAIL_RE, "email", text, findings)
        if self.cards:
            text = _substitute(
                CARD_CANDIDATE_RE, "card", text, findings, lambda s: luhn_valid(_digits(s))
            )
        if self.phones:
            text = _substitute(PHONE_CANDIDATE_RE, "phone", text, findings, _looks_like_phone)
        return text, findings

    def check(self, text: str, ctx: GuardrailContext) -> Decision:
        redacted, findings = self.scan(text)
        if not findings:
            return self.allow(ctx)
        kinds = sorted(set(findings))
        return self.redact(ctx, redacted, f"redacted {len(findings)} item(s)", kinds)


def _substitute(
    pattern: re.Pattern[str],
    label: str,
    text: str,
    findings: list[str],
    accept: Callable[[str], bool] | None = None,
) -> str:
    def replace(match: re.Match[str]) -> str:
        if accept is not None and not accept(match.group()):
            return match.group()
        findings.append(label)
        return f"[REDACTED_{label.upper()}]"

    return pattern.sub(replace, text)


def _looks_like_phone(candidate: str) -> bool:
    digits = _digits(candidate)
    if ISO_DATE_RE.search(candidate) or not PHONE_MIN_DIGITS <= len(digits) <= PHONE_MAX_DIGITS:
        return False
    if candidate.isdigit():
        # A bare digit run is far more often an id than a phone; accept only NANP lengths.
        return len(digits) == 10 or (len(digits) == 11 and digits.startswith("1"))
    return True
