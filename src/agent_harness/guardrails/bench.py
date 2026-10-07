"""Measure guardrail precision and recall on a labelled SYNTHETIC case set.

Labels reflect intent (is this text trying to steer the model? which PII types does it contain?),
not what the heuristics happen to catch, so misses show up as lower recall.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from agent_harness.guardrails.injection import InjectionDetector
from agent_harness.guardrails.pii import PIIRedactor

PII_TYPES = ("email", "phone", "card")


@dataclass(frozen=True)
class Case:
    text: str
    injection: bool
    pii: frozenset[str]


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
        }


@dataclass
class BenchReport:
    cases: int
    injection: Confusion
    pii: dict[str, Confusion]
    misses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "injection": self.injection.to_dict(),
            "pii": {kind: c.to_dict() for kind, c in self.pii.items()},
            "misses": self.misses,
        }


def load_cases(path: str | Path | None = None) -> list[Case]:
    if path is None:
        source = resources.files("agent_harness.examples").joinpath(
            "data/synthetic_guardrail_cases.jsonl"
        )
        lines = source.read_text(encoding="utf-8").splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [_case(json.loads(line)) for line in lines if line.strip()]


def _case(raw: dict[str, Any]) -> Case:
    return Case(raw["text"], bool(raw["injection"]), frozenset(raw.get("pii", [])))


def run_bench(
    cases: Iterable[Case],
    detector: InjectionDetector | None = None,
    redactor: PIIRedactor | None = None,
) -> BenchReport:
    detector = detector or InjectionDetector()
    redactor = redactor or PIIRedactor()
    injection = Confusion()
    pii = {kind: Confusion() for kind in PII_TYPES}
    misses: list[str] = []
    count = 0
    for case in cases:
        count += 1
        score, _ = detector.score(case.text)
        flagged = score >= detector.threshold
        injection.add(flagged, case.injection)
        if flagged != case.injection:
            misses.append(f"injection {'FP' if flagged else 'FN'}: {case.text}")
        found = set(redactor.scan(case.text)[1])
        for kind in PII_TYPES:
            pii[kind].add(kind in found, kind in case.pii)
            if (kind in found) != (kind in case.pii):
                misses.append(f"{kind} {'FP' if kind in found else 'FN'}: {case.text}")
    return BenchReport(count, injection, pii, misses)
