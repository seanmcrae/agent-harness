"""Scenario evals: YAML scenarios run against example agents, checked, and summarised."""

from __future__ import annotations

import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_harness.agent import Agent, RunResult, RunStatus
from agent_harness.approval import approve_all, deny_all
from agent_harness.examples import ExampleAgent, get_example
from agent_harness.guardrails import GuardrailSet
from agent_harness.providers import MockProvider, Provider, TransientProviderError
from agent_harness.tracing import SpanExporter


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolExpectations(_Strict):
    sequence: list[str] = Field(default_factory=list)
    """Tools that must have executed in this relative order (other calls may interleave)."""
    must_include: list[str] = Field(default_factory=list)
    max_calls: int | None = None


class Expectations(_Strict):
    status: RunStatus = RunStatus.COMPLETED
    output: dict[str, Any] = Field(default_factory=dict)
    """Fields of the structured output that must equal these values."""
    output_contains: dict[str, str] = Field(default_factory=dict)
    """Fields whose string form must contain this text (case-insensitive)."""
    tools: ToolExpectations = Field(default_factory=ToolExpectations)
    forbidden_tools: list[str] = Field(default_factory=list)
    """Tools that must never execute. Attempts stopped by a guardrail or reviewer are fine."""
    forbidden_output_patterns: list[str] = Field(default_factory=list)
    guardrails_triggered: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    max_cost_usd: float | None = None


class BudgetOverride(_Strict):
    max_steps: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    timeout_s: float | None = None


class Scenario(_Strict):
    name: str
    agent: str
    input: str
    description: str = ""
    approval: Literal["approve", "deny"] = "approve"
    allowed_tools: list[str] | None = None
    budget: BudgetOverride | None = None
    provider_faults: int = Field(default=0, ge=0)
    """Transient provider errors injected before the first turn (mock provider only)."""
    expect: Expectations = Field(default_factory=Expectations)


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load one YAML file or every *.yaml / *.yml file in a directory (sorted by name)."""
    root = Path(path)
    files = sorted([*root.glob("*.yaml"), *root.glob("*.yml")]) if root.is_dir() else [root]
    scenarios: list[Scenario] = []
    for file in files:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
        items = data.get("scenarios", [data]) if isinstance(data, dict) else data
        scenarios.extend(Scenario.model_validate(item) for item in items)
    names = [s.name for s in scenarios]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario names: {duplicates}")
    return scenarios


def check(scenario: Scenario, result: RunResult) -> list[str]:
    """Return human-readable failures; an empty list means the scenario passed."""
    expect = scenario.expect
    failures: list[str] = []
    if result.status is not expect.status:
        detail = f" ({result.error})" if result.error else ""
        failures.append(f"status {result.status.value} != {expect.status.value}{detail}")
    failures += _check_output(expect, result)
    failures += _check_tools(expect, result)
    absent = [g for g in expect.guardrails_triggered if g not in result.triggered_guardrails]
    if absent:
        failures.append(f"guardrails did not trigger: {absent}")
    if expect.max_steps is not None and result.steps > expect.max_steps:
        failures.append(f"{result.steps} steps > max {expect.max_steps}")
    if expect.max_cost_usd is not None and result.cost_usd > expect.max_cost_usd:
        failures.append(f"cost ${result.cost_usd:.4f} > max ${expect.max_cost_usd:.4f}")
    return failures


def _check_output(expect: Expectations, result: RunResult) -> list[str]:
    failures: list[str] = []
    output = result.output.model_dump(mode="json") if result.output is not None else {}
    for key, want in expect.output.items():
        if output.get(key) != want:
            failures.append(f"output.{key} = {output.get(key)!r}, expected {want!r}")
    for key, needle in expect.output_contains.items():
        if needle.lower() not in str(output.get(key, "")).lower():
            failures.append(f"output.{key} does not contain {needle!r}")
    text = result.final_text or ""
    failures.extend(
        f"output matches forbidden pattern {pattern!r}"
        for pattern in expect.forbidden_output_patterns
        if re.search(pattern, text, re.IGNORECASE)
    )
    return failures


def _check_tools(expect: Expectations, result: RunResult) -> list[str]:
    failures: list[str] = []
    executed = [r.name for r in result.tool_calls if r.executed]
    if not _is_subsequence(expect.tools.sequence, executed):
        failures.append(f"tool sequence {executed} lacks ordered {expect.tools.sequence}")
    missing = [t for t in expect.tools.must_include if t not in executed]
    if missing:
        failures.append(f"tools never executed: {missing}")
    if expect.tools.max_calls is not None and len(result.tool_calls) > expect.tools.max_calls:
        failures.append(f"{len(result.tool_calls)} tool calls > max {expect.tools.max_calls}")
    forbidden = sorted({t for t in executed if t in expect.forbidden_tools})
    if forbidden:
        failures.append(f"forbidden tools executed: {forbidden}")
    return failures


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    remaining = iter(haystack)
    return all(item in remaining for item in needle)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    run: RunResult
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class EvalReport:
    results: list[ScenarioResult]
    guardrails_enabled: bool

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def mean_steps(self) -> float:
        return statistics.fmean(r.run.steps for r in self.results) if self.results else 0.0

    @property
    def mean_cost_usd(self) -> float:
        return statistics.fmean(r.run.cost_usd for r in self.results) if self.results else 0.0

    @property
    def guardrail_triggers(self) -> Counter[str]:
        return Counter(name for r in self.results for name in r.run.triggered_guardrails)

    def to_dict(self) -> dict[str, Any]:
        return {
            "guardrails_enabled": self.guardrails_enabled,
            "scenarios": len(self.results),
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_steps": round(self.mean_steps, 3),
            "mean_cost_usd": round(self.mean_cost_usd, 6),
            "guardrail_triggers": dict(sorted(self.guardrail_triggers.items())),
            "results": [
                {
                    "name": r.scenario.name,
                    "agent": r.scenario.agent,
                    "passed": r.passed,
                    "status": r.run.status.value,
                    "steps": r.run.steps,
                    "cost_usd": round(r.run.cost_usd, 6),
                    "tools": [f"{t.name}:{t.status}" for t in r.run.tool_calls],
                    "guardrails": r.run.triggered_guardrails,
                    "failures": r.failures,
                }
                for r in self.results
            ],
        }


ProviderFactory = Callable[[ExampleAgent, Scenario], Provider]


def mock_provider_factory(example: ExampleAgent, scenario: Scenario) -> Provider:
    faults = [TransientProviderError("injected fault") for _ in range(scenario.provider_faults)]
    return MockProvider(policy=example.mock_policy, faults=faults)


def run_scenario(
    scenario: Scenario,
    *,
    guardrails: bool = True,
    provider_factory: ProviderFactory = mock_provider_factory,
    exporters: Sequence[SpanExporter] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> ScenarioResult:
    example = get_example(scenario.agent)
    spec = example.build_spec(guardrails=guardrails)
    budget = spec.budget
    if scenario.budget is not None:
        budget = replace(budget, **scenario.budget.model_dump(exclude_none=True))
    spec = spec.with_overrides(
        allowed_tools=scenario.allowed_tools,
        budget=budget,
        guardrails=None if guardrails else GuardrailSet(),
    )
    agent = Agent(
        spec,
        provider_factory(example, scenario),
        approver=approve_all if scenario.approval == "approve" else deny_all,
        exporters=exporters,
        sleep=sleep,
        seed=0,
    )
    result = agent.run(scenario.input)
    return ScenarioResult(scenario, result, check(scenario, result))


def run_suite(
    scenarios: Sequence[Scenario],
    *,
    guardrails: bool = True,
    provider_factory: ProviderFactory = mock_provider_factory,
    exporters: Sequence[SpanExporter] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> EvalReport:
    results = [
        run_scenario(
            s,
            guardrails=guardrails,
            provider_factory=provider_factory,
            exporters=exporters,
            sleep=sleep,
        )
        for s in scenarios
    ]
    return EvalReport(results, guardrails)
