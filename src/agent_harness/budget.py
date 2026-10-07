"""Run budgets: step limit, token and cost ceilings, wall-clock timeout."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agent_harness.guardrails.base import Action, Decision, Stage
from agent_harness.types import Usage

BUDGET_GUARDRAIL = "budget"


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    timeout_s: float | None = 60.0


class BudgetTracker:
    """Accumulates spend for one run and turns limit breaches into BLOCK decisions."""

    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._started = clock()
        self.usage = Usage()
        self.cost_usd = 0.0

    def record(self, usage: Usage, cost_usd: float) -> None:
        self.usage = self.usage + usage
        self.cost_usd += cost_usd

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def remaining_s(self) -> float | None:
        if self.budget.timeout_s is None:
            return None
        return max(self.budget.timeout_s - self.elapsed_s, 0.0)

    def check_resources(self) -> Decision | None:
        """Tokens, cost, and time; checked before each step and before running tools."""
        b = self.budget
        if b.timeout_s is not None and self.elapsed_s >= b.timeout_s:
            return _block("timeout", f"wall clock {self.elapsed_s:.1f}s >= {b.timeout_s:.1f}s")
        if b.max_total_tokens is not None and self.usage.total_tokens >= b.max_total_tokens:
            return _block(
                "tokens", f"{self.usage.total_tokens} tokens >= limit {b.max_total_tokens}"
            )
        if b.max_cost_usd is not None and self.cost_usd >= b.max_cost_usd:
            return _block("cost", f"${self.cost_usd:.4f} >= limit ${b.max_cost_usd:.4f}")
        return None

    def check_step(self, step: int) -> Decision | None:
        if step > self.budget.max_steps:
            return _block("steps", f"step limit {self.budget.max_steps} reached")
        return self.check_resources()


def _block(kind: str, reason: str) -> Decision:
    return Decision(BUDGET_GUARDRAIL, Stage.BUDGET, Action.BLOCK, reason, (kind,))
