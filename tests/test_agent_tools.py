"""Tool-execution edge cases inside the agent loop: crashes, timeouts, retries, side effects."""

import threading

import pytest

from agent_harness.agent import Agent, AgentSpec, RunStatus
from agent_harness.approval import approve_all
from agent_harness.guardrails import GuardrailSet
from agent_harness.providers import MockProvider, MockTurn
from agent_harness.tools import SideEffect, ToolRegistry, tool

FINAL = MockTurn.final("done")


def _agent(registry: ToolRegistry, script: list[MockTurn]) -> Agent:
    spec = AgentSpec(name="edge", system_prompt="test", tools=registry, guardrails=GuardrailSet())
    return Agent(spec, MockProvider(script), approver=approve_all)


def test_unexpected_tool_exception_is_contained() -> None:
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("disk on fire")

    result = _agent(ToolRegistry([explode]), [MockTurn.call("explode"), FINAL]).run("go")
    assert result.status is RunStatus.COMPLETED
    record = result.tool_calls[0]
    assert record.status == "error"
    assert record.executed  # it ran, so side effects cannot be ruled out
    assert record.error == "explode failed: RuntimeError: disk on fire"


def test_idempotent_tool_is_retried_once_after_timeout() -> None:
    calls: list[int] = []
    release = threading.Event()

    @tool(timeout_s=0.05)
    def flaky_read() -> str:
        """Hangs on the first call only."""
        calls.append(1)
        if len(calls) == 1:
            release.wait(1.0)
        return "fresh"

    result = _agent(ToolRegistry([flaky_read]), [MockTurn.call("flaky_read"), FINAL]).run("go")
    release.set()
    assert len(calls) == 2
    assert result.tool_calls[0].status == "ok"
    assert result.tool_calls[0].output == "fresh"


def test_non_idempotent_tool_is_not_retried_after_timeout() -> None:
    calls: list[int] = []
    release = threading.Event()

    @tool(side_effect=SideEffect.WRITE, timeout_s=0.05)
    def charge() -> str:
        """Hangs, simulating a payment API that never answers."""
        calls.append(1)
        release.wait(1.0)
        return "charged"

    result = _agent(ToolRegistry([charge]), [MockTurn.call("charge"), FINAL]).run("go")
    release.set()
    assert len(calls) == 1
    assert result.tool_calls[0].status == "error"
    assert "timed out" in (result.tool_calls[0].error or "")


def test_failed_write_still_counts_as_attempted_for_dedup() -> None:
    calls: list[str] = []

    @tool(side_effect=SideEffect.WRITE)
    def send_email(to: str) -> str:
        """Send an email; fails after the message may have left."""
        calls.append(to)
        raise RuntimeError("connection reset after send")

    call = MockTurn.call("send_email", to="a@example.com")
    result = _agent(ToolRegistry([send_email]), [call, call, FINAL]).run("go")
    assert calls == ["a@example.com"]
    assert [r.status for r in result.tool_calls] == ["error", "blocked"]


def test_parallel_tool_calls_in_one_turn_keep_order() -> None:
    @tool
    def echo(value: int) -> int:
        """Echo a number."""
        return value

    turn = MockTurn.call_many(("echo", {"value": 1}), ("echo", {"value": 2}))
    result = _agent(ToolRegistry([echo]), [turn, FINAL]).run("go")
    assert [m.content for m in result.messages if m.role == "tool"] == ["1", "2"]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["call_1", "call_2"]


def test_allowlist_naming_unknown_tool_is_rejected_at_construction() -> None:
    spec = AgentSpec(
        name="bad", system_prompt="x", tools=ToolRegistry(), allowed_tools=frozenset({"ghost"})
    )
    with pytest.raises(ValueError, match="ghost"):
        Agent(spec, MockProvider([FINAL]))


def test_agent_is_reusable_across_runs() -> None:
    agent = Agent(
        AgentSpec(name="r", system_prompt="x", tools=ToolRegistry()),
        MockProvider([FINAL, FINAL]),
    )
    first, second = agent.run("one"), agent.run("two")
    assert first.ok
    assert second.ok
    assert first.spans[0].trace_id != second.spans[0].trace_id
    assert second.messages[0].content == "two"
