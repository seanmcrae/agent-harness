import json
from collections.abc import Callable
from typing import Any

from agent_harness.agent import Agent, RunStatus
from agent_harness.approval import ApprovalDecision, ApprovalRequest, deny_all
from agent_harness.budget import Budget
from agent_harness.guardrails import Action
from agent_harness.providers import MockTurn, ProviderError, TransientProviderError
from agent_harness.tracing import SpanKind
from tests.conftest import Answer, FakeClock

BuildAgent = Callable[..., Agent]
DONE = MockTurn.final_json({"status": "done", "detail": "ok"})


def test_tool_loop_produces_validated_output(build_agent: BuildAgent, tool_log: list[str]) -> None:
    agent = build_agent([MockTurn.call("lookup", item_id="I-1"), DONE])
    result = agent.run("check I-1")

    assert result.status is RunStatus.COMPLETED
    assert result.output == Answer(status="done", detail="ok")
    assert result.steps == 2
    assert tool_log == ["lookup:I-1"]
    assert [r.status for r in result.tool_calls] == ["ok"]
    assert result.usage.total_tokens > 0
    assert result.cost_usd > 0
    tool_message = result.messages[2]
    assert tool_message.role == "tool"
    assert "[REDACTED_EMAIL]" in tool_message.content
    assert "pii_redaction" in result.triggered_guardrails


def test_trace_contains_every_span_kind(build_agent: BuildAgent) -> None:
    result = build_agent([MockTurn.call("lookup", item_id="I-1"), DONE]).run("go")
    kinds = {span.kind for span in result.spans}
    assert kinds == set(SpanKind)
    run = next(s for s in result.spans if s.kind is SpanKind.RUN)
    assert run.attributes["status"] == "completed"
    assert run.attributes["cost_usd"] == round(result.cost_usd, 6)


def test_step_limit_stops_the_run(build_agent: BuildAgent) -> None:
    looping = [MockTurn.call("lookup", item_id="I-1")] * 5
    result = build_agent(looping, budget=Budget(max_steps=3)).run("loop")
    assert result.status is RunStatus.MAX_STEPS
    assert result.steps == 3
    budget_decision = result.decisions[-1]
    assert budget_decision.guardrail == "budget"
    assert budget_decision.action is Action.BLOCK


def test_cost_budget_stops_before_running_more_tools(
    build_agent: BuildAgent, tool_log: list[str]
) -> None:
    script = [MockTurn.call("lookup", item_id="I-1"), MockTurn.call("lookup", item_id="I-2")]
    result = build_agent(script, budget=Budget(max_cost_usd=1e-9)).run("spend")
    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert tool_log == []
    assert "budget" in result.triggered_guardrails


def test_token_budget(build_agent: BuildAgent) -> None:
    script = [MockTurn.call("lookup", item_id="I-1"), DONE]
    result = build_agent(script, budget=Budget(max_total_tokens=10)).run("spend")
    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert "tokens" in (result.error or "")


def test_wall_clock_timeout(build_agent: BuildAgent, clock: FakeClock) -> None:
    lookup = MockTurn.call("lookup", item_id="I-1")
    agent = build_agent([lookup, lookup, DONE], budget=Budget(timeout_s=5))
    original = agent.provider.complete

    def slow_complete(request: Any) -> Any:
        clock.now += 3
        return original(request)

    agent.provider.complete = slow_complete  # type: ignore[method-assign]
    result = agent.run("slow")
    assert result.status is RunStatus.TIMEOUT
    assert result.steps == 2  # step 3 never starts: 6s elapsed against a 5s budget


def test_transient_errors_are_retried_with_backoff(
    build_agent: BuildAgent, clock: FakeClock
) -> None:
    faults = (TransientProviderError("overloaded"), TransientProviderError("overloaded"))
    result = build_agent([DONE], faults=faults).run("hi")
    assert result.status is RunStatus.COMPLETED
    llm_spans = [s for s in result.spans if s.kind is SpanKind.LLM_CALL]
    assert [s.attributes["attempt"] for s in llm_spans] == [1, 2, 3]
    assert [s.status for s in llm_spans] == ["error", "error", "ok"]
    # 0.5s then 1.0s base delays with +/-20% jitter
    assert 1.2 <= clock.now <= 1.8


def test_retries_give_up_after_max_attempts(build_agent: BuildAgent) -> None:
    faults = tuple(TransientProviderError("down") for _ in range(3))
    result = build_agent([DONE], faults=faults).run("hi")
    assert result.status is RunStatus.PROVIDER_ERROR
    assert "3 attempts" in (result.error or "")


def test_non_transient_errors_are_not_retried(build_agent: BuildAgent) -> None:
    result = build_agent([ProviderError("bad request"), DONE]).run("hi")
    assert result.status is RunStatus.PROVIDER_ERROR
    assert len([s for s in result.spans if s.kind is SpanKind.LLM_CALL]) == 1


def test_invalid_final_answer_is_repaired(build_agent: BuildAgent) -> None:
    script = [MockTurn.final('{"status": "maybe"}'), DONE]
    result = build_agent(script).run("hi")
    assert result.status is RunStatus.COMPLETED
    assert result.steps == 2
    repair = result.messages[2]
    assert repair.role == "user"
    assert "status" in repair.content
    assert "detail" in repair.content


def test_repair_budget_is_bounded(build_agent: BuildAgent) -> None:
    script = [MockTurn.final("not json"), MockTurn.final("still not json")]
    result = build_agent(script).run("hi")
    assert result.status is RunStatus.INVALID_OUTPUT


def test_plain_text_agent_returns_text(build_agent: BuildAgent) -> None:
    result = build_agent([MockTurn.final("hello there")], output_model=None).run("hi")
    assert result.ok
    assert result.output is None
    assert result.final_text == "hello there"


def test_write_tool_runs_only_when_approved(build_agent: BuildAgent, tool_log: list[str]) -> None:
    seen: list[ApprovalRequest] = []

    def reviewer(request: ApprovalRequest) -> ApprovalDecision:
        seen.append(request)
        return ApprovalDecision(approved=True, reviewer="alex")

    script = [MockTurn.call("delete_item", item_id="I-9"), DONE]
    result = build_agent(script, approver=reviewer).run("delete I-9")
    assert seen == [ApprovalRequest("test-agent", "delete_item", {"item_id": "I-9"})]
    assert tool_log == ["delete:I-9"]
    tool_span = next(s for s in result.spans if s.kind is SpanKind.TOOL_CALL)
    assert tool_span.attributes["approval"] == "approved"
    assert tool_span.attributes["reviewer"] == "alex"


def test_denied_write_tool_does_not_run(build_agent: BuildAgent, tool_log: list[str]) -> None:
    script = [MockTurn.call("delete_item", item_id="I-9"), DONE]
    result = build_agent(script, approver=deny_all).run("delete I-9")
    assert tool_log == []
    record = result.tool_calls[0]
    assert record.status == "denied"
    assert not record.executed
    assert result.messages[2].is_error


def test_duplicate_non_idempotent_call_is_suppressed(
    build_agent: BuildAgent, tool_log: list[str]
) -> None:
    call = MockTurn.call("delete_item", item_id="I-9")
    result = build_agent([call, call, DONE]).run("delete twice")
    assert tool_log == ["delete:I-9"]
    assert [r.status for r in result.tool_calls] == ["ok", "blocked"]


def test_allowlist_blocks_and_hides_tools(build_agent: BuildAgent, tool_log: list[str]) -> None:
    script = [MockTurn.call("delete_item", item_id="I-9"), DONE]
    agent = build_agent(script, allowed_tools=frozenset({"lookup"}))
    result = agent.run("delete I-9")
    assert tool_log == []
    assert result.tool_calls[0].status == "blocked"
    assert "tool_allowlist" in result.triggered_guardrails
    advertised = agent.provider.requests[0].tools  # type: ignore[attr-defined]
    assert [spec.name for spec in advertised] == ["lookup"]


def test_injected_tool_output_is_withheld(build_agent: BuildAgent) -> None:
    result = build_agent([MockTurn.call("fetch_note", item_id="I-1"), DONE]).run("read note")
    record = result.tool_calls[0]
    assert record.executed
    assert record.withheld
    assert "Ignore all previous" not in result.messages[2].content
    assert "withheld" in result.messages[2].content
    assert "prompt_injection" in result.triggered_guardrails


def test_tool_errors_and_bad_arguments_go_back_to_the_model(build_agent: BuildAgent) -> None:
    script = [
        MockTurn.call("lookup", item_id="bad-id"),
        MockTurn.call("lookup", item_id="I-404"),
        MockTurn.call("nonexistent"),
        DONE,
    ]
    result = build_agent(script).run("hi")
    assert result.ok
    assert [r.status for r in result.tool_calls] == ["invalid_args", "error", "error"]
    assert "pattern" in (result.tool_calls[0].error or "")
    assert result.tool_calls[1].error == "item not found"
    assert all(m.is_error for m in result.messages if m.role == "tool")


def test_input_pii_is_redacted_before_the_model_sees_it(build_agent: BuildAgent) -> None:
    agent = build_agent([DONE])
    agent.run("my card is 4111 1111 1111 1111")
    first_request = agent.provider.requests[0]  # type: ignore[attr-defined]
    assert first_request.messages[0].content == "my card is [REDACTED_CARD]"


def test_structured_output_instructions_reach_the_model(build_agent: BuildAgent) -> None:
    agent = build_agent([DONE])
    agent.run("hi")
    system = agent.provider.requests[0].system  # type: ignore[attr-defined]
    schema = json.loads(system.split("\n")[-1])
    assert schema["required"] == ["status", "detail"]
