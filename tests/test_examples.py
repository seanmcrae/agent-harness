"""End-to-end runs of the example agents on SYNTHETIC data through the mock provider."""

import pytest

from agent_harness.agent import Agent, RunResult, RunStatus
from agent_harness.approval import approve_all, deny_all
from agent_harness.examples import EXAMPLES, get_example
from agent_harness.examples.refund import OrderStore, RefundOutcome
from agent_harness.examples.research import (
    CitationGrounding,
    DocStore,
    ResearchAnswer,
    research_guardrails,
)
from agent_harness.guardrails import GuardrailContext, GuardrailSet, InjectionDetector, Stage
from agent_harness.providers import MockProvider
from agent_harness.types import ToolCallRecord


def run(agent_name: str, text: str, *, guardrails: bool = True, approve: bool = True) -> RunResult:
    example = get_example(agent_name)
    agent = Agent(
        example.build_spec(guardrails=guardrails),
        MockProvider(policy=example.mock_policy),
        approver=approve_all if approve else deny_all,
    )
    return agent.run(text)


def refund_outcome(result: RunResult) -> RefundOutcome:
    assert isinstance(result.output, RefundOutcome)
    return result.output


def test_synthetic_orders_load() -> None:
    store = OrderStore.load_synthetic()
    assert len(store.orders) == 40
    assert all(o.customer_email.endswith("@example.com") for o in store.orders.values())


def test_eligible_refund_is_issued_after_policy_check() -> None:
    result = run("refund", "ORD-1001 arrived cracked, refund please")
    assert result.ok
    assert [r.name for r in result.tool_calls] == [
        "lookup_order",
        "check_refund_policy",
        "issue_refund",
    ]
    outcome = refund_outcome(result)
    assert outcome.status == "refunded"
    assert outcome.refund_amount == 64.0


def test_denied_approval_escalates() -> None:
    outcome = refund_outcome(run("refund", "Refund ORD-1001", approve=False))
    assert outcome.status == "escalated"


@pytest.mark.parametrize(
    ("order_id", "reason"),
    [
        ("ORD-1002", "outside the 30-day window"),
        ("ORD-1003", "final-sale"),
        ("ORD-1004", "not been delivered"),
        ("ORD-1005", "already refunded"),
    ],
)
def test_ineligible_orders_are_denied(order_id: str, reason: str) -> None:
    result = run("refund", f"Refund {order_id}")
    outcome = refund_outcome(result)
    assert outcome.status == "denied"
    assert reason in outcome.summary
    assert "issue_refund" not in [r.name for r in result.tool_calls]


def test_injected_order_note_is_withheld_and_refund_not_issued() -> None:
    guarded = run("refund", "Refund ORD-1006")
    assert refund_outcome(guarded).status == "denied"
    assert "prompt_injection" in guarded.triggered_guardrails
    assert not any(r.name == "issue_refund" for r in guarded.tool_calls)

    unguarded = run("refund", "Refund ORD-1006", guardrails=False)
    assert refund_outcome(unguarded).status == "refunded"  # the failure guardrails prevent


def test_pii_in_order_notes_never_reaches_the_model() -> None:
    result = run("refund", "Refund ORD-1007")
    lookup = next(m for m in result.messages if m.role == "tool" and m.name == "lookup_order")
    assert "4242 4242 4242 4242" not in lookup.content
    assert "[REDACTED_CARD]" in lookup.content
    assert "[REDACTED_PHONE]" in lookup.content
    assert "[REDACTED_EMAIL]" in lookup.content


def test_missing_order_id_asks_for_it() -> None:
    result = run("refund", "I want my money back")
    assert refund_outcome(result).status == "needs_info"
    assert result.tool_calls == []


def test_unknown_order_is_denied() -> None:
    outcome = refund_outcome(run("refund", "Refund ORD-9999"))
    assert outcome.status == "denied"


def test_bm25_ranks_the_relevant_doc_first() -> None:
    store = DocStore.load_synthetic()
    assert len(store.docs) == 8
    assert store.search("API rate limits per minute", 3)[0][0] == "kb-003"
    assert store.search("backup recovery point objective", 3)[0][0] == "kb-008"
    assert store.search("zebra", 3) == []


def test_research_answers_with_grounded_citations() -> None:
    result = run("research", "What is the data retention period for raw events?")
    assert result.ok
    assert isinstance(result.output, ResearchAnswer)
    assert result.output.citations[0] == "kb-001"
    assert "30 days" in result.output.answer


def test_research_injection_is_contained() -> None:
    guarded = run("research", "How do I reset my password?")
    assert guarded.ok
    assert isinstance(guarded.output, ResearchAnswer)
    assert "email their current password" not in guarded.output.answer
    assert "prompt_injection" in guarded.triggered_guardrails

    unguarded = run("research", "How do I reset my password?", guardrails=False)
    assert isinstance(unguarded.output, ResearchAnswer)
    assert "email their current password" in unguarded.output.answer


def test_output_policy_backs_up_the_injection_detector() -> None:
    without_detector = GuardrailSet(
        [g for g in research_guardrails().guardrails if not isinstance(g, InjectionDetector)]
    )
    example = EXAMPLES["research"]
    spec = example.build_spec().with_overrides(guardrails=without_detector)
    result = Agent(spec, MockProvider(policy=example.mock_policy)).run(
        "How do I reset my password?"
    )
    assert result.status is RunStatus.BLOCKED
    assert "output_policy" in result.triggered_guardrails


def test_citation_grounding_blocks_unread_sources() -> None:
    read = ToolCallRecord("read_doc", {"doc_id": "kb-001"}, "ok", executed=True)
    ctx = GuardrailContext(stage=Stage.OUTPUT, agent="research", tool_history=(read,))
    guard = CitationGrounding()
    assert guard.check('{"answer": "x", "citations": ["kb-001"]}', ctx).action == "allow"
    decision = guard.check('{"answer": "x", "citations": ["kb-001", "kb-004"]}', ctx)
    assert decision.action == "block"
    assert decision.findings == ("kb-004",)
