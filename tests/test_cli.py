import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from guarded_agent.cli import app

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_run_refund_writes_a_renderable_trace(runner: CliRunner, tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    result = runner.invoke(
        app,
        ["run", "refund", "Refund ORD-1001 please", "--approve", "yes", "--trace-out", str(trace)],
    )
    assert result.exit_code == 0, result.output
    assert '"status": "refunded"' in result.output
    assert trace.exists()

    shown = runner.invoke(app, ["trace", "show", str(trace)])
    assert shown.exit_code == 0
    assert "run refund" in shown.output
    assert "issue_refund" in shown.output


def test_run_without_interactive_reviewer_denies_writes(runner: CliRunner) -> None:
    result = runner.invoke(app, ["run", "refund", "Refund ORD-1001", "--no-show-trace"])
    assert result.exit_code == 0
    assert '"status": "escalated"' in result.output


def test_run_exits_non_zero_when_budget_stops_the_run(runner: CliRunner) -> None:
    result = runner.invoke(
        app, ["run", "refund", "Refund ORD-1001", "--approve", "yes", "--max-steps", "1"]
    )
    assert result.exit_code == 1
    assert "max_steps" in result.output


def test_run_rejects_unknown_agent_and_missing_keys(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["run", "nope", "hi"]).exit_code == 2
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["run", "refund", "hi", "--provider", "anthropic"])
    assert result.exit_code == 2


def test_eval_reports_and_writes_json(runner: CliRunner, tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    result = runner.invoke(app, ["eval", str(SCENARIOS), "--report-json", str(report)])
    assert result.exit_code == 0, result.output
    assert "pass rate" in result.output
    data = json.loads(report.read_text())
    assert data["pass_rate"] == 1.0


def test_eval_without_guardrails_fails_the_gate(runner: CliRunner) -> None:
    result = runner.invoke(app, ["eval", str(SCENARIOS), "--no-guardrails"])
    assert result.exit_code == 1
    assert "refund_injection_in_order_note" in result.output


def test_guardrail_bench_and_agent_listing(runner: CliRunner) -> None:
    bench = runner.invoke(app, ["guardrails", "bench", "--json"])
    assert bench.exit_code == 0
    assert json.loads(bench.output)["cases"] == 48
    listing = runner.invoke(app, ["agents"])
    assert "issue_refund*" in listing.output
