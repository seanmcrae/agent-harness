"""Command-line interface: run example agents, run scenario evals, inspect traces."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from guarded_agent.agent import Agent, RunResult
from guarded_agent.approval import (
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    approve_all,
    deny_all,
)
from guarded_agent.evals import (
    EvalReport,
    Scenario,
    load_scenarios,
    mock_provider_factory,
    run_suite,
)
from guarded_agent.examples import EXAMPLES, ExampleAgent, get_example
from guarded_agent.guardrails.bench import load_cases, run_bench
from guarded_agent.providers import PROVIDER_NAMES, Provider, ProviderError, create_provider
from guarded_agent.tracing import JsonlExporter, SpanExporter, load_spans, render_trace

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run guarded tool-using agents, scenario evals, and trace inspection.",
)
trace_app = typer.Typer(no_args_is_help=True, help="Inspect trace files.")
guardrails_app = typer.Typer(no_args_is_help=True, help="Guardrail utilities.")
app.add_typer(trace_app, name="trace")
app.add_typer(guardrails_app, name="guardrails")

console = Console()


class ApprovalMode(StrEnum):
    ASK = "ask"
    YES = "yes"
    NO = "no"


ProviderOpt = Annotated[
    str, typer.Option(help=f"Model provider: {', '.join(PROVIDER_NAMES)}.", show_default=True)
]
ModelOpt = Annotated[str | None, typer.Option(help="Model id (provider default if omitted).")]


def console_approver(request: ApprovalRequest) -> ApprovalDecision:
    if not sys.stdin.isatty():
        return ApprovalDecision(False, "console", "no interactive reviewer available")
    console.print(
        f"[bold yellow]Approval needed[/] agent={request.agent} tool={request.tool} "
        f"args={json.dumps(request.arguments)}"
    )
    approved = Confirm.ask("Allow this action?", default=False, console=console)
    return ApprovalDecision(approved, "console", "" if approved else "rejected at console")


APPROVERS: dict[ApprovalMode, Approver] = {
    ApprovalMode.ASK: console_approver,
    ApprovalMode.YES: approve_all,
    ApprovalMode.NO: deny_all,
}


def _provider(name: str, model: str | None, example: ExampleAgent) -> Provider:
    try:
        return create_provider(name, model, policy=example.mock_policy)
    except (ProviderError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc


def _print_result(result: RunResult) -> None:
    style = "green" if result.ok else "red"
    console.print(
        f"[bold {style}]{result.status.value}[/]  steps={result.steps}  "
        f"tokens={result.usage.input_tokens}+{result.usage.output_tokens}  "
        f"cost=${result.cost_usd:.4f}  guardrails={','.join(result.triggered_guardrails) or '-'}"
    )
    if result.error:
        console.print(f"[red]{result.error}[/]")
    if result.output is not None:
        console.print_json(result.output.model_dump_json())
    elif result.final_text:
        console.print(result.final_text)


@app.command()
def run(
    agent: Annotated[str, typer.Argument(help=f"Example agent: {', '.join(EXAMPLES)}.")],
    prompt: Annotated[str, typer.Argument(help="The user request.")],
    provider: ProviderOpt = "mock",
    model: ModelOpt = None,
    approve: Annotated[
        ApprovalMode, typer.Option(help="How write tools are approved.")
    ] = ApprovalMode.ASK,
    max_steps: Annotated[int | None, typer.Option(help="Override the step limit.")] = None,
    max_cost: Annotated[float | None, typer.Option(help="Override the USD cost limit.")] = None,
    timeout: Annotated[float | None, typer.Option(help="Override the timeout (s).")] = None,
    guardrails: Annotated[
        bool, typer.Option(help="Enable content guardrails (PII, injection, output policy).")
    ] = True,
    trace_out: Annotated[Path | None, typer.Option(help="Write spans to this JSONL file.")] = None,
    show_trace: Annotated[bool, typer.Option(help="Print the trace tree.")] = True,
) -> None:
    """Run one request through an example agent."""
    example = _example(agent)
    spec = example.build_spec(guardrails=guardrails)
    budget = spec.budget
    if max_steps is not None:
        budget = replace(budget, max_steps=max_steps)
    if max_cost is not None:
        budget = replace(budget, max_cost_usd=max_cost)
    if timeout is not None:
        budget = replace(budget, timeout_s=timeout)
    spec = spec.with_overrides(budget=budget)

    exporters: list[SpanExporter] = [JsonlExporter(trace_out)] if trace_out else []
    runner = Agent(
        spec,
        _provider(provider, model, example),
        approver=APPROVERS[approve],
        exporters=exporters,
    )
    result = runner.run(prompt)
    if show_trace:
        for tree in render_trace(result.spans):
            console.print(tree)
    _print_result(result)
    if trace_out:
        console.print(f"trace written to {trace_out}")
    raise typer.Exit(0 if result.ok else 1)


@app.command("eval")
def eval_command(
    path: Annotated[Path, typer.Argument(help="Scenario YAML file or directory.", exists=True)],
    provider: ProviderOpt = "mock",
    model: ModelOpt = None,
    guardrails: Annotated[bool, typer.Option(help="Enable content guardrails.")] = True,
    min_pass_rate: Annotated[
        float, typer.Option(help="Exit non-zero below this pass rate.", min=0.0, max=1.0)
    ] = 1.0,
    report_json: Annotated[Path | None, typer.Option(help="Write a JSON report here.")] = None,
    trace_out: Annotated[Path | None, typer.Option(help="Append all spans to this JSONL.")] = None,
) -> None:
    """Run scenario evals and report pass rate, steps, cost, and guardrail triggers."""
    scenarios = load_scenarios(path)

    def factory(example: ExampleAgent, scenario: Scenario) -> Provider:
        if provider == "mock":
            return mock_provider_factory(example, scenario)
        return _provider(provider, model, example)

    exporters: list[SpanExporter] = []
    if trace_out:
        trace_out.unlink(missing_ok=True)
        exporters.append(JsonlExporter(trace_out, append=True))
    report = run_suite(
        scenarios,
        guardrails=guardrails,
        provider_factory=factory,
        exporters=exporters,
        # Mock runs skip real backoff sleeps so the suite stays fast; delays are still traced.
        sleep=_no_sleep if provider == "mock" else time.sleep,
    )
    _print_report(report)
    if report_json:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    raise typer.Exit(0 if report.pass_rate >= min_pass_rate else 1)


def _print_report(report: EvalReport) -> None:
    table = Table(show_lines=False, pad_edge=False)
    for column in ("scenario", "agent", "result", "status", "steps", "cost", "guardrails"):
        table.add_column(column, justify="right" if column in {"steps", "cost"} else "left")
    for r in report.results:
        table.add_row(
            r.scenario.name,
            r.scenario.agent,
            "[green]PASS[/]" if r.passed else "[red]FAIL[/]",
            r.run.status.value,
            str(r.run.steps),
            f"${r.run.cost_usd:.4f}",
            ",".join(r.run.triggered_guardrails) or "-",
        )
    console.print(table)
    for r in report.results:
        for failure in r.failures:
            console.print(f"[red]FAIL[/] {r.scenario.name}: {failure}")
    triggers = ", ".join(f"{k}={v}" for k, v in sorted(report.guardrail_triggers.items()))
    console.print(
        f"pass rate {report.passed}/{len(report.results)} ({report.pass_rate:.1%})  "
        f"mean steps {report.mean_steps:.2f}  mean cost ${report.mean_cost_usd:.4f}  "
        f"guardrails {'on' if report.guardrails_enabled else 'off'}  "
        f"triggers: {triggers or '-'}"
    )


@trace_app.command("show")
def trace_show(
    file: Annotated[Path, typer.Argument(help="JSONL trace file.", exists=True)],
    all_decisions: Annotated[
        bool, typer.Option("--all", help="Include guardrail ALLOW decisions.")
    ] = False,
) -> None:
    """Render a trace file as a tree."""
    for tree in render_trace(load_spans(file), show_allow=all_decisions):
        console.print(tree)


@guardrails_app.command("bench")
def guardrails_bench(
    cases: Annotated[
        Path | None, typer.Option(help="JSONL cases (default: bundled synthetic set).")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Measure injection and PII guardrail precision/recall on labelled cases."""
    report = run_bench(load_cases(cases))
    if as_json:
        console.print_json(json.dumps(report.to_dict()))
        return
    table = Table(pad_edge=False)
    for column in ("check", "tp", "fp", "fn", "tn", "precision", "recall"):
        table.add_column(column, justify="left" if column == "check" else "right")
    rows = [("prompt_injection", report.injection)] + [
        (f"pii:{kind}", c) for kind, c in report.pii.items()
    ]
    for name, c in rows:
        table.add_row(
            name,
            str(c.tp),
            str(c.fp),
            str(c.fn),
            str(c.tn),
            f"{c.precision:.3f}",
            f"{c.recall:.3f}",
        )
    console.print(f"{report.cases} labelled cases")
    console.print(table)
    for miss in report.misses:
        console.print(f"[yellow]{miss}[/]")


@app.command("agents")
def list_agents() -> None:
    """List the bundled example agents."""
    for example in EXAMPLES.values():
        spec = example.build_spec()
        tools = ", ".join(f"{t.name}{'*' if t.requires_approval else ''}" for t in spec.tools)
        console.print(f"[bold]{example.name}[/]  {example.description}\n  tools: {tools}")
    console.print("* write tool, requires approval")


def _no_sleep(_: float) -> None:
    return None


def _example(name: str) -> ExampleAgent:
    try:
        return get_example(name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
