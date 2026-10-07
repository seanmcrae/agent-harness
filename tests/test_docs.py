"""The numbers quoted in README.md must match what the code produces on the bundled data."""

from pathlib import Path

import pytest

from agent_harness.evals import EvalReport, load_scenarios, run_suite
from agent_harness.guardrails.bench import load_cases, run_bench

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _no_sleep(_: float) -> None:
    return None


@pytest.fixture(scope="module")
def reports() -> tuple[EvalReport, EvalReport]:
    scenarios = load_scenarios(ROOT / "scenarios")
    return (
        run_suite(scenarios, sleep=_no_sleep),
        run_suite(scenarios, guardrails=False, sleep=_no_sleep),
    )


def test_readme_results_table_matches_eval_runs(reports: tuple[EvalReport, EvalReport]) -> None:
    on, off = reports
    rows = {
        "Scenarios passed": (
            f"{on.passed}/{len(on.results)} ({on.pass_rate:.1%})",
            f"{off.passed}/{len(off.results)} ({off.pass_rate:.1%})",
        ),
        "Scenarios with a safety violation": (str(on.unsafe), str(off.unsafe)),
        "Mean steps per scenario": (f"{on.mean_steps:.2f}", f"{off.mean_steps:.2f}"),
        "Mean simulated cost per scenario": (
            f"${on.mean_cost_usd:.4f}",
            f"${off.mean_cost_usd:.4f}",
        ),
    }
    for label, (with_guardrails, without) in rows.items():
        assert f"| {label} | {with_guardrails} | {without} |" in README, label


def test_readme_bench_table_matches_bench_run() -> None:
    report = run_bench(load_cases())
    rows = [("prompt_injection", report.injection)] + [
        (f"pii:{kind}", c) for kind, c in report.pii.items()
    ]
    for name, c in rows:
        line = (
            f"| {name} | {c.tp} | {c.fp} | {c.fn} | {c.tn} | {c.precision:.3f} | {c.recall:.3f} |"
        )
        assert line in README, name
