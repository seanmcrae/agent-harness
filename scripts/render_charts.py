"""Run the scenario suite with guardrails on and off plus the guardrail bench, then chart them.

Everything runs offline on the bundled SYNTHETIC data with the mock provider.

    uv run --extra docs python scripts/render_charts.py docs/img
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from agent_harness.charts import render_all
from agent_harness.evals import EvalReport, load_scenarios, run_suite
from agent_harness.guardrails.bench import BenchReport, load_cases, run_bench

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Results:
    on: EvalReport
    off: EvalReport
    bench: BenchReport


def _no_sleep(_: float) -> None:
    return None


def collect_results(scenarios_dir: Path = ROOT / "scenarios") -> Results:
    scenarios = load_scenarios(scenarios_dir)
    return Results(
        on=run_suite(scenarios, guardrails=True, sleep=_no_sleep),
        off=run_suite(scenarios, guardrails=False, sleep=_no_sleep),
        bench=run_bench(load_cases()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, nargs="?", default=ROOT / "docs" / "img")
    args = parser.parse_args()
    results = collect_results()
    for path in render_all(results.on, results.off, results.bench, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
