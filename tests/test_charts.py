from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from agent_harness.charts import bench_chart, cost_and_steps_chart, guardrail_ablation_chart
from agent_harness.evals import EvalReport, load_scenarios, run_suite
from agent_harness.guardrails.bench import load_cases, run_bench

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _no_sleep(_: float) -> None:
    return None


@pytest.fixture(scope="module")
def reports() -> tuple[EvalReport, EvalReport]:
    scenarios = [s for s in load_scenarios(SCENARIOS) if s.agent == "refund"][:9]
    on = run_suite(scenarios, sleep=_no_sleep)
    off = run_suite(scenarios, guardrails=False, sleep=_no_sleep)
    return on, off


def test_eval_charts_are_written_and_reproducible(
    tmp_path: Path, reports: tuple[EvalReport, EvalReport]
) -> None:
    on, off = reports
    first = guardrail_ablation_chart(on, off, tmp_path / "a" / "ablation.png")
    second = guardrail_ablation_chart(on, off, tmp_path / "b" / "ablation.png")
    assert first.read_bytes().startswith(PNG_MAGIC)
    assert first.read_bytes() == second.read_bytes()
    costs = cost_and_steps_chart(on, off, tmp_path / "costs.png")
    assert costs.stat().st_size > 10_000


def test_ablation_requires_matching_scenarios(
    tmp_path: Path, reports: tuple[EvalReport, EvalReport]
) -> None:
    on, _ = reports
    other = run_suite(load_scenarios(SCENARIOS)[-2:], sleep=_no_sleep)
    with pytest.raises(ValueError, match="same scenarios"):
        guardrail_ablation_chart(on, other, tmp_path / "x.png")


def test_bench_chart(tmp_path: Path) -> None:
    path = bench_chart(run_bench(load_cases()), tmp_path / "bench.png")
    assert path.read_bytes().startswith(PNG_MAGIC)
