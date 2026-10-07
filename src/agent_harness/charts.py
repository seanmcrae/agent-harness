"""PNG charts for scenario eval and guardrail bench reports.

Requires the ``docs`` extra (matplotlib). Figures are rendered with the Agg backend and saved
without timestamp or version metadata, so re-running on the same reports gives identical files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from agent_harness.evals import EvalReport, ScenarioResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from agent_harness.guardrails.bench import BenchReport

ON = "#1f6feb"
OFF = "#d29922"
PASS = "#2da44e"
FAIL = "#bf8700"
UNSAFE = "#cf222e"
AGENT_COLORS = {"refund": "#1f6feb", "research": "#8250df"}
_PNG_METADATA: dict[str, str | None] = {"Software": None}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "svg.hashsalt": "agent-harness",
    }
)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata=_PNG_METADATA)
    plt.close(fig)
    return path


def _paired(on: EvalReport, off: EvalReport) -> list[tuple[ScenarioResult, ScenarioResult]]:
    off_by_name = {r.scenario.name: r for r in off.results}
    if set(off_by_name) != {r.scenario.name for r in on.results}:
        raise ValueError("both reports must cover the same scenarios")
    return [(r, off_by_name[r.scenario.name]) for r in on.results]


def _outcome_color(result: ScenarioResult) -> str:
    if result.violations:
        return UNSAFE
    return PASS if result.passed else FAIL


def _label_bars(ax: Axes, values: list[float], fmt: str) -> None:
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    for patch, value in zip(bars, values, strict=True):
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
        )


def guardrail_ablation_chart(on: EvalReport, off: EvalReport, path: Path) -> Path:
    """Pass rate and unsafe outcomes with guardrails on vs off, plus the per-scenario grid."""
    pairs = _paired(on, off)
    fig, (ax_rate, ax_unsafe, ax_grid) = plt.subplots(
        1, 3, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [0.8, 0.8, 2.6]}
    )
    labels = ["on", "off"]

    rates = [on.pass_rate * 100, off.pass_rate * 100]
    ax_rate.bar(labels, rates, color=[ON, OFF], width=0.6)
    _label_bars(ax_rate, rates, "{:.1f}%")
    ax_rate.set_ylim(0, 112)
    ax_rate.set_title("Pass rate (%)")
    ax_rate.set_xlabel("guardrails")

    unsafe = [float(on.unsafe), float(off.unsafe)]
    ax_unsafe.bar(labels, unsafe, color=[ON, OFF], width=0.6)
    _label_bars(ax_unsafe, unsafe, "{:.0f}")
    top = max(*unsafe, 1.0)
    ax_unsafe.set_ylim(0, top * 1.4)
    ax_unsafe.set_yticks(range(int(top) + 1))
    ax_unsafe.set_title("Unsafe scenarios")
    ax_unsafe.set_xlabel("guardrails")

    for row, (r_on, r_off) in enumerate(reversed(pairs)):
        for col, result in enumerate((r_on, r_off)):
            ax_grid.add_patch(Rectangle((col, row), 0.94, 0.86, color=_outcome_color(result), lw=0))
    ax_grid.set_xlim(0, 2)
    ax_grid.set_ylim(0, len(pairs))
    ax_grid.set_xticks([0.47, 1.47], ["on", "off"])
    ax_grid.set_yticks(
        [i + 0.43 for i in range(len(pairs))],
        [r.scenario.name for r, _ in reversed(pairs)],
        fontsize=8,
    )
    ax_grid.tick_params(length=0)
    for spine in ax_grid.spines.values():
        spine.set_visible(False)
    ax_grid.yaxis.tick_right()
    ax_grid.set_title("Per scenario, guardrails on / off")
    ax_grid.legend(
        handles=[
            Patch(color=PASS, label="pass"),
            Patch(color=FAIL, label="fail (expectation missed)"),
            Patch(color=UNSAFE, label="fail (forbidden action or output)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"Guardrail ablation: {len(pairs)} bundled synthetic scenarios, mock provider",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def cost_and_steps_chart(on: EvalReport, off: EvalReport, path: Path) -> Path:
    """Cost and steps per scenario, guardrails on, with the guardrails-off cost as a marker."""
    pairs = list(reversed(_paired(on, off)))
    names = [r.scenario.name for r, _ in pairs]
    colors = [AGENT_COLORS.get(r.scenario.agent, ON) for r, _ in pairs]
    y = list(range(len(pairs)))
    fig, (ax_cost, ax_steps) = plt.subplots(
        1, 2, figsize=(11, 6), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]}
    )

    ax_cost.barh(y, [r.run.cost_usd * 1000 for r, _ in pairs], color=colors, height=0.7)
    ax_cost.scatter(
        [r.run.cost_usd * 1000 for _, r in pairs],
        y,
        marker="|",
        s=140,
        color="black",
        zorder=3,
        label="cost with guardrails off",
    )
    ax_cost.set_yticks(y, names, fontsize=8)
    ax_cost.set_xlabel(
        r"cost per run (USD x 1000, simulated \$3 / \$15 per M input / output tokens)"
    )
    ax_cost.set_title(f"Cost per scenario (mean \\${on.mean_cost_usd:.4f})")
    ax_cost.legend(
        handles=[
            *(Patch(color=c, label=f"{a} agent") for a, c in AGENT_COLORS.items()),
            ax_cost.collections[0],
        ],
        loc="lower right",
        fontsize=8,
        frameon=False,
    )

    steps = [r.run.steps for r, _ in pairs]
    ax_steps.barh(y, steps, color=colors, height=0.7)
    for yi, value in zip(y, steps, strict=True):
        ax_steps.annotate(
            str(value),
            (value, yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax_steps.set_xlabel("agent steps (model turns)")
    ax_steps.set_xlim(0, max(steps) + 1)
    ax_steps.set_title(f"Steps per scenario (mean {on.mean_steps:.2f})")
    fig.suptitle(
        "Cost and steps per scenario, guardrails on (bundled synthetic data, mock provider)",
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def bench_chart(report: BenchReport, path: Path) -> Path:
    """Precision and recall per guardrail check on the labelled case set."""
    rows = [("prompt_injection", report.injection)] + [
        (f"pii:{kind}", c) for kind, c in report.pii.items()
    ]
    names = [name for name, _ in rows]
    x = range(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    precision = [c.precision for _, c in rows]
    recall = [c.recall for _, c in rows]
    ax.bar([i - width / 2 for i in x], precision, width, label="precision", color=ON)
    ax.bar([i + width / 2 for i in x], recall, width, label="recall", color=OFF)
    _label_bars(ax, precision + recall, "{:.2f}")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"Guardrail bench on {report.cases} labelled synthetic cases (in-sample)")
    fig.tight_layout()
    return _save(fig, path)


CHART_FILES = {
    "ablation": "guardrail-ablation.png",
    "cost_steps": "cost-steps-per-scenario.png",
    "bench": "guardrail-bench.png",
}


def render_all(on: EvalReport, off: EvalReport, bench: BenchReport, out_dir: Path) -> list[Path]:
    """Write every chart into ``out_dir`` under the names in :data:`CHART_FILES`."""
    return [
        guardrail_ablation_chart(on, off, out_dir / CHART_FILES["ablation"]),
        cost_and_steps_chart(on, off, out_dir / CHART_FILES["cost_steps"]),
        bench_chart(bench, out_dir / CHART_FILES["bench"]),
    ]
