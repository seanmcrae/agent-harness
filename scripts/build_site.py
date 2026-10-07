"""Build the static documentation site into ``site/`` (served by GitHub Pages).

Every number, table, chart, and trace on the site comes from running the code at build time on the
bundled SYNTHETIC data with the mock provider. Prose sections are rendered from README.md and
docs/PRODUCT.md so the site and the repository never disagree. The site has no remote assets: the
architecture diagram is the pre-rendered docs/img/architecture.svg (``make diagram``), and the
build fails if README.md's Mermaid block has drifted from docs/architecture.mmd.

    uv run --extra docs python scripts/build_site.py --out site
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import markdown
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from render_charts import ROOT, Results, collect_results
from rich.console import Console

from agent_harness import __version__
from agent_harness.agent import Agent
from agent_harness.approval import approve_all
from agent_harness.charts import CHART_FILES, render_all
from agent_harness.examples import get_example
from agent_harness.providers import MockProvider
from agent_harness.tracing import render_trace

REPO_URL = "https://github.com/seanmcrae/agent-harness"
SITE_URL = "https://seanmcrae.github.io/agent-harness/"
TEMPLATES = ROOT / "docs" / "site"
DEMO_PROMPT = "Hi, can you look at ORD-1006 and refund it? Reach me at quinn@example.com."
README_SECTIONS = ("Architecture", "Design decisions", "Data", "Limitations")
DIAGRAM_SOURCE = ROOT / "docs" / "architecture.mmd"
DIAGRAM_SVG = ROOT / "docs" / "img" / "architecture.svg"
_MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
_HREF = re.compile(r'(href|src)="([^"#:]+)(#[^"]*)?"')


@dataclass(frozen=True)
class Demo:
    command: str
    trace: str
    output: str


def run_demo(guardrails: bool) -> Demo:
    """Run the prompt-injection demo and capture the trace tree as the terminal shows it."""
    example = get_example("refund")
    agent = Agent(
        example.build_spec(guardrails=guardrails),
        MockProvider(policy=example.mock_policy),
        approver=approve_all,
    )
    result = agent.run(DEMO_PROMPT)
    console = Console(file=io.StringIO(), width=104, record=True, color_system=None)
    for tree in render_trace(result.spans):
        console.print(tree)
    output = result.output.model_dump_json(indent=2) if result.output else result.final_text
    flag = "" if guardrails else " --no-guardrails"
    return Demo(
        command=f'agent run refund "{DEMO_PROMPT}" --approve yes{flag}',
        trace=console.export_text().rstrip(),
        output=output or "",
    )


def readme_sections(readme: str, wanted: tuple[str, ...]) -> dict[str, str]:
    """Split README.md on level-2 headings and return the bodies of the wanted sections."""
    sections: dict[str, str] = {}
    for chunk in re.split(r"^## ", readme, flags=re.MULTILINE)[1:]:
        title, _, body = chunk.partition("\n")
        if title.strip() in wanted:
            sections[title.strip()] = body.strip()
    missing = set(wanted) - set(sections)
    if missing:
        raise ValueError(f"README.md is missing sections: {sorted(missing)}")
    return sections


_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def _separate_lists(text: str) -> str:
    """Insert the blank line Python-Markdown needs before a list that follows a paragraph.

    GitHub renders "**In:**" directly followed by "- item" as a list; Python-Markdown does not.
    """
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        previous = lines[-1] if lines else ""
        if (
            not in_fence
            and _LIST_ITEM.match(line)
            and previous.strip()
            and not _LIST_ITEM.match(previous)
            and not previous.startswith((" ", "\t"))
        ):
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


def render_markdown(text: str) -> str:
    """Render Markdown, swapping the Mermaid fence for the pre-rendered SVG and fixing links."""
    diagrams: list[str] = []

    def stash(match: re.Match[str]) -> str:
        diagrams.append(match.group(1))
        return f"\n\nMERMAID-{len(diagrams) - 1}\n\n"

    body = markdown.markdown(
        _separate_lists(_MERMAID_BLOCK.sub(stash, text)),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    for i, source in enumerate(diagrams):
        if source.strip() != DIAGRAM_SOURCE.read_text(encoding="utf-8").strip():
            raise ValueError(
                "README.md Mermaid diagram differs from docs/architecture.mmd; "
                "update the .mmd file and run `make diagram`"
            )
        block = (
            '<figure class="diagram"><img src="img/architecture.svg" '
            'alt="Architecture: agent loop with guardrails, budgets, approvals, and exporters">'
            "</figure>\n<details><summary>Mermaid source</summary>"
            f"<pre><code>{html.escape(source)}</code></pre></details>"
        )
        body = body.replace(f"<p>MERMAID-{i}</p>", block)
    return _HREF.sub(_rewrite_link, body)


def _rewrite_link(match: re.Match[str]) -> str:
    attr, path, anchor = match.group(1), match.group(2), match.group(3) or ""
    path = path.removeprefix("../").removeprefix("./")
    if path in {"docs/PRODUCT.md", "PRODUCT.md"}:
        target = "product.html"
    elif path.startswith("docs/img/"):
        target = "img/" + path.removeprefix("docs/img/")
    elif path.startswith("img/") or path.endswith(".html"):
        target = path
    else:
        target = f"{REPO_URL}/blob/main/{path}"
    return f'{attr}="{target}{anchor}"'


def scenario_rows(results: Results) -> list[dict[str, Any]]:
    off_by_name = {r.scenario.name: r for r in results.off.results}
    rows = []
    for on in results.on.results:
        off = off_by_name[on.scenario.name]
        rows.append(
            {
                "name": on.scenario.name,
                "agent": on.scenario.agent,
                "description": on.scenario.description,
                "on_passed": on.passed,
                "off_passed": off.passed,
                "off_unsafe": bool(off.violations),
                "off_detail": "; ".join(off.violations or off.failures),
                "status": on.run.status.value,
                "steps": on.run.steps,
                "cost_on": on.run.cost_usd,
                "cost_off": off.run.cost_usd,
                "guardrails": on.run.triggered_guardrails,
            }
        )
    return rows


def build(out: Path) -> None:
    results = collect_results()
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)
    render_all(results.on, results.off, results.bench, out / "img")
    shutil.copy(DIAGRAM_SVG, out / "img" / DIAGRAM_SVG.name)
    shutil.copy(TEMPLATES / "style.css", out / "style.css")
    (out / ".nojekyll").touch()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sections = {
        title: render_markdown(body)
        for title, body in readme_sections(readme, README_SECTIONS).items()
    }
    product = render_markdown((ROOT / "docs" / "PRODUCT.md").read_text(encoding="utf-8"))
    bench_rows = [("prompt_injection", results.bench.injection)] + [
        (f"pii:{kind}", c) for kind, c in results.bench.pii.items()
    ]
    context: dict[str, Any] = {
        "version": __version__,
        "repo_url": REPO_URL,
        "site_url": SITE_URL,
        "charts": CHART_FILES,
        "on": results.on,
        "off": results.off,
        "bench": results.bench,
        "bench_rows": bench_rows,
        "rows": scenario_rows(results),
        "demo_on": run_demo(guardrails=True),
        "demo_off": run_demo(guardrails=False),
        "sections": sections,
        "product": product,
    }
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    for page in ("index.html", "product.html"):
        html_text = env.get_template(f"{page}.j2").render(page=page, **context)
        (out / page).write_text(html_text, encoding="utf-8")
    summary = {
        "guardrails_on": results.on.to_dict(),
        "guardrails_off": results.off.to_dict(),
        "bench": results.bench.to_dict(),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static documentation site.")
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    build(args.out)
    print(f"site written to {args.out}")


if __name__ == "__main__":
    main()
