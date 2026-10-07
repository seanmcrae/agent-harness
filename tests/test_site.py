import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("jinja2")
pytest.importorskip("markdown")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_site.py"), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    return out


def test_site_has_pages_assets_and_results(site: Path) -> None:
    for name in ("index.html", "product.html", "style.css", ".nojekyll", "results.json"):
        assert (site / name).exists(), name
    assert {p.name for p in (site / "img").glob("*.png")} == {
        "guardrail-ablation.png",
        "cost-steps-per-scenario.png",
        "guardrail-bench.png",
    }
    assert (site / "img" / "architecture.svg").exists()


def test_headline_numbers_match_the_eval_results(site: Path) -> None:
    results = json.loads((site / "results.json").read_text())
    on, off = results["guardrails_on"], results["guardrails_off"]
    index = (site / "index.html").read_text()
    assert f"{on['passed']}/{on['scenarios']}" in index
    assert f"{off['passed']}/{off['scenarios']}" in index
    assert on["unsafe_scenarios"] == 0
    assert off["unsafe_scenarios"] >= 1


def test_local_references_resolve_and_no_remote_assets(site: Path) -> None:
    for page in ("index.html", "product.html"):
        text = (site / page).read_text()
        for ref in re.findall(r'(?:href|src)="([^"#]+)', text):
            if not ref.startswith(("http://", "https://")):
                assert (site / ref).exists(), f"{page}: {ref}"
        assert not re.search(r'<(?:script|link|img)[^>]+(?:src|href)="https?://', text)
    index = (site / "index.html").read_text()
    assert 'src="img/architecture.svg"' in index
    assert "flowchart TD" in index


def test_product_brief_lists_render_as_lists(site: Path) -> None:
    product = (site / "product.html").read_text()
    assert "<p><strong>In:</strong></p>\n<ul>" in product
    assert "<h2>Roadmap</h2>" in product


def test_readme_diagram_matches_the_rendered_source() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"```mermaid\n(.*?)```", readme, re.DOTALL)
    assert block is not None
    source = (ROOT / "docs" / "architecture.mmd").read_text(encoding="utf-8")
    assert block.group(1).strip() == source.strip()
    assert (ROOT / "docs" / "img" / "architecture.svg").read_text().startswith("<svg")
