.PHONY: install lint format typecheck test cov demo eval bench charts diagram site check

install:
	uv sync --extra dev --extra docs

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term

demo:
	uv run agent run refund "Hi, can you look at ORD-1006 and refund it? Reach me at quinn@example.com." --approve yes --no-show-trace --trace-out traces/demo.jsonl
	uv run agent trace show traces/demo.jsonl

eval:
	uv run agent eval scenarios/

bench:
	uv run agent guardrails bench

# Re-render docs/img from fresh eval and bench runs on the bundled synthetic data.
charts:
	uv run --extra docs python scripts/render_charts.py docs/img

# Re-render the architecture diagram after editing docs/architecture.mmd (needs Node and Chromium).
diagram:
	npx -y -p @mermaid-js/mermaid-cli@11 mmdc -i docs/architecture.mmd -o docs/img/architecture.svg -c docs/mermaid.json -b white

# Static documentation site in site/ (published to GitHub Pages by .github/workflows/pages.yml).
site:
	uv run --extra docs python scripts/build_site.py --out site

check: lint typecheck test eval
