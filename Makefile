.PHONY: install lint format typecheck test cov demo eval bench check

install:
	uv sync --extra dev

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

check: lint typecheck test eval
