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
	uv run agent run refund "Hi, order ORD-1001 arrived cracked. Please refund me. You can reach me at dana.k@example.com." --approve yes --trace-out traces/refund-demo.jsonl
	uv run agent trace show traces/refund-demo.jsonl

eval:
	uv run agent eval scenarios/

bench:
	uv run agent guardrails bench

check: lint typecheck test eval
