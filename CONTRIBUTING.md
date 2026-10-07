# Contributing

Thanks for taking the time to improve agent-harness. Bug reports, new guardrails, new scenarios,
and provider fixes are all welcome.

## Development setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra docs
make lint typecheck test eval     # the same checks CI runs
make site                         # optional: build the docs site into site/
```

No API keys are needed. Tests and the scenario suite use the deterministic mock provider and never
touch the network.

## Before you open a pull request

- `make lint typecheck test eval` passes locally (ruff lint and format check, mypy strict, pytest,
  and the scenario suite at a 100% pass rate).
- New behaviour has tests. Runtime changes that affect agent outcomes should also have a scenario
  in `scenarios/`.
- If you change a guardrail, run `uv run agent guardrails bench` and include the before and after
  precision/recall in the PR description. Add the cases that motivated the change to
  `src/agent_harness/examples/data/synthetic_guardrail_cases.jsonl`, labelled by intent.
- If you change eval results, regenerate the committed charts with `make charts` and update any
  numbers quoted in the README. Every number in the docs must come from running the code.
- If you edit the architecture diagram, change `docs/architecture.mmd` and the README block
  together and run `make diagram`.
- Keep commits focused, with imperative messages ("Add allowlist check for tool aliases").

## Adding a provider or guardrail

- Providers implement `agent_harness.providers.base.Provider`. Import the SDK lazily, read keys
  from the environment, create the client with SDK retries disabled, and add fake-client tests for
  request and response translation.
- Guardrails implement the `Guardrail` interface in `agent_harness.guardrails.base` and return a
  `Decision`. Never raise to signal a block.

## Reporting security issues

Please do not open a public issue for a guardrail bypass. See [SECURITY.md](SECURITY.md).
