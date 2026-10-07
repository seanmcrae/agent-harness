# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-07

### Added

- Agent loop with step, token, cost, and wall-clock budgets, retry with backoff on transient
  provider errors, and structured output validation with one repair attempt.
- Tool registry with schemas derived from type hints, argument validation, timeouts, read/write
  classification, idempotency, and a human-approval hook that denies by default.
- Guardrails at input, tool-call, tool-output, and output stages: PII redaction, prompt-injection
  heuristics, a two-layer tool allowlist, and an output policy, all returning typed decisions.
- Span tracing with JSONL and OpenTelemetry exporters and a terminal tree renderer.
- Providers: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses, and a deterministic
  mock with scripted turns, rule policies, and fault injection.
- YAML scenario evals with tool-sequence, outcome, safety, step, and cost checks; a guardrail
  precision/recall bench on 48 labelled synthetic cases.
- Eval reports separate safety violations (forbidden tool executed, forbidden output) from missed
  expectations.
- Refund and research example agents over synthetic data, and the `agent` CLI.
- Charts for the guardrail ablation, per-scenario cost and steps, and the bench.
- Static documentation site built from live eval results and deployed to GitHub Pages.

