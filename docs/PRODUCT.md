# agent-harness: product write-up

## Problem

Teams can get a tool-using agent to work in a notebook in a week. Getting it to production, and
keeping it there, is where things go wrong. Three failures come up again and again:

1. **Unpredictable execution.** The same request can take 3 steps one day and 14 the next. A run
   may loop on a failing tool, or end with a JSON blob that does not parse. Cost per task has a long
   tail that nobody budgeted for.
2. **Ungoverned side effects.** An agent with a `refund`, `send_email`, or `delete` tool acts on the
   world. Without controls, the only thing between a prompt injection in a customer note and money
   leaving the account is the model's judgement.
3. **Opaque failures.** When a run goes wrong, the evidence is a chat log. Nobody can answer "which
   tool call, what did the guardrail decide, how much did it cost, did we retry?" without adding
   logging after the incident.

The frameworks that address these problems also bring their own graph DSL, state model, and
dependency footprint. Many teams want the controls without adopting a framework.

## Users and jobs-to-be-done

| User | Job | What they need from this repo |
| --- | --- | --- |
| AI product engineer shipping one agent | "Get my agent safe enough to put in front of customers." | Typed tools, approvals on writes, budgets, structured output that validates |
| Platform / infra engineer | "Give every product team the same guardrails and telemetry without reviewing each agent by hand." | One runtime contract, spans in OpenTelemetry, a guardrail interface teams extend |
| AI PM / tech lead | "Know whether the agent got better or worse after a prompt or model change, and what it costs per task." | Scenario evals in CI with pass rate, mean steps, cost per task, guardrail triggers |
| Trust and safety / risk reviewer | "Show me what stops the agent from doing X." | Forbidden-action scenarios, the guardrail bench, trace evidence for each decision |

## Scope

**In:**
- A synchronous agent loop with step, token, cost, and wall-clock limits.
- Retry with backoff on transient provider errors.
- Structured final answers with pydantic validation and one repair attempt.
- Tool registry: schemas from type hints, argument validation, timeouts, read/write
  classification, idempotency, and a human-approval hook.
- Guardrails at four stages (input, tool call, tool output, final output) plus budgets, all
  returning typed decisions.
- Tracing: a span model, a JSONL exporter, an OpenTelemetry exporter, and a terminal tree view.
- Provider adapters: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses, and a
  deterministic mock.
- YAML scenario evals and a guardrail precision/recall bench, all runnable in CI without keys.
- Two reference agents on synthetic data.

**Out (deliberately):**
- Multi-agent orchestration, graph or state-machine DSLs, memory stores, vector databases.
- Hosted UI or trace storage. We export to OTel and let teams use the backend they already run.
- ML-based classifiers for injection or PII. The `Guardrail` interface is the extension point
  for them.
- Streaming and async execution, which are on the roadmap below.

## Requirements

| # | Requirement | How it is met |
| --- | --- | --- |
| R1 | A run must stop at its step, token, cost, or time limit, and say which one | `BudgetTracker` checks before every step and before executing tools. Each limit maps to a distinct `RunStatus` and a `budget` decision span |
| R2 | Transient provider failures are retried without hiding them | The runtime owns retries (SDK `max_retries=0`). Each attempt is its own span, and backoff has jitter and respects the deadline |
| R3 | Write tools never run without an approval verdict | Default approver denies. The verdict and reviewer are recorded on the tool span |
| R4 | A non-idempotent write never runs twice in a run | Calls are keyed on canonicalised arguments and recorded before execution, so a crashed write cannot be repeated |
| R5 | Untrusted text is screened at every boundary | `GuardrailSet` runs at input, tool call, tool output, and output. Redactions chain, and the first block wins |
| R6 | Every guardrail outcome is auditable | Each `Decision` becomes a `guardrail` span with stage, action, reason, and findings |
| R7 | Final answers conform to a schema or the run says so | Pydantic validation, one repair prompt with the errors, then `invalid_output` |
| R8 | Everything runs without API keys | `MockProvider` is the default for the CLI, tests, and CI |
| R9 | Behaviour changes are caught before merge | `agent eval scenarios/` exits non-zero below `--min-pass-rate`. CI runs it on every push |

## Success metrics and evals

These are the metrics the repo measures about itself. Current values are from the bundled
synthetic data with the mock provider and simulated pricing ($3 / $15 per Mtok).

| Metric | Definition | How measured | Current |
| --- | --- | --- | --- |
| Task success rate | Scenarios meeting every expectation (outcome, tool sequence, forbidden actions, limits) | `agent eval scenarios/` | 19/19 (100%) |
| Unsafe outcomes prevented | Ablation: scenarios with a safety violation (a forbidden tool executed or a forbidden pattern in the answer), content guardrails off vs on | `agent eval scenarios/ --no-guardrails` (`unsafe` in the summary) | 2 off vs 0 on (unauthorised refund, relayed phishing); 15/19 pass overall without guardrails |
| Guardrail precision | TP / (TP + FP) per check on labelled cases | `agent guardrails bench` | Injection 1.000, PII 1.000 (in-sample) |
| Guardrail recall | TP / (TP + FN) | same | Injection 0.733 (11/15), PII 1.000 |
| Cost per task | Mean `cost_usd` per scenario | eval report | $0.0055 (simulated) |
| Mean steps | Mean model turns per scenario | eval report | 2.89 |

How to read these numbers:
- **Mock-mode success rate checks the runtime, not the model.** A 100% pass rate shows that the
  loop, guardrails, approvals, and budgets behave as specified against a model that does what the
  policy scripts. The same suite runs against a real model with `--provider anthropic|openai` and a
  `--min-pass-rate` threshold. That number is the model-quality metric, and it has not been
  measured here.
- **Guardrail precision is the metric to protect.** A false positive withholds a legitimate tool
  output or blocks a valid answer, and users experience that as the agent being broken. Recall gaps
  are covered by the other layers (allowlist, approval, policy check, output policy). The bench test
  in CI fails on any new false positive.
- **The bench is in-sample.** The rules were tuned on these 48 cases. A held-out set, and labelled
  traffic from real deployments, are needed before quoting recall as a capability.

Guardrail metrics we intend to track once real traffic exists: trigger rate per 1k runs by stage,
approval denial rate per write tool, share of runs ending in a non-`completed` status, and p95 cost
per task.

## Trade-offs and alternatives considered

| Option | Why not (for this scope) | What we gave up |
| --- | --- | --- |
| **LangGraph / LangChain agents** | They are strong for complex graphs with checkpointing. But they bring a large dependency surface and a state model of their own, and guardrails and budgets are add-ons rather than core loop semantics. | Graph composition, persistence, human-in-the-loop resume, a large integration catalogue |
| **OpenAI Agents SDK / vendor SDKs** | Good ergonomics, but tied to one vendor's API shapes. We need the same guardrail and eval contract across Anthropic and OpenAI. | Built-in hosted tools and handoffs |
| **Guardrails-as-a-service (classifier APIs)** | Add a network hop and a vendor to every boundary, and cannot run in keyless CI. | Higher recall on semantic injections |
| **Exceptions instead of decisions** | Raising on a violation is simpler, but it loses the ability to redact-and-continue and to count triggers. | A little code simplicity |
| **Async-first loop** | Most agent workloads are latency-bound on the model, not on concurrency within a single run. Sync keeps the core easy to read and to test deterministically. | Parallel tool execution within a turn, native streaming |

The bet is that about 800 lines of core runtime with strict contracts are easier to audit,
extend, and trust than a framework, for teams whose agents are a loop over a handful of tools.
When that stops being true (long-running workflows, branching, resumable state), LangGraph is the
better choice. The span model and scenario format here are meant to carry over if a team switches.

## Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Heuristic injection detection gives false confidence | High | High | Documented recall, in-sample label, layered controls. Write tools fail closed regardless |
| Mock-mode green CI hides real-model regressions | Medium | High | Same scenarios run against real providers with a pass-rate gate. Recommend a nightly job with keys |
| Static pricing table drifts from vendor prices | High | Medium | Prices live in one module with per-provider override. Unknown models are priced at $0 and flagged in docs |
| Abandoned tool threads after a timeout keep running side effects | Medium | Medium | Writes are not retried after timeout, and records mark `executed`. Docs require tools to enforce their own I/O timeouts |
| Approval fatigue leads reviewers to rubber-stamp | Medium | High | Approval requests carry canonical arguments. The roadmap adds policy-based auto-approval for low-risk writes, so humans see fewer, higher-stakes requests |
| PII redaction misses categories (names, addresses) | High | Medium | Scope stated plainly. The `Guardrail` interface accepts an NER-based redactor |

## Roadmap

**Now (this repo)**
- Core loop, budgets, retries, structured output repair.
- Tool registry with approval and idempotency semantics.
- Four-stage guardrails, synthetic bench, JSONL and OTel tracing, tree renderer.
- Anthropic and OpenAI adapters, mock provider, 19 scenarios gating CI.
- Charts and a GitHub Pages site rebuilt from live eval runs on every push to main.

**Next**
- Held-out guardrail set, plus a pluggable classifier guardrail (for example, a small local model)
  with the bench comparing it against the heuristics.
- Policy-based approval: rules such as "auto-approve refunds under $50 on eligible orders", with
  every auto-approval traced.
- Nightly real-provider eval job with per-model pass-rate, cost, and step trends.
- Async loop with concurrent tool execution within a turn, and streaming of final answers.

**Later**
- Resumable runs: persist `RunResult` messages so a denied approval can be retried after review
  without replaying earlier steps.
- Scenario generation from production traces: turn a failing trace into a regression scenario with
  one command.
- Cost and step SLOs per agent, with alerts driven by exported spans.
