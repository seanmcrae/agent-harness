# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/seanmcrae/agent-harness/security/advisories/new)
rather than in a public issue.

Useful reports include:

- A prompt injection, PII pattern, or tool-call sequence that bypasses a guardrail, with the exact
  input and the trace (`--trace-out`).
- A way for a write tool to run without an approval, or to run twice in one run.
- A budget (steps, tokens, cost, time) that can be exceeded without the run stopping.
- Secrets or user data leaking into traces or exported spans.

Expect an acknowledgement within 7 days. Fixes for confirmed issues ship with a regression
scenario or bench case and a CHANGELOG entry.

## Scope and known limits

The injection detector and PII redactor are heuristics with documented gaps (see Limitations in
the README). Paraphrased or purely semantic injections that the bench already lists as misses are
known; reports that extend that list with new patterns are still welcome as bench cases.
