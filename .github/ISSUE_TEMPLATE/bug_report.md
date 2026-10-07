---
name: Bug report
about: Something in the runtime, CLI, evals, or docs does not behave as documented
title: ""
labels: bug
assignees: ""
---

**What happened**

A clear description of the bug. For a guardrail bypass, please use private vulnerability
reporting instead (see SECURITY.md).

**How to reproduce**

```bash
# exact commands, e.g.
uv run agent run refund "..." --approve yes --trace-out trace.jsonl
```

If a run is involved, attach the trace (`--trace-out`) or the output of `agent trace show`.

**What you expected**

**Environment**

- agent-harness version or commit:
- Python version:
- OS:
- Provider and model (`mock`, `anthropic`, `openai`, `openai-responses`):
