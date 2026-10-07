"""Terminal rendering of a trace as a tree."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from rich.markup import escape
from rich.tree import Tree

from agent_harness.tracing.spans import Span, SpanKind

_ACTION_STYLE = {"allow": "dim", "redact": "yellow", "block": "bold red"}
_OUTCOME_STYLE = {"ok": "green", "denied": "yellow"}


def _ms(span: Span) -> str:
    ms = span.duration_ms
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _args(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_short(val, 40)}" for key, val in arguments.items())


def label(span: Span) -> str:
    attrs = span.attributes
    error = f" [red]error: {escape(_short(attrs['error']))}[/]" if span.status == "error" else ""
    if span.kind is SpanKind.RUN:
        return (
            f"[bold]run[/] {escape(span.name)}  [bold]{attrs.get('status', '?')}[/]  {_ms(span)}  "
            f"steps={attrs.get('steps', 0)}  tokens={attrs.get('input_tokens', 0)}"
            f"+{attrs.get('output_tokens', 0)}  cost=${attrs.get('cost_usd', 0.0):.4f}"
        )
    if span.kind is SpanKind.STEP:
        return f"[cyan]{escape(span.name)}[/]  {_ms(span)}{error}"
    if span.kind is SpanKind.LLM_CALL:
        tokens = f"tokens={attrs.get('input_tokens', 0)}+{attrs.get('output_tokens', 0)}"
        result = f"  -> {escape(attrs['result'])}" if "result" in attrs else ""
        return (
            f"[magenta]llm[/] {escape(span.name)}  attempt={attrs.get('attempt', 1)}  {tokens}  "
            f"${attrs.get('cost_usd', 0.0):.4f}  {_ms(span)}{result}{error}"
        )
    if span.kind is SpanKind.TOOL_CALL:
        outcome = str(attrs.get("outcome", "?"))
        style = _OUTCOME_STYLE.get(outcome, "red")
        extras = "".join(
            f"  {key}={escape(str(attrs[key]))}"
            for key in ("approval", "side_effect")
            if key in attrs
        )
        withheld = "  [bold red]output withheld[/]" if attrs.get("withheld") else ""
        return (
            f"[blue]tool[/] {escape(span.name)}({escape(_args(attrs.get('arguments', {})))})  "
            f"[{style}]{outcome}[/]  {_ms(span)}{extras}{withheld}"
        )
    action = str(attrs.get("action", "allow"))
    findings = ", ".join(attrs.get("findings", []))
    detail = f"  {escape(findings)}" if findings else ""
    reason = f"  ({escape(attrs['reason'])})" if attrs.get("reason") else ""
    return (
        f"[{_ACTION_STYLE.get(action, 'white')}]guardrail {escape(span.name)}@"
        f"{attrs.get('stage', '?')}  {action.upper()}{detail}{reason}[/]"
    )


def render_trace(spans: Sequence[Span], *, show_allow: bool = False) -> list[Tree]:
    """Build one rich Tree per trace. Allow-decisions are hidden unless ``show_allow``."""
    children: dict[str | None, list[Span]] = defaultdict(list)
    for span in sorted(spans, key=lambda s: s.start_ns):
        hidden = span.kind is SpanKind.GUARDRAIL and span.attributes.get("action") == "allow"
        if hidden and not show_allow:
            continue
        children[span.parent_id].append(span)

    def build(node: Tree, span: Span) -> None:
        for child in children.get(span.span_id, []):
            build(node.add(label(child)), child)

    trees = []
    for root in children.get(None, []):
        tree = Tree(label(root))
        build(tree, root)
        trees.append(tree)
    return trees
