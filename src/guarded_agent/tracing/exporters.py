"""Span exporters: JSONL files (always available) and OpenTelemetry (optional extra)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from guarded_agent.tracing.spans import Span


class SpanExporter(Protocol):
    def export(self, spans: Sequence[Span]) -> None: ...


class JsonlExporter:
    """Write one span per line. ``append=True`` lets many runs share a file (eval suites)."""

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self._mode = "a" if append else "w"

    def export(self, spans: Sequence[Span]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(self._mode, encoding="utf-8") as handle:
            for span in spans:
                handle.write(json.dumps(span.to_dict(), default=str) + "\n")
        self._mode = "a"


def load_spans(path: str | Path) -> list[Span]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Span.from_dict(json.loads(line)) for line in handle if line.strip()]


class OpenTelemetryExporter:
    """Replay finished spans into an OpenTelemetry tracer with their original timings.

    Requires the ``otel`` extra. Configure the TracerProvider (OTLP, console, ...) as usual and
    pass a tracer from it, or rely on the globally configured provider.
    """

    def __init__(self, tracer: Any | None = None) -> None:
        from opentelemetry import trace

        self._trace = trace
        self._tracer = tracer or trace.get_tracer("guarded_agent")

    def export(self, spans: Sequence[Span]) -> None:
        otel_spans: dict[str, Any] = {}
        for span in sorted(spans, key=lambda s: s.start_ns):
            parent = otel_spans.get(span.parent_id) if span.parent_id else None
            context = self._trace.set_span_in_context(parent) if parent is not None else None
            otel_span = self._tracer.start_span(
                f"{span.kind.value} {span.name}",
                context=context,
                start_time=span.start_ns,
                attributes=dict(_otel_attributes(span)),
            )
            if span.status == "error":
                otel_span.set_status(self._trace.Status(self._trace.StatusCode.ERROR))
            otel_spans[span.span_id] = otel_span
        for span in spans:
            otel_spans[span.span_id].end(end_time=span.end_ns or span.start_ns)


def _otel_attributes(span: Span) -> Iterable[tuple[str, Any]]:
    yield "guarded_agent.kind", span.kind.value
    for key, value in span.attributes.items():
        if isinstance(value, str | bool | int | float):
            yield f"guarded_agent.{key}", value
        elif value is not None:
            yield f"guarded_agent.{key}", json.dumps(value, default=str)
