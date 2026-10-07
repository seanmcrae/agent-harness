"""Span model and an in-process tracer for agent runs."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class SpanKind(StrEnum):
    RUN = "run"
    STEP = "step"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    GUARDRAIL = "guardrail"


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: SpanKind
    name: str
    start_ns: int
    end_ns: int | None = None
    status: Literal["ok", "error"] = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return 0.0 if self.end_ns is None else (self.end_ns - self.start_ns) / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "name": self.name,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_id=data.get("parent_id"),
            kind=SpanKind(data["kind"]),
            name=data["name"],
            start_ns=int(data["start_ns"]),
            end_ns=None if data.get("end_ns") is None else int(data["end_ns"]),
            status=data.get("status", "ok"),
            attributes=dict(data.get("attributes", {})),
        )


class Tracer:
    """Collects nested spans for one run. Not thread-safe; one tracer per run."""

    def __init__(
        self,
        trace_id: str | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self._clock_ns = clock_ns
        self._stack: list[Span] = []
        self.spans: list[Span] = []

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    def _new(self, kind: SpanKind, name: str, attributes: dict[str, Any]) -> Span:
        parent = self.current
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            kind=kind,
            name=name,
            start_ns=self._clock_ns(),
            attributes=attributes,
        )
        self.spans.append(span)
        return span

    @contextmanager
    def span(self, kind: SpanKind, name: str, **attributes: Any) -> Iterator[Span]:
        span = self._new(kind, name, attributes)
        self._stack.append(span)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.attributes.setdefault("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._stack.pop()
            span.end_ns = self._clock_ns()

    def event(self, kind: SpanKind, name: str, **attributes: Any) -> Span:
        """Record an instantaneous child span of the current span."""
        span = self._new(kind, name, attributes)
        span.end_ns = span.start_ns
        return span
