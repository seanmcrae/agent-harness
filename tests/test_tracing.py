import itertools
from pathlib import Path

import pytest
from rich.console import Console

from guarded_agent.tracing import (
    JsonlExporter,
    OpenTelemetryExporter,
    SpanKind,
    Tracer,
    load_spans,
    render_trace,
)


def _tracer() -> Tracer:
    ticks = itertools.count(0, 1_000_000)  # 1 ms per clock read
    return Tracer(trace_id="t1", clock_ns=lambda: next(ticks))


def _sample_trace() -> Tracer:
    tracer = _tracer()
    with tracer.span(SpanKind.RUN, "refund") as run:
        with tracer.span(SpanKind.STEP, "step 1"):
            with tracer.span(SpanKind.LLM_CALL, "mock-1", input_tokens=10, output_tokens=5):
                pass
            with tracer.span(SpanKind.TOOL_CALL, "lookup", arguments={"id": 1}) as call:
                call.attributes["outcome"] = "ok"
                tracer.event(SpanKind.GUARDRAIL, "pii_redaction", action="redact", stage="input")
                tracer.event(SpanKind.GUARDRAIL, "prompt_injection", action="allow", stage="input")
        run.attributes.update(status="completed", steps=1)
    return tracer


def test_spans_nest_and_time() -> None:
    tracer = _sample_trace()
    by_name = {s.name: s for s in tracer.spans}
    assert by_name["refund"].parent_id is None
    assert by_name["step 1"].parent_id == by_name["refund"].span_id
    assert by_name["lookup"].parent_id == by_name["step 1"].span_id
    assert by_name["pii_redaction"].parent_id == by_name["lookup"].span_id
    assert by_name["pii_redaction"].duration_ms == 0
    assert by_name["refund"].duration_ms > by_name["step 1"].duration_ms > 0
    assert all(s.end_ns is not None for s in tracer.spans)


def test_span_records_errors_and_reraises() -> None:
    tracer = _tracer()
    with pytest.raises(RuntimeError), tracer.span(SpanKind.TOOL_CALL, "boom"):
        raise RuntimeError("kaput")
    assert tracer.spans[0].status == "error"
    assert tracer.spans[0].attributes["error"] == "RuntimeError: kaput"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    tracer = _sample_trace()
    path = tmp_path / "trace.jsonl"
    JsonlExporter(path).export(tracer.spans)
    loaded = load_spans(path)
    assert [s.to_dict() for s in loaded] == [s.to_dict() for s in tracer.spans]


def test_jsonl_append_mode_accumulates_runs(tmp_path: Path) -> None:
    path = tmp_path / "suite.jsonl"
    exporter = JsonlExporter(path, append=True)
    exporter.export(_sample_trace().spans)
    exporter.export(_sample_trace().spans)
    assert len(load_spans(path)) == 2 * len(_sample_trace().spans)


def test_render_hides_allow_decisions_by_default() -> None:
    console = Console(record=True, width=160, color_system=None)
    for tree in render_trace(_sample_trace().spans):
        console.print(tree)
    text = console.export_text()
    assert "run refund  completed" in text
    assert "tool lookup(id=1)  ok" in text
    assert "guardrail pii_redaction@input  REDACT" in text
    assert "prompt_injection" not in text

    console = Console(record=True, width=160, color_system=None)
    for tree in render_trace(_sample_trace().spans, show_allow=True):
        console.print(tree)
    assert "prompt_injection@input  ALLOW" in console.export_text()


def test_opentelemetry_export_preserves_hierarchy() -> None:
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    export = pytest.importorskip("opentelemetry.sdk.trace.export")

    memory = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(memory))
    OpenTelemetryExporter(provider.get_tracer("test")).export(_sample_trace().spans)

    finished = {s.name: s for s in memory.get_finished_spans()}
    assert set(finished) >= {"run refund", "step step 1", "tool_call lookup"}
    run, step = finished["run refund"], finished["step step 1"]
    assert step.parent is not None
    assert step.parent.span_id == run.context.span_id
    assert finished["tool_call lookup"].attributes["guarded_agent.arguments"] == '{"id": 1}'
