# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry span helpers for IICP SDK nodes — ADR-014 TRACE-01/TRACE-02.

Node-side spans:
  iicp.task.validate  (TRACE-01) — request parsing, nonce check, auth
  iicp.task.execute   (TRACE-02) — handler dispatch and response

Behaviour:
  - When opentelemetry-api is installed AND OTEL_EXPORTER_OTLP_ENDPOINT is set:
    exports spans to the configured collector via OTLP/HTTP.
  - Otherwise: yields a no-op span — call sites need no conditionals.

W3C traceparent propagation is handled in node.py at the HTTP layer; this
module manages the span lifecycle within the node process.
"""
from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Generator

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_initialised = False
_tracer: object | None = None


def _init() -> None:
    global _initialised, _tracer
    if _initialised:
        return
    _initialised = True

    if not _OTEL_AVAILABLE:
        _tracer = _NoopTracer()
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = _TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            _otel_trace.set_tracer_provider(provider)  # type: ignore[attr-defined]
            _tracer = _otel_trace.get_tracer("iicp.client")  # type: ignore[attr-defined]
            logger.info("otel_tracer: OTLP exporter configured → %s", endpoint)
        except Exception as exc:  # noqa: BLE001
            logger.warning("otel_tracer: OTLP init failed (%s) — using no-op", exc)
            _tracer = _NoopTracer()
    else:
        provider = _TracerProvider()
        _otel_trace.set_tracer_provider(provider)  # type: ignore[attr-defined]
        _tracer = _otel_trace.get_tracer("iicp.client")  # type: ignore[attr-defined]


class _NoopSpan:
    def set_attribute(self, _key: str, _value: object) -> None:
        pass

    def record_exception(self, _exc: BaseException) -> None:
        pass


class _NoopTracer:
    @contextlib.contextmanager
    def start_as_current_span(self, _name: str, **_kw: object) -> Generator[_NoopSpan, None, None]:
        yield _NoopSpan()


@contextlib.contextmanager
def task_validate_span(task_id: str) -> Generator[object, None, None]:
    """TRACE-01: iicp.task.validate — wraps request parsing and auth check."""
    _init()
    assert _tracer is not None
    with _tracer.start_as_current_span("iicp.task.validate") as span:  # type: ignore[attr-defined]
        span.set_attribute("iicp.task_id", task_id)
        yield span


@contextlib.contextmanager
def task_execute_span(task_id: str, intent: str) -> Generator[object, None, None]:
    """TRACE-02: iicp.task.execute — wraps handler dispatch and response."""
    _init()
    assert _tracer is not None
    with _tracer.start_as_current_span("iicp.task.execute") as span:  # type: ignore[attr-defined]
        span.set_attribute("iicp.task_id", task_id)
        span.set_attribute("iicp.intent", intent)
        yield span
