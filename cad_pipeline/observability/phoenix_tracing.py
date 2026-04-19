"""phoenix_tracing.py — Optional OpenTelemetry tracing for Phoenix."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator

_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_TRACER = None


def _init_tracer():
    global _INITIALIZED, _TRACER
    if _INITIALIZED:
        return _TRACER
    with _INIT_LOCK:
        if _INITIALIZED:
            return _TRACER
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except Exception:
            _INITIALIZED = True
            _TRACER = None
            return _TRACER

        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
        project = os.getenv("PHOENIX_PROJECT_NAME", "CAD").strip() or "CAD"
        api_key = os.getenv("PHOENIX_API_KEY", "").strip()
        if not endpoint:
            _INITIALIZED = True
            _TRACER = None
            return _TRACER

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["api_key"] = api_key

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "cad_pipeline",
                    "phoenix.project_name": project,
                }
            )
        )
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("cad_pipeline")
        _INITIALIZED = True
        return _TRACER


@contextmanager
def traced_span(name: str, **attrs) -> Iterator[object]:
    """Context manager for Phoenix span; no-op when tracing unavailable."""
    tracer = _init_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            try:
                if v is not None:
                    span.set_attribute(k, v)
            except Exception:
                pass
        yield span
