"""Observability module - Prometheus metrics, structured logging, OpenTelemetry tracing."""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Global registry (can be overridden for testing)
_metrics_registry: CollectorRegistry | None = None


def get_registry() -> CollectorRegistry:
    """Get or create the Prometheus registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = CollectorRegistry(auto_describe=True)
    return _metrics_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _metrics_registry
    _metrics_registry = None


# =============================================================================
# Prometheus Metrics
# =============================================================================

# Query metrics
queries_total = Counter(
    "hermes_fusion_queries_total",
    "Total number of fusion queries executed",
    ["strategy", "result"],
    registry=get_registry(),
)

query_duration_seconds = Histogram(
    "hermes_fusion_query_duration_seconds",
    "Time spent executing fusion queries",
    ["strategy"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=get_registry(),
)

# Cache metrics
cache_hits_total = Counter(
    "hermes_fusion_cache_hits_total",
    "Total cache hits",
    ["cache_type"],
    registry=get_registry(),
)

cache_misses_total = Counter(
    "hermes_fusion_cache_misses_total",
    "Total cache misses",
    ["cache_type"],
    registry=get_registry(),
)

cache_size = Gauge(
    "hermes_fusion_cache_size",
    "Current number of entries in cache",
    ["cache_type"],
    registry=get_registry(),
)

# Provider metrics
provider_requests_total = Counter(
    "hermes_fusion_provider_requests_total",
    "Total requests to each provider",
    ["provider", "status"],
    registry=get_registry(),
)

provider_duration_seconds = Histogram(
    "hermes_fusion_provider_duration_seconds",
    "Time spent calling each provider",
    ["provider"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=get_registry(),
)

provider_errors_total = Counter(
    "hermes_fusion_provider_errors_total",
    "Total errors per provider",
    ["provider", "error_type"],
    registry=get_registry(),
)

provider_health = Gauge(
    "hermes_fusion_provider_health",
    "Current health status of provider (1=healthy, 0=unhealthy)",
    ["provider"],
    registry=get_registry(),
)

# Strategy metrics
strategy_fusions_total = Counter(
    "hermes_fusion_strategy_fusions_total",
    "Total fusions executed per strategy",
    ["strategy"],
    registry=get_registry(),
)

strategy_confidence = Histogram(
    "hermes_fusion_strategy_confidence",
    "Confidence scores from fusion strategies",
    ["strategy"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=get_registry(),
)

# Fusion engine metrics
active_queries = Gauge(
    "hermes_fusion_active_queries",
    "Number of currently executing queries",
    registry=get_registry(),
)

fusion_errors_total = Counter(
    "hermes_fusion_errors_total",
    "Total fusion engine errors",
    ["error_type"],
    registry=get_registry(),
)


# =============================================================================
# Metrics Helpers
# =============================================================================

class MetricsCollector:
    """Centralized metrics collection for Fusion Engine."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or get_registry()

    def record_query(self, strategy: str, success: bool, duration: float) -> None:
        """Record a completed query."""
        queries_total.labels(strategy=strategy, result="success" if success else "error").inc()
        query_duration_seconds.labels(strategy=strategy).observe(duration)

    def record_cache_hit(self, cache_type: str = "semantic") -> None:
        """Record a cache hit."""
        cache_hits_total.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "semantic") -> None:
        """Record a cache miss."""
        cache_misses_total.labels(cache_type=cache_type).inc()

    def set_cache_size(self, size: int, cache_type: str = "semantic") -> None:
        """Set current cache size."""
        cache_size.labels(cache_type=cache_type).set(size)

    def record_provider_request(
        self, provider: str, success: bool, duration: float, error_type: str | None = None
    ) -> None:
        """Record a provider request."""
        status = "success" if success else "error"
        provider_requests_total.labels(provider=provider, status=status).inc()
        provider_duration_seconds.labels(provider=provider).observe(duration)
        if not success and error_type:
            provider_errors_total.labels(provider=provider, error_type=error_type).inc()

    def set_provider_health(self, provider: str, healthy: bool) -> None:
        """Set provider health status."""
        provider_health.labels(provider=provider).set(1 if healthy else 0)

    def record_fusion(self, strategy: str, confidence: float) -> None:
        """Record a fusion execution."""
        strategy_fusions_total.labels(strategy=strategy).inc()
        strategy_confidence.labels(strategy=strategy).observe(confidence)

    def set_active_queries(self, count: int) -> None:
        """Set number of active queries."""
        active_queries.set(count)

    def record_error(self, error_type: str) -> None:
        """Record a fusion engine error."""
        fusion_errors_total.labels(error_type=error_type).inc()


# Global metrics collector
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (for testing)."""
    global _metrics_collector
    _metrics_collector = None


def generate_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(get_registry())


def get_metrics_content_type() -> str:
    """Get the Prometheus metrics content type."""
    return CONTENT_TYPE_LATEST


# =============================================================================
# Structured Logging
# =============================================================================

def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structlog for structured logging."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True) if include_timestamp else lambda *_, **__: None,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if json_output:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set stdlib logging level
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# =============================================================================
# OpenTelemetry Tracing
# =============================================================================

def configure_tracing(
    service_name: str = "hermes-fusion",
    enable_console_export: bool = False,
    enable_prometheus_export: bool = True,
) -> trace.Tracer:
    """Configure OpenTelemetry tracing and metrics."""
    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })

    # Configure tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    if enable_console_export:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    trace.set_tracer_provider(tracer_provider)

    # Configure meter provider with Prometheus exporter
    if enable_prometheus_export:
        reader = PrometheusMetricReader()
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    else:
        meter_provider = MeterProvider(resource=resource)

    from opentelemetry import metrics
    metrics.set_meter_provider(meter_provider)

    # Instrument asyncio
    AsyncioInstrumentor().instrument()

    return trace.get_tracer(service_name)


def get_tracer(name: str | None = None) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name or "hermes-fusion")


# =============================================================================
# Context Managers for Tracing
# =============================================================================

@asynccontextmanager
async def trace_query(question: str, strategy: str, provider_count: int):
    """Context manager for tracing a fusion query."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "fusion.query",
        attributes={
            "hermes.strategy": strategy,
            "hermes.provider_count": provider_count,
            "hermes.question_length": len(question),
        },
    ) as span:
        start_time = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", str(e))
            raise
        finally:
            duration = time.perf_counter() - start_time
            span.set_attribute("hermes.duration_seconds", duration)


@asynccontextmanager
async def trace_provider_call(provider: str, model: str | None = None):
    """Context manager for tracing a provider call."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "provider.call",
        attributes={
            "hermes.provider": provider,
            "hermes.model": model or "default",
        },
    ) as span:
        start_time = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", str(e))
            raise
        finally:
            duration = time.perf_counter() - start_time
            span.set_attribute("hermes.duration_seconds", duration)


@asynccontextmanager
async def trace_cache_operation(operation: str, cache_type: str = "semantic"):
    """Context manager for tracing cache operations."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"cache.{operation}",
        attributes={
            "hermes.cache_type": cache_type,
            "hermes.operation": operation,
        },
    ) as span:
        start_time = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", str(e))
            raise
        finally:
            duration = time.perf_counter() - start_time
            span.set_attribute("hermes.duration_seconds", duration)


# =============================================================================
# Logging Context Helpers
# =============================================================================

@dataclass
class QueryContext:
    """Context for query logging."""
    query_id: str
    question: str
    strategy: str
    provider_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def bind(self, logger: structlog.BoundLogger) -> structlog.BoundLogger:
        """Bind context to logger."""
        return logger.bind(
            query_id=self.query_id,
            question_preview=self.question[:100],
            strategy=self.strategy,
            provider_count=self.provider_count,
            **self.metadata,
        )


def create_query_context(
    query_id: str,
    question: str,
    strategy: str,
    provider_count: int,
    **metadata,
) -> QueryContext:
    """Create a query context for structured logging."""
    return QueryContext(
        query_id=query_id,
        question=question,
        strategy=strategy,
        provider_count=provider_count,
        metadata=metadata,
    )


def log_query_start(logger: structlog.BoundLogger, ctx: QueryContext) -> None:
    """Log query start."""
    ctx.bind(logger).info("fusion.query.start", question_length=len(ctx.question))


def log_query_end(
    logger: structlog.BoundLogger,
    ctx: QueryContext,
    result: Any,
    duration: float,
    cached: bool = False,
) -> None:
    """Log query completion."""
    ctx.bind(logger).info(
        "fusion.query.end",
        duration_seconds=duration,
        cached=cached,
        confidence=getattr(result, "confidence", None),
        answer_length=len(getattr(result, "final_answer", "")),
    )


def log_provider_result(
    logger: structlog.BoundLogger,
    provider: str,
    success: bool,
    duration: float,
    error: str | None = None,
) -> None:
    """Log provider call result."""
    logger.bind(provider=provider).info(
        "fusion.provider.result",
        success=success,
        duration_seconds=duration,
        error=error,
    )


def log_cache_event(
    logger: structlog.BoundLogger,
    event: str,
    cache_type: str,
    hit: bool | None = None,
    size: int | None = None,
) -> None:
    """Log cache event."""
    logger.bind(cache_type=cache_type).info(
        f"fusion.cache.{event}",
        hit=hit,
        size=size,
    )