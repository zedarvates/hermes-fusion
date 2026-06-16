"""Hermes Fusion - Multi-LLM Fusion Engine for Hermes Agent."""

__version__ = "1.0.0"
__author__ = "Sylvain Galliez"

# Observability (optional - requires `pip install hermes-fusion[observability]`)
try:
    from .observability import (
        MetricsCollector,
        configure_logging,
        configure_tracing,
        create_query_context,
        generate_metrics,
        get_logger,
        get_metrics_collector,
        get_metrics_content_type,
        log_cache_event,
        log_provider_result,
        log_query_end,
        log_query_start,
        trace_cache_operation,
        trace_provider_call,
        trace_query,
    )
    _observability_available = True
except ImportError:
    _observability_available = False

# Cost Tracker (optional - requires `pip install hermes-fusion[observability]`)
try:
    from .cost_tracker import (
        CLOUD_PROVIDERS,
        DEFAULT_PRICING,
        LOCAL_PROVIDERS,
        Budget,
        CostMetrics,
        CostTracker,
        TokenUsage,
        create_cost_tracker_from_config,
        estimate_cost,
    )
    _cost_tracker_available = True
except ImportError:
    _cost_tracker_available = False

__all__ = [
    "__version__",
    "__author__",
    "_observability_available",
    "_cost_tracker_available",
]