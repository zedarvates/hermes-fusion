"""Tests for observability module."""

from hermes_fusion.observability import (
    MetricsCollector,
    configure_logging,
    generate_metrics,
    get_logger,
    get_metrics_collector,
    get_metrics_content_type,
    reset_metrics_collector,
)


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_metrics_collector()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_metrics_collector()

    def test_metrics_collector_creation(self):
        """Test creating a MetricsCollector."""
        collector = MetricsCollector()
        assert collector is not None

    def test_record_query(self):
        """Test recording query metrics."""
        collector = MetricsCollector()
        collector.record_query("weighted_vote", True, 0.5)
        collector.record_query("weighted_vote", False, 1.0)

        # Verify no exceptions raised
        assert True

    def test_record_cache_hit_miss(self):
        """Test recording cache hits and misses."""
        collector = MetricsCollector()
        collector.record_cache_hit("semantic")
        collector.record_cache_miss("semantic")
        collector.set_cache_size(100, "semantic")

        assert True

    def test_record_provider_request(self):
        """Test recording provider requests."""
        collector = MetricsCollector()
        collector.record_provider_request("localai", True, 0.3)
        collector.record_provider_request("xai", False, 2.0, "timeout")
        collector.set_provider_health("localai", True)
        collector.set_provider_health("xai", False)

        assert True

    def test_record_fusion(self):
        """Test recording fusion executions."""
        collector = MetricsCollector()
        collector.record_fusion("weighted_vote", 0.85)
        collector.record_fusion("cot_consensus", 0.92)

        assert True

    def test_set_active_queries(self):
        """Test setting active queries count."""
        collector = MetricsCollector()
        collector.set_active_queries(5)
        collector.set_active_queries(0)

        assert True

    def test_record_error(self):
        """Test recording errors."""
        collector = MetricsCollector()
        collector.record_error("all_providers_failed")
        collector.record_error("timeout")

        assert True


class TestLogging:
    """Test structured logging functions."""

    def test_configure_logging(self):
        """Test logging configuration."""
        configure_logging(level="DEBUG", json_output=False)
        configure_logging(level="INFO", json_output=True)
        # No exceptions = success

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2


class TestMetricsGeneration:
    """Test Prometheus metrics generation."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_metrics_collector()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_metrics_collector()

    def test_generate_metrics(self):
        """Test generating Prometheus metrics output."""
        # Record some metrics first
        collector = get_metrics_collector()
        collector.record_query("weighted_vote", True, 0.5)
        collector.record_cache_hit("semantic")

        output = generate_metrics()
        assert isinstance(output, bytes)
        assert b"hermes_fusion_queries_total" in output
        assert b"hermes_fusion_cache_hits_total" in output

    def test_get_metrics_content_type(self):
        """Test getting metrics content type."""
        content_type = get_metrics_content_type()
        assert "text/plain" in content_type or "application/openmetrics-text" in content_type