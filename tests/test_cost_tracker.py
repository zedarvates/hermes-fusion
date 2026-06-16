"""Tests for cost tracker module."""

import asyncio
import os
import tempfile

import pytest

from hermes_fusion.cost_tracker import (
    DEFAULT_PRICING,
    Budget,
    CostMetrics,
    CostTracker,
    TokenUsage,
    estimate_cost,
)


class TestTokenUsage:
    def test_token_usage_creation(self):
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
            query_id="test-123",
        )
        assert usage.provider == "xai"
        assert usage.model == "grok-3"
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500
    
    def test_cost_calculation(self):
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
        )
        # grok-3: $0.30/1K input, $0.60/1K output
        expected = (1000/1000)*0.30 + (500/1000)*0.60
        assert usage.cost_usd == expected
    
    def test_cached_request_zero_cost(self):
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
            cached=True,
        )
        assert usage.cost_usd == 0.0
    
    def test_unknown_model_zero_cost(self):
        usage = TokenUsage(
            provider="xai",
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert usage.cost_usd == 0.0


class TestBudget:
    def test_budget_creation(self):
        budget = Budget(limit_usd=10.0, period="daily", alert_threshold=0.8)
        assert budget.limit_usd == 10.0
        assert budget.period == "daily"
        assert budget.alert_threshold == 0.8
    
    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            Budget(limit_usd=10.0, period="yearly")
    
    def test_period_start(self):
        budget = Budget(limit_usd=10.0, period="daily")
        start = budget.period_start
        # Should be midnight today
        from datetime import datetime
        expected = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        assert abs(start - expected.timestamp()) < 2


class TestCostMetrics:
    def test_add_usage(self):
        metrics = CostMetrics()
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
        )
        metrics.add_usage(usage)
        
        assert metrics.total_cost_usd == usage.cost_usd
        assert metrics.total_tokens == 1500
        assert metrics.total_input_tokens == 1000
        assert metrics.total_output_tokens == 500
        assert metrics.request_count == 1
        assert metrics.provider_costs["xai"] == usage.cost_usd
        assert metrics.model_costs["grok-3"] == usage.cost_usd
    
    def test_cached_usage(self):
        metrics = CostMetrics()
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
            cached=True,
        )
        metrics.add_usage(usage)
        
        assert metrics.total_cost_usd == 0.0
        assert metrics.cached_requests == 1
        assert metrics.request_count == 1
    
    def test_get_budget_status(self):
        metrics = CostMetrics()
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
        )
        metrics.add_usage(usage)
        
        budget = Budget(limit_usd=10.0, period="daily", alert_threshold=0.5)
        status = metrics.get_budget_status(budget)
        
        assert status["budget_limit"] == 10.0
        assert status["spent"] == usage.cost_usd
        assert "utilization" in status
        assert "alert_triggered" in status


class TestCostTracker:
    def test_tracker_creation(self):
        tracker = CostTracker()
        assert tracker is not None
    
    def test_record_usage(self):
        tracker = CostTracker()
        usage = TokenUsage(
            provider="xai",
            model="grok-3",
            input_tokens=1000,
            output_tokens=500,
        )
        
        import asyncio
        asyncio.run(tracker.record_usage(usage))
        
        metrics = tracker.get_metrics()
        assert metrics.total_cost_usd == usage.cost_usd
        assert metrics.request_count == 1
    
    def test_persistence(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            tracker = CostTracker(persistence_path=temp_path)
            
            async def test():
                await tracker.start()
                usage = TokenUsage(
                    provider="xai",
                    model="grok-3",
                    input_tokens=1000,
                    output_tokens=500,
                )
                await tracker.record_usage(usage)
                await tracker.stop()
                return usage.cost_usd
            
            expected_cost = asyncio.run(test())
            
            # Load in new tracker
            tracker2 = CostTracker(persistence_path=temp_path)
            asyncio.run(tracker2.start())
            
            metrics = tracker2.get_metrics()
            assert metrics.total_cost_usd == expected_cost
            
            asyncio.run(tracker2.stop())
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_metrics_by_provider(self):
        tracker = CostTracker()
        
        async def test():
            await tracker.start()
            usage1 = TokenUsage(provider="xai", model="grok-3", input_tokens=1000, output_tokens=500)
            usage2 = TokenUsage(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=500)
            await tracker.record_usage(usage1)
            await tracker.record_usage(usage2)
            await tracker.stop()
        
        asyncio.run(test())
        
        by_provider = tracker.get_metrics_by_provider()
        assert "xai" in by_provider
        assert "openai" in by_provider
        assert by_provider["xai"]["cost_usd"] > 0
        assert by_provider["openai"]["cost_usd"] > 0
    
    def test_get_metrics_by_model(self):
        tracker = CostTracker()
        
        async def test():
            await tracker.start()
            usage = TokenUsage(provider="xai", model="grok-3", input_tokens=1000, output_tokens=500)
            await tracker.record_usage(usage)
            await tracker.stop()
        
        asyncio.run(test())
        
        by_model = tracker.get_metrics_by_model()
        assert "grok-3" in by_model
        assert by_model["grok-3"]["pricing_per_1k"] == DEFAULT_PRICING["grok-3"]


class TestEstimateCost:
    def test_estimate_cost_cloud(self):
        cost = estimate_cost("xai", "grok-3", 1000, 500)
        expected = (1000/1000)*0.30 + (500/1000)*0.60
        assert cost == expected
    
    def test_estimate_cost_local_zero(self):
        cost = estimate_cost("localai", "gemma-4", 1000, 500)
        assert cost == 0.0
    
    def test_estimate_cost_unknown_model(self):
        cost = estimate_cost("xai", "unknown", 1000, 500)
        assert cost == 0.0


class TestDefaultPricing:
    def test_grok3_pricing(self):
        assert "grok-3" in DEFAULT_PRICING
        assert DEFAULT_PRICING["grok-3"] == (0.30, 0.60)
    
    def test_gpt4o_pricing(self):
        assert "gpt-4o" in DEFAULT_PRICING
        assert DEFAULT_PRICING["gpt-4o"] == (2.50, 10.00)
    
    def test_claude_pricing(self):
        assert "claude-3-5-sonnet" in DEFAULT_PRICING
        assert DEFAULT_PRICING["claude-3-5-sonnet"] == (3.00, 15.00)
    
    def test_local_models_zero(self):
        assert "gemma-4-e2b-it:latest" in DEFAULT_PRICING
        assert DEFAULT_PRICING["gemma-4-e2b-it:latest"] == (0.0, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])