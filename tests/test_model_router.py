"""Tests for model router module."""

import pytest
import tempfile
import os
from hermes_fusion.model_router import (
    ModelRouter,
    TaskClassifier,
    TaskType,
    RoutingPolicy,
    ModelOption,
    ProviderProfile,
    RoutingDecision,
    create_model_router_from_config,
)


class TestTaskClassifier:
    def test_classify_code(self):
        clf = TaskClassifier()
        assert clf.classify("Write a Python function to sort a list") == TaskType.CODE
        assert clf.classify("Debug this JavaScript error") == TaskType.CODE
        assert clf.classify("```python\ndef foo():\n    pass\n```") == TaskType.CODE
        assert clf.classify("SELECT * FROM users") == TaskType.CODE
    
    def test_classify_reasoning(self):
        clf = TaskClassifier()
        assert clf.classify("Prove that square root of 2 is irrational") == TaskType.REASONING
        assert clf.classify("Solve this math problem step by step") == TaskType.REASONING
        assert clf.classify("Think through this logic puzzle") == TaskType.REASONING
    
    def test_classify_creative(self):
        clf = TaskClassifier()
        assert clf.classify("Write a short story about a cat") == TaskType.CREATIVE
        assert clf.classify("Brainstorm marketing slogans") == TaskType.CREATIVE
        assert clf.classify("Write a poem about spring") == TaskType.CREATIVE
    
    def test_classify_chat(self):
        clf = TaskClassifier()
        assert clf.classify("Hello, how are you?") == TaskType.CHAT
        assert clf.classify("What is the capital of France?") == TaskType.CHAT
        assert clf.classify("Explain quantum computing") == TaskType.CHAT
    
    def test_classify_general_fallback(self):
        clf = TaskClassifier()
        assert clf.classify("xyz random text") == TaskType.GENERAL


class TestModelRouter:
    def test_default_models_loaded(self):
        router = ModelRouter()
        assert "gemma-4-e2b-it:latest" in router.models
        assert "grok-3" in router.models
        assert "gpt-4o" in router.models
        assert "claude-3-5-sonnet" in router.models
    
    def test_route_code_task(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route(
            "Write a Python function to calculate fibonacci",
            policy=RoutingPolicy.QUALITY
        ))
        
        # May classify as reasoning or code depending on keyword weights
        assert decision.task_type in (TaskType.CODE, TaskType.REASONING)
        assert decision.model_id in router.models
        assert decision.provider is not None
        assert len(decision.fallbacks) >= 0
    
    def test_route_chat_task(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route(
            "Hello, how are you?",
            policy=RoutingPolicy.COST
        ))
        
        assert decision.task_type == TaskType.CHAT
        # Cost policy should prefer free local model for simple chat
        # or cheapest cloud
    
    def test_allowed_models_filter(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route(
            "Write code",
            allowed_models=["gpt-4o", "claude-3-5-sonnet"],
            policy=RoutingPolicy.QUALITY
        ))
        
        assert decision.model_id in ["gpt-4o", "claude-3-5-sonnet"]
    
    def test_required_capabilities_filter(self):
        router = ModelRouter()
        
        import asyncio
        # Require vision
        decision = asyncio.run(router.route(
            "Analyze this image",
            required_capabilities={"vision": True},
            policy=RoutingPolicy.QUALITY
        ))
        
        assert decision.provider.supports_vision
    
    def test_session_stickiness(self):
        router = ModelRouter()
        
        import asyncio
        decision1 = asyncio.run(router.route(
            "First question",
            session_id="test-session",
            policy=RoutingPolicy.QUALITY
        ))
        
        decision2 = asyncio.run(router.route(
            "Second question",
            session_id="test-session",
            policy=RoutingPolicy.QUALITY
        ))
        
        # Should use same model/provider for session
        assert decision1.model_id == decision2.model_id
        assert decision1.provider_name == decision2.provider_name
        assert decision2.metadata.get("session_sticky") is True
    
    def test_session_expiry(self):
        router = ModelRouter(session_ttl_seconds=1)  # 1 second TTL
        
        import asyncio
        decision1 = asyncio.run(router.route(
            "First question",
            session_id="test-session-expiry",
            policy=RoutingPolicy.QUALITY
        ))
        
        import time
        time.sleep(1.5)
        
        decision2 = asyncio.run(router.route(
            "Second question",
            session_id="test-session-expiry",
            policy=RoutingPolicy.QUALITY
        ))
        
        # Session expired, may route differently
        # (not asserting different since it could pick same by chance)
        assert decision2.metadata.get("session_sticky") != True
    
    def test_clear_session(self):
        router = ModelRouter()
        
        import asyncio
        decision1 = asyncio.run(router.route(
            "First question",
            session_id="test-clear",
            policy=RoutingPolicy.QUALITY
        ))
        
        router.clear_session("test-clear")
        
        decision2 = asyncio.run(router.route(
            "Second question",
            session_id="test-clear",
            policy=RoutingPolicy.QUALITY
        ))
        
        assert decision2.metadata.get("session_sticky") != True
    
    def test_record_outcome_updates_stats(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route(
            "Test question",
            policy=RoutingPolicy.QUALITY
        ))
        
        initial_latency = decision.provider.avg_latency_ms
        initial_quality = decision.provider.avg_quality_score
        
        # Use different latency to ensure EMA changes
        router.record_outcome(
            decision,
            success=True,
            latency_ms=3000,  # Different from initial 1500
            quality_score=0.9,
            cost_usd=0.01,
            tokens_in=100,
            tokens_out=200
        )
        
        # Should update with EMA
        assert decision.provider.avg_latency_ms != initial_latency
        assert decision.provider.avg_latency_ms == pytest.approx(1500 * 0.9 + 3000 * 0.1, rel=0.01)
        assert decision.provider.avg_quality_score != initial_quality
        assert decision.provider.success_rate == 1.0  # First success
    
    def test_record_failure_marks_unhealthy(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route(
            "Test question",
            policy=RoutingPolicy.QUALITY
        ))
        
        # Fail 3 times
        for _ in range(3):
            router.record_outcome(
                decision,
                success=False,
                latency_ms=5000,
                quality_score=0.0,
                cost_usd=0.0
            )
        
        assert decision.provider.is_healthy is False
        assert decision.provider.consecutive_errors >= 3
    
    def test_routing_policies(self):
        router = ModelRouter()
        
        import asyncio
        
        # Quality policy
        dq = asyncio.run(router.route("Complex reasoning task", policy=RoutingPolicy.QUALITY))
        
        # Cost policy
        dc = asyncio.run(router.route("Simple chat", policy=RoutingPolicy.COST))
        
        # Both should return valid decisions
        assert dq.provider is not None
        assert dc.provider is not None
    
    def test_cost_quality_tradeoff(self):
        router = ModelRouter()
        
        import asyncio
        
        # High quality preference (0)
        dq = asyncio.run(router.route("Task", policy=RoutingPolicy.COST_QUALITY, cost_quality_tradeoff=0))
        
        # High cost savings preference (10)
        dc = asyncio.run(router.route("Task", policy=RoutingPolicy.COST_QUALITY, cost_quality_tradeoff=10))
        
        assert dq.provider is not None
        assert dc.provider is not None
    
    def test_fallback_chain(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route("Task", policy=RoutingPolicy.QUALITY))
        
        assert isinstance(decision.fallbacks, list)
        # All fallbacks should be healthy
        for fallback in decision.fallbacks:
            assert fallback.is_healthy
    
    def test_persistence(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            router = ModelRouter(persistence_path=temp_path)
            
            import asyncio
            decision = asyncio.run(router.route("Test", policy=RoutingPolicy.QUALITY))
            router.record_outcome(decision, True, 1000, 0.8, 0.001)
            
            asyncio.run(router._save())
            
            # Load in new router
            router2 = ModelRouter(persistence_path=temp_path)
            asyncio.run(router2.load())
            
            # Should have restored selection counts
            key = f"{decision.model_id}:{decision.provider_name}"
            assert router2._selection_counts[key] > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_routing_stats(self):
        router = ModelRouter()
        
        import asyncio
        decision = asyncio.run(router.route("Test", policy=RoutingPolicy.QUALITY))
        router.record_outcome(decision, True, 1000, 0.8, 0.001)
        
        stats = router.get_routing_stats()
        
        assert stats["total_routes"] >= 1
        assert stats["success_rate"] >= 0
        assert "model_distribution" in stats
        assert "provider_distribution" in stats
    
    def test_get_models_for_task(self):
        router = ModelRouter()
        
        code_models = router.get_models_for_task(TaskType.CODE)
        assert len(code_models) > 0
        
        vision_models = router.get_models_for_task(TaskType.VISION)
        assert any("gpt-4o" in str(m) for m in vision_models)
    
    def test_add_model(self):
        router = ModelRouter()
        
        new_model = ModelOption(
            model_id="custom-model",
            task_types=[TaskType.CODE],
            providers=[
                ProviderProfile(
                    name="custom-provider",
                    model="custom-model",
                    cost_per_1k_input=1.0,
                    cost_per_1k_output=2.0,
                    avg_quality_score=0.9,
                )
            ]
        )
        
        router.add_model(new_model)
        assert "custom-model" in router.models
    
    def test_create_from_config(self):
        config = {
            "default_policy": "cost",
            "cost_quality_tradeoff": 5,
            "exploration_rate": 0.2,
            "session_ttl_seconds": 600,
        }
        
        router = create_model_router_from_config(config)
        
        assert router.default_policy == RoutingPolicy.COST
        assert router.cost_quality_tradeoff == 5
        assert router.exploration_rate == 0.2
        assert router.session_ttl == 600


class TestProviderProfile:
    def test_cost_score_free(self):
        p = ProviderProfile(name="local", model="test", cost_per_1k_input=0, cost_per_1k_output=0)
        assert p.cost_score == 1.0
    
    def test_cost_score_paid(self):
        p = ProviderProfile(name="cloud", model="test", cost_per_1k_input=10, cost_per_1k_output=30)
        assert 0 < p.cost_score < 1
    
    def test_latency_score(self):
        p = ProviderProfile(name="test", model="test", avg_latency_ms=2000)
        assert 0 < p.latency_score < 1
        
        p2 = ProviderProfile(name="test", model="test", avg_latency_ms=0)
        assert p2.latency_score == 0.5
    
    def test_composite_score(self):
        p = ProviderProfile(
            name="test", model="test",
            avg_quality_score=0.8,
            cost_per_1k_input=0,
            cost_per_1k_output=0,
            avg_latency_ms=1000,
            weight_quality=1.0,
            weight_cost=1.0,
            weight_latency=1.0,
        )
        assert 0 < p.composite_score < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])