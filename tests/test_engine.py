"""Tests for Fusion Engine - main orchestration layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from hermes_fusion.engine import FusionEngine
from hermes_fusion.config import FusionConfig, FusionSettings, ProvidersConfig, LocalAIConfig
from hermes_fusion.providers.base import ProviderResponse
from hermes_fusion.providers.localai import LocalAIProvider
from hermes_fusion.providers.cloud import XAIProvider, OpenAIProvider
from hermes_fusion.providers.qdrant import QdrantProvider
from hermes_fusion.strategies import get_strategy


class MockProvider:
    """Mock provider for testing."""
    def __init__(self, name: str, response_content: str = "test answer", healthy: bool = True):
        self.name = name
        self._response_content = response_content
        self._healthy = healthy
        self.chat = AsyncMock(side_effect=self._chat_impl)
        self.health_check = AsyncMock(side_effect=self._health_impl)

    async def _chat_impl(self, messages, model, **kwargs):
        return ProviderResponse(
            content=self._response_content,
            provider=self.name,
            model=model,
            tokens_used=50,
        )

    async def _health_impl(self):
        return self._healthy

    async def embed(self, texts, model):
        return [[0.1] * 768] * len(texts)


@pytest.fixture
def mock_localai():
    return MockProvider("localai", "local answer", True)


@pytest.fixture
def mock_xai():
    return MockProvider("xai", "cloud answer", True)


@pytest.fixture
def mock_qdrant():
    qdrant = MagicMock()
    qdrant.health_check = AsyncMock(return_value=True)
    qdrant.get_similar = AsyncMock(return_value=None)  # Cache miss
    qdrant.store = AsyncMock()
    qdrant.cleanup_ttl = AsyncMock(return_value=0)
    return qdrant


@pytest.fixture
def fusion_config():
    return FusionConfig(
        fusion=FusionSettings(
            default_strategy="weighted_vote",
            max_parallel_providers=3,
            timeout_seconds=30,
            semantic_cache_enabled=True,
            cache_ttl_hours=24,
        ),
        providers=ProvidersConfig(
            localai=LocalAIConfig(base_url="http://test:8080"),
        ),
    )


class TestFusionEngine:
    @pytest.mark.asyncio
    async def test_engine_initialization(self, fusion_config, mock_localai, mock_xai, mock_qdrant):
        engine = FusionEngine(
            config=fusion_config,
            providers={"localai": mock_localai, "xai": mock_xai},
            qdrant=mock_qdrant,
        )
        
        assert engine.config == fusion_config
        assert "localai" in engine.providers
        assert "xai" in engine.providers
        assert engine.qdrant == mock_qdrant

    @pytest.mark.asyncio
    async def test_engine_query_simple(self, fusion_config, mock_localai, mock_xai, mock_qdrant):
        engine = FusionEngine(
            config=fusion_config,
            providers={"localai": mock_localai, "xai": mock_xai},
            qdrant=mock_qdrant,
        )
        
        result = await engine.query("What is 2+2?", strategy="weighted_vote")
        
        assert result.final_answer in ["local answer", "cloud answer"]
        assert result.method == "weighted_vote"
        assert "localai" in result.participating_providers
        assert "xai" in result.participating_providers
        mock_localai.chat.assert_called()
        mock_xai.chat.assert_called()

    @pytest.mark.asyncio
    async def test_engine_query_with_cache_hit(self, fusion_config, mock_localai, mock_xai, mock_qdrant):
        # Setup cache hit
        mock_qdrant.get_similar.return_value = {
            "final_answer": "cached answer",
            "confidence": 0.95,
            "method": "weighted_vote",
            "participating_providers": ["localai"],
        }
        
        engine = FusionEngine(
            config=fusion_config,
            providers={"localai": mock_localai, "xai": mock_xai},
            qdrant=mock_qdrant,
        )
        
        result = await engine.query("What is 2+2?", strategy="weighted_vote")
        
        assert result.final_answer == "cached answer"
        assert result.metadata.get("cached") is True
        # Providers should NOT be called on cache hit
        mock_localai.chat.assert_not_called()
        mock_xai.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_engine_query_cache_miss_stores_result(self, fusion_config, mock_localai, mock_xai, mock_qdrant):
        engine = FusionEngine(
            config=fusion_config,
            providers={"localai": mock_localai, "xai": mock_xai},
            qdrant=mock_qdrant,
        )
        
        result = await engine.query("What is 2+2?", strategy="weighted_vote")
        
        # Should store in cache
        mock_qdrant.store.assert_called_once()
        call_args = mock_qdrant.store.call_args
        assert call_args[0][0] == "What is 2+2?"  # query
        assert "final_answer" in call_args[0][1]  # result dict

    @pytest.mark.asyncio
    async def test_engine_parallel_provider_calls(self, fusion_config, mock_qdrant):
        # Track call order
        call_times = []
        
        async def slow_chat(messages, model, **kwargs):
            import time
            call_times.append(time.time())
            await asyncio.sleep(0.01)
            return ProviderResponse(content="answer", provider="slow", model=model, tokens_used=10)
        
        import asyncio
        
        provider1 = MockProvider("p1", "answer 1")
        provider1.chat = slow_chat
        provider2 = MockProvider("p2", "answer 2")
        provider2.chat = slow_chat
        provider3 = MockProvider("p3", "answer 3")
        provider3.chat = slow_chat
        
        engine = FusionEngine(
            config=fusion_config,
            providers={"p1": provider1, "p2": provider2, "p3": provider3},
            qdrant=mock_qdrant,
        )
        
        await engine.query("Test", strategy="weighted_vote")
        
        # All should be called in parallel (within ~0.01s not 0.03s sequential)
        # This is a weak test but checks parallelism conceptually
        assert len(call_times) == 3

    @pytest.mark.asyncio
    async def test_engine_timeout_handling(self, fusion_config, mock_qdrant):
        async def timeout_chat(messages, model, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return ProviderResponse(content="late", provider="slow", model=model, tokens_used=10)
        
        import asyncio
        
        slow_provider = MockProvider("slow", "late")
        slow_provider.chat = timeout_chat
        
        # Short timeout
        config = FusionConfig(
            fusion=FusionSettings(
                default_strategy="weighted_vote",
                max_parallel_providers=3,
                timeout_seconds=1,  # 1 second
                semantic_cache_enabled=True,
                cache_ttl_hours=24,
            ),
            providers=ProvidersConfig(localai=LocalAIConfig()),
        )
        
        engine = FusionEngine(
            config=config,
            providers={"slow": slow_provider},
            qdrant=mock_qdrant,
        )
        
        result = await engine.query("Test", strategy="weighted_vote")
        
        # Should handle timeout gracefully (may return partial or empty)
        assert result is not None

    @pytest.mark.asyncio
    async def test_engine_different_strategies(self, fusion_config, mock_localai, mock_xai, mock_qdrant):
        engine = FusionEngine(
            config=fusion_config,
            providers={"localai": mock_localai, "xai": mock_xai},
            qdrant=mock_qdrant,
        )
        
        for strategy_name in ["weighted_vote", "handoff", "cot_consensus", "best_of_n"]:
            result = await engine.query("Test", strategy=strategy_name)
            assert result.method in [strategy_name, f"{strategy_name}_fallback"]

    @pytest.mark.asyncio
    async def test_engine_health_check(self, fusion_config, mock_localai, mock_xai, mock_qdrant):
        mock_localai._healthy = True
        mock_xai._healthy = False
        
        engine = FusionEngine(
            config=fusion_config,
            providers={"localai": mock_localai, "xai": mock_xai},
            qdrant=mock_qdrant,
        )
        
        health = await engine.health_check()
        
        assert health["localai"] is True
        assert health["xai"] is False
        assert health["qdrant"] is True

    @pytest.mark.asyncio
    async def test_engine_provider_failure_resilience(self, fusion_config, mock_qdrant):
        failing_provider = MockProvider("fail", "answer")
        failing_provider.chat = AsyncMock(side_effect=Exception("API error"))
        failing_provider.health_check = AsyncMock(return_value=False)
        
        working_provider = MockProvider("work", "working answer")
        
        engine = FusionEngine(
            config=fusion_config,
            providers={"fail": failing_provider, "work": working_provider},
            qdrant=mock_qdrant,
        )
        
        result = await engine.query("Test", strategy="weighted_vote")
        
        # Should succeed with working provider
        assert result.final_answer == "working answer"
        assert "work" in result.participating_providers


class TestFusionEngineFactory:
    @pytest.mark.asyncio
    async def test_create_from_config(self, fusion_config, mock_qdrant):
        # This tests the factory pattern if we add it
        from hermes_fusion.engine import create_engine
        
        # Would need actual provider implementations
        # Skipping for now - requires real clients
        pass