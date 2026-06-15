"""Fusion Engine - Main orchestration layer for multi-LLM fusion."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from hermes_fusion.config import FusionConfig, FusionSettings
from hermes_fusion.providers.base import Provider, ProviderResponse
from hermes_fusion.providers.qdrant import QdrantProvider
from hermes_fusion.strategies import FusionStrategy, FusionResult, get_strategy


@dataclass
class EngineMetrics:
    """Runtime metrics for monitoring."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_errors: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


class FusionEngine:
    """
    Main fusion engine orchestrating:
    - Multiple LLM providers (local + cloud)
    - Semantic cache (Qdrant)
    - Fusion strategies (weighted_vote, handoff, etc.)
    - Health monitoring and metrics
    """

    def __init__(
        self,
        config: FusionConfig,
        providers: dict[str, Provider] | None = None,
        qdrant: QdrantProvider | None = None,
        strategy: FusionStrategy | None = None,
    ):
        self.config = config
        self.providers = providers or {}
        self.qdrant = qdrant
        self._strategy = strategy
        self.metrics = EngineMetrics()
        self._provider_health: dict[str, bool] = {}

    def add_provider(self, name: str, provider: Provider) -> None:
        """Register a provider."""
        self.providers[name] = provider

    def set_strategy(self, strategy: FusionStrategy) -> None:
        """Set the default fusion strategy."""
        self._strategy = strategy

    def _get_strategy(self, strategy_name: str | None) -> FusionStrategy:
        """Get strategy instance by name or default."""
        if strategy_name:
            return get_strategy(strategy_name)
        if self._strategy:
            return self._strategy
        return get_strategy(self.config.fusion.default_strategy)

    async def query(
        self,
        question: str,
        strategy: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> FusionResult:
        """
        Execute a fusion query:
        1. Check semantic cache
        2. Query providers in parallel
        3. Apply fusion strategy
        4. Store result in cache
        """
        start_time = time.perf_counter()
        self.metrics.total_queries += 1
        
        # Get strategy
        fusion_strategy = self._get_strategy(strategy)
        
        # 1. Check semantic cache
        if self.config.fusion.semantic_cache_enabled and self.qdrant:
            try:
                cached = await self.qdrant.get_similar(
                    question, 
                    threshold=0.92,
                )
                if cached:
                    self.metrics.cache_hits += 1
                    cached["_cached"] = True
                    cached["metadata"] = cached.get("metadata", {})
                    cached["metadata"]["cached"] = True
                    return self._dict_to_result(cached, fusion_strategy.name)
            except Exception:
                pass  # Cache miss on error
        
        self.metrics.cache_misses += 1
        
        # 2. Query providers in parallel
        provider_tasks = self._create_provider_tasks(question, model, **kwargs)
        responses = await self._execute_with_timeout(provider_tasks)
        
        # Filter successful responses
        valid_responses = [r for r in responses if r and r.content]
        
        if not valid_responses:
            # All providers failed
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=fusion_strategy.name,
                participating_providers=[],
                raw_responses=responses,
                metadata={"error": "All providers failed"},
            )
        
        # 3. Apply fusion strategy
        result = await fusion_strategy.fuse(question, valid_responses)
        
        # 4. Store in semantic cache
        if self.config.fusion.semantic_cache_enabled and self.qdrant and valid_responses:
            try:
                await self.qdrant.store(question, self._result_to_dict(result))
            except Exception:
                pass  # Fail silently on cache write
        
        # Update metrics
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        self._update_metrics(latency_ms)
        
        return result

    def _create_provider_tasks(
        self, 
        question: str, 
        model: str | None, 
        **kwargs
    ) -> list[Callable[[], Awaitable[ProviderResponse | None]]]:
        """Create async tasks for each healthy provider."""
        tasks = []
        
        for name, provider in self.providers.items():
            # Skip unhealthy providers
            if self._provider_health.get(name) is False:
                continue
                
            async def call_provider(p=provider, n=name):
                try:
                    messages = [{"role": "user", "content": question}]
                    return await p.chat(messages, model or "default", **kwargs)
                except Exception as e:
                    self.metrics.provider_errors[n] = self.metrics.provider_errors.get(n, 0) + 1
                    self._provider_health[n] = False
                    return None
            
            tasks.append(call_provider)
        
        return tasks

    async def _execute_with_timeout(
        self, 
        tasks: list[Callable[[], Awaitable[ProviderResponse | None]]]
    ) -> list[ProviderResponse | None]:
        """Execute provider tasks with global timeout."""
        if not tasks:
            return []
        
        # Run all in parallel with timeout
        async def run_with_timeout(task):
            try:
                return await asyncio.wait_for(
                    task(), 
                    timeout=self.config.fusion.timeout_seconds
                )
            except asyncio.TimeoutError:
                return None
            except Exception:
                return None
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.config.fusion.max_parallel_providers)
        
        async def limited(task):
            async with semaphore:
                return await run_with_timeout(task)
        
        return await asyncio.gather(*[limited(t) for t in tasks], return_exceptions=False)

    def _result_to_dict(self, result: FusionResult) -> dict[str, Any]:
        """Convert FusionResult to dict for caching."""
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "method": result.method,
            "participating_providers": result.participating_providers,
            "metadata": result.metadata,
        }

    def _dict_to_result(self, data: dict[str, Any], method: str) -> FusionResult:
        """Convert cached dict to FusionResult."""
        return FusionResult(
            final_answer=data.get("final_answer", ""),
            confidence=data.get("confidence", 0.0),
            method=data.get("method", method),
            participating_providers=data.get("participating_providers", []),
            metadata=data.get("metadata", {}),
        )

    def _update_metrics(self, latency_ms: int) -> None:
        """Update rolling average latency."""
        n = self.metrics.total_queries
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (n - 1) + latency_ms) / n
        )

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers and cache."""
        health = {}
        
        # Check providers
        for name, provider in self.providers.items():
            try:
                healthy = await provider.health_check()
                health[name] = healthy
                self._provider_health[name] = healthy
            except Exception:
                health[name] = False
                self._provider_health[name] = False
        
        # Check Qdrant
        if self.qdrant:
            try:
                health["qdrant"] = await self.qdrant.health_check()
            except Exception:
                health["qdrant"] = False
        
        return health

    async def cleanup_cache(self, hours: int | None = None) -> int:
        """Clean up old cache entries."""
        if not self.qdrant:
            return 0
        ttl = hours or self.config.fusion.cache_ttl_hours
        return await self.qdrant.cleanup_ttl(ttl)

    def get_metrics(self) -> EngineMetrics:
        """Get current engine metrics."""
        return self.metrics

    def get_available_strategies(self) -> list[str]:
        """List available fusion strategies."""
        from hermes_fusion.strategies import list_strategies
        return list_strategies()


# Factory function for easy setup
async def create_engine(config: FusionConfig) -> FusionEngine:
    """Create and initialize FusionEngine from config."""
    engine = FusionEngine(config=config)
    
    # Initialize providers from config (lazy - actual clients created on first use)
    # This is a placeholder - real implementation would create provider instances
    # based on config.providers
    
    return engine