"""Fusion Engine - Main orchestration layer for multi-LLM fusion."""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from hermes_fusion.config import FusionConfig
from hermes_fusion.providers.base import Provider, ProviderResponse
from hermes_fusion.providers.qdrant import QdrantProvider
from hermes_fusion.strategies import FusionResult, FusionStrategy, get_strategy

# Observability
try:
    from hermes_fusion.observability import (
        create_query_context,
        get_logger,
        get_metrics_collector,
        log_cache_event,
        log_provider_result,
        log_query_end,
        log_query_start,
        trace_cache_operation,
        trace_provider_call,
        trace_query,
    )
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False
    # No-op stubs
    def get_logger(name=None):
        import logging
        return logging.getLogger(name)

    def get_metrics_collector():
        return None

    @asynccontextmanager
    async def trace_query(*args, **kwargs):
        yield None

    @asynccontextmanager
    async def trace_provider_call(*args, **kwargs):
        yield None

    @asynccontextmanager
    async def trace_cache_operation(*args, **kwargs):
        yield None

    def create_query_context(*args, **kwargs):
        return None

    def log_query_start(*args, **kwargs):
        pass

    def log_query_end(*args, **kwargs):
        pass

    def log_provider_result(*args, **kwargs):
        pass

    def log_cache_event(*args, **kwargs):
        pass


@dataclass
class EngineMetrics:
    """Runtime metrics for monitoring."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_errors: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


class CircuitBreaker:
    """Circuit breaker for provider resilience (Hystrix-style)."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> str:
        if self._state == "open":
            if self._last_failure_time and \
               time.perf_counter() - self._last_failure_time >= self.recovery_timeout:
                return "half-open"
        return self._state
    
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        async with self._lock:
            if self.state == "open":
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._on_success()
            return result
        except self.expected_exception:
            async with self._lock:
                self._on_failure()
            raise
    
    def _on_success(self):
        self._failure_count = 0
        self._state = "closed"
    
    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.perf_counter()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
    
    def reset(self):
        self._failure_count = 0
        self._state = "closed"
        self._last_failure_time = None


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

        # Observability
        self._logger = get_logger("hermes_fusion.engine")
        self._metrics = get_metrics_collector()
        self._active_queries = 0

        # Circuit Breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._init_circuit_breakers()

    def _init_circuit_breakers(self):
        """Initialize circuit breakers for each provider."""
        # Handle both Pydantic models (model_dump) and dataclasses (asdict)
        try:
            fusion_dict = self.config.fusion.model_dump()
        except AttributeError:
            from dataclasses import asdict
            fusion_dict = asdict(self.config.fusion)
        
        cb_config = fusion_dict.get("circuit_breaker", {})
        threshold = cb_config.get("failure_threshold", 5)
        timeout = cb_config.get("recovery_timeout", 30.0)
        
        for name in self.providers:
            self._circuit_breakers[name] = CircuitBreaker(
                failure_threshold=threshold,
                recovery_timeout=timeout,
            )

    def get_circuit_breaker(self, name: str) -> CircuitBreaker | None:
        return self._circuit_breakers.get(name)

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
        query_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        self.metrics.total_queries += 1
        self._active_queries += 1
        if self._metrics:
            self._metrics.set_active_queries(self._active_queries)

        # Get strategy
        fusion_strategy = self._get_strategy(strategy)
        strategy_name = fusion_strategy.name

        # Observability context
        healthy_providers = sum(
            1 for n in self.providers
            if self._provider_health.get(n) is not False
        )
        ctx = create_query_context(
            query_id=query_id,
            question=question,
            strategy=strategy_name,
            provider_count=healthy_providers,
        )
        log_query_start(self._logger, ctx)

        success = False
        cached = False
        result = None

        try:
            async with trace_query(question, strategy_name, len(self.providers)):
                # 1. Check semantic cache
                if self.config.fusion.semantic_cache_enabled and self.qdrant:
                    async with trace_cache_operation("get", "semantic"):
                        try:
                            cached_result = await self.qdrant.get_similar(
                                question,
                                threshold=0.92,
                            )
                            if cached_result:
                                self.metrics.cache_hits += 1
                                if self._metrics:
                                    self._metrics.record_cache_hit("semantic")
                                cached_result["_cached"] = True
                                cached_result["metadata"] = cached_result.get("metadata", {})
                                cached_result["metadata"]["cached"] = True
                                result = self._dict_to_result(cached_result, strategy_name)
                                success = True
                                cached = True
                                log_cache_event(self._logger, "hit", "semantic", hit=True)
                            else:
                                self.metrics.cache_misses += 1
                                if self._metrics:
                                    self._metrics.record_cache_miss("semantic")
                                log_cache_event(self._logger, "miss", "semantic", hit=False)
                        except Exception:
                            self.metrics.cache_misses += 1
                            if self._metrics:
                                self._metrics.record_cache_miss("semantic")
                            log_cache_event(self._logger, "error", "semantic")
                            # Cache miss on error - continue to providers

                if not cached:
                    self.metrics.cache_misses += 1

                    # 2. Query providers in parallel
                    provider_tasks = self._create_provider_tasks(question, model, **kwargs)
                    responses = await self._execute_with_timeout(provider_tasks)

                    # Filter successful responses
                    valid_responses = [r for r in responses if r and r.content]

                    if not valid_responses:
                        # All providers failed
                        if self._metrics:
                            self._metrics.record_error("all_providers_failed")
                        result = FusionResult(
                            final_answer="",
                            confidence=0.0,
                            method=fusion_strategy.name,
                            participating_providers=[],
                            raw_responses=responses,
                            metadata={"error": "All providers failed"},
                        )
                    else:
                        # 3. Apply fusion strategy
                        result = await fusion_strategy.fuse(question, valid_responses)
                        success = True

                        # Record fusion metrics
                        if self._metrics:
                            self._metrics.record_fusion(strategy_name, result.confidence)

                        # 4. Store in semantic cache
                        if self.config.fusion.semantic_cache_enabled and self.qdrant and valid_responses:
                            async with trace_cache_operation("set", "semantic"):
                                try:
                                    await self.qdrant.store(question, self._result_to_dict(result))
                                except Exception:
                                    pass  # Fail silently on cache write

        except Exception as e:
            if self._metrics:
                self._metrics.record_error(type(e).__name__)
            self._logger.exception("fusion.query.error", query_id=query_id, error=str(e))
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._update_metrics(latency_ms)
            self._active_queries -= 1
            if self._metrics:
                self._metrics.set_active_queries(self._active_queries)
                duration = time.perf_counter() - start_time
                self._metrics.record_query(strategy_name, success, duration)

            log_query_end(self._logger, ctx, result, time.perf_counter() - start_time, cached)

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
            
            # Skip if circuit breaker is open
            cb = self._circuit_breakers.get(name)
            if cb and cb.state == "open":
                continue

            async def call_provider(p=provider, n=name):
                async with trace_provider_call(n, model) as span:
                    try:
                        messages = [{"role": "user", "content": question}]
                        start = time.perf_counter()
                        
                        # Use circuit breaker if available
                        if cb:
                            response = await cb.call(p.chat, messages, model or "default", **kwargs)
                        else:
                            response = await p.chat(messages, model or "default", **kwargs)
                        
                        duration = time.perf_counter() - start

                        if self._metrics:
                            self._metrics.record_provider_request(n, True, duration)
                        log_provider_result(self._logger, n, True, duration)

                        if span:
                            span.set_attribute("hermes.response_length", len(response.content) if response and response.content else 0)

                        return response
                    except Exception as e:
                        duration = time.perf_counter() - start if 'start' in locals() else 0
                        self.metrics.provider_errors[n] = self.metrics.provider_errors.get(n, 0) + 1
                        self._provider_health[n] = False
                        if self._metrics:
                            self._metrics.record_provider_request(n, False, duration, type(e).__name__)
                        log_provider_result(self._logger, n, False, duration, str(e))
                        if span:
                            span.record_exception(e)
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
            except TimeoutError:
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
                if self._metrics:
                    self._metrics.set_provider_health(name, healthy)
            except Exception:
                health[name] = False
                self._provider_health[name] = False
                if self._metrics:
                    self._metrics.set_provider_health(name, False)

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

    def get_prometheus_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        if _OBSERVABILITY_AVAILABLE:
            from hermes_fusion.observability import generate_metrics
            return generate_metrics()
        return b""

    def get_metrics_content_type(self) -> str:
        """Get the Prometheus metrics content type."""
        if _OBSERVABILITY_AVAILABLE:
            from hermes_fusion.observability import get_metrics_content_type
            return get_metrics_content_type()
        return "text/plain"

    async def query_stream(
        self,
        question: str,
        strategy: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream fusion query response token-by-token.
        Uses first provider that supports streaming, then applies strategy on complete responses.
        """
        query_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        fusion_strategy = self._get_strategy(strategy)
        strategy_name = fusion_strategy.name

        ctx = create_query_context(
            query_id=query_id,
            question=question,
            strategy=strategy_name,
            provider_count=len(self.providers),
        )
        log_query_start(self._logger, ctx)

        try:
            async with trace_query(question, strategy_name, len(self.providers)):
                # Check cache first
                if self.config.fusion.semantic_cache_enabled and self.qdrant:
                    cached = await self.qdrant.get_similar(question, threshold=0.92)
                    if cached:
                        yield cached.get("final_answer", "")
                        log_cache_event(self._logger, "hit", "semantic", hit=True)
                        return

                # Get streaming provider (first one that supports it)
                streaming_provider = None
                for name, provider in self.providers.items():
                    if hasattr(provider, 'chat_stream') and callable(provider.chat_stream):
                        streaming_provider = (name, provider)
                        break

                if streaming_provider:
                    name, provider = streaming_provider
                    messages = [{"role": "user", "content": question}]

                    # Stream from first available provider
                    async with trace_provider_call(name, model) as span:
                        try:
                            async for chunk in provider.chat_stream(messages, model or "default", **kwargs):
                                yield chunk
                            log_provider_result(self._logger, name, True, time.perf_counter() - start_time)
                        except Exception as e:
                            log_provider_result(self._logger, name, False, time.perf_counter() - start_time, str(e))
                            # Fall back to non-streaming
                            pass
                else:
                    # No streaming provider - collect all then yield complete answer
                    result = await self.query(question, strategy, model, **kwargs)
                    yield result.final_answer

        except Exception as e:
            self._logger.exception("fusion.stream.error", query_id=query_id, error=str(e))
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._update_metrics(latency_ms)
            log_query_end(self._logger, ctx, None, time.perf_counter() - start_time, False)


# Factory function for easy setup
async def create_engine(config: FusionConfig) -> FusionEngine:
    """Create and initialize FusionEngine from config."""
    engine = FusionEngine(config=config)

    # Initialize providers from config (lazy - actual clients created on first use)
    # This is a placeholder - real implementation would create provider instances
    # based on config.providers

    return engine