"""Fusion Engine - Main orchestration layer for multi-LLM fusion."""

import asyncio
import os
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
from hermes_fusion.model_router import ModelRouter, RoutingDecision, TaskType, RoutingPolicy

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

# Cost Tracker
try:
    from hermes_fusion.cost_tracker import (
        Budget,
        CostMetrics,
        CostTracker,
        TokenUsage,
        create_cost_tracker_from_config,
    )
    _COST_TRACKER_AVAILABLE = True
except ImportError:
    _COST_TRACKER_AVAILABLE = False
    CostTracker = None
    TokenUsage = None
    Budget = None
    CostMetrics = None
    def create_cost_tracker_from_config(config):
        return None


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

        # Cost Tracker
        self._cost_tracker: CostTracker | None = None
        self._init_cost_tracker()

        # Model Router
        self._model_router: ModelRouter | None = None
        self._init_model_router()

    def _init_cost_tracker(self):
        """Initialize cost tracker from config."""
        if not _COST_TRACKER_AVAILABLE:
            return
        ct_config = self.config.fusion.cost_tracker
        if not ct_config.enabled:
            return
        
        # Build pricing from config + defaults
        custom_pricing = {}
        for model, costs in ct_config.custom_pricing.items():
            custom_pricing[model] = (costs[0], costs[1])
        
        # Also get pricing from cloud provider configs
        for provider_name in ("xai", "openai", "anthropic"):
            provider = getattr(self.config.providers.cloud, provider_name, None)
            if provider and provider.model:
                custom_pricing[provider.model] = (
                    provider.input_cost_per_1k,
                    provider.output_cost_per_1k,
                )
        
        for name, cp in self.config.providers.cloud.custom.items():
            custom_pricing[cp.model] = (cp.input_cost_per_1k, cp.output_cost_per_1k)
        
        # Build budgets
        budgets = []
        for b in ct_config.budgets:
            budgets.append(Budget(
                limit_usd=b["limit_usd"],
                period=b.get("period", "daily"),
                alert_threshold=b.get("alert_threshold", 0.8),
            ))
        
        persistence = os.path.expanduser(ct_config.persistence_path)
        
        self._cost_tracker = CostTracker(
            pricing=custom_pricing,
            persistence_path=persistence,
            budgets=budgets,
            auto_save_interval=ct_config.auto_save_interval,
        )
    
    async def start_cost_tracker(self):
        """Start the cost tracker (call after engine creation)."""
        if self._cost_tracker:
            await self._cost_tracker.start()
    
    async def stop_cost_tracker(self):
        """Stop the cost tracker."""
        if self._cost_tracker:
            await self._cost_tracker.stop()
    
    def get_cost_tracker(self) -> CostTracker | None:
        return self._cost_tracker
    
    def get_cost_metrics(self, since: float | None = None) -> CostMetrics | None:
        if self._cost_tracker:
            return self._cost_tracker.get_metrics(since)
        return None

    def _init_model_router(self):
        """Initialize model router from config."""
        mr_config = getattr(self.config.fusion, 'model_router', None)
        if not mr_config:
            return
        
        # Build model options from providers
        models = {}
        for name, provider in self.providers.items():
            # This is a simplified mapping - in practice would use config
            pass
        
        self._model_router = ModelRouter(
            models=models or None,
            default_policy=RoutingPolicy(mr_config.get("default_policy", "balanced")),
            cost_quality_tradeoff=mr_config.get("cost_quality_tradeoff", 7),
            exploration_rate=mr_config.get("exploration_rate", 0.1),
            ucb_c=mr_config.get("ucb_c", 2.0),
            session_ttl_seconds=mr_config.get("session_ttl_seconds", 300),
            persistence_path=os.path.expanduser(mr_config.get("persistence_path", "~/.hermes_fusion/router.json")),
        )
    
    async def start_model_router(self):
        """Start the model router (load persisted state)."""
        if self._model_router:
            await self._model_router.load()
    
    async def stop_model_router(self):
        """Stop the model router (save state)."""
        if self._model_router:
            await self._model_router._save()
    
    def get_model_router(self) -> ModelRouter | None:
        return self._model_router

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
                    provider_tasks = self._create_provider_tasks(question, model, query_id, **kwargs)
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
        query_id: str,
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

                        # Track cost
                        if self._cost_tracker and _COST_TRACKER_AVAILABLE:
                            input_tokens = response.tokens_prompt if hasattr(response, 'tokens_prompt') else 0
                            output_tokens = response.tokens_completion if hasattr(response, 'tokens_completion') else 0
                            if not input_tokens and not output_tokens:
                                import tiktoken
                                try:
                                    enc = tiktoken.get_encoding("cl100k_base")
                                    input_tokens = len(enc.encode(question))
                                    output_tokens = len(enc.encode(response.content))
                                except Exception:
                                    pass
                            self._cost_tracker.record_usage(
                                provider=n,
                                model=response.model or model or "unknown",
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                query_id=query_id,
                            )

                        if self._metrics:
                            self._metrics.record_provider_request(n, True, duration)
                        log_provider_result(self._logger, n, response, True, duration)
                        return response
                    except Exception as e:
                        duration = time.perf_counter() - start
                        if self._metrics:
                            self._metrics.record_provider_request(n, False, duration)
                        log_provider_result(self._logger, n, None, False, duration, str(e))
                        return None
            
            tasks.append(call_provider)
        
        return tasks

    async def _execute_with_timeout(
        self,
        tasks: list[Callable[[], Awaitable[ProviderResponse | None]]]
    ) -> list[ProviderResponse | None]:
        """Execute provider tasks with timeout."""
        timeout = self.config.fusion.provider_timeout
        
        async def run_with_timeout(coro, task_timeout):
            try:
                return await asyncio.wait_for(coro(), timeout=task_timeout)
            except asyncio.TimeoutError:
                return None
            except Exception:
                return None
        
        return await asyncio.gather(*[
            run_with_timeout(task, timeout) for task in tasks
        ])

    def _update_metrics(self, latency_ms: int):
        """Update running metrics."""
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (self.metrics.total_queries - 1) + latency_ms)
            / self.metrics.total_queries
        )

    def _result_to_dict(self, result: FusionResult) -> dict[str, Any]:
        """Convert FusionResult to dict for storage."""
        return {
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "method": result.method,
            "participating_providers": result.participating_providers,
            "raw_responses": [
                {
                    "content": r.content,
                    "model": r.model,
                    "provider": r.provider,
                    "tokens_used": r.tokens_used,
                    "latency_ms": r.latency_ms,
                    "raw": r.raw,
                }
                for r in result.raw_responses
            ],
            "metadata": result.metadata,
        }

    def _dict_to_result(self, d: dict[str, Any], strategy_name: str) -> FusionResult:
        """Convert dict to FusionResult."""
        return FusionResult(
            final_answer=d["final_answer"],
            confidence=d["confidence"],
            method=d["method"],
            participating_providers=d["participating_providers"],
            raw_responses=[
                ProviderResponse(
                    content=r["content"],
                    model=r["model"],
                    provider=r["provider"],
                    tokens_used=r.get("tokens_used", 0),
                    latency_ms=r.get("latency_ms", 0),
                    raw=r.get("raw", {}),
                )
                for r in d["raw_responses"]
            ],
            metadata=d.get("metadata", {}),
        )

    async def query_stream(
        self,
        question: str,
        strategy: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a fusion query response token by token.
        
        Tries to find a streaming-capable provider, falls back to non-streaming.
        """
        # Find streaming providers
        streaming_providers = [
            (name, provider) for name, provider in self.providers.items()
            if hasattr(provider, 'chat_stream') and 
            self._provider_health.get(name) is not False
        ]
        
        if not streaming_providers:
            # Fallback: just yield full response
            result = await self.query(question, strategy, model, **kwargs)
            yield result.final_answer
            return
        
        # Use first streaming provider
        name, provider = streaming_providers[0]
        messages = [{"role": "user", "content": question}]
        
        try:
            async for token in provider.chat_stream(messages, model or "default", **kwargs):
                yield token
        except Exception:
            # Fallback to non-streaming
            result = await self.query(question, strategy, model, **kwargs)
            yield result.final_answer

    async def route_query(
        self,
        question: str,
        strategy: str | None = None,
        model: str | None = None,
        policy: RoutingPolicy | None = None,
        cost_quality_tradeoff: int | None = None,
        session_id: str | None = None,
        allowed_models: list[str] | None = None,
        required_capabilities: dict[str, bool] | None = None,
        preferred_providers: list[str] | None = None,
        ignored_providers: list[str] | None = None,
        **kwargs,
    ) -> FusionResult:
        """
        Route query through model router, then execute with selected model/provider.
        
        This uses the ModelRouter to intelligently select the best model/provider
        based on task classification, cost/quality policy, and historical performance.
        """
        if not self._model_router:
            return await self.query(question, strategy, model, **kwargs)
        
        # Route to best model/provider
        decision = await self._model_router.route(
            prompt=question,
            messages=kwargs.get("messages"),
            policy=policy,
            cost_quality_tradeoff=cost_quality_tradeoff,
            session_id=session_id,
            allowed_models=allowed_models,
            required_capabilities=required_capabilities,
            preferred_providers=preferred_providers,
            ignored_providers=ignored_providers,
        )
        
        # Execute with selected model (override model parameter)
        effective_model = model or decision.model_id
        
        # Add routing metadata to kwargs
        kwargs["_routing_decision"] = decision
        
        result = await self.query(question, strategy, effective_model, **kwargs)
        
        # Record outcome for learning
        if result.metadata:
            tokens_in = result.metadata.get("tokens_in", 0)
            tokens_out = result.metadata.get("tokens_out", 0)
            cost = result.metadata.get("cost_usd", 0.0)
            latency = result.metadata.get("latency_ms", 0)
            
            self._model_router.record_outcome(
                decision=decision,
                success=bool(result.final_answer),
                latency_ms=latency,
                quality_score=result.confidence,
                cost_usd=cost,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        
        return result

    async def route_query_stream(
        self,
        question: str,
        strategy: str | None = None,
        model: str | None = None,
        policy: RoutingPolicy | None = None,
        cost_quality_tradeoff: int | None = None,
        session_id: str | None = None,
        allowed_models: list[str] | None = None,
        required_capabilities: dict[str, bool] | None = None,
        preferred_providers: list[str] | None = None,
        ignored_providers: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a routed query."""
        if not self._model_router:
            async for chunk in self.query_stream(question, strategy, model, **kwargs):
                yield chunk
            return
        
        decision = await self._model_router.route(
            prompt=question,
            messages=kwargs.get("messages"),
            policy=policy,
            cost_quality_tradeoff=cost_quality_tradeoff,
            session_id=session_id,
            allowed_models=allowed_models,
            required_capabilities=required_capabilities,
            preferred_providers=preferred_providers,
            ignored_providers=ignored_providers,
        )
        
        effective_model = model or decision.model_id
        kwargs["_routing_decision"] = decision
        
        async for chunk in self.query_stream(question, strategy, effective_model, **kwargs):
            yield chunk

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


# Factory function for easy setup
async def create_engine_from_config(config_path: str = "configs/default.yaml") -> FusionEngine:
    """Create engine from configuration file."""
    from hermes_fusion.config import load_config
    from hermes_fusion.providers import create_providers_from_config
    from hermes_fusion.providers.qdrant import create_qdrant_from_config
    
    config = load_config(config_path)
    providers, qdrant = await create_providers_from_config(config)
    
    engine = FusionEngine(config, providers, qdrant)
    
    if engine._cost_tracker:
        await engine.start_cost_tracker()
    if engine._model_router:
        await engine.start_model_router()
    
    return engine