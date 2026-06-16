"""Model Router - Intelligent model/provider selection based on task type, cost, quality, and history."""

import asyncio
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class TaskType(str, Enum):
    """Types of tasks for routing."""
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"
    TOOL_USE = "tool_use"
    GENERAL = "general"


class RoutingPolicy(str, Enum):
    """Routing optimization policies."""
    QUALITY = "quality"          # Best quality regardless of cost
    COST = "cost"                # Cheapest viable option
    LATENCY = "latency"          # Fastest response
    BALANCED = "balanced"        # Balance quality/cost/latency
    COST_QUALITY = "cost_quality"  # Configurable tradeoff (0-10)


@dataclass(slots=True)
class ProviderProfile:
    """Profile of a provider for a specific model."""
    name: str
    model: str
    # Performance metrics (updated from history)
    avg_latency_ms: float = 0.0
    avg_quality_score: float = 0.5  # 0-1
    success_rate: float = 1.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    tokens_per_second: float = 0.0
    # Capabilities
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    max_context: int = 4096
    # Routing weights
    weight_quality: float = 1.0
    weight_cost: float = 1.0
    weight_latency: float = 1.0
    # Health
    is_healthy: bool = True
    last_success: float = 0.0
    last_error: float = 0.0
    consecutive_errors: int = 0
    
    @property
    def cost_score(self) -> float:
        """Normalized cost score (lower is better)."""
        if self.cost_per_1k_input == 0 and self.cost_per_1k_output == 0:
            return 1.0  # Free/local
        return 1.0 / (1.0 + self.cost_per_1k_input + self.cost_per_1k_output)
    
    @property
    def latency_score(self) -> float:
        """Normalized latency score (lower is better)."""
        if self.avg_latency_ms == 0:
            return 0.5
        return 1.0 / (1.0 + self.avg_latency_ms / 1000.0)
    
    @property
    def composite_score(self) -> float:
        """Composite score for routing."""
        return (
            self.avg_quality_score * self.weight_quality +
            self.cost_score * self.weight_cost +
            self.latency_score * self.weight_latency
        ) / (self.weight_quality + self.weight_cost + self.weight_latency)


@dataclass(slots=True)
class ModelOption:
    """A model available for routing with its provider profiles."""
    model_id: str
    task_types: list[TaskType]
    providers: list[ProviderProfile] = field(default_factory=list)
    
    def get_best_provider(self, policy: RoutingPolicy, cost_quality_tradeoff: int = 7) -> ProviderProfile | None:
        """Get best provider based on policy."""
        healthy = [p for p in self.providers if p.is_healthy]
        if not healthy:
            return None
        
        if policy == RoutingPolicy.QUALITY:
            return max(healthy, key=lambda p: p.avg_quality_score)
        elif policy == RoutingPolicy.COST:
            return max(healthy, key=lambda p: p.cost_score)
        elif policy == RoutingPolicy.LATENCY:
            return max(healthy, key=lambda p: p.latency_score)
        elif policy == RoutingPolicy.BALANCED:
            return max(healthy, key=lambda p: p.composite_score)
        elif policy == RoutingPolicy.COST_QUALITY:
            # 0 = pure quality, 10 = pure cost
            alpha = cost_quality_tradeoff / 10.0
            return max(healthy, key=lambda p: alpha * p.cost_score + (1 - alpha) * p.avg_quality_score)
        return healthy[0]
    
    def get_fallback_chain(self, policy: RoutingPolicy, cost_quality_tradeoff: int = 7, max_fallbacks: int = 3) -> list[ProviderProfile]:
        """Get ordered fallback chain."""
        healthy = [p for p in self.providers if p.is_healthy]
        if not healthy:
            return []
        
        if policy == RoutingPolicy.QUALITY:
            sorted_providers = sorted(healthy, key=lambda p: p.avg_quality_score, reverse=True)
        elif policy == RoutingPolicy.COST:
            sorted_providers = sorted(healthy, key=lambda p: p.cost_score, reverse=True)
        elif policy == RoutingPolicy.LATENCY:
            sorted_providers = sorted(healthy, key=lambda p: p.latency_score, reverse=True)
        elif policy == RoutingPolicy.BALANCED:
            sorted_providers = sorted(healthy, key=lambda p: p.composite_score, reverse=True)
        elif policy == RoutingPolicy.COST_QUALITY:
            alpha = cost_quality_tradeoff / 10.0
            sorted_providers = sorted(
                healthy,
                key=lambda p: alpha * p.cost_score + (1 - alpha) * p.avg_quality_score,
                reverse=True
            )
        else:
            sorted_providers = healthy
        
        return sorted_providers[:max_fallbacks]


@dataclass(slots=True)
class RoutingDecision:
    """Result of a routing decision."""
    model_id: str
    provider_name: str
    provider: ProviderProfile
    task_type: TaskType
    policy: RoutingPolicy
    confidence: float
    fallbacks: list[ProviderProfile]
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)


class TaskClassifier:
    """Classify prompt into task type."""
    
    KEYWORDS = {
        TaskType.CODE: [
            "code", "function", "class", "debug", "implement", "refactor",
            "python", "javascript", "typescript", "rust", "go", "java",
            "api", "endpoint", "database", "sql", "git", "docker",
            "algorithm", "data structure", "leetcode", "bug", "error",
            "compile", "syntax", "import", "library", "framework"
        ],
        TaskType.REASONING: [
            "prove", "solve", "calculate", "analyze", "deduce", "infer",
            "logic", "math", "mathematical", "theorem", "step by step",
            "reasoning", "think through", "break down", "derive"
        ],
        TaskType.CREATIVE: [
            "write", "story", "poem", "creative", "imagine", "brainstorm",
            "narrative", "character", "plot", "dialogue", "scene",
            "marketing", "copy", "slogan", "brand", "design"
        ],
        TaskType.CHAT: [
            "hello", "hi", "how are", "what is", "tell me", "explain",
            "chat", "talk", "discuss", "opinion", "thoughts"
        ],
        TaskType.TOOL_USE: [
            "search", "lookup", "find", "browse", "fetch", "get data",
            "api call", "function call", "tool", "execute", "run"
        ],
    }
    
    # Patterns that strongly indicate task type
    STRONG_PATTERNS = {
        TaskType.CODE: [
            r"```", r"def\s+\w+\(", r"function\s+\w+\(", r"class\s+\w+",
            r"import\s+\w+", r"from\s+\w+\s+import", r"SELECT\s+.*\s+FROM",
            r"CREATE\s+TABLE", r"\.py$", r"\.js$", r"\.ts$", r"\.rs$",
            r"async\s+def", r"await\s+", r"asyncio", r"Promise", r"async/await"
        ],
        TaskType.REASONING: [
            r"step\s+by\s+step", r"think\s+through", r"reason\s+about",
            r"prove\s+that", r"show\s+that", r"calculate\s+"
        ],
    }
    
    def classify(self, prompt: str, messages: list[dict] | None = None) -> TaskType:
        """Classify prompt into task type."""
        text = prompt.lower()
        if messages:
            # Include recent conversation context
            recent = " ".join(m.get("content", "").lower() for m in messages[-3:])
            text = recent + " " + text
        
        scores = defaultdict(int)
        
        # Keyword scoring
        for task_type, keywords in self.KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[task_type] += 1
        
        # Strong pattern scoring (higher weight)
        for task_type, patterns in self.STRONG_PATTERNS.items():
            for pattern in patterns:
                import re
                if re.search(pattern, text, re.IGNORECASE):
                    scores[task_type] += 5
        
        # Heuristic: short questions -> chat
        if len(text.split()) < 10 and scores.get(TaskType.CHAT, 0) > 0:
            scores[TaskType.CHAT] += 3
        
        if not scores:
            return TaskType.GENERAL
        
        return max(scores, key=scores.get)


class ModelRouter:
    """
    Intelligent model router for Hermes Fusion.
    
    Features:
    - Task classification (code, reasoning, creative, chat, etc.)
    - Per-provider performance tracking
    - Multiple routing policies (quality, cost, latency, balanced)
    - UCB exploration for discovering better providers
    - Session stickiness for conversation consistency
    - Fallback chains with health awareness
    - Cost/quality tradeoff control (0-10)
    - Persistence of routing history
    """
    
    def __init__(
        self,
        models: dict[str, ModelOption] | None = None,
        default_policy: RoutingPolicy = RoutingPolicy.BALANCED,
        cost_quality_tradeoff: int = 7,
        exploration_rate: float = 0.1,  # 10% exploration
        ucb_c: float = 2.0,  # UCB exploration parameter
        session_ttl_seconds: int = 300,  # 5 min session stickiness
        persistence_path: str | Path | None = None,
    ):
        self.models = models or self._default_models()
        self.default_policy = default_policy
        self.cost_quality_tradeoff = cost_quality_tradeoff
        self.exploration_rate = exploration_rate
        self.ucb_c = ucb_c
        self.session_ttl = session_ttl_seconds
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        self.classifier = TaskClassifier()
        self._session_cache: dict[str, tuple[RoutingDecision, float]] = {}  # session_id -> (decision, timestamp)
        self._routing_history: list[dict] = []
        self._provider_stats: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self._lock = asyncio.Lock()
        
        # Initialize UCB counts
        self._selection_counts: dict[str, int] = defaultdict(int)
        self._reward_sums: dict[str, float] = defaultdict(float)
    
    def _default_models(self) -> dict[str, ModelOption]:
        """Default model configuration for Hermes Fusion."""
        return {
            # Local models (free, good for simple tasks)
            "gemma-4-e2b-it:latest": ModelOption(
                model_id="gemma-4-e2b-it:latest",
                task_types=[TaskType.CHAT, TaskType.GENERAL, TaskType.CODE, TaskType.REASONING],
                providers=[
                    ProviderProfile(
                        name="localai",
                        model="gemma-4-e2b-it:latest",
                        cost_per_1k_input=0.0,
                        cost_per_1k_output=0.0,
                        avg_latency_ms=2000,
                        avg_quality_score=0.7,
                        max_context=8192,
                        supports_streaming=True,
                    ),
                ],
            ),
            # xAI Grok models
            "grok-3": ModelOption(
                model_id="grok-3",
                task_types=[TaskType.REASONING, TaskType.CODE, TaskType.CHAT, TaskType.GENERAL],
                providers=[
                    ProviderProfile(
                        name="xai",
                        model="grok-3",
                        cost_per_1k_input=0.30,
                        cost_per_1k_output=0.60,
                        avg_latency_ms=3000,
                        avg_quality_score=0.9,
                        max_context=131072,
                        supports_streaming=True,
                        supports_tools=True,
                    ),
                ],
            ),
            "grok-3-mini": ModelOption(
                model_id="grok-3-mini",
                task_types=[TaskType.CHAT, TaskType.CODE, TaskType.GENERAL],
                providers=[
                    ProviderProfile(
                        name="xai",
                        model="grok-3-mini",
                        cost_per_1k_input=0.03,
                        cost_per_1k_output=0.06,
                        avg_latency_ms=1500,
                        avg_quality_score=0.75,
                        max_context=131072,
                        supports_streaming=True,
                        supports_tools=True,
                    ),
                ],
            ),
            # OpenAI models
            "gpt-4o": ModelOption(
                model_id="gpt-4o",
                task_types=[TaskType.CODE, TaskType.REASONING, TaskType.CREATIVE, TaskType.CHAT, TaskType.VISION, TaskType.TOOL_USE],
                providers=[
                    ProviderProfile(
                        name="openai",
                        model="gpt-4o",
                        cost_per_1k_input=2.50,
                        cost_per_1k_output=10.00,
                        avg_latency_ms=2500,
                        avg_quality_score=0.95,
                        max_context=128000,
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                ],
            ),
            "gpt-4o-mini": ModelOption(
                model_id="gpt-4o-mini",
                task_types=[TaskType.CHAT, TaskType.CODE, TaskType.GENERAL, TaskType.VISION],
                providers=[
                    ProviderProfile(
                        name="openai",
                        model="gpt-4o-mini",
                        cost_per_1k_input=0.15,
                        cost_per_1k_output=0.60,
                        avg_latency_ms=1000,
                        avg_quality_score=0.82,
                        max_context=128000,
                        supports_streaming=True,
                        supports_tools=True,
                        supports_vision=True,
                    ),
                ],
            ),
            # Anthropic models
            "claude-3-5-sonnet": ModelOption(
                model_id="claude-3-5-sonnet",
                task_types=[TaskType.CODE, TaskType.REASONING, TaskType.CREATIVE, TaskType.CHAT, TaskType.TOOL_USE],
                providers=[
                    ProviderProfile(
                        name="anthropic",
                        model="claude-3-5-sonnet",
                        cost_per_1k_input=3.00,
                        cost_per_1k_output=15.00,
                        avg_latency_ms=3000,
                        avg_quality_score=0.94,
                        max_context=200000,
                        supports_streaming=True,
                        supports_tools=True,
                    ),
                ],
            ),
            "claude-3-5-haiku": ModelOption(
                model_id="claude-3-5-haiku",
                task_types=[TaskType.CHAT, TaskType.CODE, TaskType.GENERAL],
                providers=[
                    ProviderProfile(
                        name="anthropic",
                        model="claude-3-5-haiku",
                        cost_per_1k_input=0.25,
                        cost_per_1k_output=1.25,
                        avg_latency_ms=800,
                        avg_quality_score=0.80,
                        max_context=200000,
                        supports_streaming=True,
                        supports_tools=True,
                    ),
                ],
            ),
        }
    
    async def route(
        self,
        prompt: str,
        messages: list[dict] | None = None,
        policy: RoutingPolicy | None = None,
        cost_quality_tradeoff: int | None = None,
        session_id: str | None = None,
        allowed_models: list[str] | None = None,
        required_capabilities: dict[str, bool] | None = None,
        preferred_providers: list[str] | None = None,
        ignored_providers: list[str] | None = None,
    ) -> RoutingDecision:
        """
        Route a request to the best model/provider.
        
        Args:
            prompt: User prompt
            messages: Conversation history
            policy: Routing policy override
            cost_quality_tradeoff: 0-10 tradeoff override
            session_id: Session ID for stickiness
            allowed_models: Restrict to these models
            required_capabilities: {"streaming": True, "tools": True, "vision": True}
            preferred_providers: Try these providers first
            ignored_providers: Never use these providers
            
        Returns:
            RoutingDecision with selected model, provider, and fallbacks
        """
        async with self._lock:
            # Check session stickiness
            if session_id and session_id in self._session_cache:
                decision, timestamp = self._session_cache[session_id]
                if time.time() - timestamp < self.session_ttl:
                    decision.metadata["session_sticky"] = True
                    return decision
            
            # Classify task
            task_type = self.classifier.classify(prompt, messages)
            
            # Filter models by task type
            candidates = {
                mid: m for mid, m in self.models.items()
                if task_type in m.task_types
            }
            
            # Filter by allowed models
            if allowed_models:
                candidates = {mid: m for mid, m in candidates.items() if mid in allowed_models}
            
            # Filter by required capabilities
            if required_capabilities:
                filtered = {}
                for mid, m in candidates.items():
                    for provider in m.providers:
                        match = True
                        for cap, required in required_capabilities.items():
                            if required and not getattr(provider, f"supports_{cap}", False):
                                match = False
                                break
                        if match:
                            filtered[mid] = m
                            break
                candidates = filtered
            
            if not candidates:
                raise ValueError(f"No models available for task {task_type} with given constraints")
            
            # Apply provider preferences/filters
            for m in candidates.values():
                providers = m.providers
                if ignored_providers:
                    providers = [p for p in providers if p.name not in ignored_providers]
                if preferred_providers:
                    # Reorder: preferred first
                    pref = [p for p in providers if p.name in preferred_providers]
                    other = [p for p in providers if p.name not in preferred_providers]
                    providers = pref + other
                m.providers = providers
            
            # Remove models with no providers
            candidates = {mid: m for mid, m in candidates.items() if m.providers}
            
            # Select model using UCB for exploration
            model_id = self._select_model(candidates, task_type, policy or self.default_policy)
            model = candidates[model_id]
            
            # Select best provider for this model
            provider = model.get_best_provider(
                policy or self.default_policy,
                cost_quality_tradeoff or self.cost_quality_tradeoff
            )
            
            if provider is None:
                # Fallback to any healthy provider
                for m in candidates.values():
                    provider = m.get_best_provider(policy or self.default_policy)
                    if provider:
                        model_id = m.model_id
                        model = m
                        break
            
            if provider is None:
                raise RuntimeError("No healthy providers available")
            
            # Get fallback chain
            fallbacks = model.get_fallback_chain(
                policy or self.default_policy,
                cost_quality_tradeoff or self.cost_quality_tradeoff
            )
            
            # Build decision
            decision = RoutingDecision(
                model_id=model_id,
                provider_name=provider.name,
                provider=provider,
                task_type=task_type,
                policy=policy or self.default_policy,
                confidence=provider.composite_score,
                fallbacks=fallbacks,
                session_id=session_id,
                metadata={
                    "candidates": list(candidates.keys()),
                    "task_type": task_type.value,
                    "exploration": False,
                }
            )
            
            # Update session cache
            if session_id:
                self._session_cache[session_id] = (decision, time.time())
            
            # Track selection for UCB
            self._selection_counts[f"{model_id}:{provider.name}"] += 1
            
            return decision
    
    def _select_model(
        self,
        candidates: dict[str, ModelOption],
        task_type: TaskType,
        policy: RoutingPolicy,
    ) -> str:
        """Select model using UCB for exploration/exploitation."""
        # Exploitation: pick best by policy
        if random.random() > self.exploration_rate:
            model_scores = {}
            for mid, m in candidates.items():
                provider = m.get_best_provider(policy, self.cost_quality_tradeoff)
                if provider:
                    model_scores[mid] = provider.composite_score
            
            if model_scores:
                return max(model_scores, key=model_scores.get)
        
        # Exploration: UCB
        total_selections = sum(self._selection_counts.values()) or 1
        ucb_scores = {}
        
        for mid, m in candidates.items():
            provider = m.get_best_provider(policy, self.cost_quality_tradeoff)
            if not provider:
                continue
            
            key = f"{mid}:{provider.name}"
            count = self._selection_counts[key]
            reward_avg = self._reward_sums[key] / count if count > 0 else 0.5
            
            # UCB formula: reward_avg + c * sqrt(ln(total) / count)
            exploration_bonus = self.ucb_c * np.sqrt(np.log(total_selections) / (count + 1)) if count > 0 else float('inf')
            ucb_scores[mid] = reward_avg + exploration_bonus
        
        if ucb_scores:
            return max(ucb_scores, key=ucb_scores.get)
        
        return list(candidates.keys())[0]
    
    def record_outcome(
        self,
        decision: RoutingDecision,
        success: bool,
        latency_ms: float,
        quality_score: float | None = None,
        cost_usd: float | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ):
        """Record outcome for learning."""
        key = f"{decision.model_id}:{decision.provider_name}"
        
        # Update UCB rewards
        if success:
            reward = 1.0
            if quality_score is not None:
                reward = quality_score
            if cost_usd is not None and cost_usd > 0:
                # Penalize high cost
                reward *= max(0.1, 1.0 - min(cost_usd, 1.0))
        else:
            reward = 0.0
        
        self._selection_counts[key] += 1
        self._reward_sums[key] += reward
        
        # Update provider profile
        provider = decision.provider
        alpha = 0.1  # EMA weight
        provider.avg_latency_ms = (1 - alpha) * provider.avg_latency_ms + alpha * latency_ms
        if quality_score is not None:
            provider.avg_quality_score = (1 - alpha) * provider.avg_quality_score + alpha * quality_score
        provider.success_rate = (1 - alpha) * provider.success_rate + alpha * (1.0 if success else 0.0)
        
        if success:
            provider.last_success = time.time()
            provider.consecutive_errors = 0
            if provider.consecutive_errors >= 3:
                provider.is_healthy = False
        else:
            provider.last_error = time.time()
            provider.consecutive_errors += 1
            if provider.consecutive_errors >= 3:
                provider.is_healthy = False
        
        # Update cost tracking
        if tokens_in > 0:
            provider.cost_per_1k_input = cost_usd * 1000 / tokens_in if cost_usd else provider.cost_per_1k_input
        if tokens_out > 0:
            provider.cost_per_1k_output = cost_usd * 1000 / tokens_out if cost_usd else provider.cost_per_1k_output
        
        if latency_ms > 0:
            provider.tokens_per_second = (tokens_in + tokens_out) / (latency_ms / 1000)
        
        # Record history
        self._routing_history.append({
            "timestamp": time.time(),
            "model_id": decision.model_id,
            "provider": decision.provider_name,
            "task_type": decision.task_type.value,
            "policy": decision.policy.value,
            "success": success,
            "latency_ms": latency_ms,
            "quality_score": quality_score,
            "cost_usd": cost_usd,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        })
        
        # Persist periodically
        if len(self._routing_history) % 50 == 0:
            asyncio.create_task(self._save())
    
    async def _save(self):
        """Persist routing history and provider stats."""
        if not self.persistence_path:
            return
        
        data = {
            "version": 1,
            "saved_at": time.time(),
            "provider_stats": dict(self._provider_stats),
            "selection_counts": dict(self._selection_counts),
            "reward_sums": dict(self._reward_sums),
            "routing_history": self._routing_history[-1000:],  # Keep last 1000
        }
        
        tmp = self.persistence_path.with_suffix(".tmp")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: tmp.write_text(json.dumps(data)))
        tmp.replace(self.persistence_path)
    
    async def load(self):
        """Load persisted state."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            loop = asyncio.get_event_loop()
            data = json.loads(await loop.run_in_executor(None, self.persistence_path.read_text))
            
            self._provider_stats = defaultdict(lambda: defaultdict(list), data.get("provider_stats", {}))
            self._selection_counts = defaultdict(int, data.get("selection_counts", {}))
            self._reward_sums = defaultdict(float, data.get("reward_sums", {}))
            self._routing_history = data.get("routing_history", [])
        except Exception:
            pass
    
    def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_routes": 0}
        
        total = len(self._routing_history)
        successful = sum(1 for h in self._routing_history if h["success"])
        
        # Model distribution
        model_dist = defaultdict(int)
        provider_dist = defaultdict(int)
        task_dist = defaultdict(int)
        
        for h in self._routing_history:
            model_dist[h["model_id"]] += 1
            provider_dist[h["provider"]] += 1
            task_dist[h["task_type"]] += 1
        
        avg_latency = np.mean([h["latency_ms"] for h in self._routing_history])
        avg_cost = np.mean([h["cost_usd"] for h in self._routing_history if h["cost_usd"]]) if any(h["cost_usd"] for h in self._routing_history) else 0
        
        return {
            "total_routes": total,
            "success_rate": successful / total,
            "avg_latency_ms": float(avg_latency),
            "avg_cost_usd": float(avg_cost),
            "model_distribution": dict(model_dist),
            "provider_distribution": dict(provider_dist),
            "task_distribution": dict(task_dist),
            "session_cache_size": len(self._session_cache),
        }
    
    def clear_session(self, session_id: str):
        """Clear session stickiness."""
        self._session_cache.pop(session_id, None)
    
    def add_model(self, model: ModelOption):
        """Add or update a model option."""
        self.models[model.model_id] = model
    
    def get_models_for_task(self, task_type: TaskType) -> list[ModelOption]:
        """Get all models suitable for a task type."""
        return [m for m in self.models.values() if task_type in m.task_types]


def create_model_router_from_config(config: dict[str, Any]) -> ModelRouter:
    """Create ModelRouter from configuration dict."""
    return ModelRouter(
        default_policy=RoutingPolicy(config.get("default_policy", "balanced")),
        cost_quality_tradeoff=config.get("cost_quality_tradeoff", 7),
        exploration_rate=config.get("exploration_rate", 0.1),
        ucb_c=config.get("ucb_c", 2.0),
        session_ttl_seconds=config.get("session_ttl_seconds", 300),
        persistence_path=config.get("persistence_path"),
    )