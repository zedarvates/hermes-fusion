"""Cost Tracker - Token usage tracking, cost estimation, budgets, and alerts."""

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Pricing per 1K tokens (input / output) - update as models change
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # xAI (Grok)
    "grok-3": (0.30, 0.60),
    "grok-3-mini": (0.03, 0.06),
    "grok-2": (0.20, 0.60),
    "grok-2-vision": (0.20, 0.60),
    "grok-1.5": (0.20, 0.60),
    "grok-1.5-vision": (0.20, 0.60),
    
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.02),
    "text-embedding-3-large": (0.13, 0.13),
    
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    
    # Local (free - token cost only for accounting)
    "gemma-4-e2b-it:latest": (0.0, 0.0),
    "gemma-2b": (0.0, 0.0),
    "llama-3.1": (0.0, 0.0),
    "mistral": (0.0, 0.0),
}

# Local models (free compute but track tokens for accounting)
LOCAL_PROVIDERS = {"localai", "hailo"}

# Cloud providers (have costs)
CLOUD_PROVIDERS = {"xai", "openai", "anthropic"}


@dataclass(slots=True)
class TokenUsage:
    """Token usage for a single request."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float = field(default_factory=time.time)
    query_id: str = ""
    success: bool = True
    cached: bool = False
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def cost_usd(self) -> float:
        """Calculate cost in USD for this request."""
        if self.cached:
            return 0.0
        pricing = DEFAULT_PRICING.get(self.model, (0.0, 0.0))
        input_cost = (self.input_tokens / 1000.0) * pricing[0]
        output_cost = (self.output_tokens / 1000.0) * pricing[1]
        return input_cost + output_cost


@dataclass(slots=True)
class Budget:
    """Budget configuration and tracking."""
    limit_usd: float
    period: str = "daily"  # "daily", "weekly", "monthly"
    alert_threshold: float = 0.8  # Alert at 80% of budget
    alert_callback: Callable[[str, float, float], Any] | None = None
    
    def __post_init__(self):
        if self.period not in ("daily", "weekly", "monthly"):
            raise ValueError(f"Invalid period: {self.period}")
    
    @property
    def period_start(self) -> float:
        """Get Unix timestamp for start of current budget period."""
        now = datetime.now()
        if self.period == "daily":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif self.period == "weekly":
            start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # monthly
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()


@dataclass
class CostMetrics:
    """Aggregated cost metrics."""
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0
    cached_requests: int = 0
    provider_costs: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    provider_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_costs: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    model_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hourly_costs: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    daily_costs: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    def add_usage(self, usage: TokenUsage):
        """Add a token usage record to metrics."""
        self.total_cost_usd += usage.cost_usd
        self.total_tokens += usage.total_tokens
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.request_count += 1
        if usage.cached:
            self.cached_requests += 1
        if not usage.cached:
            self.provider_costs[usage.provider] += usage.cost_usd
            self.model_costs[usage.model] += usage.cost_usd
        self.provider_tokens[usage.provider] += usage.total_tokens
        self.model_tokens[usage.model] += usage.total_tokens
        
        hour_key = datetime.fromtimestamp(usage.timestamp).strftime("%Y-%m-%d-%H")
        day_key = datetime.fromtimestamp(usage.timestamp).strftime("%Y-%m-%d")
        self.hourly_costs[hour_key] += usage.cost_usd
        self.daily_costs[day_key] += usage.cost_usd
    
    def get_budget_status(self, budget: Budget) -> dict[str, Any]:
        """Get budget status for current period."""
        period_start = budget.period_start
        period_cost = sum(
            cost for day, cost in self.daily_costs.items()
            if datetime.strptime(day, "%Y-%m-%d").timestamp() >= period_start
        )
        return {
            "budget_limit": budget.limit_usd,
            "spent": period_cost,
            "remaining": max(0, budget.limit_usd - period_cost),
            "utilization": period_cost / budget.limit_usd if budget.limit_usd > 0 else 0,
            "alert_triggered": period_cost / budget.limit_usd >= budget.alert_threshold if budget.limit_usd > 0 else False,
            "period": budget.period,
        }


class CostTracker:
    """
    Tracks token usage and costs across all providers.
    Stores data in memory + optional persistent file.
    """
    
    def __init__(
        self,
        pricing: dict[str, tuple[float, float]] | None = None,
        persistence_path: str | Path | None = None,
        budgets: list[Budget] | None = None,
        auto_save_interval: int = 60,  # seconds
    ):
        self.pricing = {**DEFAULT_PRICING, **(pricing or {})}
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.budgets = budgets or []
        self.auto_save_interval = auto_save_interval
        
        self._usage_log: list[TokenUsage] = []
        self._metrics = CostMetrics()
        self._lock = asyncio.Lock()
        self._save_task: asyncio.Task | None = None
        self._alerts_triggered: set[str] = set()
    
    async def start(self):
        """Start auto-save task and load persisted data."""
        if self.persistence_path and self.persistence_path.exists():
            await self.load()
        self._save_task = asyncio.create_task(self._auto_save_loop())
    
    async def stop(self):
        """Stop auto-save and persist."""
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self.save()
    
    async def _auto_save_loop(self):
        """Periodically save to disk."""
        while True:
            await asyncio.sleep(self.auto_save_interval)
            await self.save()
    
    async def record_usage(self, usage: TokenUsage) -> None:
        """Record a token usage event."""
        async with self._lock:
            self._usage_log.append(usage)
            self._metrics.add_usage(usage)
            await self._check_budgets()
    
    async def _check_budgets(self):
        """Check all budgets and trigger alerts if needed."""
        for budget in self.budgets:
            status = self._metrics.get_budget_status(budget)
            if status["alert_triggered"]:
                alert_key = f"{budget.period}_{budget.alert_threshold}"
                if alert_key not in self._alerts_triggered:
                    self._alerts_triggered.add(alert_key)
                    if budget.alert_callback:
                        try:
                            await budget.alert_callback(
                                f"BUDGET ALERT: {budget.period} budget at {status['utilization']:.1%}",
                                status["spent"],
                                budget.limit_usd,
                            )
                        except Exception:
                            pass  # Don't let alert failures break tracking
    
    def get_metrics(self, since: float | None = None) -> CostMetrics:
        """Get aggregated metrics, optionally since a timestamp."""
        if since is None:
            return self._metrics
        
        # Filter usage log and rebuild metrics
        filtered = CostMetrics()
        for usage in self._usage_log:
            if usage.timestamp >= since:
                filtered.add_usage(usage)
        return filtered
    
    def get_metrics_by_provider(self) -> dict[str, dict[str, Any]]:
        """Get per-provider breakdown."""
        result = {}
        for provider, cost in self._metrics.provider_costs.items():
            tokens = self._metrics.provider_tokens.get(provider, 0)
            result[provider] = {
                "cost_usd": cost,
                "total_tokens": tokens,
                "request_count": sum(
                    1 for u in self._usage_log if u.provider == provider
                ),
            }
        return result
    
    def get_metrics_by_model(self) -> dict[str, dict[str, Any]]:
        """Get per-model breakdown."""
        result = {}
        for model, cost in self._metrics.model_costs.items():
            tokens = self._metrics.model_tokens.get(model, 0)
            result[model] = {
                "cost_usd": cost,
                "total_tokens": tokens,
                "pricing_per_1k": self.pricing.get(model, (0.0, 0.0)),
            }
        return result
    
    async def save(self) -> None:
        """Persist usage log to disk."""
        if not self.persistence_path:
            return
        
        async with self._lock:
            data = {
                "version": 1,
                "saved_at": time.time(),
                "pricing": {k: list(v) for k, v in self.pricing.items()},
                "usage_log": [
                    {
                        "provider": u.provider,
                        "model": u.model,
                        "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens,
                        "timestamp": u.timestamp,
                        "query_id": u.query_id,
                        "success": u.success,
                        "cached": u.cached,
                    }
                    for u in self._usage_log
                ],
                "budgets": [
                    {
                        "limit_usd": b.limit_usd,
                        "period": b.period,
                        "alert_threshold": b.alert_threshold,
                    }
                    for b in self.budgets
                ],
            }
            
            # Atomic write
            tmp_path = self.persistence_path.with_suffix(".tmp")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: tmp_path.write_text(json.dumps(data)))
            tmp_path.replace(self.persistence_path)
    
    async def load(self) -> None:
        """Load usage log from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self.persistence_path.read_text)
            parsed = json.loads(data)
            
            # Restore pricing
            self.pricing.update({k: tuple(v) for k, v in parsed.get("pricing", {}).items()})
            
            # Restore usage log
            self._usage_log = [
                TokenUsage(
                    provider=u["provider"],
                    model=u["model"],
                    input_tokens=u["input_tokens"],
                    output_tokens=u["output_tokens"],
                    timestamp=u["timestamp"],
                    query_id=u.get("query_id", ""),
                    success=u.get("success", True),
                    cached=u.get("cached", False),
                )
                for u in parsed.get("usage_log", [])
            ]
            
            # Rebuild metrics
            self._metrics = CostMetrics()
            for usage in self._usage_log:
                self._metrics.add_usage(usage)
            
            # Restore budgets (callbacks need to be re-attached)
            saved_budgets = parsed.get("budgets", [])
            self.budgets = [
                Budget(
                    limit_usd=b["limit_usd"],
                    period=b["period"],
                    alert_threshold=b.get("alert_threshold", 0.8),
                )
                for b in saved_budgets
            ]
        except Exception:
            pass  # Fail silently, start fresh
    
    def reset_metrics(self, keep_log: bool = True) -> None:
        """Reset in-memory metrics (keep or clear usage log)."""
        if not keep_log:
            self._usage_log.clear()
            self._alerts_triggered.clear()
        self._metrics = CostMetrics()
        for usage in self._usage_log:
            if not usage.cached:
                self._metrics.add_usage(usage)


def create_cost_tracker_from_config(config: dict[str, Any]) -> CostTracker:
    """Create CostTracker from configuration dict."""
    pricing = config.get("pricing", {})
    budgets = [
        Budget(
            limit_usd=b["limit_usd"],
            period=b.get("period", "daily"),
            alert_threshold=b.get("alert_threshold", 0.8),
        )
        for b in config.get("budgets", [])
    ]
    return CostTracker(
        pricing=pricing,
        persistence_path=config.get("persistence_path"),
        budgets=budgets,
        auto_save_interval=config.get("auto_save_interval", 60),
    )


# Convenience function for simple usage
def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Quick cost estimation without full tracker."""
    pricing = DEFAULT_PRICING.get(model, (0.0, 0.0))
    if provider in LOCAL_PROVIDERS:
        return 0.0
    input_cost = (input_tokens / 1000.0) * pricing[0]
    output_cost = (output_tokens / 1000.0) * pricing[1]
    return input_cost + output_cost