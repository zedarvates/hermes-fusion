"""Fusion Strategies - pluggable methods for combining multi-LLM responses."""

from hermes_fusion.strategies.base import (
    FusionResult,
    FusionStrategy,
    ProviderResponse,
    normalize_answer,
)
from hermes_fusion.strategies.best_of_n import BestOfNStrategy
from hermes_fusion.strategies.cot_consensus import CoTConsensusStrategy
from hermes_fusion.strategies.handoff import HandoffStrategy
from hermes_fusion.strategies.weighted_vote import WeightedVoteStrategy

# Strategy registry
_STRATEGIES = {
    "weighted_vote": lambda: WeightedVoteStrategy(),
    "best_of_n": lambda: BestOfNStrategy(),
    "cot_consensus": lambda: CoTConsensusStrategy(),
    "handoff": lambda: HandoffStrategy(),
}


def get_strategy(name: str) -> FusionStrategy:
    """Get a fusion strategy instance by name."""
    if name not in _STRATEGIES:
        available = ", ".join(sorted(_STRATEGIES.keys()))
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return _STRATEGIES[name]()


def register_strategy(name: str, factory) -> None:
    """Register a custom fusion strategy."""
    _STRATEGIES[name] = factory


def list_strategies() -> list[str]:
    """List available strategy names."""
    return sorted(_STRATEGIES.keys())


__all__ = [
    "FusionStrategy",
    "FusionResult", 
    "ProviderResponse",
    "normalize_answer",
    "WeightedVoteStrategy",
    "BestOfNStrategy",
    "CoTConsensusStrategy",
    "HandoffStrategy",
    "get_strategy",
    "register_strategy",
    "list_strategies",
]