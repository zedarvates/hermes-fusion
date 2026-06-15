"""Base classes for fusion strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Response from a single LLM provider."""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: int = 0
    raw: Any = None


@dataclass
class FusionResult:
    """Result of fusing multiple provider responses."""
    final_answer: str
    confidence: float
    method: str
    participating_providers: list[str]
    raw_responses: list[ProviderResponse] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FusionStrategy(ABC):
    """Abstract base class for fusion strategies."""
    
    @abstractmethod
    async def fuse(self, question: str, responses: list[ProviderResponse]) -> FusionResult:
        """Fuse multiple provider responses into a single result."""
        pass


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison (lowercase, strip punctuation)."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text