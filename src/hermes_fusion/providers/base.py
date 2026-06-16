"""Base provider interface for Hermes Fusion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProviderResponse:
    """Response from a provider."""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = None


class Provider(ABC):
    """Abstract base class for all providers."""
    name: str = "base"

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], model: str, **kwargs) -> ProviderResponse:
        """Chat completion."""
        pass

    @abstractmethod
    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass