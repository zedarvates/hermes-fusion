"""Handoff fusion strategy - sequential fallback from primary to backup providers."""


from hermes_fusion.strategies.base import FusionResult, FusionStrategy, ProviderResponse


class HandoffStrategy(FusionStrategy):
    """
    Handoff fusion: try providers in order, return first valid response.
    Useful for cost optimization (local first) with cloud fallback.
    """
    name = "handoff"

    def __init__(self, order: list[str] | None = None, min_length: int = 1):
        self.order = order or ["localai", "xai", "openai", "anthropic"]
        self.min_length = min_length

    async def fuse(self, question: str, responses: list[ProviderResponse]) -> FusionResult:
        if not responses:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[],
            )

        # Build provider -> response map
        by_provider = {r.provider: r for r in responses if r.content and len(r.content.strip()) >= self.min_length}

        # Try in order
        for provider in self.order:
            if provider in by_provider:
                resp = by_provider[provider]
                return FusionResult(
                    final_answer=resp.content,
                    confidence=0.8,  # Single provider, not fused
                    method=self.name,
                    participating_providers=[provider],
                    raw_responses=responses,
                    metadata={
                        "selected_provider": provider,
                        "available_providers": list(by_provider.keys()),
                        "order": self.order,
                    },
                )

        # No provider in order gave valid response - return first available
        if by_provider:
            first_provider = next(iter(by_provider))
            resp = by_provider[first_provider]
            return FusionResult(
                final_answer=resp.content,
                confidence=0.5,
                method=self.name,
                participating_providers=[first_provider],
                raw_responses=responses,
                metadata={
                    "selected_provider": first_provider,
                    "note": "not in preferred order",
                },
            )

        # Nothing valid
        return FusionResult(
            final_answer="",
            confidence=0.0,
            method=self.name,
            participating_providers=[r.provider for r in responses],
            raw_responses=responses,
            metadata={"note": "all responses empty or too short"},
        )