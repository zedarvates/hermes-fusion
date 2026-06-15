"""Best-of-N fusion strategy - uses a judge model to pick the best answer."""

from typing import Any

from hermes_fusion.strategies.base import FusionStrategy, FusionResult, ProviderResponse, normalize_answer


class BestOfNStrategy(FusionStrategy):
    """
    Best-of-N fusion: use a judge LLM to evaluate and pick the best answer.
    Falls back to weighted vote if judge unavailable.
    """
    name = "best_of_n"

    def __init__(self, n: int = 3, judge_provider: str = "xai", judge_model: str | None = None):
        self.n = n
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self._judge = None  # Injected for testing

    async def fuse(self, question: str, responses: list[ProviderResponse]) -> FusionResult:
        if not responses:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[],
            )

        # Filter non-empty responses
        valid_responses = [r for r in responses if r.content.strip()]
        if not valid_responses:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[r.provider for r in responses],
                raw_responses=responses,
            )

        if len(valid_responses) == 1:
            resp = valid_responses[0]
            return FusionResult(
                final_answer=resp.content,
                confidence=1.0,
                method=self.name,
                participating_providers=[resp.provider],
                raw_responses=responses,
            )

        # Try judge if available
        if self._judge:
            try:
                judgment = await self._judge(question, valid_responses)
                return self._parse_judgment(judgment, valid_responses, responses)
            except Exception:
                pass  # Fall back to weighted vote

        # Fallback: weighted vote
        from hermes_fusion.strategies.weighted_vote import WeightedVoteStrategy
        fallback = WeightedVoteStrategy()
        result = await fallback.fuse(question, responses)
        result.method = f"{self.name}_fallback"
        return result

    def _parse_judgment(self, judgment: str, valid_responses: list[ProviderResponse], 
                        all_responses: list[ProviderResponse]) -> FusionResult:
        """Parse judge's response to find winning answer."""
        # Simple parsing: look for provider name or answer content
        judgment_lower = judgment.lower()
        
        for resp in valid_responses:
            if resp.provider.lower() in judgment_lower:
                return FusionResult(
                    final_answer=resp.content,
                    confidence=0.9,
                    method=self.name,
                    participating_providers=[resp.provider],
                    raw_responses=all_responses,
                    metadata={"judgment": judgment},
                )
            
            # Check if answer content mentioned
            norm = normalize_answer(resp.content)
            if norm in judgment_lower:
                return FusionResult(
                    final_answer=resp.content,
                    confidence=0.8,
                    method=self.name,
                    participating_providers=[resp.provider],
                    raw_responses=all_responses,
                    metadata={"judgment": judgment},
                )

        # Default to first valid response
        return FusionResult(
            final_answer=valid_responses[0].content,
            confidence=0.5,
            method=self.name,
            participating_providers=[valid_responses[0].provider],
            raw_responses=all_responses,
            metadata={"judgment": judgment, "note": "could not parse judge output"},
        )