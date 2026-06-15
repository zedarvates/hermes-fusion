"""Weighted Vote fusion strategy - weights answers by provider reliability."""

from collections import Counter
from typing import Any

from hermes_fusion.strategies.base import FusionStrategy, FusionResult, ProviderResponse, normalize_answer


class WeightedVoteStrategy(FusionStrategy):
    """
    Weighted vote fusion: each provider's answer gets a weight.
    The answer with highest cumulative weight wins.
    """
    name = "weighted_vote"

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {"localai": 1.0, "xai": 1.5, "openai": 1.2, "anthropic": 1.3}

    async def fuse(self, question: str, responses: list[ProviderResponse]) -> FusionResult:
        if not responses:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[],
            )

        # Group responses by normalized answer
        answer_weights = Counter()
        provider_for_answer = {}
        
        for resp in responses:
            if not resp.content.strip():
                continue
            norm = normalize_answer(resp.content)
            weight = self.weights.get(resp.provider, 1.0)
            answer_weights[norm] += weight
            if norm not in provider_for_answer:
                provider_for_answer[norm] = []
            provider_for_answer[norm].append(resp.provider)

        if not answer_weights:
            return FusionResult(
                final_answer="",
                confidence=0.0,
                method=self.name,
                participating_providers=[r.provider for r in responses],
                raw_responses=responses,
            )

        # Get winning answer
        winning_norm, winning_weight = answer_weights.most_common(1)[0]
        total_weight = sum(answer_weights.values())
        confidence = winning_weight / total_weight if total_weight > 0 else 0.0

        # Find original response for winning answer
        winning_content = ""
        for resp in responses:
            if normalize_answer(resp.content) == winning_norm:
                winning_content = resp.content
                break

        # Return ALL participating providers (test expectation)
        all_providers = [r.provider for r in responses if r.content.strip()]

        return FusionResult(
            final_answer=winning_content,
            confidence=confidence,
            method=self.name,
            participating_providers=all_providers,
            raw_responses=responses,
            metadata={
                "answer_weights": dict(answer_weights),
                "total_weight": total_weight,
            },
        )