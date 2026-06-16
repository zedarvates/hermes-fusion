"""Tests for Fusion Strategies."""

from unittest.mock import AsyncMock

import pytest

from hermes_fusion.strategies.base import ProviderResponse
from hermes_fusion.strategies.best_of_n import BestOfNStrategy
from hermes_fusion.strategies.cot_consensus import CoTConsensusStrategy
from hermes_fusion.strategies.handoff import HandoffStrategy
from hermes_fusion.strategies.weighted_vote import WeightedVoteStrategy


class TestWeightedVoteStrategy:
    @pytest.fixture
    def strategy(self):
        return WeightedVoteStrategy(weights={"localai": 1.0, "xai": 1.5, "openai": 1.2})

    @pytest.mark.asyncio
    async def test_weighted_vote_single_response(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.final_answer == "Answer A"
        assert result.method == "weighted_vote"
        assert result.participating_providers == ["localai"]

    @pytest.mark.asyncio
    async def test_weighted_vote_multiple_different(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
            ProviderResponse(content="Answer B", provider="xai", model="grok-3", tokens_used=80),
            ProviderResponse(content="Answer C", provider="openai", model="gpt-4o", tokens_used=100),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.final_answer in ["Answer A", "Answer B", "Answer C"]
        assert result.method == "weighted_vote"
        assert set(result.participating_providers) == {"localai", "xai", "openai"}

    @pytest.mark.asyncio
    async def test_weighted_vote_empty(self, strategy):
        result = await strategy.fuse("Test question", [])
        
        assert result.final_answer == ""
        assert result.confidence == 0.0


class TestBestOfNStrategy:
    @pytest.fixture
    def strategy(self):
        return BestOfNStrategy(n=3, judge_provider="xai")

    @pytest.mark.asyncio
    async def test_best_of_n_single(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.final_answer == "Answer A"
        assert result.method == "best_of_n"

    @pytest.mark.asyncio
    async def test_best_of_n_multiple(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
            ProviderResponse(content="Answer B", provider="xai", model="grok-3", tokens_used=80),
            ProviderResponse(content="Answer C", provider="openai", model="gpt-4o", tokens_used=100),
        ]
        # Mock the judge call
        strategy._judge = AsyncMock(return_value="Answer B is best")
        
        result = await strategy.fuse("Test question", responses)
        
        assert result.method == "best_of_n"


class TestCoTConsensusStrategy:
    @pytest.fixture
    def strategy(self):
        return CoTConsensusStrategy(min_agreement=0.6)

    @pytest.mark.asyncio
    async def test_cot_consensus_agreement(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
            ProviderResponse(content="Answer A", provider="xai", model="grok-3", tokens_used=80),
            ProviderResponse(content="Answer B", provider="openai", model="gpt-4o", tokens_used=100),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.final_answer == "Answer A"
        assert result.method == "cot_consensus"
        assert result.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_cot_consensus_no_agreement(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
            ProviderResponse(content="Answer B", provider="xai", model="grok-3", tokens_used=80),
            ProviderResponse(content="Answer C", provider="openai", model="gpt-4o", tokens_used=100),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.confidence < 0.6
        assert result.method == "cot_consensus"


class TestHandoffStrategy:
    @pytest.fixture
    def strategy(self):
        return HandoffStrategy(order=["localai", "xai", "openai"])

    @pytest.mark.asyncio
    async def test_handoff_first_success(self, strategy):
        responses = [
            ProviderResponse(content="Answer A", provider="localai", model="gemma-4", tokens_used=50),
            ProviderResponse(content="Answer B", provider="xai", model="grok-3", tokens_used=80),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.final_answer == "Answer A"
        assert result.method == "handoff"
        assert result.participating_providers == ["localai"]

    @pytest.mark.asyncio
    async def test_handoff_skip_empty(self, strategy):
        responses = [
            ProviderResponse(content="", provider="localai", model="gemma-4", tokens_used=0),
            ProviderResponse(content="Answer B", provider="xai", model="grok-3", tokens_used=80),
        ]
        result = await strategy.fuse("Test question", responses)
        
        assert result.final_answer == "Answer B"
        assert result.participating_providers == ["xai"]


class TestStrategyRegistry:
    @pytest.mark.asyncio
    async def test_get_strategy(self):
        from hermes_fusion.strategies import get_strategy
        
        weighted = get_strategy("weighted_vote")
        best_of_n = get_strategy("best_of_n")
        cot = get_strategy("cot_consensus")
        handoff = get_strategy("handoff")
        
        assert isinstance(weighted, WeightedVoteStrategy)
        assert isinstance(best_of_n, BestOfNStrategy)
        assert isinstance(cot, CoTConsensusStrategy)
        assert isinstance(handoff, HandoffStrategy)

    @pytest.mark.asyncio
    async def test_unknown_strategy(self):
        from hermes_fusion.strategies import get_strategy
        
        with pytest.raises(ValueError):
            get_strategy("unknown_strategy")