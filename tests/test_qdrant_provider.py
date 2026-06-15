"""Tests for Qdrant semantic cache provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from hermes_fusion.providers.qdrant import QdrantProvider
from hermes_fusion.config import QdrantConfig


@pytest.fixture
def qdrant_config():
    return QdrantConfig(url="http://localhost:6333", collection="test_cache", vector_size=768)


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def qdrant_provider(qdrant_config, mock_client):
    provider = QdrantProvider(qdrant_config)
    provider._client = mock_client
    return provider


@pytest.mark.asyncio
async def test_qdrant_provider_health_check(qdrant_provider, mock_client):
    """Test health check via collections list."""
    mock_client.get_collections.return_value = AsyncMock(collections=[MagicMock(name="test_cache")])
    
    healthy = await qdrant_provider.health_check()
    
    assert healthy is True
    mock_client.get_collections.assert_called_once()


@pytest.mark.asyncio
async def test_qdrant_provider_health_check_failed(qdrant_provider, mock_client):
    """Test health check returns False on error."""
    mock_client.get_collections.side_effect = Exception("Connection failed")
    
    healthy = await qdrant_provider.health_check()
    
    assert healthy is False


@pytest.mark.asyncio
async def test_qdrant_provider_get_similar_found(qdrant_provider, mock_client):
    """Test semantic cache hit."""
    mock_client.search.return_value = [
        MagicMock(
            payload={"query": "test question", "result": {"final_answer": "cached answer", "confidence": 0.9}},
            score=0.95
        )
    ]
    mock_client.embed.return_value = [MagicMock(embedding=[0.1] * 768)]
    
    result = await qdrant_provider.get_similar("test question", threshold=0.9)
    
    assert result is not None
    assert result["final_answer"] == "cached answer"
    assert result["confidence"] == 0.9


@pytest.mark.asyncio
async def test_qdrant_provider_get_similar_not_found(qdrant_provider, mock_client):
    """Test semantic cache miss (low score)."""
    # Mock returns empty list because score_threshold=0.9 filters out 0.5 score
    mock_client.search.return_value = []
    mock_client.embed.return_value = [MagicMock(embedding=[0.1] * 768)]
    
    result = await qdrant_provider.get_similar("test question", threshold=0.9)
    
    assert result is None


@pytest.mark.asyncio
async def test_qdrant_provider_get_similar_empty(qdrant_provider, mock_client):
    """Test semantic cache miss (no results)."""
    mock_client.search.return_value = []
    mock_client.embed.return_value = [MagicMock(embedding=[0.1] * 768)]
    
    result = await qdrant_provider.get_similar("test question", threshold=0.9)
    
    assert result is None


@pytest.mark.asyncio
async def test_qdrant_provider_store(qdrant_provider, mock_client):
    """Test storing result in semantic cache."""
    mock_client.embed.return_value = [MagicMock(embedding=[0.1] * 768)]
    mock_client.upsert.return_value = None
    
    result = {
        "final_answer": "test answer",
        "confidence": 0.9,
        "method": "weighted_vote",
        "participating_providers": ["localai", "xai"],
        "raw_responses": [],
        "metadata": {"cached": False}
    }
    
    await qdrant_provider.store("test question", result)
    
    mock_client.embed.assert_called_once()
    mock_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_qdrant_provider_cleanup_ttl(qdrant_provider, mock_client):
    """Test TTL cleanup of old cache entries."""
    import time
    old_time = time.time() - (48 * 3600)  # 48 hours ago
    new_time = time.time() - (1 * 3600)   # 1 hour ago
    
    mock_client.scroll.return_value = (
        [
            MagicMock(id="1", payload={"timestamp": old_time}),
            MagicMock(id="2", payload={"timestamp": new_time}),
        ],
        None
    )
    mock_client.delete.return_value = None
    
    deleted = await qdrant_provider.cleanup_ttl(hours=24)
    
    assert deleted == 1  # Only the old one should be deleted
    mock_client.delete.assert_called_once()
    # Verify it was called with the old point id
    call_args = mock_client.delete.call_args
    assert call_args[1]["points_selector"] == ["1"]