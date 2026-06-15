"""Tests for LocalAI provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_fusion.config import LocalAIConfig
from hermes_fusion.providers.localai import LocalAIProvider


@pytest.fixture
def localai_config():
    return LocalAIConfig(base_url="http://localhost:8080", models=["test-model"], timeout=30)


@pytest.fixture
def localai_provider(localai_config):
    return LocalAIProvider(localai_config)


@pytest.mark.asyncio
async def test_localai_provider_chat_completion(localai_provider):
    """Test chat completion returns proper response."""
    with patch.object(localai_provider, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {
            "data": {
                "choices": [{"message": {"content": "Hello from LocalAI"}}],
                "model": "test-model",
                "usage": {"total_tokens": 42}
            },
            "latency_ms": 150
        }
        
        response = await localai_provider.chat([
            {"role": "user", "content": "Say hello"}
        ], model="test-model")
        
        assert response.content == "Hello from LocalAI"
        assert response.model == "test-model"
        assert response.provider == "localai"
        assert response.tokens_used == 42
        assert response.latency_ms == 150
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_localai_provider_embeddings(localai_provider):
    """Test embedding generation."""
    with patch.object(localai_provider, '_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {
            "data": {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            },
            "latency_ms": 100
        }
        
        embeddings = await localai_provider.embed(["text1", "text2"], model="embedding-model")
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        mock_request.assert_called_once()


@pytest.mark.asyncio
async def test_localai_provider_health_check_healthy(localai_provider):
    """Test health check returns True when healthy."""
    with patch('hermes_fusion.providers.localai.urlopen') as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value = mock_response
        
        healthy = await localai_provider.health_check()
        
        assert healthy is True
        mock_urlopen.assert_called_once()


@pytest.mark.asyncio
async def test_localai_provider_health_check_unhealthy(localai_provider):
    """Test health check returns False when unhealthy."""
    with patch('hermes_fusion.providers.localai.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = Exception("Connection refused")
        
        healthy = await localai_provider.health_check()
        
        assert healthy is False


@pytest.mark.asyncio
async def test_localai_provider_request_error_handling(localai_provider):
    """Test _request handles URLError properly."""
    from urllib.error import URLError
    
    with patch('hermes_fusion.providers.localai.urlopen') as mock_urlopen:
        mock_urlopen.side_effect = URLError("Connection failed")
        
        with pytest.raises(ConnectionError, match="LocalAI request failed"):
            await localai_provider._request("/v1/chat/completions", {"model": "test"})