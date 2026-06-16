"""Tests for Cloud providers (xAI, OpenAI, Anthropic)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_fusion.config import CloudConfig, CloudProviderConfig
from hermes_fusion.providers.cloud import AnthropicProvider, OpenAIProvider, XAIProvider


@pytest.fixture
def mock_openai_client():
    return MagicMock()


@pytest.fixture
def mock_anthropic_client():
    return MagicMock()


class TestXAIProvider:
    @pytest.fixture
    def config(self):
        return CloudConfig(xai=CloudProviderConfig(api_key_env="XAI_API_KEY", model="grok-3"))

    @pytest.fixture
    def provider(self, config, mock_openai_client):
        provider = XAIProvider(config)
        provider._client = mock_openai_client
        return provider

    @pytest.mark.asyncio
    async def test_xai_chat(self, provider, mock_openai_client, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="xAI response"))]
        mock_response.usage = MagicMock(total_tokens=100)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat([{"role": "user", "content": "Hello"}], "grok-3")

        assert result.content == "xAI response"
        assert result.tokens_used == 100
        assert result.provider == "xai"

    @pytest.mark.asyncio
    async def test_xai_health_check(self, provider, mock_openai_client, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        mock_openai_client.models.list = AsyncMock(return_value=MagicMock(data=[MagicMock(id="grok-3")]))
        healthy = await provider.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_xai_health_check_failed(self, provider, mock_openai_client, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        mock_openai_client.models.list = AsyncMock(side_effect=Exception("API error"))
        healthy = await provider.health_check()
        assert healthy is False


class TestOpenAIProvider:
    @pytest.fixture
    def config(self):
        return CloudConfig(openai=CloudProviderConfig(api_key_env="OPENAI_API_KEY", model="gpt-4o"))

    @pytest.fixture
    def provider(self, config, mock_openai_client):
        provider = OpenAIProvider(config)
        provider._client = mock_openai_client
        return provider

    @pytest.mark.asyncio
    async def test_openai_chat(self, provider, mock_openai_client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
        mock_response.usage = MagicMock(total_tokens=150)
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat([{"role": "user", "content": "Hello"}], "gpt-4o")

        assert result.content == "OpenAI response"
        assert result.tokens_used == 150
        assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_openai_embed(self, provider, mock_openai_client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        embeddings = await provider.embed(["text1", "text2"], "text-embedding-3-small")

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536

    @pytest.mark.asyncio
    async def test_openai_health_check(self, provider, mock_openai_client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_openai_client.models.list = AsyncMock(return_value=MagicMock(data=[MagicMock(id="gpt-4o")]))
        healthy = await provider.health_check()
        assert healthy is True


class TestAnthropicProvider:
    @pytest.fixture
    def config(self):
        return CloudConfig(anthropic=CloudProviderConfig(api_key_env="ANTHROPIC_API_KEY", model="claude-3-5-sonnet"))

    @pytest.fixture
    def provider(self, config, mock_anthropic_client):
        provider = AnthropicProvider(config)
        provider._client = mock_anthropic_client
        return provider

    @pytest.mark.asyncio
    async def test_anthropic_chat(self, provider, mock_anthropic_client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Anthropic response")]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=50)
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat([{"role": "user", "content": "Hello"}], "claude-3-5-sonnet")

        assert result.content == "Anthropic response"
        assert result.tokens_used == 100
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_anthropic_health_check(self, provider, mock_anthropic_client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_anthropic_client.messages.create = AsyncMock(return_value=MagicMock())
        healthy = await provider.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_anthropic_health_check_failed(self, provider, mock_anthropic_client, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_anthropic_client.messages.create = AsyncMock(side_effect=Exception("API error"))
        healthy = await provider.health_check()
        assert healthy is False