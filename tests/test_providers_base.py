"""Tests for base provider interfaces."""

import pytest

from hermes_fusion.providers.base import Provider, ProviderResponse


def test_provider_response_creation():
    """Test ProviderResponse dataclass."""
    resp = ProviderResponse(
        content="Hello world",
        model="test-model",
        provider="test-provider",
        tokens_used=10,
        latency_ms=500,
        raw={"id": "test-123"}
    )

    assert resp.content == "Hello world"
    assert resp.model == "test-model"
    assert resp.provider == "test-provider"
    assert resp.tokens_used == 10
    assert resp.latency_ms == 500
    assert resp.raw == {"id": "test-123"}

    # Test defaults
    resp2 = ProviderResponse(content="Test", model="m", provider="p")
    assert resp2.tokens_used == 0
    assert resp2.latency_ms == 0
    assert resp2.raw is None


def test_provider_is_abstract():
    """Test that Provider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Provider()  # Should fail - abstract class