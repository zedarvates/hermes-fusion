"""Tests for FusionConfig."""

import tempfile

from hermes_fusion.config import (
    FusionConfig,
)


def test_fusion_config_loads_from_toml():
    """Test FusionConfig loads correctly from TOML."""
    toml_content = """
[fusion]
default_strategy = "weighted_vote"
max_parallel_providers = 3
timeout_seconds = 120
semantic_cache_enabled = true
cache_ttl_hours = 24

[fusion.providers.localai]
base_url = "http://192.168.1.47:8080"
models = ["gemma-4-e2b-it:latest", "gpt-4o", "whisper-1", "tts-1"]
timeout = 60

[fusion.providers.hailo]
host = "192.168.1.47"
port = 8767
models = ["resnet18", "yolov8", "ocr"]
timeout = 30

[fusion.providers.qdrant]
url = "http://192.168.1.47:6333"
collection = "hermes_fusion_cache"
vector_size = 768

[fusion.providers.cloud.xai]
api_key_env = "XAI_API_KEY"
model = "grok-3"
weight = 1.0

[fusion.providers.cloud.openai]
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o"
weight = 0.8

[fusion.providers.cloud.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4"
weight = 0.9
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = FusionConfig.from_toml(f.name)

    assert config.fusion.default_strategy == "weighted_vote"
    assert config.fusion.max_parallel_providers == 3
    assert config.providers.localai.base_url == "http://192.168.1.47:8080"
    assert "gemma-4-e2b-it:latest" in config.providers.localai.models
    assert config.providers.hailo.host == "192.168.1.47"
    assert config.providers.qdrant.url == "http://192.168.1.47:6333"
    assert config.providers.cloud.xai.model == "grok-3"
    assert config.providers.cloud.openai.weight == 0.8
    assert config.providers.cloud.anthropic.weight == 0.9


def test_fusion_config_defaults():
    """Test FusionConfig default values."""
    toml_content = """
[fusion]
default_strategy = "weighted_vote"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = FusionConfig.from_toml(f.name)

    # Should have defaults for everything not specified
    assert config.fusion.default_strategy == "weighted_vote"
    assert config.fusion.max_parallel_providers == 3  # default
    assert config.providers.localai.base_url == "http://192.168.1.47:8080"  # default
    assert config.providers.hailo.port == 8767  # default
    assert config.providers.qdrant.vector_size == 768  # default


def test_get_cloud_api_key():
    """Test getting cloud API keys from environment."""
    import os
    toml_content = """
[fusion]
[fusion.providers.cloud.xai]
api_key_env = "TEST_XAI_KEY"
model = "grok-3"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = FusionConfig.from_toml(f.name)

    # Without env var
    assert config.get_cloud_api_key("xai") is None

    # With env var
    os.environ["TEST_XAI_KEY"] = "test-key-123"
    assert config.get_cloud_api_key("xai") == "test-key-123"
    del os.environ["TEST_XAI_KEY"]