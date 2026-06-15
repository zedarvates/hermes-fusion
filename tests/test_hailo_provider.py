"""Tests for Hailo-8 provider."""

from unittest.mock import AsyncMock

import pytest

from hermes_fusion.config import HailoConfig
from hermes_fusion.providers.hailo import HailoProvider


@pytest.fixture
def hailo_config():
    return HailoConfig(host="192.168.1.47", port=8767, models=["resnet18", "yolov8", "ocr"])


@pytest.fixture
def mock_classify_fn():
    return AsyncMock()


@pytest.fixture
def mock_detect_fn():
    return AsyncMock()


@pytest.fixture
def mock_ocr_fn():
    return AsyncMock()


@pytest.fixture
def mock_status_fn():
    return AsyncMock()


@pytest.fixture
def hailo_provider(hailo_config, mock_classify_fn, mock_detect_fn, mock_ocr_fn, mock_status_fn):
    return HailoProvider(
        hailo_config,
        classify_fn=mock_classify_fn,
        detect_fn=mock_detect_fn,
        ocr_fn=mock_ocr_fn,
        status_fn=mock_status_fn,
    )


@pytest.mark.asyncio
async def test_hailo_provider_classify(hailo_provider, mock_classify_fn):
    """Test image classification."""
    mock_classify_fn.return_value = {"predictions": [{"label": "cat", "confidence": 0.95}, {"label": "dog", "confidence": 0.03}]}
    
    result = await hailo_provider.classify("/path/to/image.jpg")
    
    assert "predictions" in result
    assert result["predictions"][0]["label"] == "cat"
    assert result["predictions"][0]["confidence"] == 0.95
    mock_classify_fn.assert_called_once_with(image_path="/path/to/image.jpg")


@pytest.mark.asyncio
async def test_hailo_provider_detect(hailo_provider, mock_detect_fn):
    """Test object detection."""
    mock_detect_fn.return_value = {"detections": [{"class": "person", "confidence": 0.88, "bbox": [10, 20, 100, 200]}]}
    
    result = await hailo_provider.detect("/path/to/image.jpg")
    
    assert "detections" in result
    assert result["detections"][0]["class"] == "person"
    assert result["detections"][0]["confidence"] == 0.88
    mock_detect_fn.assert_called_once_with(image_path="/path/to/image.jpg")


@pytest.mark.asyncio
async def test_hailo_provider_ocr(hailo_provider, mock_ocr_fn):
    """Test OCR text extraction."""
    mock_ocr_fn.return_value = {"text": "Hello World", "confidence": 0.92}
    
    result = await hailo_provider.ocr("/path/to/image.jpg")
    
    assert "text" in result
    assert result["text"] == "Hello World"
    assert result["confidence"] == 0.92
    mock_ocr_fn.assert_called_once_with(image_path="/path/to/image.jpg")


@pytest.mark.asyncio
async def test_hailo_provider_health_check(hailo_provider, mock_status_fn):
    """Test health check via MCP status."""
    mock_status_fn.return_value = {"status": "ok", "firmware": "2024.10", "temperature": 45, "models_loaded": ["resnet18", "yolov8", "ocr"]}
    
    healthy = await hailo_provider.health_check()
    
    assert healthy is True
    mock_status_fn.assert_called_once()


@pytest.mark.asyncio
async def test_hailo_provider_health_check_failed(hailo_provider, mock_status_fn):
    """Test health check returns False on error."""
    mock_status_fn.side_effect = Exception("Connection failed")
    
    healthy = await hailo_provider.health_check()
    
    assert healthy is False


@pytest.mark.asyncio
async def test_hailo_provider_chat_not_supported(hailo_provider):
    """Test that chat raises NotImplementedError for vision provider."""
    with pytest.raises(NotImplementedError, match="vision models don't use chat"):
        await hailo_provider.chat([{"role": "user", "content": "test"}], model="test")


@pytest.mark.asyncio
async def test_hailo_provider_embed_not_supported(hailo_provider):
    """Test that embed raises NotImplementedError for vision provider."""
    with pytest.raises(NotImplementedError, match="not supported"):
        await hailo_provider.embed(["text"], model="test")