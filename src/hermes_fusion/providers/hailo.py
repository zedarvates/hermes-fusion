"""Hailo-8 provider implementation using MCP vision tools."""

from collections.abc import Callable
from typing import Any

from hermes_fusion.config import HailoConfig
from hermes_fusion.providers.base import Provider, ProviderResponse


class HailoProvider(Provider):
    """Hailo-8 AI accelerator provider for vision tasks (classify, detect, OCR)."""
    name = "hailo"

    def __init__(
        self,
        config: HailoConfig,
        classify_fn: Callable | None = None,
        detect_fn: Callable | None = None,
        ocr_fn: Callable | None = None,
        status_fn: Callable | None = None,
    ):
        self.config = config
        # Allow injection for testing; defaults to MCP functions in Hermes env
        self._classify_fn = classify_fn
        self._detect_fn = detect_fn
        self._ocr_fn = ocr_fn
        self._status_fn = status_fn

    def _get_classify_fn(self):
        if self._classify_fn:
            return self._classify_fn
        # In Hermes environment, mcp_hailo_vision_hailo_classify is available globally
        try:
            return globals()['mcp_hailo_vision_hailo_classify']
        except KeyError:
            raise RuntimeError("mcp_hailo_vision_hailo_classify not available. Run in Hermes or inject classify_fn.")

    def _get_detect_fn(self):
        if self._detect_fn:
            return self._detect_fn
        try:
            return globals()['mcp_hailo_vision_hailo_detect']
        except KeyError:
            raise RuntimeError("mcp_hailo_vision_hailo_detect not available. Run in Hermes or inject detect_fn.")

    def _get_ocr_fn(self):
        if self._ocr_fn:
            return self._ocr_fn
        try:
            return globals()['mcp_hailo_vision_hailo_ocr']
        except KeyError:
            raise RuntimeError("mcp_hailo_vision_hailo_ocr not available. Run in Hermes or inject ocr_fn.")

    def _get_status_fn(self):
        if self._status_fn:
            return self._status_fn
        try:
            return globals()['mcp_hailo_vision_hailo_status']
        except KeyError:
            raise RuntimeError("mcp_hailo_vision_hailo_status not available. Run in Hermes or inject status_fn.")

    async def chat(self, messages: list[dict[str, str]], model: str, **kwargs) -> ProviderResponse:
        raise NotImplementedError("HailoProvider.chat - vision models don't use chat, use classify/detect/ocr")

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError("HailoProvider.embed - not supported")

    async def classify(self, image_path: str) -> dict[str, Any]:
        """Classify image using Hailo-8 ResNet-18."""
        fn = self._get_classify_fn()
        return await fn(image_path=image_path)

    async def detect(self, image_path: str) -> dict[str, Any]:
        """Detect objects in image using Hailo-8 YOLOv8."""
        fn = self._get_detect_fn()
        return await fn(image_path=image_path)

    async def ocr(self, image_path: str) -> dict[str, Any]:
        """Extract text from image using Hailo-8 OCR."""
        fn = self._get_ocr_fn()
        return await fn(image_path=image_path)

    async def health_check(self) -> bool:
        """Check Hailo-8 device status."""
        try:
            fn = self._get_status_fn()
            result = await fn()
            return result.get("status") == "ok"
        except Exception:
            return False