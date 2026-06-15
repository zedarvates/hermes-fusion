"""LocalAI provider implementation using stdlib urllib."""

import asyncio
import json
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from hermes_fusion.config import LocalAIConfig
from hermes_fusion.providers.base import Provider, ProviderResponse


class LocalAIProvider(Provider):
    """LocalAI provider (OpenAI-compatible API on EUREKAI:8080)."""
    name = "localai"

    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')

    async def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to LocalAI API."""
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode('utf-8')
        req = Request(url, data=data, headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }, method='POST')

        loop = asyncio.get_event_loop()
        start = time.perf_counter()
        try:
            def _fetch():
                return urlopen(req, timeout=self.config.timeout)
            response = await loop.run_in_executor(None, _fetch)
        except URLError as e:
            raise ConnectionError("LocalAI request failed") from e
        else:
            latency_ms = int((time.perf_counter() - start) * 1000)
            result = json.loads(response.read().decode('utf-8'))
            return {"data": result, "latency_ms": latency_ms}

    async def chat(self, messages: list[dict[str, str]], model: str, **kwargs) -> ProviderResponse:
        """Chat completion via LocalAI /v1/chat/completions."""
        payload = {"model": model, "messages": messages, **kwargs}
        result = await self._request("/v1/chat/completions", payload)
        data = result["data"]
        choice = data["choices"][0]
        return ProviderResponse(
            content=choice["message"]["content"],
            model=data.get("model", model),
            provider=self.name,
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=result["latency_ms"],
            raw=data
        )

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Generate embeddings via LocalAI /v1/embeddings."""
        payload = {"model": model, "input": texts}
        result = await self._request("/v1/embeddings", payload)
        return [item["embedding"] for item in result["data"]["data"]]

    async def health_check(self) -> bool:
        """Check if LocalAI is healthy via /v1/models."""
        try:
            url = f"{self.base_url}/v1/models"
            req = Request(url, headers={'Accept': 'application/json'})
            loop = asyncio.get_event_loop()
            def _fetch():
                return urlopen(req, timeout=5)
            response = await loop.run_in_executor(None, _fetch)
        except Exception:
            return False
        else:
            return response.status == 200