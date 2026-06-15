"""Cloud LLM providers (xAI Grok, OpenAI GPT-4o, Anthropic Claude)."""

import os
from typing import Any
from hermes_fusion.config import CloudConfig
from hermes_fusion.providers.base import Provider, ProviderResponse


class CloudProvider(Provider):
    """Base class for cloud LLM providers using OpenAI-compatible APIs."""
    
    def __init__(self, config: CloudConfig, provider_name: str):
        self.config = config
        self.provider_name = provider_name
        self._client = None

    def _get_client(self):
        """Lazy-load the appropriate client."""
        if self._client is not None:
            return self._client
        
        # Import inside to avoid hard dependency
        try:
            if self.provider_name == "xai":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._get_api_key("XAI_API_KEY"),
                    base_url="https://api.x.ai/v1",
                )
            elif self.provider_name == "openai":
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._get_api_key("OPENAI_API_KEY"),
                )
            elif self.provider_name == "anthropic":
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self._get_api_key("ANTHROPIC_API_KEY"),
                )
        except ImportError as e:
            raise RuntimeError(f"{self.provider_name} client not installed: {e}")
        
        return self._client

    def _get_api_key(self, env_var: str) -> str:
        """Get API key from CloudConfig or environment."""
        # Try config first (injected), then env, then raise
        config_key = getattr(self.config, f"{self.provider_name}_api_key", None)
        if config_key:
            return config_key
        env_key = os.getenv(env_var)
        if env_key:
            return env_key
        raise RuntimeError(f"{self.provider_name} API key not configured. Set {env_var} env var or config.{self.provider_name}_api_key")

    def _get_model(self, model: str | None = None) -> str:
        """Get model name from parameter or config."""
        if model:
            return model
        config_model = getattr(self.config, f"{self.provider_name}_model", None)
        if config_model:
            return config_model
        # Defaults
        defaults = {"xai": "grok-3", "openai": "gpt-4o", "anthropic": "claude-3-5-sonnet"}
        return defaults.get(self.provider_name, "gpt-4o")

    async def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> ProviderResponse:
        raise NotImplementedError

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check API connectivity."""
        client = self._get_client()
        try:
            if self.provider_name in ("xai", "openai"):
                await client.models.list()
            elif self.provider_name == "anthropic":
                # Anthropic doesn't have models.list, try a minimal call
                await client.messages.create(
                    model=self._get_model(),
                    max_tokens=1,
                    messages=[{"role": "user", "content": "ping"}],
                )
            return True
        except Exception:
            return False


class XAIProvider(CloudProvider):
    """xAI Grok provider (OpenAI-compatible API)."""
    name = "xai"

    def __init__(self, config: CloudConfig):
        super().__init__(config, "xai")

    async def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> ProviderResponse:
        client = self._get_client()
        model_name = self._get_model(model)
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
        
        return ProviderResponse(
            content=response.choices[0].message.content,
            provider=self.name,
            model=model_name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            raw=response,
        )

    async def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        # xAI doesn't have embeddings, delegate to OpenAI
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY") or self.config.openai_api_key)
        response = await client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in response.data]


class OpenAIProvider(CloudProvider):
    """OpenAI GPT-4o provider."""
    name = "openai"

    def __init__(self, config: CloudConfig):
        super().__init__(config, "openai")

    async def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> ProviderResponse:
        client = self._get_client()
        model_name = self._get_model(model)
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
        
        return ProviderResponse(
            content=response.choices[0].message.content,
            provider=self.name,
            model=model_name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            raw=response,
        )

    async def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        client = self._get_client()
        response = await client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in response.data]


class AnthropicProvider(CloudProvider):
    """Anthropic Claude 3.5 Sonnet provider."""
    name = "anthropic"

    def __init__(self, config: CloudConfig):
        super().__init__(config, "anthropic")

    async def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs) -> ProviderResponse:
        client = self._get_client()
        model_name = self._get_model(model)
        
        # Convert OpenAI format to Anthropic format
        system_msg = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = await client.messages.create(
            model=model_name,
            messages=anthropic_messages,
            system=system_msg,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
        )
        
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        
        return ProviderResponse(
            content=response.content[0].text if response.content else "",
            provider=self.name,
            model=model_name,
            tokens_used=input_tokens + output_tokens,
            raw=response,
        )

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        # Anthropic doesn't have embeddings, delegate to OpenAI
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY") or self.config.openai_api_key)
        response = await client.embeddings.create(model=model or "text-embedding-3-small", input=texts)
        return [d.embedding for d in response.data]