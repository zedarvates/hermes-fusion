"""Fusion Configuration - TOML-based config for local + cloud providers."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class FusionSettings:
    """Global fusion engine settings."""
    default_strategy: str = "weighted_vote"
    max_parallel_providers: int = 3
    timeout_seconds: int = 120
    semantic_cache_enabled: bool = True
    cache_ttl_hours: int = 24


@dataclass(slots=True)
class LocalAIConfig:
    """LocalAI provider configuration (EUREKAI:8080)."""
    base_url: str = "http://192.168.1.47:8080"
    models: list[str] = field(default_factory=lambda: ["gemma-4-e2b-it:latest", "gpt-4o", "whisper-1", "tts-1"])
    timeout: int = 60


@dataclass(slots=True)
class HailoConfig:
    """Hailo-8 AI accelerator configuration (EUREKAI:8767)."""
    host: str = "192.168.1.47"
    port: int = 8767
    models: list[str] = field(default_factory=lambda: ["resnet18", "yolov8", "ocr"])
    timeout: int = 30


@dataclass(slots=True)
class QdrantConfig:
    """Qdrant vector database configuration (EUREKAI:6333)."""
    url: str = "http://192.168.1.47:6333"
    collection: str = "hermes_fusion_cache"
    vector_size: int = 768


@dataclass(slots=True)
class CloudProviderConfig:
    """Cloud LLM provider configuration."""
    api_key_env: str
    model: str
    weight: float = 1.0
    timeout: int = 60


@dataclass(slots=True)
class CloudConfig:
    """Cloud providers container."""
    xai: CloudProviderConfig | None = None
    openai: CloudProviderConfig | None = None
    anthropic: CloudProviderConfig | None = None
    custom: dict[str, CloudProviderConfig] = field(default_factory=dict)


@dataclass(slots=True)
class ProvidersConfig:
    """All provider configurations."""
    localai: LocalAIConfig = field(default_factory=LocalAIConfig)
    hailo: HailoConfig = field(default_factory=HailoConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)


@dataclass(slots=True)
class FusionConfig:
    """Main fusion configuration loaded from TOML."""
    fusion: FusionSettings = field(default_factory=FusionSettings)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "FusionConfig":
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Handle both flat and nested TOML structures
        fusion_data = data.get("fusion", {})
        # Providers can be under [fusion.providers] or [providers]
        providers_data = data.get("fusion", {}).get("providers", data.get("providers", {}))

        # Filter out nested tables from fusion_data (like "providers" key if it's a dict)
        fusion_kwargs = {k: v for k, v in fusion_data.items() if not isinstance(v, dict)}
        fusion = FusionSettings(**fusion_kwargs)

        localai_data = providers_data.get("localai", {})
        localai = LocalAIConfig(**localai_data)

        hailo_data = providers_data.get("hailo", {})
        hailo = HailoConfig(**hailo_data)

        qdrant_data = providers_data.get("qdrant", {})
        qdrant = QdrantConfig(**qdrant_data)

        cloud_data = providers_data.get("cloud", {})
        cloud = CloudConfig()
        for key in ("xai", "openai", "anthropic"):
            if key in cloud_data:
                setattr(cloud, key, CloudProviderConfig(**cloud_data[key]))
        for key, val in cloud_data.items():
            if key not in ("xai", "openai", "anthropic"):
                cloud.custom[key] = CloudProviderConfig(**val)

        return cls(fusion=fusion, providers=ProvidersConfig(
            localai=localai,
            hailo=hailo,
            qdrant=qdrant,
            cloud=cloud
        ))

    def get_cloud_api_key(self, provider: str) -> str | None:
        """Get API key for a cloud provider from environment."""
        cloud_provider = getattr(self.providers.cloud, provider, None)
        if not cloud_provider:
            for _, cp in self.providers.cloud.custom.items():
                if cp.model == provider:
                    return os.getenv(cp.api_key_env)
            return None
        return os.getenv(cloud_provider.api_key_env)