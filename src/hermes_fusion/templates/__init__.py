"""Prompt Templates - Jinja2-based templates with versioning, A/B testing, and provider optimization."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TemplateType(str, Enum):
    """Types of prompt templates."""
    SYSTEM = "system"
    USER = "user"
    CHAT = "chat"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REFLECTION = "reflection"
    CUSTOM = "custom"


class TemplateStatus(str, Enum):
    """Template lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class TemplateVariable:
    """Definition of a template variable."""
    name: str
    type: str = "string"  # string, number, boolean, list, object
    required: bool = True
    default: Any = None
    description: str = ""
    enum: list[Any] | None = None


@dataclass
class PromptTemplate:
    """A versioned prompt template with metadata."""
    name: str
    content: str
    version: str = "1.0.0"
    template_type: TemplateType = TemplateType.USER
    status: TemplateStatus = TemplateStatus.DRAFT

    # Metadata
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Variables
    variables: list[TemplateVariable] = field(default_factory=list)

    # Provider-specific overrides
    provider_overrides: dict[str, str] = field(default_factory=dict)  # provider -> template content

    # A/B testing
    ab_test_id: str | None = None
    ab_variant: str | None = None  # "A" or "B"
    ab_weight: float = 1.0  # 0.0-1.0 for traffic split

    # Performance tracking
    metrics: dict[str, float] = field(default_factory=dict)  # accuracy, latency, cost, etc.

    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.provider_overrides is None:
            self.provider_overrides = {}
        if self.metrics is None:
            self.metrics = {}

    def get_hash(self) -> str:
        """Get content hash for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "content": self.content,
            "version": self.version,
            "template_type": self.template_type.value,
            "status": self.status.value,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "variables": [
                {"name": v.name, "type": v.type, "required": v.required, "default": v.default,
                 "description": v.description, "enum": v.enum}
                for v in self.variables
            ],
            "provider_overrides": self.provider_overrides,
            "ab_test_id": self.ab_test_id,
            "ab_variant": self.ab_variant,
            "ab_weight": self.ab_weight,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptTemplate":
        """Deserialize from dictionary."""
        variables = [
            TemplateVariable(**v) for v in data.get("variables", [])
        ]
        return cls(
            name=data["name"],
            content=data["content"],
            version=data.get("version", "1.0.0"),
            template_type=TemplateType(data.get("template_type", "user")),
            status=TemplateStatus(data.get("status", "draft")),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            variables=variables,
            provider_overrides=data.get("provider_overrides", {}),
            ab_test_id=data.get("ab_test_id"),
            ab_variant=data.get("ab_variant"),
            ab_weight=data.get("ab_weight", 1.0),
            metrics=data.get("metrics", {}),
        )

    def render(self, variables: dict[str, Any], provider: str | None = None) -> str:
        """Render template with variables, using provider override if available."""
        from jinja2 import Environment, BaseLoader, StrictUndefined

        # Use provider-specific override if available
        content = self.content
        if provider and provider in self.provider_overrides:
            content = self.provider_overrides[provider]

        env = Environment(loader=BaseLoader(), undefined=StrictUndefined)
        template = env.from_string(content)
        return template.render(**variables)

    def validate_variables(self, variables: dict[str, Any]) -> list[str]:
        """Validate provided variables against template requirements. Returns list of errors."""
        errors = []
        provided = set(variables.keys())
        required = {v.name for v in self.variables if v.required}

        # Check missing required
        missing = required - provided
        if missing:
            errors.append(f"Missing required variables: {missing}")

        # Check unknown variables
        known = {v.name for v in self.variables}
        unknown = provided - known
        if unknown:
            errors.append(f"Unknown variables: {unknown}")

        # Validate enums
        for var in self.variables:
            if var.name in variables and var.enum is not None:
                if variables[var.name] not in var.enum:
                    errors.append(f"Variable '{var.name}' must be one of {var.enum}")

        return errors


@dataclass
class ABTestConfig:
    """Configuration for A/B testing a template."""
    test_id: str
    name: str
    description: str
    variant_a: PromptTemplate
    variant_b: PromptTemplate
    traffic_split: float = 0.5  # 0.0-1.0 for variant A
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "running"  # running, completed, stopped
    min_samples: int = 100
    confidence_threshold: float = 0.95

    def assign_variant(self, session_id: str) -> str:
        """Deterministically assign variant based on session ID."""
        hash_val = int(hashlib.md5(f"{self.test_id}:{session_id}".encode()).hexdigest(), 16)
        return "A" if (hash_val % 100) / 100 < self.traffic_split else "B"

    def get_variant(self, session_id: str) -> PromptTemplate:
        """Get the assigned variant template for a session."""
        variant = self.assign_variant(session_id)
        return self.variant_a if variant == "A" else self.variant_b


@dataclass
class TemplateMetrics:
    """Aggregated metrics for a template."""
    template_name: str
    version: str
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    avg_quality_score: float = 0.0
    last_used: float = 0.0
    provider_breakdown: dict[str, dict] = field(default_factory=dict)

    def record_use(self, latency_ms: float, cost_usd: float, quality: float, provider: str, success: bool = True):
        """Record a template usage."""
        self.total_uses += 1
        if success:
            self.successful_uses += 1
        else:
            self.failed_uses += 1

        # Rolling averages
        alpha = 0.1
        self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
        self.avg_cost_usd = (1 - alpha) * self.avg_cost_usd + alpha * cost_usd
        self.avg_quality_score = (1 - alpha) * self.avg_quality_score + alpha * quality
        self.last_used = time.time()

        # Provider breakdown
        if provider not in self.provider_breakdown:
            self.provider_breakdown[provider] = {"uses": 0, "latency": 0.0, "cost": 0.0, "quality": 0.0}
        p = self.provider_breakdown[provider]
        p["uses"] += 1
        p["latency"] = (1 - alpha) * p["latency"] + alpha * latency_ms
        p["cost"] = (1 - alpha) * p["cost"] + alpha * cost_usd
        p["quality"] = (1 - alpha) * p["quality"] + alpha * quality


@dataclass
class RenderResult:
    """Result of rendering a template."""
    template_name: str
    template_version: str
    rendered_content: str
    provider: str | None
    variables_used: dict[str, Any]
    ab_test_id: str | None = None
    ab_variant: str | None = None
    render_time_ms: float = 0.0