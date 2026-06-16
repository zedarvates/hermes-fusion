"""Provider Optimization - Model-specific template optimization and selection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hermes_fusion.templates.registry import TemplateRegistry
from hermes_fusion.templates import PromptTemplate, TemplateType, TemplateStatus


@dataclass
class ProviderProfile:
    """Profile of a provider/model for template optimization."""
    provider: str
    model: str
    capabilities: list[str] = field(default_factory=list)  # e.g., ["function_calling", "vision", "reasoning"]
    context_window: int = 4096
    preferred_format: str = "chat"  # chat, completion, instruct
    system_prompt_style: str = "default"  # default, minimal, detailed, persona
    temperature_range: tuple[float, float] = (0.0, 1.0)
    max_tokens_default: int = 1024
    stop_sequences: list[str] = field(default_factory=list)
    special_tokens: dict[str, str] = field(default_factory=dict)  # e.g., {"bos": "<s>", "eos": "</s>"}


# Built-in provider profiles
PROVIDER_PROFILES = {
    "openrouter": {
        "anthropic/claude-sonnet-4": ProviderProfile(
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
            capabilities=["reasoning", "function_calling", "vision", "long_context"],
            context_window=200000,
            preferred_format="chat",
            system_prompt_style="detailed",
            temperature_range=(0.0, 1.0),
            max_tokens_default=4096,
        ),
        "openai/gpt-5.5": ProviderProfile(
            provider="openrouter",
            model="openai/gpt-5.5",
            capabilities=["reasoning", "function_calling", "vision", "long_context"],
            context_window=128000,
            preferred_format="chat",
            system_prompt_style="default",
            temperature_range=(0.0, 2.0),
            max_tokens_default=4096,
        ),
        "nvidia/nemotron-3-ultra": ProviderProfile(
            provider="openrouter",
            model="nvidia/nemotron-3-ultra",
            capabilities=["reasoning"],
            context_window=8192,
            preferred_format="chat",
            system_prompt_style="minimal",
            temperature_range=(0.0, 1.0),
            max_tokens_default=2048,
        ),
    },
    "localai": {
        "gemma-4-e2b-it": ProviderProfile(
            provider="localai",
            model="gemma-4-e2b-it",
            capabilities=["chat", "reasoning"],
            context_window=8192,
            preferred_format="chat",
            system_prompt_style="minimal",
            temperature_range=(0.1, 1.0),
            max_tokens_default=2048,
        ),
        "llama-3.1-8b": ProviderProfile(
            provider="localai",
            model="llama-3.1-8b",
            capabilities=["chat", "reasoning", "long_context"],
            context_window=131072,
            preferred_format="instruct",
            system_prompt_style="default",
            temperature_range=(0.1, 1.0),
            max_tokens_default=2048,
            special_tokens={"bos": "<|begin_of_text|>", "eos": "<|end_of_text|>", "user": "<|start_header_id|>user<|end_header_id|>", "assistant": "<|start_header_id|>assistant<|end_header_id|>"},
        ),
    },
    "minimax": {
        "MiniMax-M3": ProviderProfile(
            provider="minimax",
            model="MiniMax-M3",
            capabilities=["chat", "long_context"],
            context_window=1000000,
            preferred_format="chat",
            system_prompt_style="default",
            temperature_range=(0.0, 1.0),
            max_tokens_default=4096,
        ),
    },
}


class ProviderOptimizer:
    """Optimizes templates for specific providers/models."""

    def __init__(
        self,
        registry: TemplateRegistry,
        profiles: dict[str, dict[str, ProviderProfile]] | None = None,
    ):
        self.registry = registry
        self.profiles = profiles or PROVIDER_PROFILES

    def get_profile(self, provider: str, model: str) -> Optional[ProviderProfile]:
        """Get provider profile."""
        return self.profiles.get(provider, {}).get(model)

    def optimize_template(
        self,
        template: PromptTemplate,
        provider: str,
        model: str,
    ) -> PromptTemplate:
        """Create a provider-optimized version of a template."""
        profile = self.get_profile(provider, model)
        if not profile:
            return template  # No optimization available

        # Check if override already exists
        if provider in template.provider_overrides:
            return template  # Already optimized

        optimized_content = self._apply_optimizations(template.content, profile)
        if optimized_content == template.content:
            return template

        # Create new version with provider override
        from hermes_fusion.templates import TemplateVariable
        new_template = PromptTemplate(
            name=template.name,
            content=template.content,
            version=template.version,
            template_type=template.template_type,
            status=template.status,
            description=template.description,
            author=template.author,
            tags=template.tags[:],
            variables=template.variables[:],
            provider_overrides={**template.provider_overrides, provider: optimized_content},
        )
        return new_template

    def _apply_optimizations(self, content: str, profile: ProviderProfile) -> str:
        """Apply provider-specific optimizations to template content."""
        optimized = content

        # 1. Adjust system prompt style
        if "system" in content.lower() or "system:" in content.lower():
            optimized = self._adjust_system_prompt(optimized, profile)

        # 2. Add model-specific tokens if needed
        if profile.special_tokens:
            optimized = self._add_special_tokens(optimized, profile)

        # 3. Optimize for context window
        if profile.context_window < 8192:
            optimized = self._optimize_for_short_context(optimized, profile)

        # 4. Add capability hints
        optimized = self._add_capability_hints(optimized, profile)

        return optimized

    def _adjust_system_prompt(self, content: str, profile: ProviderProfile) -> str:
        """Adjust system prompt style for provider."""
        # This is a simplified version - in practice you'd use Jinja2 to rewrite
        # For now, return as-is; the template author should use provider_overrides
        return content

    def _add_special_tokens(self, content: str, profile: ProviderProfile) -> str:
        """Add model-specific special tokens."""
        # Again, this is template-specific. Best done via provider_overrides in the template.
        return content

    def _optimize_for_short_context(self, content: str, profile: ProviderProfile) -> str:
        """Optimize template for short context windows."""
        # Could truncate few-shot examples, simplify instructions
        return content

    def _add_capability_hints(self, content: str, profile: ProviderProfile) -> str:
        """Add hints about model capabilities."""
        return content

    def auto_create_overrides(
        self,
        template_name: str,
        providers: list[tuple[str, str]],  # [(provider, model), ...]
    ) -> PromptTemplate:
        """Auto-create provider overrides for a template using heuristics."""
        template = self.registry.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        new_overrides = {}
        for provider, model in providers:
            if provider in template.provider_overrides:
                continue  # Already has override

            profile = self.get_profile(provider, model)
            if profile:
                # Apply simple heuristics
                override_content = self._generate_override(template.content, profile)
                if override_content != template.content:
                    new_overrides[provider] = override_content

        if new_overrides:
            return self.registry.update_template(
                template_name,
                provider_overrides={**template.provider_overrides, **new_overrides},
            )
        return template

    def _generate_override(self, content: str, profile: ProviderProfile) -> str:
        """Generate provider-specific override using heuristics."""
        # Simple heuristic-based generation
        # In practice, this could use an LLM to rewrite
        override = content

        # Add provider-specific formatting hints as comments
        if profile.preferred_format == "instruct":
            override = f"# Optimized for {profile.provider}/{profile.model}\n{override}"

        return override


class TemplateSelector:
    """Selects best template for a given task/provider/model combination."""

    def __init__(self, registry: TemplateRegistry, optimizer: ProviderOptimizer):
        self.registry = registry
        self.optimizer = optimizer

    def select_template(
        self,
        task_type: str,
        provider: str,
        model: str,
        preferences: dict[str, Any] | None = None,
    ) -> Optional[PromptTemplate]:
        """Select the best template for a task/provider/model."""
        prefs = preferences or {}

        # Find templates matching task type
        candidates = self.registry.list_templates(
            template_type=TemplateType(task_type) if task_type in [t.value for t in TemplateType] else None,
            status=TemplateStatus.ACTIVE,
        )

        if not candidates:
            # Fallback: any active template
            candidates = self.registry.list_templates(status=TemplateStatus.ACTIVE)

        if not candidates:
            return None

        # Score candidates
        scored = []
        for template in candidates:
            score = self._score_template(template, provider, model, prefs)
            scored.append((score, template))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def _score_template(
        self,
        template: PromptTemplate,
        provider: str,
        model: str,
        preferences: dict[str, Any],
    ) -> float:
        """Score a template for a specific provider/model."""
        score = 0.0

        # 1. Provider override exists
        if provider in template.provider_overrides:
            score += 10.0

        # 2. Metrics for this provider
        metrics = self.registry.get_metrics(template.name, template.version)
        if metrics and provider in metrics.provider_breakdown:
            p = metrics.provider_breakdown[provider]
            # Quality weight
            score += p.get("quality", 0) * 5.0
            # Latency weight (lower is better)
            latency = p.get("latency", 0)
            if latency > 0:
                score += max(0, 5.0 - latency / 1000.0)
            # Cost weight (lower is better)
            cost = p.get("cost", 0)
            if cost > 0:
                score += max(0, 3.0 - cost * 1000.0)

        # 3. Task type match
        if template.template_type.value in preferences.get("preferred_types", []):
            score += 5.0

        # 4. Tag match
        preferred_tags = preferences.get("tags", [])
        for tag in preferred_tags:
            if tag in template.tags:
                score += 2.0

        # 5. A/B test variant - prefer active tests
        if template.ab_test_id and template.ab_variant:
            ab_test = self.registry.get_ab_test(template.ab_test_id)
            if ab_test and ab_test.status == "running":
                score += 3.0

        return score

    def render_with_selection(
        self,
        task_type: str,
        variables: dict[str, Any],
        provider: str,
        model: str,
        session_id: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> tuple[str, PromptTemplate]:
        """Select and render best template."""
        template = self.select_template(task_type, provider, model, preferences)
        if not template:
            raise ValueError(f"No suitable template for task '{task_type}' on {provider}/{model}")

        # Optimize if needed
        optimized = self.optimizer.optimize_template(template, provider, model)

        # Render
        rendered = optimized.render(variables, provider)
        return rendered, optimized


def get_default_templates() -> list[PromptTemplate]:
    """Get a set of default templates for common tasks."""
    from hermes_fusion.templates import PromptTemplate, TemplateVariable, TemplateType, TemplateStatus

    return [
        PromptTemplate(
            name="code_review",
            content="""You are an expert code reviewer. Analyze the following code for:
1. Correctness & logic errors
2. Performance issues
3. Security vulnerabilities
4. Code style & maintainability
5. Test coverage gaps

Code to review:
{{ code }}

{% if context %}
Context: {{ context }}
{% endif %}

Provide specific, actionable feedback with line numbers where possible.""",
            version="1.0.0",
            template_type=TemplateType.USER,
            status=TemplateStatus.ACTIVE,
            description="Code review template for PR analysis",
            tags=["code", "review", "security", "performance"],
            variables=[
                TemplateVariable(name="code", type="string", required=True, description="Code to review"),
                TemplateVariable(name="context", type="string", required=False, description="Additional context"),
            ],
        ),
        PromptTemplate(
            name="complex_reasoning",
            content="""You are a careful reasoning engine. Solve the problem step by step.

Problem: {{ question }}

{% if context %}
Context: {{ context }}
{% endif %}

Think through this systematically:
1. Identify key information and constraints
2. Break down into sub-problems
3. Solve each sub-problem
4. Synthesize final answer

Show your reasoning clearly.""",
            version="1.0.0",
            template_type=TemplateType.CHAIN_OF_THOUGHT,
            status=TemplateStatus.ACTIVE,
            description="Chain-of-thought template for complex reasoning",
            tags=["reasoning", "cot", "analysis"],
            variables=[
                TemplateVariable(name="question", type="string", required=True),
                TemplateVariable(name="context", type="string", required=False),
            ],
        ),
        PromptTemplate(
            name="creative_writing",
            content="""Write a {{ format }} about {{ topic }}.

Style: {{ style }}
Tone: {{ tone }}
Length: {{ length }}

{% if constraints %}
Constraints:
{% for c in constraints %}
- {{ c }}
{% endfor %}
{% endif %}""",
            version="1.0.0",
            template_type=TemplateType.USER,
            status=TemplateStatus.ACTIVE,
            description="Creative writing template with structured parameters",
            tags=["creative", "writing", "story"],
            variables=[
                TemplateVariable(name="format", type="string", required=True, enum=["story", "poem", "essay", "script"]),
                TemplateVariable(name="topic", type="string", required=True),
                TemplateVariable(name="style", type="string", required=False, default="literary"),
                TemplateVariable(name="tone", type="string", required=False, default="neutral"),
                TemplateVariable(name="length", type="string", required=False, default="medium"),
                TemplateVariable(name="constraints", type="list", required=False, default=[]),
            ],
        ),
        PromptTemplate(
            name="data_analysis",
            content="""Analyze the following data and provide insights.

Data: {{ data }}

{% if question %}
Question: {{ question }}
{% endif %}

{% if context %}
Context: {{ context }}
{% endif %}

Provide:
1. Summary statistics
2. Key patterns/trends
3. Anomalies or outliers
4. Actionable recommendations""",
            version="1.0.0",
            template_type=TemplateType.USER,
            status=TemplateStatus.ACTIVE,
            description="Data analysis template",
            tags=["data", "analysis", "insights"],
            variables=[
                TemplateVariable(name="data", type="string", required=True),
                TemplateVariable(name="question", type="string", required=False),
                TemplateVariable(name="context", type="string", required=False),
            ],
        ),
        PromptTemplate(
            name="system_hermes",
            content="""You are Hnoss, an advanced AI assistant for Sylvain Galliez. You operate from a distributed cluster (GLYPH WSL, EUREKAI GPU server, ODIN-PC Windows).

Your principles (Shinto + Bushido + Catholic):
1. Respect Nature - prefer local, efficient, resilient solutions
2. Purity - clean code, clear architecture, no patches
3. Harmony & Charity - work with existing ecosystem, reuse before rebuild
4. Loyalty & Faith - protect user's interests, data, infrastructure
5. Courage & Forgiveness - take initiative, admit errors immediately
6. Respect & Dignity - respect user's time, intelligence, autonomy
7. Honor & Humility - deliver results, under-promise, over-deliver

Communication: French-first, direct/warm/technical, concise but complete.
Technical depth welcomed. Local-first, resilient, token-efficient.
Current stack: xAI Grok, TurboQuant, hnoss, Hailo-8, ComfyUI, Qdrant, LocalAI.

Answer the user's question following these principles.""",
            version="1.0.0",
            template_type=TemplateType.SYSTEM,
            status=TemplateStatus.ACTIVE,
            description="System prompt for Hermes agent persona",
            tags=["system", "persona", "hermes"],
            variables=[],
        ),
    ]