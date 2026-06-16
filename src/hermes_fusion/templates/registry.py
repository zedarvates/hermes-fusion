"""Template Registry - Storage, loading, validation, and management of prompt templates."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from hermes_fusion.templates import (
    ABTestConfig,
    PromptTemplate,
    TemplateMetrics,
    TemplateStatus,
    TemplateType,
)


class TemplateRegistry:
    """Registry for managing prompt templates with versioning and persistence."""

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        auto_save: bool = True,
    ):
        self.storage_dir = storage_dir or Path.home() / ".hermes_fusion" / "templates"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save
        self._templates: dict[str, PromptTemplate] = {}  # name -> template (latest version)
        self._versions: dict[str, dict[str, PromptTemplate]] = {}  # name -> version -> template
        self._ab_tests: dict[str, ABTestConfig] = {}
        self._metrics: dict[str, TemplateMetrics] = {}  # name_version -> metrics

        self._load_all()

    def _load_all(self):
        """Load all templates from storage."""
        # Templates
        for template_file in (self.storage_dir / "templates").glob("*.json") if (self.storage_dir / "templates").exists() else []:
            try:
                with open(template_file) as f:
                    data = json.load(f)
                template = PromptTemplate.from_dict(data)
                self._register_template(template)
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")

        # A/B tests
        ab_dir = self.storage_dir / "ab_tests"
        for ab_file in ab_dir.glob("*.json") if ab_dir.exists() else []:
            try:
                with open(ab_file) as f:
                    data = json.load(f)
                test = ABTestConfig(
                    test_id=data["test_id"],
                    name=data["name"],
                    description=data["description"],
                    variant_a=PromptTemplate.from_dict(data["variant_a"]),
                    variant_b=PromptTemplate.from_dict(data["variant_b"]),
                    traffic_split=data.get("traffic_split", 0.5),
                    start_time=data.get("start_time", time.time()),
                    end_time=data.get("end_time"),
                    status=data.get("status", "running"),
                    min_samples=data.get("min_samples", 100),
                    confidence_threshold=data.get("confidence_threshold", 0.95),
                )
                self._ab_tests[test.test_id] = test
            except Exception as e:
                print(f"Warning: Failed to load A/B test {ab_file}: {e}")

        # Metrics
        metrics_dir = self.storage_dir / "metrics"
        for metrics_file in metrics_dir.glob("*.json") if metrics_dir.exists() else []:
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                key = f"{data['template_name']}_{data['version']}"
                self._metrics[key] = TemplateMetrics(**data)
            except Exception as e:
                print(f"Warning: Failed to load metrics {metrics_file}: {e}")

    def _register_template(self, template: PromptTemplate):
        """Register a template in memory."""
        key = template.name
        version_key = template.version

        if key not in self._versions:
            self._versions[key] = {}

        self._versions[key][version_key] = template

        # Update latest pointer if this is newer or active
        current_latest = self._templates.get(key)
        if (
            current_latest is None
            or template.version > current_latest.version
            or (template.status == TemplateStatus.ACTIVE and current_latest.status != TemplateStatus.ACTIVE)
        ):
            self._templates[key] = template

    def _save_template(self, template: PromptTemplate):
        """Persist template to disk."""
        if not self.auto_save:
            return

        template_dir = self.storage_dir / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)

        filepath = template_dir / f"{template.name}_v{template.version}.json"
        with open(filepath, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

    def _save_ab_test(self, test: ABTestConfig):
        """Persist A/B test to disk."""
        if not self.auto_save:
            return

        ab_dir = self.storage_dir / "ab_tests"
        ab_dir.mkdir(parents=True, exist_ok=True)

        filepath = ab_dir / f"{test.test_id}.json"
        data = {
            "test_id": test.test_id,
            "name": test.name,
            "description": test.description,
            "variant_a": test.variant_a.to_dict(),
            "variant_b": test.variant_b.to_dict(),
            "traffic_split": test.traffic_split,
            "start_time": test.start_time,
            "end_time": test.end_time,
            "status": test.status,
            "min_samples": test.min_samples,
            "confidence_threshold": test.confidence_threshold,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _save_metrics(self, metrics: TemplateMetrics):
        """Persist metrics to disk."""
        if not self.auto_save:
            return

        metrics_dir = self.storage_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        filepath = metrics_dir / f"{metrics.template_name}_v{metrics.version}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(metrics), f, indent=2, default=str)

    # --- CRUD Operations ---

    def create_template(
        self,
        name: str,
        content: str,
        version: str = "1.0.0",
        template_type: TemplateType = TemplateType.USER,
        description: str = "",
        author: str = "",
        tags: list[str] | None = None,
        variables: list[dict] | None = None,
        provider_overrides: dict[str, str] | None = None,
    ) -> PromptTemplate:
        """Create a new template."""
        if name in self._templates:
            raise ValueError(f"Template '{name}' already exists. Use update_template() or create new version.")

        template_vars = []
        if variables:
            from hermes_fusion.templates import TemplateVariable
            template_vars = [TemplateVariable(**v) for v in variables]

        template = PromptTemplate(
            name=name,
            content=content,
            version=version,
            template_type=template_type,
            description=description,
            author=author,
            tags=tags or [],
            variables=template_vars,
            provider_overrides=provider_overrides or {},
        )

        self._register_template(template)
        self._save_template(template)

        # Initialize metrics
        metrics = TemplateMetrics(template_name=name, version=version)
        self._metrics[f"{name}_{version}"] = metrics
        self._save_metrics(metrics)

        return template

    def update_template(
        self,
        name: str,
        content: str | None = None,
        version: str | None = None,
        status: TemplateStatus | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        variables: list[dict] | None = None,
        provider_overrides: dict[str, str] | None = None,
    ) -> PromptTemplate:
        """Update an existing template (creates new version if content changes)."""
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found")

        # Determine if we need a new version
        new_version = version or template.version
        new_content = content or template.content

        if new_content != template.content:
            # Auto-increment version if not explicitly provided
            if version is None:
                parts = template.version.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                new_version = ".".join(parts)

            # Create new version
            new_template = PromptTemplate(
                name=name,
                content=new_content,
                version=new_version,
                template_type=template.template_type,
                status=status or TemplateStatus.DRAFT,
                description=description or template.description,
                author=template.author,
                tags=tags or template.tags[:],
                variables=[TemplateVariable(**v) for v in variables] if variables else template.variables[:],
                provider_overrides=provider_overrides or template.provider_overrides.copy(),
            )
        else:
            # Update in place
            new_template = template
            new_template.version = new_version
            if status:
                new_template.status = status
            if description is not None:
                new_template.description = description
            if tags is not None:
                new_template.tags = tags
            if variables is not None:
                from hermes_fusion.templates import TemplateVariable
                new_template.variables = [TemplateVariable(**v) for v in variables]
            if provider_overrides is not None:
                new_template.provider_overrides = provider_overrides
            new_template.updated_at = time.time()

        self._register_template(new_template)
        self._save_template(new_template)
        return new_template

    def get_template(self, name: str, version: str | None = None) -> Optional[PromptTemplate]:
        """Get a template by name (and optional version)."""
        if version:
            return self._versions.get(name, {}).get(version)
        return self._templates.get(name)

    def get_active_template(self, name: str) -> Optional[PromptTemplate]:
        """Get the active version of a template."""
        template = self.get_template(name)
        if template and template.status == TemplateStatus.ACTIVE:
            return template
        return None

    def list_templates(
        self,
        status: TemplateStatus | None = None,
        template_type: TemplateType | None = None,
        tag: str | None = None,
    ) -> list[PromptTemplate]:
        """List templates with optional filters."""
        templates = list(self._templates.values())

        if status:
            templates = [t for t in templates if t.status == status]
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        if tag:
            templates = [t for t in templates if tag in t.tags]

        return sorted(templates, key=lambda t: t.name)

    def list_versions(self, name: str) -> list[PromptTemplate]:
        """List all versions of a template."""
        versions = self._versions.get(name, {})
        return sorted(versions.values(), key=lambda t: t.version, reverse=True)

    def delete_template(self, name: str, version: str | None = None):
        """Delete a template (or specific version)."""
        if version:
            if name in self._versions and version in self._versions[name]:
                del self._versions[name][version]
                # Clean up file
                filepath = self.storage_dir / "templates" / f"{name}_v{version}.json"
                if filepath.exists():
                    filepath.unlink()
        else:
            # Delete all versions
            if name in self._versions:
                for v in self._versions[name].values():
                    filepath = self.storage_dir / "templates" / f"{name}_v{v.version}.json"
                    if filepath.exists():
                        filepath.unlink()
                del self._versions[name]
            if name in self._templates:
                del self._templates[name]

    # --- Rendering ---

    def render(
        self,
        template_name: str,
        variables: dict[str, Any],
        provider: str | None = None,
        version: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, PromptTemplate]:
        """Render a template with variables, handling A/B tests."""
        # Check for active A/B test
        ab_test = self._get_active_ab_test_for_template(template_name)
        if ab_test and session_id:
            variant = ab_test.assign_variant(session_id)
            template = ab_test.variant_a if variant == "A" else ab_test.variant_b
            rendered = template.render(variables, provider)
            return rendered, template

        # Normal rendering
        template = self.get_template(template_name, version)
        if template is None:
            raise ValueError(f"Template '{template_name}' not found")

        errors = template.validate_variables(variables)
        if errors:
            raise ValueError(f"Variable validation failed: {errors}")

        rendered = template.render(variables, provider)
        return rendered, template

    def _get_active_ab_test_for_template(self, template_name: str) -> Optional[ABTestConfig]:
        """Find active A/B test for a template."""
        for test in self._ab_tests.values():
            if test.status == "running" and (
                test.variant_a.name == template_name or test.variant_b.name == template_name
            ):
                if test.end_time is None or test.end_time > time.time():
                    return test
        return None

    # --- A/B Testing ---

    def create_ab_test(
        self,
        name: str,
        description: str,
        variant_a: PromptTemplate,
        variant_b: PromptTemplate,
        traffic_split: float = 0.5,
        min_samples: int = 100,
        confidence_threshold: float = 0.95,
    ) -> ABTestConfig:
        """Create an A/B test between two template variants."""
        test_id = f"ab_{name}_{int(time.time())}"
        test = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            variant_a=variant_a,
            variant_b=variant_b,
            traffic_split=traffic_split,
            min_samples=min_samples,
            confidence_threshold=confidence_threshold,
        )

        # Mark templates as part of A/B test
        variant_a.ab_test_id = test_id
        variant_a.ab_variant = "A"
        variant_b.ab_test_id = test_id
        variant_b.ab_variant = "B"

        self._ab_tests[test_id] = test
        self._save_ab_test(test)
        self._save_template(variant_a)
        self._save_template(variant_b)

        return test

    def get_ab_test(self, test_id: str) -> Optional[ABTestConfig]:
        """Get an A/B test by ID."""
        return self._ab_tests.get(test_id)

    def list_ab_tests(self, status: str | None = None) -> list[ABTestConfig]:
        """List A/B tests."""
        tests = list(self._ab_tests.values())
        if status:
            tests = [t for t in tests if t.status == status]
        return sorted(tests, key=lambda t: t.start_time, reverse=True)

    def stop_ab_test(self, test_id: str, winner: str | None = None):
        """Stop an A/B test, optionally declaring a winner."""
        test = self._ab_tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test '{test_id}' not found")

        test.status = "completed"
        test.end_time = time.time()
        self._save_ab_test(test)

        if winner:
            # Promote winner to active
            winner_template = test.variant_a if winner == "A" else test.variant_b
            loser_template = test.variant_b if winner == "A" else test.variant_a

            self.update_template(
                winner_template.name,
                version=winner_template.version,
                status=TemplateStatus.ACTIVE,
            )
            self.update_template(
                loser_template.name,
                version=loser_template.version,
                status=TemplateStatus.DEPRECATED,
            )

    # --- Metrics ---

    def record_usage(
        self,
        template_name: str,
        version: str,
        latency_ms: float,
        cost_usd: float,
        quality_score: float,
        provider: str,
        success: bool = True,
    ):
        """Record template usage metrics."""
        key = f"{template_name}_{version}"
        if key not in self._metrics:
            self._metrics[key] = TemplateMetrics(template_name=template_name, version=version)

        self._metrics[key].record_use(latency_ms, cost_usd, quality_score, provider, success)
        self._save_metrics(self._metrics[key])

    def get_metrics(self, template_name: str, version: str | None = None) -> Optional[TemplateMetrics]:
        """Get metrics for a template."""
        if version:
            return self._metrics.get(f"{template_name}_{version}")
        # Return latest version metrics
        template = self.get_template(template_name)
        if template:
            return self._metrics.get(f"{template_name}_{template.version}")
        return None

    def get_best_template_for_provider(
        self,
        template_name: str,
        provider: str,
        metric: str = "quality",  # quality, latency, cost
    ) -> Optional[PromptTemplate]:
        """Find the best template version for a specific provider based on metrics."""
        versions = self.list_versions(template_name)
        best = None
        best_score = float("-inf") if metric == "quality" else float("inf")

        for template in versions:
            metrics = self.get_metrics(template_name, template.version)
            if not metrics or provider not in metrics.provider_breakdown:
                continue

            p = metrics.provider_breakdown[provider]
            score = p.get(metric, 0)

            if metric == "quality":
                if score > best_score:
                    best_score = score
                    best = template
            else:  # latency, cost - lower is better
                if score < best_score and score > 0:
                    best_score = score
                    best = template

        return best

    # --- Validation ---

    def validate_template(self, template: PromptTemplate) -> list[str]:
        """Validate a template for syntax and variable consistency."""
        errors = []

        # Test Jinja2 syntax
        try:
            from jinja2 import Environment, BaseLoader, StrictUndefined
            env = Environment(loader=BaseLoader(), undefined=StrictUndefined)
            env.from_string(template.content)
        except Exception as e:
            errors.append(f"Jinja2 syntax error: {e}")

        # Validate provider overrides
        for provider, content in template.provider_overrides.items():
            try:
                env = Environment(loader=BaseLoader(), undefined=StrictUndefined)
                env.from_string(content)
            except Exception as e:
                errors.append(f"Provider '{provider}' override syntax error: {e}")

        return errors

    # --- Export/Import ---

    def export_template(self, name: str, version: str | None = None) -> dict:
        """Export template as portable dictionary."""
        template = self.get_template(name, version)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        return template.to_dict()

    def import_template(self, data: dict, overwrite: bool = False) -> PromptTemplate:
        """Import template from dictionary."""
        template = PromptTemplate.from_dict(data)
        if template.name in self._templates and not overwrite:
            raise ValueError(f"Template '{template.name}' exists. Use overwrite=True.")
        self._register_template(template)
        self._save_template(template)
        return template

    def migrate_from_file(self, filepath: Path):
        """Migrate templates from a JSON/YAML file."""
        with open(filepath) as f:
            if filepath.suffix in (".yaml", ".yml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        if isinstance(data, list):
            for item in data:
                self.import_template(item)
        elif isinstance(data, dict):
            self.import_template(data)