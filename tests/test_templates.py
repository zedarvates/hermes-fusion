"""Tests for Prompt Templates."""

import pytest
import tempfile
from pathlib import Path

from hermes_fusion.templates import (
    PromptTemplate,
    TemplateVariable,
    TemplateType,
    TemplateStatus,
    ABTestConfig,
    TemplateMetrics,
    RenderResult,
)
from hermes_fusion.templates.registry import TemplateRegistry
from hermes_fusion.templates.optimization import (
    ProviderProfile,
    ProviderOptimizer,
    TemplateSelector,
    PROVIDER_PROFILES,
    get_default_templates,
)


class TestPromptTemplate:
    """Test PromptTemplate core functionality."""

    def test_template_creation(self):
        """Test basic template creation."""
        template = PromptTemplate(
            name="test",
            content="Hello {{ name }}!",
            version="1.0.0",
        )
        assert template.name == "test"
        assert template.content == "Hello {{ name }}!"
        assert template.version == "1.0.0"
        assert template.status == TemplateStatus.DRAFT

    def test_template_render(self):
        """Test template rendering with variables."""
        template = PromptTemplate(
            name="greeting",
            content="Hello {{ name }}, you are {{ age }} years old!",
        )
        rendered = template.render({"name": "Alice", "age": 30})
        assert rendered == "Hello Alice, you are 30 years old!"

    def test_template_render_with_missing_var(self):
        """Test template rendering fails with missing required variable."""
        template = PromptTemplate(
            name="greeting",
            content="Hello {{ name }}!",
        )
        with pytest.raises(Exception):  # jinja2.exceptions.UndefinedError
            template.render({})

    def test_variable_validation(self):
        """Test variable validation."""
        template = PromptTemplate(
            name="test",
            content="Hello {{ name }}!",
            variables=[
                TemplateVariable(name="name", type="string", required=True),
                TemplateVariable(name="age", type="number", required=False, default=18),
            ],
        )
        # Should pass with required variable
        errors = template.validate_variables({"name": "Alice"})
        assert len(errors) == 0

        # Should fail with missing required
        errors = template.validate_variables({})
        assert "Missing required variables" in errors[0]

        # Should fail with unknown variable
        errors = template.validate_variables({"name": "Alice", "unknown": "value"})
        assert "Unknown variables" in errors[0]

    def test_enum_validation(self):
        """Test enum validation for variables."""
        template = PromptTemplate(
            name="test",
            content="Format: {{ format }}",
            variables=[
                TemplateVariable(name="format", type="string", required=True, enum=["json", "xml", "yaml"]),
            ],
        )
        errors = template.validate_variables({"format": "json"})
        assert len(errors) == 0

        errors = template.validate_variables({"format": "csv"})
        assert "must be one of" in errors[0]

    def test_serialization(self):
        """Test template to_dict/from_dict roundtrip."""
        template = PromptTemplate(
            name="test",
            content="Hello {{ name }}!",
            version="2.0.0",
            template_type=TemplateType.SYSTEM,
            status=TemplateStatus.ACTIVE,
            description="Test template",
            author="Test Author",
            tags=["test", "demo"],
            variables=[
                TemplateVariable(name="name", type="string", required=True, description="User name"),
            ],
            provider_overrides={"openrouter": "Hello {{ name }} from OpenRouter!"},
        )

        data = template.to_dict()
        restored = PromptTemplate.from_dict(data)

        assert restored.name == template.name
        assert restored.content == template.content
        assert restored.version == template.version
        assert restored.template_type == template.template_type
        assert restored.status == template.status
        assert restored.description == template.description
        assert restored.author == template.author
        assert restored.tags == template.tags
        assert len(restored.variables) == len(template.variables)
        assert restored.provider_overrides == template.provider_overrides


class TestTemplateRegistry:
    """Test TemplateRegistry functionality."""

    def test_create_and_get_template(self, tmp_path):
        """Test creating and retrieving templates."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        template = registry.create_template(
            name="test",
            content="Hello {{ name }}!",
            template_type=TemplateType.USER,
            description="Test template",
        )

        assert template.name == "test"
        assert template.version == "1.0.0"

        # Get by name
        retrieved = registry.get_template("test")
        assert retrieved is not None
        assert retrieved.content == "Hello {{ name }}!"

    def test_update_template_creates_new_version(self, tmp_path):
        """Test updating template creates new version."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        registry.create_template(name="test", content="v1")
        template = registry.update_template("test", content="v2")

        assert template.version == "1.0.1"
        assert template.content == "v2"

        # Old version should still exist
        v1 = registry.get_template("test", version="1.0.0")
        assert v1.content == "v1"

    def test_status_transition(self, tmp_path):
        """Test template status changes."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        registry.create_template(name="test", content="Hello")
        registry.update_template("test", status=TemplateStatus.ACTIVE)

        template = registry.get_template("test")
        assert template.status == TemplateStatus.ACTIVE

    def test_list_templates_with_filters(self, tmp_path):
        """Test listing templates with filters."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        registry.create_template(name="a", content="a", template_type=TemplateType.USER, tags=["tag1"])
        registry.create_template(name="b", content="b", template_type=TemplateType.SYSTEM, tags=["tag2"])
        registry.update_template("b", status=TemplateStatus.ACTIVE)

        # Filter by status
        active = registry.list_templates(status=TemplateStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "b"

        # Filter by type
        systems = registry.list_templates(template_type=TemplateType.SYSTEM)
        assert len(systems) == 1
        assert systems[0].name == "b"

        # Filter by tag
        tagged = registry.list_templates(tag="tag1")
        assert len(tagged) == 1
        assert tagged[0].name == "a"

    def test_render_with_provider_override(self, tmp_path):
        """Test rendering uses provider override."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        registry.create_template(
            name="test",
            content="Default",
            provider_overrides={"openrouter": "OpenRouter version"},
        )

        rendered, template = registry.render("test", {}, provider="openrouter")
        assert rendered == "OpenRouter version"

        rendered, template = registry.render("test", {}, provider="localai")
        assert rendered == "Default"

    def test_ab_test_creation(self, tmp_path):
        """Test A/B test creation."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        # Create two active templates
        registry.create_template(name="var_a", content="A", template_type=TemplateType.USER)
        registry.update_template("var_a", status=TemplateStatus.ACTIVE)
        registry.create_template(name="var_b", content="B", template_type=TemplateType.USER)
        registry.update_template("var_b", status=TemplateStatus.ACTIVE)

        var_a = registry.get_active_template("var_a")
        var_b = registry.get_active_template("var_b")

        test = registry.create_ab_test(
            name="test_ab",
            description="Test A/B",
            variant_a=var_a,
            variant_b=var_b,
            traffic_split=0.5,
        )

        assert test.test_id.startswith("ab_test_ab_")
        assert test.variant_a.ab_test_id == test.test_id
        assert test.variant_a.ab_variant == "A"
        assert test.variant_b.ab_variant == "B"

    def test_ab_test_variant_assignment(self, tmp_path):
        """Test deterministic variant assignment."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        registry.create_template(name="var_a", content="A", template_type=TemplateType.USER)
        registry.update_template("var_a", status=TemplateStatus.ACTIVE)
        registry.create_template(name="var_b", content="B", template_type=TemplateType.USER)
        registry.update_template("var_b", status=TemplateStatus.ACTIVE)

        var_a = registry.get_active_template("var_a")
        var_b = registry.get_active_template("var_b")

        test = registry.create_ab_test("test", "desc", var_a, var_b, traffic_split=0.5)

        # Same session should get same variant
        v1 = test.assign_variant("session_123")
        v2 = test.assign_variant("session_123")
        assert v1 == v2

        # Different sessions should distribute
        variants = [test.assign_variant(f"session_{i}") for i in range(1000)]
        a_count = variants.count("A")
        # Should be roughly 50/50
        assert 400 < a_count < 600

    def test_metrics_recording(self, tmp_path):
        """Test template usage metrics."""
        registry = TemplateRegistry(storage_dir=tmp_path)

        registry.record_usage("test", "1.0.0", 100.0, 0.01, 0.9, "openrouter", True)
        registry.record_usage("test", "1.0.0", 200.0, 0.02, 0.8, "openrouter", True)

        metrics = registry.get_metrics("test", "1.0.0")
        assert metrics is not None
        assert metrics.total_uses == 2
        assert metrics.successful_uses == 2
        assert metrics.provider_breakdown["openrouter"]["uses"] == 2


class TestProviderOptimizer:
    """Test provider-specific template optimization."""

    def test_get_profile(self):
        """Test getting provider profile."""
        registry = TemplateRegistry(storage_dir=Path(tempfile.mkdtemp()))
        optimizer = ProviderOptimizer(registry)

        profile = optimizer.get_profile("openrouter", "anthropic/claude-sonnet-4")
        assert profile is not None
        assert profile.provider == "openrouter"
        assert profile.model == "anthropic/claude-sonnet-4"
        assert "reasoning" in profile.capabilities

    def test_unknown_profile_returns_none(self):
        """Test unknown provider/model returns None."""
        registry = TemplateRegistry(storage_dir=Path(tempfile.mkdtemp()))
        optimizer = ProviderOptimizer(registry)

        profile = optimizer.get_profile("unknown", "model")
        assert profile is None

    def test_template_selector_scoring(self, tmp_path):
        """Test template selection scoring."""
        registry = TemplateRegistry(storage_dir=tmp_path)
        optimizer = ProviderOptimizer(registry)
        selector = TemplateSelector(registry, optimizer)

        # Create templates
        registry.create_template(name="with_override", content="x", template_type=TemplateType.USER)
        registry.update_template("with_override", status=TemplateStatus.ACTIVE)
        t = registry.get_active_template("with_override")
        t.provider_overrides["openrouter"] = "custom"
        registry.update_template("with_override", provider_overrides=t.provider_overrides)

        registry.create_template(name="no_override", content="x", template_type=TemplateType.USER)
        registry.update_template("no_override", status=TemplateStatus.ACTIVE)

        # Select for openrouter
        selected = selector.select_template("user", "openrouter", "some-model")
        assert selected is not None
        assert selected.name == "with_override"


class TestDefaultTemplates:
    """Test built-in default templates."""

    def test_get_default_templates(self):
        """Test getting default templates."""
        templates = get_default_templates()
        assert len(templates) >= 5

        names = {t.name for t in templates}
        assert "code_review" in names
        assert "complex_reasoning" in names
        assert "creative_writing" in names
        assert "data_analysis" in names
        assert "system_hermes" in names

    def test_default_templates_have_variables(self):
        """Test default templates have proper variable definitions."""
        templates = get_default_templates()
        for t in templates:
            assert t.version == "1.0.0"
            assert t.template_type in TemplateType
            assert t.status == TemplateStatus.ACTIVE
            # Variables should be valid
            for v in t.variables:
                assert v.name
                assert v.type in ("string", "number", "boolean", "list", "object")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])