"""Template CLI Handlers."""

import sys
from pathlib import Path

from hermes_fusion.templates.registry import TemplateRegistry
from hermes_fusion.templates import TemplateType, TemplateStatus
from hermes_fusion.templates.optimization import ProviderOptimizer, TemplateSelector
from hermes_fusion.config import FusionConfig
from hermes_fusion.engine import FusionEngine

# Reuse engine creation from main cli
async def create_engine_from_config(config_path: Path) -> FusionEngine:
    """Create FusionEngine from config file."""
    config = (
        FusionConfig.from_toml(config_path)
        if config_path.exists()
        else FusionConfig()
    )

    # Import providers lazily to avoid hard dependencies
    from hermes_fusion.providers.cloud import (
        AnthropicProvider,
        OpenAIProvider,
        XAIProvider,
    )
    from hermes_fusion.providers.localai import LocalAIProvider
    from hermes_fusion.providers.qdrant import QdrantProvider

    engine = FusionEngine(config=config)

    # Initialize providers from config
    providers_config = config.providers

    # LocalAI
    if providers_config.localai:
        try:
            provider = LocalAIProvider(providers_config.localai)
            engine.add_provider("localai", provider)
        except Exception as e:
            print(f"Warning: Failed to initialize LocalAI: {e}", file=sys.stderr)

    # Cloud providers
    cloud = providers_config.cloud
    if cloud.xai:
        try:
            provider = XAIProvider(cloud.xai)
            engine.add_provider("xai", provider)
        except Exception as e:
            print(f"Warning: Failed to initialize xAI: {e}", file=sys.stderr)

    if cloud.openai:
        try:
            provider = OpenAIProvider(cloud.openai)
            engine.add_provider("openai", provider)
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI: {e}", file=sys.stderr)

    if cloud.anthropic:
        try:
            provider = AnthropicProvider(cloud.anthropic)
            engine.add_provider("anthropic", provider)
        except Exception as e:
            print(f"Warning: Failed to initialize Anthropic: {e}", file=sys.stderr)

    # Qdrant
    if providers_config.qdrant:
        try:
            provider = QdrantProvider(providers_config.qdrant)
            engine.qdrant = provider
        except Exception as e:
            print(f"Warning: Failed to initialize Qdrant: {e}", file=sys.stderr)

    return engine


async def run_template(args) -> int:
    """Execute template commands."""
    engine = await create_engine_from_config(args.config)

    try:
        registry = TemplateRegistry()
        optimizer = ProviderOptimizer(registry)
        selector = TemplateSelector(registry, optimizer)

        if args.template_action == "create":
            content = args.content
            if args.file:
                content = args.file.read_text()

            template = registry.create_template(
                name=args.name,
                content=content,
                template_type=TemplateType(args.type),
                description=args.description,
                tags=args.tags,
            )
            print(f"✅ Created template: {template.name} v{template.version}")
            return 0

        elif args.template_action == "list":
            templates = registry.list_templates()
            if not templates:
                print("No templates found.")
                return 0

            print(f"{'Name':<30} {'Type':<15} {'Status':<12} {'Version':<10} {'Tags'}")
            print("-" * 90)
            for t in templates:
                tags = ", ".join(t.tags[:3])
                if len(t.tags) > 3:
                    tags += "..."
                print(f"{t.name:<30} {t.template_type.value:<15} {t.status.value:<12} {t.version:<10} {tags}")
            return 0

        elif args.template_action == "show":
            template = registry.get_template(args.name)
            if not template:
                print(f"Template '{args.name}' not found")
                return 1

            print(f"Name: {template.name}")
            print(f"Version: {template.version}")
            print(f"Type: {template.template_type.value}")
            print(f"Status: {template.status.value}")
            print(f"Description: {template.description}")
            print(f"Author: {template.author}")
            print(f"Tags: {', '.join(template.tags) or 'none'}")
            print(f"Variables: {len(template.variables)}")
            for v in template.variables:
                req = " (required)" if v.required else ""
                print(f"  - {v.name}: {v.type}{req}")
            print(f"Provider overrides: {list(template.provider_overrides.keys()) or 'none'}")
            print(f"\nContent:\n{template.content}")
            return 0

        elif args.template_action == "render":
            # Simple render with empty variables for testing
            variables = {}  # In real use, would parse from args
            try:
                rendered, template = registry.render(args.name, variables)
                print(f"Rendered ({template.name} v{template.version}):\n")
                print(rendered)
            except ValueError as e:
                print(f"Error: {e}")
                print("Variables needed:")
                template = registry.get_template(args.name)
                if template:
                    for v in template.variables:
                        print(f"  - {v.name} ({v.type}){' [required]' if v.required else ''}")
                return 1
            return 0

        elif args.template_action == "versions":
            versions = registry.list_versions(args.name)
            if not versions:
                print(f"No versions found for '{args.name}'")
                return 1

            print(f"{'Version':<12} {'Status':<12} {'Type':<15} {'AB Test'}")
            print("-" * 60)
            for v in versions:
                ab = f"{v.ab_test_id}:{v.ab_variant}" if v.ab_test_id else "none"
                print(f"{v.version:<12} {v.status.value:<12} {v.template_type.value:<15} {ab}")
            return 0

        elif args.template_action == "delete":
            registry.delete_template(args.name, args.version)
            print(f"✅ Deleted template '{args.name}'" + (f" v{args.version}" if args.version else " (all versions)"))
            return 0

        elif args.template_action == "ab-test":
            return await run_ab_test(args, registry)

        elif args.template_action == "optimize":
            providers = []
            for p in args.providers:
                if "/" in p:
                    prov, model = p.split("/", 1)
                    providers.append((prov, model))

            template = optimizer.auto_create_overrides(args.name, providers)
            overrides = list(template.provider_overrides.keys())
            print(f"✅ Added provider overrides for: {', '.join(overrides) or 'none'}")
            return 0

        elif args.template_action == "select":
            template = selector.select_template(
                args.task_type,
                args.provider,
                args.model,
            )
            if template:
                print(f"Selected: {template.name} v{template.version}")
                print(f"Type: {template.template_type.value}")
                print(f"Status: {template.status.value}")
                if args.provider in template.provider_overrides:
                    print(f"Has override for {args.provider}")
            else:
                print(f"No suitable template found for {args.task_type} on {args.provider}/{args.model}")
                return 1
            return 0

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_ab_test(args, registry: TemplateRegistry) -> int:
    """Execute A/B test commands."""

    if args.ab_action == "create":
        variant_a = registry.get_active_template(args.variant_a)
        variant_b = registry.get_active_template(args.variant_b)

        if not variant_a or not variant_b:
            print("Error: Both variants must exist and be ACTIVE")
            return 1

        test = registry.create_ab_test(
            name=args.name,
            description=args.description,
            variant_a=variant_a,
            variant_b=variant_b,
            traffic_split=args.split,
        )
        print(f"✅ Created A/B test: {test.test_id}")
        print(f"  Variant A: {variant_a.name} v{variant_a.version} ({args.split:.0%})")
        print(f"  Variant B: {variant_b.name} v{variant_b.version} ({1-args.split:.0%})")
        return 0

    elif args.ab_action == "list":
        tests = registry.list_ab_tests()
        if not tests:
            print("No A/B tests found.")
            return 0

        print(f"{'Test ID':<35} {'Name':<20} {'Status':<12} {'Split':<8} {'Min Samples'}")
        print("-" * 90)
        for t in tests:
            print(f"{t.test_id:<35} {t.name:<20} {t.status:<12} {t.traffic_split:<8.0%} {t.min_samples}")
        return 0

    elif args.ab_action == "show":
        test = registry.get_ab_test(args.test_id)
        if not test:
            print(f"A/B test '{args.test_id}' not found")
            return 1

        print(f"Test ID: {test.test_id}")
        print(f"Name: {test.name}")
        print(f"Description: {test.description}")
        print(f"Status: {test.status}")
        print(f"Traffic Split: {test.traffic_split:.0%} A / {1-test.traffic_split:.0%} B")
        print(f"Min Samples: {test.min_samples}")
        print(f"Confidence: {test.confidence_threshold:.0%}")
        print(f"Variant A: {test.variant_a.name} v{test.variant_a.version}")
        print(f"Variant B: {test.variant_b.name} v{test.variant_b.version}")
        return 0

    elif args.ab_action == "stop":
        registry.stop_ab_test(args.test_id, args.winner)
        print(f"✅ Stopped A/B test: {args.test_id}")
        if args.winner:
            print(f"   Winner declared: Variant {args.winner}")
        return 0

    return 0