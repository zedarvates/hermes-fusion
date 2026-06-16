"""CLI entry point for Hermes Fusion."""

import argparse
import asyncio
import sys
from pathlib import Path

from hermes_fusion.config import FusionConfig
from hermes_fusion.engine import FusionEngine


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hermes-fusion",
        description="Multi-LLM Fusion Engine - Local cluster + Cloud orchestration",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.toml"),
        help="Path to config TOML file (default: config.toml)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="hermes-fusion 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a fusion query")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["weighted_vote", "handoff", "cot_consensus", "best_of_n"],
        help="Fusion strategy (default: from config)",
    )
    query_parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model override",
    )
    query_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )

    # Health command
    subparsers.add_parser("health", help="Check health of all providers")

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_sub = cache_parser.add_subparsers(dest="cache_action", required=True)
    cleanup_parser = cache_sub.add_parser("cleanup", help="Clean up old cache entries")
    cleanup_parser.add_argument(
        "--hours", type=int, default=24,
        help="Remove entries older than N hours (default: 24)",
    )

    # Strategies command
    subparsers.add_parser("strategies", help="List available fusion strategies")

    # Metrics command
    subparsers.add_parser("metrics", help="Output Prometheus metrics")

    return parser


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


async def run_query(args) -> int:
    """Execute query command."""
    engine = await create_engine_from_config(args.config)
    
    try:
        result = await engine.query(
            args.question,
            strategy=args.strategy,
            model=args.model,
        )
        
        if args.json:
            import json
            output = {
                "answer": result.final_answer,
                "confidence": result.confidence,
                "method": result.method,
                "providers": result.participating_providers,
                "metadata": result.metadata,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"Answer: {result.final_answer}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Method: {result.method}")
            print(f"Providers: {', '.join(result.participating_providers)}")
            if result.metadata.get("cached"):
                print("(from cache)")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_health(args) -> int:
    """Execute health check command."""
    engine = await create_engine_from_config(args.config)
    
    try:
        health = await engine.health_check()
        for name, healthy in health.items():
            status = "✓" if healthy else "✗"
            print(f"  {status} {name}")
        return 0 if all(health.values()) else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_cache_cleanup(args) -> int:
    """Execute cache cleanup command."""
    engine = await create_engine_from_config(args.config)
    
    try:
        deleted = await engine.cleanup_cache(args.hours)
        print(f"Deleted {deleted} cache entries older than {args.hours}h")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_strategies(args) -> int:
    """List available strategies."""
    engine = await create_engine_from_config(args.config)

    try:
        strategies = engine.get_available_strategies()
        print("Available fusion strategies:")
        for s in strategies:
            print(f"  - {s}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_metrics(args) -> int:
    """Output Prometheus metrics."""
    engine = await create_engine_from_config(args.config)

    try:
        metrics_output = engine.get_prometheus_metrics()
        content_type = engine.get_metrics_content_type()
        # Print raw metrics for scraping
        sys.stdout.buffer.write(metrics_output)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    command_map = {
        "query": run_query,
        "health": run_health,
        "cache": {
            "cleanup": run_cache_cleanup,
        },
        "strategies": run_strategies,
        "metrics": run_metrics,
    }
    
    try:
        if args.command == "cache":
            handler = command_map["cache"][args.cache_action]
        else:
            handler = command_map[args.command]
        
        return await handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cli_entry():
    """Sync entry point for setuptools console_scripts."""
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(cli_entry())