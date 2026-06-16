"""CLI entry point for Hermes Fusion."""

import argparse
import asyncio
import sys
from pathlib import Path

from hermes_fusion.config import FusionConfig
from hermes_fusion.engine import FusionEngine
from hermes_fusion.model_router import TaskType, RoutingPolicy
from hermes_fusion.templates import TemplateType
from hermes_fusion.cli_template import run_template


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

    # Cost command
    cost_parser = subparsers.add_parser("cost", help="Cost tracking and budgets")
    cost_sub = cost_parser.add_subparsers(dest="cost_action", required=True)
    
    cost_summary = cost_sub.add_parser("summary", help="Show cost summary")
    cost_summary.add_argument("--by-provider", action="store_true", help="Break down by provider")
    cost_summary.add_argument("--by-model", action="store_true", help="Break down by model")
    
    cost_sub.add_parser("budget", help="Show budget status")
    cost_sub.add_parser("daily", help="Show daily costs")

    # Router command
    router_parser = subparsers.add_parser("router", help="Model router commands")
    router_sub = router_parser.add_subparsers(dest="router_action", required=True)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Benchmark evaluation")
    eval_sub = eval_parser.add_subparsers(dest="eval_action", required=True)

    run_parser = eval_sub.add_parser("run", help="Run a benchmark")
    run_parser.add_argument(
        "benchmark",
        type=str,
        choices=["mmlu", "gsm8k", "humaneval", "arc"],
        help="Benchmark to run",
    )
    run_parser.add_argument(
        "--provider", "-p",
        type=str,
        required=True,
        help="Provider to test (e.g., openrouter, localai, minimax)",
    )
    run_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model to test (e.g., anthropic/claude-sonnet-4)",
    )
    run_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of questions (default: all)",
    )
    run_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "validation"],
        help="Dataset split (default: test)",
    )
    run_parser.add_argument(
        "--task-type",
        type=str,
        default="complex_reasoning",
        help="Task type for routing (default: complex_reasoning)",
    )

    eval_sub.add_parser("list", help="List saved benchmark results")
    eval_sub.add_parser("report", help="Show regression report")
    eval_sub.add_parser("compare", help="Compare two benchmark runs")

    # Template command
    template_parser = subparsers.add_parser("template", help="Prompt template management")
    template_sub = template_parser.add_subparsers(dest="template_action", required=True)

    create_parser = template_sub.add_parser("create", help="Create a new template")
    create_parser.add_argument("name", type=str, help="Template name")
    create_parser.add_argument("content", type=str, help="Template content (Jinja2)")
    create_parser.add_argument("--type", type=str, default="user", choices=[t.value for t in TemplateType], help="Template type")
    create_parser.add_argument("--description", type=str, default="", help="Description")
    create_parser.add_argument("--tags", type=str, nargs="+", default=[], help="Tags")
    create_parser.add_argument("--file", "-f", type=Path, help="Read content from file")

    list_parser = template_sub.add_parser("list", help="List templates")

    show_parser = template_sub.add_parser("show", help="Show template details")
    show_parser.add_argument("name", type=str)

    render_parser = template_sub.add_parser("render", help="Render a template")
    render_parser.add_argument("name", type=str)

    delete_parser = template_sub.add_parser("delete", help="Delete a template")
    delete_parser.add_argument("name", type=str)
    delete_parser.add_argument("--version", type=str)

    versions_parser = template_sub.add_parser("versions", help="List template versions")
    versions_parser.add_argument("name", type=str)

    # A/B test commands
    ab_parser = template_sub.add_parser("ab-test", help="A/B test management")
    ab_sub = ab_parser.add_subparsers(dest="ab_action", required=True)

    ab_create = ab_sub.add_parser("create", help="Create A/B test")
    ab_create.add_argument("name", type=str)
    ab_create.add_argument("description", type=str)
    ab_create.add_argument("--variant-a", type=str, required=True)
    ab_create.add_argument("--variant-b", type=str, required=True)
    ab_create.add_argument("--split", type=float, default=0.5)

    ab_sub.add_parser("list", help="List A/B tests")

    ab_show = ab_sub.add_parser("show", help="Show A/B test details")
    ab_show.add_argument("test_id", type=str)

    ab_stop = ab_sub.add_parser("stop", help="Stop A/B test")
    ab_stop.add_argument("test_id", type=str)
    ab_stop.add_argument("--winner", type=str, choices=["A", "B"])

    # Optimization commands
    opt_parser = template_sub.add_parser("optimize", help="Create provider overrides for template")
    opt_parser.add_argument("name", type=str)
    opt_parser.add_argument("--providers", type=str, nargs="+", default=[])

    select_parser = template_sub.add_parser("select", help="Select best template for task/provider")
    select_parser.add_argument("task_type", type=str)
    select_parser.add_argument("--provider", type=str, required=True)
    select_parser.add_argument("--model", type=str, required=True)

    route_parser = router_sub.add_parser("route", help="Get routing decision for a prompt")
    route_parser.add_argument("prompt", type=str, help="Prompt to route")
    route_parser.add_argument(
        "--policy", "-p",
        type=str,
        choices=[p.value for p in RoutingPolicy],
        help="Routing policy (default: from config)",
    )
    route_parser.add_argument(
        "--cost-quality", type=int, choices=range(0, 11),
        help="Cost/quality tradeoff 0-10 (0=quality, 10=cost)",
    )
    route_parser.add_argument("--session-id", type=str, help="Session ID for stickiness")
    route_parser.add_argument("--allowed-models", type=str, nargs="+", help="Restrict to models")
    
    router_sub.add_parser("stats", help="Show routing statistics")
    route_sub = router_sub.add_parser("models", help="List models for a task type")
    route_sub.add_argument(
        "--task-type", "-t",
        type=str,
        choices=[t.value for t in TaskType],
        default="general",
        help="Task type to filter models",
    )
    clear_parser = router_sub.add_parser("clear-session", help="Clear session stickiness")
    clear_parser.add_argument("session_id", type=str, help="Session ID to clear")

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
        # Print raw metrics for scraping
        sys.stdout.buffer.write(metrics_output)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_eval(args) -> int:
    """Execute eval commands."""
    engine = await create_engine_from_config(args.config)

    try:
        if args.eval_action == "run":
            from hermes_fusion.eval.runner import EvalRunner

            runner = EvalRunner(engine)
            suite = await runner.run_benchmark(
                benchmark_name=args.benchmark,
                provider=args.provider,
                model=args.model,
                limit=args.limit,
                split=args.split,
                task_type=args.task_type,
            )

            print(f"\n✅ Benchmark Complete: {suite.suite_id}")
            print(f"   Accuracy: {suite.accuracy:.2%} ({suite.correct}/{suite.total_questions})")
            print(f"   Avg Latency: {suite.avg_latency_ms:.0f}ms")
            print(f"   Total Cost: ${suite.total_cost_usd:.4f}")
            print(f"   Results saved to ~/.hermes_fusion/eval_results/{suite.suite_id}.json")
            return 0

        elif args.eval_action == "list":
            from hermes_fusion.eval.runner import EvalRunner

            runner = EvalRunner(engine)
            suites = runner.list_suites()

            if not suites:
                print("No benchmark results found.")
                return 0

            print(f"{'Suite ID':<50} {'Benchmark':<12} {'Provider':<12} {'Model':<30} {'Acc':>8} {'Cost':>10}")
            print("-" * 130)
            for s in suites:
                model = s["model"][:28] + ".." if len(s["model"]) > 30 else s["model"]
                print(f"{s['suite_id']:<50} {s['benchmark_name']:<12} {s['provider']:<12} {model:<30} {s['accuracy']:>7.1%} ${s['total_cost_usd']:>9.4f}")
            return 0

        elif args.eval_action == "report":
            from hermes_fusion.eval.regression import RegressionDetector

            detector = RegressionDetector()
            alerts = detector.check_all_regressions()

            if not alerts:
                print("✅ No regressions detected.")
                return 0

            print(f"⚠️  {len(alerts)} regression(s) detected:\n")
            for alert in alerts:
                status = "🔴 SIGNIFICANT" if alert.significant else "🟡 WARNING"
                print(f"  {status}: {alert.benchmark_name} / {alert.provider} / {alert.model}")
                print(f"    Current: {alert.current_accuracy:.2%}, Baseline: {alert.baseline_accuracy:.2%}")
                print(f"    Delta: {alert.delta:+.2%}, p-value: {alert.p_value:.4f}")
                print()
            return 0

        elif args.eval_action == "compare":
            print("Compare command not yet implemented. Use 'eval list' to see runs.")
            return 1

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_cost(args) -> int:
    """Show cost metrics."""
    engine = await create_engine_from_config(args.config)

    try:
        if args.cost_action == "summary":
            metrics = engine.get_cost_metrics()
            if not metrics:
                print("Cost tracker not enabled")
                return 1
            print(f"Total Cost: ${metrics.total_cost_usd:.4f}")
            print(f"Total Tokens: {metrics.total_tokens:,}")
            print(f"  Input: {metrics.total_input_tokens:,}")
            print(f"  Output: {metrics.total_output_tokens:,}")
            print(f"Requests: {metrics.request_count}")
            print(f"Cached: {metrics.cached_requests}")
            print()
            
            if args.by_provider:
                for provider, data in metrics.get_metrics_by_provider().items():
                    print(f"  {provider}: ${data['cost_usd']:.4f} ({data['total_tokens']:,} tokens)")
            
            if args.by_model:
                for model, data in metrics.get_metrics_by_model().items():
                    pricing = data['pricing_per_1k']
                    print(f"  {model}: ${data['cost_usd']:.4f} ({data['total_tokens']:,} tokens) @ ${pricing[0]}/${pricing[1]} per 1K")
        
        elif args.cost_action == "budget":
            if not engine._cost_tracker:
                print("Cost tracker not enabled")
                return 1
            for budget in engine._cost_tracker.budgets:
                metrics = engine.get_cost_metrics()
                if metrics:
                    status = metrics.get_budget_status(budget)
                    print(f"Budget ({budget.period}): ${status['spent']:.2f} / ${status['budget_limit']:.2f} ({status['utilization']:.1%})")
                    if status['alert_triggered']:
                        print(f"  ⚠️  ALERT: Over {budget.alert_threshold:.0%} threshold!")
        
        elif args.cost_action == "daily":
            if not engine._cost_tracker:
                print("Cost tracker not enabled")
                return 1
            metrics = engine.get_cost_metrics()
            if metrics:
                print("Daily costs:")
                for day, cost in sorted(metrics.daily_costs.items()):
                    print(f"  {day}: ${cost:.4f}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def run_router(args) -> int:
    """Execute router commands."""
    engine = await create_engine_from_config(args.config)
    
    # Start model router if available
    if engine._model_router:
        await engine.start_model_router()
    
    try:
        if args.router_action == "route":
            if not engine._model_router:
                print("Model router not enabled in config")
                return 1
            
            policy = RoutingPolicy(args.policy) if args.policy else None
            decision = await engine._model_router.route(
                prompt=args.prompt,
                policy=policy,
                cost_quality_tradeoff=args.cost_quality,
                session_id=args.session_id,
                allowed_models=args.allowed_models,
            )
            
            print(f"Model: {decision.model_id}")
            print(f"Provider: {decision.provider_name}")
            print(f"Task Type: {decision.task_type.value}")
            print(f"Policy: {decision.policy.value}")
            print(f"Confidence: {decision.confidence:.2%}")
            print(f"Session ID: {decision.session_id or 'N/A'}")
            print(f"Fallbacks: {', '.join(f'{f.name}:{f.model}' for f in decision.fallbacks) or 'None'}")
            
        elif args.router_action == "stats":
            if not engine._model_router:
                print("Model router not enabled in config")
                return 1
            
            stats = engine._model_router.get_routing_stats()
            print(f"Total Routes: {stats['total_routes']}")
            if stats['total_routes'] > 0:
                print(f"Success Rate: {stats['success_rate']:.1%}")
                print(f"Avg Latency: {stats['avg_latency_ms']:.0f}ms")
                print(f"Avg Cost: ${stats['avg_cost_usd']:.4f}")
                print()
                print("Model Distribution:")
                for model, count in stats['model_distribution'].items():
                    print(f"  {model}: {count}")
                print()
                print("Provider Distribution:")
                for provider, count in stats['provider_distribution'].items():
                    print(f"  {provider}: {count}")
                print()
                print("Task Distribution:")
                for task, count in stats['task_distribution'].items():
                    print(f"  {task}: {count}")
        
        elif args.router_action == "models":
            if not engine._model_router:
                print("Model router not enabled in config")
                return 1
            
            task_type = TaskType(args.task_type)
            models = engine._model_router.get_models_for_task(task_type)
            print(f"Models for {task_type.value}:")
            for m in models:
                for p in m.providers:
                    print(f"  {m.model_id} via {p.name} (quality={p.avg_quality_score:.2f}, cost={p.cost_per_1k_input:.4f}/${p.cost_per_1k_output:.4f})")
        
        elif args.router_action == "clear-session":
            if not engine._model_router:
                print("Model router not enabled in config")
                return 1
            
            engine._model_router.clear_session(args.session_id)
            print(f"Cleared session: {args.session_id}")
        
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
        "cost": run_cost,
        "router": run_router,
        "eval": run_eval,
        "template": run_template,
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