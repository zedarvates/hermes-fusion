"""Eval Runner - Executes benchmarks against providers."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hermes_fusion.engine import FusionEngine
from hermes_fusion.eval import BenchmarkResult, BenchmarkSuite
from hermes_fusion.eval.loaders import BenchmarkLoader, BenchmarkSample, get_loader


class EvalRunner:
    """Runs benchmarks against configured providers."""

    def __init__(
        self,
        engine: FusionEngine,
        results_dir: Optional[Path] = None,
    ):
        self.engine = engine
        self.results_dir = results_dir or Path.home() / ".hermes_fusion" / "eval_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run_benchmark(
        self,
        benchmark_name: str,
        provider: str,
        model: str,
        limit: Optional[int] = None,
        split: str = "test",
        task_type: str = "complex_reasoning",
    ) -> BenchmarkSuite:
        """Run a single benchmark against a provider/model."""
        loader = get_loader(benchmark_name)
        samples = loader.load_samples(limit=limit, split=split)

        suite_id = f"{benchmark_name}_{provider}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = []
        correct = 0
        total_latency = 0
        total_cost = 0
        total_tokens_in = 0
        total_tokens_out = 0

        started_at = time.time()

        for sample in samples:
            start = time.time()
            try:
                response = await self.engine.query(
                    sample.question,
                    provider=provider,
                    model=model,
                    task_type=task_type,
                )
                latency_ms = int((time.time() - start) * 1000)

                score = loader.evaluate(response.content, sample.expected, sample)
                if score >= 0.5:
                    correct += 1

                # Estimate tokens and cost (would come from provider response in real impl)
                tokens_in = len(sample.question.split()) * 1.3
                tokens_out = len(response.content.split()) * 1.3
                total_tokens_in += tokens_in
                total_tokens_out += tokens_out

                cost = self._estimate_cost(provider, model, tokens_in, tokens_out)
                total_cost += cost

                result = BenchmarkResult(
                    benchmark_id=f"{suite_id}_{sample.id}",
                    benchmark_name=benchmark_name,
                    provider=provider,
                    model=model,
                    task_type=task_type,
                    question=sample.question,
                    expected=sample.expected,
                    response=response.content,
                    score=score,
                    latency_ms=latency_ms,
                    tokens_in=int(tokens_in),
                    tokens_out=int(tokens_out),
                    cost_usd=cost,
                    timestamp=time.time(),
                    metadata=sample.metadata,
                )
                results.append(result)

            except Exception as e:
                # Record failed attempt
                result = BenchmarkResult(
                    benchmark_id=f"{suite_id}_{sample.id}",
                    benchmark_name=benchmark_name,
                    provider=provider,
                    model=model,
                    task_type=task_type,
                    question=sample.question,
                    expected=sample.expected,
                    response=f"ERROR: {e}",
                    score=0.0,
                    latency_ms=0,
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    timestamp=time.time(),
                    metadata={"error": str(e), **sample.metadata},
                )
                results.append(result)

            total_latency += latency_ms

        completed_at = time.time()
        suite = BenchmarkSuite(
            suite_id=suite_id,
            benchmark_name=benchmark_name,
            provider=provider,
            model=model,
            total_questions=len(samples),
            correct=correct,
            accuracy=correct / len(samples) if samples else 0.0,
            avg_latency_ms=total_latency / len(results) if results else 0.0,
            avg_cost_usd=total_cost / len(results) if results else 0.0,
            total_cost_usd=total_cost,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
            metadata={"limit": limit, "split": split, "task_type": task_type},
        )

        # Save suite
        self._save_suite(suite)

        return suite

    async def run_suite(
        self,
        benchmarks: list[str],
        providers: list[tuple[str, str]],  # (provider, model)
        limit: Optional[int] = None,
        split: str = "test",
        task_type: str = "complex_reasoning",
    ) -> list[BenchmarkSuite]:
        """Run multiple benchmarks across multiple providers."""
        suites = []
        for benchmark in benchmarks:
            for provider, model in providers:
                print(f"Running {benchmark} on {provider}/{model}...")
                suite = await self.run_benchmark(
                    benchmark, provider, model, limit, split, task_type
                )
                suites.append(suite)
                print(f"  Accuracy: {suite.accuracy:.2%}, Cost: ${suite.total_cost_usd:.4f}")
        return suites

    def _estimate_cost(self, provider: str, model: str, tokens_in: float, tokens_out: float) -> float:
        """Estimate cost based on provider/model pricing."""
        pricing = {
            "openrouter": {
                "anthropic/claude-sonnet-4": (3.0, 15.0),
                "openai/gpt-5.5": (5.0, 15.0),
                "nvidia/nemotron-3-ultra": (0.5, 2.0),
            },
            "localai": {
                "gemma-4-e2b-it": (0.0, 0.0),
            },
            "minimax": {
                "MiniMax-M3": (0.3, 1.2),
            },
        }
        if provider in pricing and model in pricing[provider]:
            in_cost, out_cost = pricing[provider][model]
            return (tokens_in / 1000) * in_cost + (tokens_out / 1000) * out_cost
        return 0.0

    def _save_suite(self, suite: BenchmarkSuite) -> None:
        """Save benchmark suite to JSON."""
        filepath = self.results_dir / f"{suite.suite_id}.json"
        with open(filepath, "w") as f:
            json.dump(asdict(suite), f, indent=2, default=str)

    def load_suite(self, suite_id: str) -> Optional[BenchmarkSuite]:
        """Load a saved benchmark suite."""
        filepath = self.results_dir / f"{suite_id}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            # Reconstruct objects
            from hermes_fusion.eval import BenchmarkResult, BenchmarkSuite
            results = [BenchmarkResult(**r) for r in data.pop("results", [])]
            return BenchmarkSuite(results=results, **data)
        return None

    def list_suites(self, benchmark_name: Optional[str] = None) -> list[dict]:
        """List all saved benchmark suites."""
        suites = []
        for filepath in self.results_dir.glob("*.json"):
            with open(filepath) as f:
                data = json.load(f)
            if benchmark_name and data.get("benchmark_name") != benchmark_name:
                continue
            suites.append({
                "suite_id": data.get("suite_id"),
                "benchmark_name": data.get("benchmark_name"),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "accuracy": data.get("accuracy"),
                "total_cost_usd": data.get("total_cost_usd"),
                "total_questions": data.get("total_questions"),
                "completed_at": data.get("completed_at"),
            })
        return sorted(suites, key=lambda x: x.get("completed_at", 0), reverse=True)