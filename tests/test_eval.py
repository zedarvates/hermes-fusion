"""Tests for Eval Harness."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hermes_fusion.eval.loaders import (
    BenchmarkSample,
    GSM8KLoader,
    HumanEvalLoader,
    ARCLoader,
    CustomBenchmarkLoader,
)
from hermes_fusion.eval.regression import RegressionDetector


class TestBenchmarkLoaders:
    """Test benchmark data loaders."""

    def test_gsm8k_loader_format(self):
        """Test GSM8K sample formatting."""
        loader = GSM8KLoader()
        sample = BenchmarkSample(
            id="test_1",
            question="What is 2 + 2?",
            expected="4",
        )
        # Should not crash
        assert loader.evaluate("4", "4", sample) == 1.0
        assert loader.evaluate("5", "4", sample) == 0.0

    def test_humaneval_loader_format(self):
        """Test HumanEval sample formatting."""
        loader = HumanEvalLoader()
        sample = BenchmarkSample(
            id="test_1",
            question="def add(a, b):\n    return a + b",
            expected="def add(a, b):\n    return a + b",
        )
        # Simple code match test
        assert loader._simple_code_match("def add(a, b): return a + b", "def add(a, b): return a + b") == 0.5
        assert loader.evaluate("def add(a, b): return a + b", "def add(a, b): return a + b", sample) == 0.5

    def test_arc_loader_format(self):
        """Test ARC sample formatting."""
        loader = ARCLoader()
        sample = BenchmarkSample(
            id="test_1",
            question="input: [[1,2],[3,4]]",
            expected="[[1,2],[3,4]]",
        )
        assert loader.evaluate("[[1,2],[3,4]]", "[[1,2],[3,4]]", sample) == 1.0
        assert loader.evaluate("[[1,2],[3,5]]", "[[1,2],[3,4]]", sample) == 0.0

    def test_custom_loader_json(self, tmp_path):
        """Test custom JSON benchmark loader."""
        filepath = tmp_path / "test_bench.json"
        filepath.write_text('[{"id": "q1", "question": "Q1", "answer": "A1"}]')

        loader = CustomBenchmarkLoader("test", filepath)
        samples = loader.load_samples()
        assert len(samples) == 1
        assert samples[0].id == "q1"
        assert samples[0].question == "Q1"
        assert samples[0].expected == "A1"

    def test_custom_loader_jsonl(self, tmp_path):
        """Test custom JSONL benchmark loader."""
        filepath = tmp_path / "test_bench.jsonl"
        filepath.write_text('{"id": "q1", "question": "Q1", "answer": "A1"}\n{"id": "q2", "question": "Q2", "answer": "A2"}\n')

        loader = CustomBenchmarkLoader("test", filepath)
        samples = loader.load_samples()
        assert len(samples) == 2
        assert samples[0].id == "q1"
        assert samples[1].id == "q2"


class TestRegressionDetector:
    """Test regression detection."""

    def test_normal_cdf(self):
        """Test normal CDF approximation."""
        detector = RegressionDetector()
        # CDF at 0 should be 0.5
        assert abs(detector._normal_cdf(0) - 0.5) < 0.01
        # CDF at large positive should approach 1
        assert detector._normal_cdf(5) > 0.99
        # CDF at large negative should approach 0
        assert detector._normal_cdf(-5) < 0.01

    def test_detect_no_regression(self):
        """Test that similar accuracies don't trigger regression."""
        from hermes_fusion.eval import BenchmarkSuite, BenchmarkResult

        detector = RegressionDetector(min_samples=10, alpha=0.05)

        # Create two similar suites
        current = BenchmarkSuite(
            suite_id="current",
            benchmark_name="test",
            provider="test",
            model="test",
            total_questions=100,
            correct=85,
            accuracy=0.85,
            avg_latency_ms=100,
            avg_cost_usd=0.01,
            total_cost_usd=1.0,
            results=[],
            started_at=0,
            completed_at=0,
        )

        baseline = BenchmarkSuite(
            suite_id="baseline",
            benchmark_name="test",
            provider="test",
            model="test",
            total_questions=100,
            correct=86,
            accuracy=0.86,
            avg_latency_ms=100,
            avg_cost_usd=0.01,
            total_cost_usd=1.0,
            results=[],
            started_at=0,
            completed_at=0,
        )

        alert = detector._compare_suites(current, baseline)
        # Small difference should not be significant
        assert alert is None or not alert.significant

    def test_detect_regression(self):
        """Test that large accuracy drop triggers regression."""
        from hermes_fusion.eval import BenchmarkSuite, BenchmarkResult

        detector = RegressionDetector(min_samples=30, alpha=0.05)

        # Current much worse than baseline
        current = BenchmarkSuite(
            suite_id="current",
            benchmark_name="test",
            provider="test",
            model="test",
            total_questions=100,
            correct=50,  # 50% accuracy
            accuracy=0.50,
            avg_latency_ms=100,
            avg_cost_usd=0.01,
            total_cost_usd=1.0,
            results=[],
            started_at=0,
            completed_at=0,
        )

        baseline = BenchmarkSuite(
            suite_id="baseline",
            benchmark_name="test",
            provider="test",
            model="test",
            total_questions=100,
            correct=90,  # 90% accuracy
            accuracy=0.90,
            avg_latency_ms=100,
            avg_cost_usd=0.01,
            total_cost_usd=1.0,
            results=[],
            started_at=0,
            completed_at=0,
        )

        alert = detector._compare_suites(current, baseline)
        assert alert is not None
        assert alert.significant is True
        assert alert.delta < -0.3  # Significant drop


class TestEvalIntegration:
    """Integration tests for eval components."""

    @pytest.mark.asyncio
    async def test_runner_creation(self):
        """Test EvalRunner can be created."""
        from hermes_fusion.eval.runner import EvalRunner
        from hermes_fusion.engine import FusionEngine
        from hermes_fusion.config import FusionConfig

        engine = FusionEngine(config=FusionConfig())
        runner = EvalRunner(engine)
        assert runner is not None
        assert runner.results_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])