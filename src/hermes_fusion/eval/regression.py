"""Regression Detection - Statistical significance testing for benchmark results."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hermes_fusion.eval import BenchmarkSuite, RegressionAlert


class RegressionDetector:
    """Detects statistically significant regressions in benchmark performance."""

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        alpha: float = 0.05,
        min_samples: int = 30,
    ):
        self.results_dir = results_dir or Path.home() / ".hermes_fusion" / "eval_results"
        self.alpha = alpha
        self.min_samples = min_samples

    def detect_regressions(
        self,
        current_suite: BenchmarkSuite,
        baseline_suites: list[BenchmarkSuite] = None,
    ) -> list[RegressionAlert]:
        """Detect regressions by comparing current results to baselines."""
        if baseline_suites is None:
            baseline_suites = []
        if not baseline_suites:
            baseline_suites = self._load_baselines(current_suite)

        alerts = []
        for baseline in baseline_suites:
            alert = self._compare_suites(current_suite, baseline)
            if alert:
                alerts.append(alert)
        return alerts

    def _load_baselines(self, current: BenchmarkSuite) -> list[BenchmarkSuite]:
        """Load historical baselines for same benchmark/provider/model."""
        baselines = []
        for filepath in self.results_dir.glob("*.json"):
            # Skip current
            if current.suite_id in str(filepath):
                continue
            with open(filepath) as f:
                data = json.load(f)
            if (data.get("benchmark_name") == current.benchmark_name and
                data.get("provider") == current.provider and
                data.get("model") == current.model):
                baselines.append(self._dict_to_suite(data))
        # Sort by date, take recent ones
        baselines.sort(key=lambda s: s.completed_at, reverse=True)
        return baselines[:5]  # Last 5 runs

    def _dict_to_suite(self, data: dict) -> BenchmarkSuite:
        from hermes_fusion.eval import BenchmarkResult, BenchmarkSuite
        results = [BenchmarkResult(**r) for r in data.pop("results", [])]
        return BenchmarkSuite(results=results, **data)

    def _compare_suites(
        self,
        current: BenchmarkSuite,
        baseline: BenchmarkSuite,
    ) -> Optional[RegressionAlert]:
        """Compare two suites using two-proportion z-test."""
        # Need enough samples
        if current.total_questions < self.min_samples or baseline.total_questions < self.min_samples:
            return None

        p1 = current.accuracy
        p2 = baseline.accuracy
        n1 = current.total_questions
        n2 = baseline.total_questions

        # Pooled proportion
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        se = math.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))

        if se == 0:
            return None

        # Z-score
        z = (p1 - p2) / se

        # Two-tailed p-value
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        # Check significance and direction (regression = current worse)
        significant = p_value < self.alpha and p1 < p2
        delta = p1 - p2

        if significant or abs(delta) > 0.05:  # Also flag large drops
            return RegressionAlert(
                benchmark_name=current.benchmark_name,
                provider=current.provider,
                model=current.model,
                current_accuracy=p1,
                baseline_accuracy=p2,
                delta=delta,
                p_value=p_value,
                significant=significant,
                timestamp=datetime.now().timestamp(),
                details={
                    "current_suite_id": current.suite_id,
                    "baseline_suite_id": baseline.suite_id,
                    "current_correct": current.correct,
                    "baseline_correct": baseline.correct,
                    "z_score": z,
                    "alpha": self.alpha,
                },
            )
        return None

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def check_all_regressions(self) -> list[RegressionAlert]:
        """Scan all recent suites for regressions."""
        all_alerts = []
        suite_files = list(self.results_dir.glob("*.json"))

        # Group by benchmark/provider/model
        groups = {}
        for filepath in suite_files:
            with open(filepath) as f:
                data = json.load(f)
            key = (data.get("benchmark_name"), data.get("provider"), data.get("model"))
            if key not in groups:
                groups[key] = []
            groups[key].append(self._dict_to_suite(data))

        # For each group, compare latest to previous
        for key, suites in groups.items():
            suites.sort(key=lambda s: s.completed_at, reverse=True)
            if len(suites) >= 2:
                alerts = self.detect_regressions(suites[0], suites[1:])
                all_alerts.extend(alerts)

        return all_alerts