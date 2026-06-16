"""Eval Harness - Nightly benchmarks for multi-LLM fusion."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_fusion.engine import FusionEngine


@dataclass(slots=True)
class BenchmarkResult:
    """Single benchmark evaluation result."""
    benchmark_id: str
    benchmark_name: str
    provider: str
    model: str
    task_type: str
    question: str
    expected: str
    response: str
    score: float  # 0.0 - 1.0
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSuite:
    """Collection of benchmark results."""
    suite_id: str
    benchmark_name: str
    provider: str
    model: str
    total_questions: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    avg_cost_usd: float
    total_cost_usd: float
    results: list[BenchmarkResult]
    started_at: float
    completed_at: float
    metadata: dict = field(default_factory=dict)

    @property
    def failed(self) -> int:
        return self.total_questions - self.correct


@dataclass(slots=True)
class RegressionAlert:
    """Detected regression in benchmark performance."""
    benchmark_name: str
    provider: str
    model: str
    current_accuracy: float
    baseline_accuracy: float
    delta: float
    p_value: float
    significant: bool
    timestamp: float
    details: dict = field(default_factory=dict)