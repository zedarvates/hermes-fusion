#!/usr/bin/env python3
"""Test suite pour OR-Tools Optimizer."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ortools_optimizer import (
    ORToolsOptimizer, QuerySpec, ProviderProfile, _ORTOOLS_AVAILABLE
)


def test_ortools_available():
    assert _ORTOOLS_AVAILABLE, "OR-Tools should be installed"
    print("✅ OR-Tools available")


def test_basic_allocation():
    queries = [
        QuerySpec(id="q1", task_type="chat", priority=1, max_latency=5.0),
        QuerySpec(id="q2", task_type="code", priority=2, max_latency=10.0),
    ]
    providers = [
        ProviderProfile(name="deepseek", model="deepseek-chat",
                        cost_per_1k_input=0.0005, cost_per_1k_output=0.0015,
                        avg_latency=2.0, quality_score=0.85,
                        supports_task_types=["chat", "code"]),
        ProviderProfile(name="local", model="gemma",
                        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
                        avg_latency=8.0, quality_score=0.7,
                        supports_task_types=["chat"]),
    ]

    opt = ORToolsOptimizer(budget_daily=10.0)
    allocs = opt.optimize(queries, providers)

    assert len(allocs) == 2, f"Expected 2 allocations, got {len(allocs)}"
    for a in allocs:
        assert a.query_id in ("q1", "q2"), f"Unexpected query_id: {a.query_id}"
        assert a.provider_name in ("deepseek", "local")

    print(f"✅ Basic allocation: {len(allocs)} queries, ${opt.budget_spent:.4f}")


def test_budget_constraint():
    """Budget serré: doit forcer vers le provider gratuit."""
    queries = [
        QuerySpec(id="q1", task_type="chat", priority=1, max_latency=30.0),
        QuerySpec(id="q2", task_type="chat", priority=1, max_latency=30.0),
    ]
    providers = [
        ProviderProfile(name="paid", model="gpt-4",
                        cost_per_1k_input=0.01, cost_per_1k_output=0.03,
                        avg_latency=1.0, quality_score=0.95,
                        supports_task_types=["chat"]),
        ProviderProfile(name="free", model="gemma",
                        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
                        avg_latency=8.0, quality_score=0.7,
                        supports_task_types=["chat"]),
    ]

    opt = ORToolsOptimizer(budget_daily=0.001)  # $0.001 — très serré
    allocs = opt.optimize(queries, providers)

    # Au moins une requête doit aller vers le free
    free_allocs = [a for a in allocs if a.provider_name == "free"]
    assert len(free_allocs) >= 1, "Budget constraint should force free provider"
    assert opt.budget_spent <= 0.0015, f"Budget exceeded: ${opt.budget_spent:.6f}"

    print(f"✅ Budget constraint respected: ${opt.budget_spent:.6f}")


def test_fallback():
    """Test fallback avec providers vides ou sans OR-Tools."""
    queries = [QuerySpec(id="q1", task_type="chat")]
    opt = ORToolsOptimizer()

    # Providers vides
    allocs = opt.optimize(queries, [])
    assert len(allocs) == 0, "Should return empty with no providers"
    print("✅ Empty providers fallback OK")

    # Queries vides
    allocs = opt.optimize([], [
        ProviderProfile(name="test", model="t", supports_task_types=["chat"])
    ])
    assert len(allocs) == 0, "Should return empty with no queries"
    print("✅ Empty queries fallback OK")


def test_task_type_incompatibility():
    """Query reasoning ne peut pas aller sur un provider chat-only."""
    queries = [QuerySpec(id="q1", task_type="reasoning")]
    providers = [
        ProviderProfile(name="chat-only", model="chat",
                        supports_task_types=["chat"]),
    ]

    opt = ORToolsOptimizer()
    allocs = opt.optimize(queries, providers)
    assert len(allocs) == 0, "Should not allocate incompatible task type"
    print("✅ Task type incompatibility respected")


if __name__ == "__main__":
    test_ortools_available()
    test_basic_allocation()
    test_budget_constraint()
    test_fallback()
    test_task_type_incompatibility()
    print("\n🎉 All tests passed!")
