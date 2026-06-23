#!/usr/bin/env python3
"""
ortools_optimizer.py — Solveur CP-SAT pour allocation de modèles LLM sous contraintes.

Utilise OR-Tools pour optimiser l'allocation des requêtes entre providers.
Import optionnel: si OR-Tools n'est pas installé, le fallback est silencieux.

Usage:
    from ortools_optimizer import ORToolsOptimizer
    opt = ORToolsOptimizer(budget_daily=10.0)
    allocations = opt.optimize(queries, providers)
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    from ortools.sat.python import cp_model
    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False


@dataclass
class QuerySpec:
    """Spécification d'une requête à router."""
    id: str
    task_type: str = "chat"  # chat, code, reasoning, creative
    max_latency: float = 30.0  # secondes
    min_quality: float = 0.5  # 0-1
    estimated_input_tokens: int = 1000
    estimated_output_tokens: int = 500
    priority: int = 1  # 1=normal, 2=important, 3=critique


@dataclass
class ProviderProfile:
    """Profil d'un provider de modèle."""
    name: str
    model: str
    cost_per_1k_input: float = 0.0  # dollars
    cost_per_1k_output: float = 0.0  # dollars
    avg_latency: float = 5.0  # secondes
    quality_score: float = 0.8  # 0-1
    max_concurrent: int = 5
    supports_task_types: list[str] = field(default_factory=lambda: ["chat"])


@dataclass
class Allocation:
    """Résultat d'allocation d'une requête à un provider."""
    query_id: str
    provider_name: str
    model: str
    estimated_cost: float
    estimated_latency: float
    quality_score: float


class ORToolsOptimizer:
    """
    Optimiseur d'allocation de requêtes entre providers via CP-SAT.

    Contraintes supportées:
      - Budget journalier (dollars)
      - Latence max par requête
      - Qualité minimale
      - Compatibilité task_type → provider

    Fallback: allocation round-robin basique si OR-Tools indisponible.
    """

    def __init__(
        self,
        budget_daily: float = 10.0,
        budget_alert_threshold: float = 0.8,
        solver_timeout_seconds: int = 5,
    ):
        self.budget_daily = budget_daily
        self.budget_spent = 0.0
        self.budget_alert_threshold = budget_alert_threshold
        self.solver_timeout = solver_timeout_seconds

    def optimize(
        self,
        queries: list[QuerySpec],
        providers: list[ProviderProfile],
    ) -> list[Allocation]:
        """
        Optimise l'allocation de queries vers providers.

        Args:
            queries: Liste des requêtes à router
            providers: Liste des providers disponibles

        Returns:
            Liste des allocations (1 par query)
        """
        if not _ORTOOLS_AVAILABLE:
            return self._fallback_allocate(queries, providers)

        if not queries or not providers:
            return []

        return self._cp_sat_optimize(queries, providers)

    def _cp_sat_optimize(
        self,
        queries: list[QuerySpec],
        providers: list[ProviderProfile],
    ) -> list[Allocation]:
        """Optimisation via CP-SAT."""
        model = cp_model.CpModel()

        num_queries = len(queries)
        num_providers = len(providers)

        # Variables: x[q][p] = 1 si la query q va au provider p
        x = {}
        for q in range(num_queries):
            for p in range(num_providers):
                x[(q, p)] = model.NewBoolVar(f"q{q}_p{p}")

        # Contrainte: chaque query va exactement à 1 provider
        for q in range(num_queries):
            model.Add(sum(x[(q, p)] for p in range(num_providers)) == 1)

        # Contrainte: compatibilité task_type
        for q, query in enumerate(queries):
            for p, provider in enumerate(providers):
                if query.task_type not in provider.supports_task_types:
                    model.Add(x[(q, p)] == 0)

        # Contrainte: latence max
        for q, query in enumerate(queries):
            for p, provider in enumerate(providers):
                if provider.avg_latency > query.max_latency:
                    model.Add(x[(q, p)] == 0)

        # Contrainte: qualité minimale
        for q, query in enumerate(queries):
            for p, provider in enumerate(providers):
                if provider.quality_score < query.min_quality:
                    model.Add(x[(q, p)] == 0)

        # Contrainte: budget journalier (OR-Tools = entiers, on utilise des micro-dollars)
        COST_SCALE = 1_000_000  # Convertir dollars → micro-dollars (entiers)

        total_cost_micro = sum(
            x[(q, p)] * int((
                queries[q].estimated_input_tokens / 1000 * provider.cost_per_1k_input
                + queries[q].estimated_output_tokens / 1000 * provider.cost_per_1k_output
            ) * COST_SCALE + 0.5)
            for q in range(num_queries)
            for p in range(num_providers)
        )
        remaining_budget_micro = int((self.budget_daily - self.budget_spent) * COST_SCALE + 0.5)
        if remaining_budget_micro > 0:
            model.Add(total_cost_micro <= remaining_budget_micro)

        # Objectif: maximiser la qualité - minimiser le coût
        # Quality scale = 1M (même scale que cost), priorité multiplie
        QUALITY_SCALE = COST_SCALE
        model.Maximize(
            sum(x[(q, p)] * int(provider.quality_score * QUALITY_SCALE + 0.5) * queries[q].priority
                for q in range(num_queries) for p in range(num_providers))
            - total_cost_micro
        )

        # Résoudre
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solver_timeout
        status = solver.Solve(model)

        # Extraire les allocations
        allocations = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for q, query in enumerate(queries):
                for p, provider in enumerate(providers):
                    if solver.Value(x[(q, p)]) == 1:
                        cost = (
                            query.estimated_input_tokens / 1000 * provider.cost_per_1k_input
                            + query.estimated_output_tokens / 1000 * provider.cost_per_1k_output
                        )
                        allocations.append(Allocation(
                            query_id=query.id,
                            provider_name=provider.name,
                            model=provider.model,
                            estimated_cost=round(cost, 6),
                            estimated_latency=provider.avg_latency,
                            quality_score=provider.quality_score,
                        ))
                        self.budget_spent += cost
                        break

        # Fallback: les queries non assignées vont en round-robin
        if len(allocations) < num_queries:
            allocated_ids = {a.query_id for a in allocations}
            unallocated = [q for q in queries if q.id not in allocated_ids]
            fallback_alloc = self._fallback_allocate(unallocated, providers)
            allocations.extend(fallback_alloc)

        return allocations

    def _fallback_allocate(
        self,
        queries: list[QuerySpec],
        providers: list[ProviderProfile],
    ) -> list[Allocation]:
        """Fallback round-robin basique quand OR-Tools n'est pas disponible."""
        if not providers:
            return []

        allocations = []
        for i, query in enumerate(queries):
            # Filtrer les providers compatibles avec le task_type
            compatible = [p for p in providers if query.task_type in p.supports_task_types]
            if not compatible:
                continue  # Skip this query, no compatible provider
            provider = compatible[i % len(compatible)]
            cost = (
                query.estimated_input_tokens / 1000 * provider.cost_per_1k_input
                + query.estimated_output_tokens / 1000 * provider.cost_per_1k_output
            )
            allocations.append(Allocation(
                query_id=query.id,
                provider_name=provider.name,
                model=provider.model,
                estimated_cost=round(cost, 6),
                estimated_latency=provider.avg_latency,
                quality_score=provider.quality_score,
            ))
        return allocations

    def reset_budget(self, new_daily: Optional[float] = None):
        """Reset le compteur de budget (appelé quotidiennement)."""
        self.budget_spent = 0.0
        if new_daily is not None:
            self.budget_daily = new_daily


# Module-level singleton (pour intégration facile)
_optimizer: Optional[ORToolsOptimizer] = None


def get_optimizer(budget_daily: float = 10.0) -> ORToolsOptimizer:
    """Get or create the global optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ORToolsOptimizer(budget_daily=budget_daily)
    return _optimizer


if __name__ == "__main__":
    # Test
    print(f"OR-Tools available: {_ORTOOLS_AVAILABLE}")

    queries = [
        QuerySpec(id="q1", task_type="chat", priority=1, max_latency=5.0),
        QuerySpec(id="q2", task_type="code", priority=2, max_latency=10.0),
        QuerySpec(id="q3", task_type="reasoning", priority=3, max_latency=30.0),
    ]

    providers = [
        ProviderProfile(name="deepseek", model="deepseek-chat",
                        cost_per_1k_input=0.0005, cost_per_1k_output=0.0015,
                        avg_latency=2.0, quality_score=0.85,
                        supports_task_types=["chat", "code"]),
        ProviderProfile(name="openai", model="gpt-4o",
                        cost_per_1k_input=0.01, cost_per_1k_output=0.03,
                        avg_latency=1.0, quality_score=0.95,
                        supports_task_types=["chat", "code", "reasoning", "creative"]),
        ProviderProfile(name="local", model="gemma-3-27b",
                        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
                        avg_latency=8.0, quality_score=0.7,
                        supports_task_types=["chat", "reasoning"]),
    ]

    opt = ORToolsOptimizer(budget_daily=1.0)
    allocations = opt.optimize(queries, providers)

    print("\nAllocations:")
    for a in allocations:
        print(f"  {a.query_id} → {a.provider_name}/{a.model}")
        print(f"       cost=${a.estimated_cost:.6f}, latency={a.estimated_latency}s, quality={a.quality_score}")

    print(f"\nBudget spent: ${opt.budget_spent:.4f} / ${opt.budget_daily:.2f}")
