"""Deterministic ablation runner for memory-routing strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import fsum

from experiments.synthetic_negative_transfer import BenchmarkCase
from remem.routing.counterfactual import CounterfactualRouter


class AblationStrategy(StrEnum):
    """Routing policies compared by the synthetic evaluation harness."""

    MEMORY_ALWAYS = "memory_always"
    SELF_REASONING_ALWAYS = "self_reasoning_always"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Aggregate utility and negative-transfer metrics for one strategy."""

    strategy: AblationStrategy
    total_cases: int
    selected_memory: int
    mean_utility: float
    negative_transfer_cases: int
    selected_negative_transfer_cases: int
    routing_regret: float

    @property
    def negative_transfer_rate(self) -> float:
        """Return the share of cases where the selected path was harmful."""

        return (
            self.selected_negative_transfer_cases / self.total_cases
            if self.total_cases
            else 0.0
        )


def run_ablations(
    cases: list[BenchmarkCase],
    router: CounterfactualRouter | None = None,
) -> list[AblationResult]:
    """Evaluate fixed and counterfactual routing policies on matched cases."""

    _validate_unique_case_ids(cases)
    selected_router = router or CounterfactualRouter()
    return [
        _evaluate_strategy(cases, AblationStrategy.MEMORY_ALWAYS, selected_router),
        _evaluate_strategy(cases, AblationStrategy.SELF_REASONING_ALWAYS, selected_router),
        _evaluate_strategy(cases, AblationStrategy.COUNTERFACTUAL, selected_router),
    ]


def _evaluate_strategy(
    cases: list[BenchmarkCase],
    strategy: AblationStrategy,
    router: CounterfactualRouter,
) -> AblationResult:
    """Evaluate one routing strategy over matched cases."""

    utilities: list[float] = []
    selected_memory = 0
    selected_negative_transfer_cases = 0
    routing_regret = 0.0

    for case in cases:
        use_memory = _select_memory(case, strategy, router)
        utility = case.utility_with_memory if use_memory else case.utility_without_memory
        utilities.append(utility)
        if use_memory:
            selected_memory += 1
        memory_is_harmful = case.utility_with_memory < case.utility_without_memory
        if use_memory and memory_is_harmful:
            selected_negative_transfer_cases += 1
            routing_regret += case.utility_without_memory - case.utility_with_memory

    return AblationResult(
        strategy=strategy,
        total_cases=len(cases),
        selected_memory=selected_memory,
        mean_utility=fsum(utilities) / len(utilities) if utilities else 0.0,
        negative_transfer_cases=sum(
            case.utility_with_memory < case.utility_without_memory for case in cases
        ),
        selected_negative_transfer_cases=selected_negative_transfer_cases,
        routing_regret=routing_regret,
    )


def _select_memory(
    case: BenchmarkCase,
    strategy: AblationStrategy,
    router: CounterfactualRouter,
) -> bool:
    """Return whether a strategy selects the memory path for one case."""

    if strategy is AblationStrategy.MEMORY_ALWAYS:
        return True
    if strategy is AblationStrategy.SELF_REASONING_ALWAYS:
        return False

    _, decision = router.route(
        evaluate_with_memory=lambda: case.utility_with_memory,
        evaluate_without_memory=lambda: case.utility_without_memory,
    )
    return decision.route == "memory"


def _validate_unique_case_ids(cases: list[BenchmarkCase]) -> None:
    """Reject duplicate identifiers so ablation rows remain traceable."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case_id values must be unique")
