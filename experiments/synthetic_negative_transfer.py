"""Minimal deterministic benchmark for measuring memory-induced negative transfer."""

from __future__ import annotations

from dataclasses import dataclass

from remem.routing.counterfactual import CounterfactualRouter


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A matched decision where memory may help or hurt."""

    case_id: str
    utility_with_memory: float
    utility_without_memory: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate routing metrics for a benchmark run."""

    total_cases: int
    memory_selected: int
    self_reasoning_selected: int
    negative_transfer_cases: int

    @property
    def negative_transfer_rate(self) -> float:
        return self.negative_transfer_cases / self.total_cases if self.total_cases else 0.0


def run_benchmark(cases: list[BenchmarkCase], router: CounterfactualRouter) -> BenchmarkResult:
    """Route every matched case and measure how often memory would hurt."""
    memory_selected = 0
    self_reasoning_selected = 0
    negative_transfer_cases = 0

    for case in cases:
        _, decision = router.route(
            evaluate_with_memory=lambda value=case.utility_with_memory: value,
            evaluate_without_memory=lambda value=case.utility_without_memory: value,
        )
        if case.utility_with_memory < case.utility_without_memory:
            negative_transfer_cases += 1
        if decision.route == "memory":
            memory_selected += 1
        else:
            self_reasoning_selected += 1

    return BenchmarkResult(
        total_cases=len(cases),
        memory_selected=memory_selected,
        self_reasoning_selected=self_reasoning_selected,
        negative_transfer_cases=negative_transfer_cases,
    )


if __name__ == "__main__":
    benchmark_cases = [
        BenchmarkCase("positive_1", 0.90, 0.70),
        BenchmarkCase("positive_2", 0.85, 0.60),
        BenchmarkCase("negative_1", 0.35, 0.80),
        BenchmarkCase("negative_2", 0.40, 0.75),
    ]
    result = run_benchmark(benchmark_cases, CounterfactualRouter(minimum_delta=0.05))
    print(f"negative_transfer_rate={result.negative_transfer_rate:.3f}")
