"""Deterministic benchmark for measuring memory-induced negative transfer."""

from __future__ import annotations

from dataclasses import dataclass

from remem.routing.counterfactual import CounterfactualRouter


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A matched decision where memory may help or hurt."""

    case_id: str
    utility_with_memory: float
    utility_without_memory: float
    memory_id: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if self.memory_id is not None and not self.memory_id.strip():
            raise ValueError("memory_id must not be empty when provided")


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    """Auditable routing outcome for one benchmark case."""

    case_id: str
    selected_route: str
    utility_delta: float
    negative_transfer: bool
    regret: float
    memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate routing metrics for a benchmark run."""

    total_cases: int
    memory_selected: int
    self_reasoning_selected: int
    negative_transfer_cases: int
    selected_negative_transfer_cases: int
    avoided_negative_transfer_cases: int
    routing_regret: float
    case_results: tuple[BenchmarkCaseResult, ...] = ()

    @property
    def negative_transfer_rate(self) -> float:
        """Return the share of cases where memory has lower utility."""

        return self.negative_transfer_cases / self.total_cases if self.total_cases else 0.0

    @property
    def memory_induced_negative_transfer_rate(self) -> float:
        """Return negative transfer among cases where memory was selected."""

        return (
            self.selected_negative_transfer_cases / self.memory_selected
            if self.memory_selected
            else 0.0
        )

    @property
    def negative_transfer_avoidance_rate(self) -> float:
        """Return the share of negative-transfer opportunities the router avoided."""

        return (
            self.avoided_negative_transfer_cases / self.negative_transfer_cases
            if self.negative_transfer_cases
            else 0.0
        )


def run_benchmark(cases: list[BenchmarkCase], router: CounterfactualRouter) -> BenchmarkResult:
    """Route matched cases and measure both exposure and avoided negative transfer."""
    _validate_unique_case_ids(cases)
    memory_selected = 0
    self_reasoning_selected = 0
    negative_transfer_cases = 0
    selected_negative_transfer_cases = 0
    avoided_negative_transfer_cases = 0
    routing_regret = 0.0
    case_results: list[BenchmarkCaseResult] = []

    for case in cases:
        _, decision = router.route(
            evaluate_with_memory=lambda value=case.utility_with_memory: value,
            evaluate_without_memory=lambda value=case.utility_without_memory: value,
        )
        utility_delta = case.utility_with_memory - case.utility_without_memory
        memory_is_worse = utility_delta < 0.0
        if memory_is_worse:
            negative_transfer_cases += 1
        if decision.route == "memory":
            memory_selected += 1
            regret = max(0.0, -utility_delta)
            if memory_is_worse:
                selected_negative_transfer_cases += 1
                routing_regret += regret
        else:
            self_reasoning_selected += 1
            regret = max(0.0, utility_delta)
            if memory_is_worse:
                avoided_negative_transfer_cases += 1

        case_results.append(
            BenchmarkCaseResult(
                case_id=case.case_id,
                selected_route=decision.route,
                utility_delta=utility_delta,
                negative_transfer=memory_is_worse,
                regret=regret,
                memory_id=case.memory_id,
            )
        )

    return BenchmarkResult(
        total_cases=len(cases),
        memory_selected=memory_selected,
        self_reasoning_selected=self_reasoning_selected,
        negative_transfer_cases=negative_transfer_cases,
        selected_negative_transfer_cases=selected_negative_transfer_cases,
        avoided_negative_transfer_cases=avoided_negative_transfer_cases,
        routing_regret=routing_regret,
        case_results=tuple(case_results),
    )


def _validate_unique_case_ids(cases: list[BenchmarkCase]) -> None:
    """Reject duplicate identifiers so aggregate results remain auditable."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case_id values must be unique")


if __name__ == "__main__":
    benchmark_cases = [
        BenchmarkCase("positive_1", 0.90, 0.70, memory_id="memory_a"),
        BenchmarkCase("positive_2", 0.85, 0.60, memory_id="memory_b"),
        BenchmarkCase("negative_1", 0.35, 0.80, memory_id="memory_a"),
        BenchmarkCase("negative_2", 0.40, 0.75, memory_id="memory_b"),
    ]
    result = run_benchmark(benchmark_cases, CounterfactualRouter(minimum_delta=0.05))
    print(f"negative_transfer_rate={result.negative_transfer_rate:.3f}")
    print(f"negative_transfer_avoidance_rate={result.negative_transfer_avoidance_rate:.3f}")
    print(f"routing_regret={result.routing_regret:.3f}")
