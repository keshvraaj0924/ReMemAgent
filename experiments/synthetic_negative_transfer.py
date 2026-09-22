"""Deterministic benchmark for measuring memory-induced negative transfer."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.report_io import atomic_write_json
from remem.routing.counterfactual import CounterfactualRouter


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A matched decision where memory may help or hurt."""

    case_id: str
    utility_with_memory: float
    utility_without_memory: float

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not math.isfinite(self.utility_with_memory):
            raise ValueError("utility_with_memory must be finite")
        if not math.isfinite(self.utility_without_memory):
            raise ValueError("utility_without_memory must be finite")


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

    @property
    def mean_routing_regret(self) -> float:
        """Return mean utility lost relative to the per-case oracle route."""

        return self.routing_regret / self.total_cases if self.total_cases else 0.0


def _constant_utility(value: float) -> Callable[[], float]:
    """Build a typed zero-argument evaluator for one benchmark utility."""

    def evaluate() -> float:
        return value

    return evaluate


def _selected_utility(case: BenchmarkCase, route: str) -> float:
    """Return utility realized by the route selected for one matched case."""

    return case.utility_with_memory if route == "memory" else case.utility_without_memory


def _routing_regret(case: BenchmarkCase, route: str) -> float:
    """Return non-negative utility loss against the best route for one case."""

    oracle_utility = max(case.utility_with_memory, case.utility_without_memory)
    return oracle_utility - _selected_utility(case, route)


def run_benchmark(cases: list[BenchmarkCase], router: CounterfactualRouter) -> BenchmarkResult:
    """Route matched cases and measure both exposure and avoided negative transfer."""
    _validate_unique_case_ids(cases)
    memory_selected = 0
    self_reasoning_selected = 0
    negative_transfer_cases = 0
    selected_negative_transfer_cases = 0
    avoided_negative_transfer_cases = 0
    routing_regret = 0.0

    for case in cases:
        _, decision = router.route(
            evaluate_with_memory=_constant_utility(case.utility_with_memory),
            evaluate_without_memory=_constant_utility(case.utility_without_memory),
        )
        memory_is_worse = case.utility_with_memory < case.utility_without_memory
        if memory_is_worse:
            negative_transfer_cases += 1
        if decision.route == "memory":
            memory_selected += 1
            if memory_is_worse:
                selected_negative_transfer_cases += 1
        else:
            self_reasoning_selected += 1
            if memory_is_worse:
                avoided_negative_transfer_cases += 1
        routing_regret += _routing_regret(case, decision.route)

    return BenchmarkResult(
        total_cases=len(cases),
        memory_selected=memory_selected,
        self_reasoning_selected=self_reasoning_selected,
        negative_transfer_cases=negative_transfer_cases,
        selected_negative_transfer_cases=selected_negative_transfer_cases,
        avoided_negative_transfer_cases=avoided_negative_transfer_cases,
        routing_regret=routing_regret,
    )


def save_benchmark_evidence(
    cases: list[BenchmarkCase],
    router: CounterfactualRouter,
    output_path: str | Path,
) -> Path:
    """Execute the synthetic benchmark and persist its inputs and measured outputs."""

    result = run_benchmark(cases, router)
    payload = {
        "benchmark": "synthetic_negative_transfer",
        "cases": [asdict(case) for case in cases],
        "router": {"minimum_delta": router.minimum_delta},
        "result": {
            **asdict(result),
            "mean_routing_regret": result.mean_routing_regret,
            "memory_induced_negative_transfer_rate": result.memory_induced_negative_transfer_rate,
            "negative_transfer_avoidance_rate": result.negative_transfer_avoidance_rate,
            "negative_transfer_rate": result.negative_transfer_rate,
        },
    }
    return atomic_write_json(output_path, payload)


def load_benchmark_cases(input_path: str | Path) -> list[BenchmarkCase]:
    """Load and strictly validate benchmark cases from a JSON array."""

    path = Path(input_path)
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load benchmark cases from {path}") from exc
    if not isinstance(payload, list):
        raise ValueError("benchmark case file must contain a JSON array")

    expected_fields = {"case_id", "utility_with_memory", "utility_without_memory"}
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError(f"benchmark case at index {index} must contain exactly {sorted(expected_fields)}")
        case_id = item["case_id"]
        with_memory = item["utility_with_memory"]
        without_memory = item["utility_without_memory"]
        if not isinstance(case_id, str):
            raise ValueError(f"benchmark case at index {index} has non-string case_id")
        if isinstance(with_memory, bool) or not isinstance(with_memory, (int, float)):
            raise ValueError(f"benchmark case at index {index} has invalid utility_with_memory")
        if isinstance(without_memory, bool) or not isinstance(without_memory, (int, float)):
            raise ValueError(f"benchmark case at index {index} has invalid utility_without_memory")
        cases.append(BenchmarkCase(case_id, float(with_memory), float(without_memory)))

    _validate_unique_case_ids(cases)
    return cases


def _validate_unique_case_ids(cases: list[BenchmarkCase]) -> None:
    """Reject duplicate identifiers so aggregate results remain auditable."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case_id values must be unique")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for evidence-producing benchmark runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSON array of matched benchmark cases")
    parser.add_argument("--output", required=True, type=Path, help="path for deterministic JSON evidence")
    parser.add_argument("--minimum-delta", type=float, default=0.05, help="minimum memory utility advantage")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one configured benchmark and persist auditable evidence."""

    arguments = _build_argument_parser().parse_args(argv)
    if not math.isfinite(arguments.minimum_delta):
        raise ValueError("minimum_delta must be finite")
    cases = load_benchmark_cases(arguments.cases)
    save_benchmark_evidence(
        cases,
        CounterfactualRouter(minimum_delta=arguments.minimum_delta),
        arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
