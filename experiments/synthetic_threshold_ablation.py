"""Threshold ablation for the synthetic negative-transfer benchmark."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.report_io import atomic_write_json
from experiments.synthetic_negative_transfer import (
    BenchmarkCase,
    BenchmarkResult,
    load_benchmark_cases,
    run_benchmark,
)
from remem.routing.counterfactual import CounterfactualRouter


@dataclass(frozen=True, slots=True)
class ThresholdAblationResult:
    """Measured benchmark result for one counterfactual routing threshold."""

    minimum_delta: float
    benchmark_result: BenchmarkResult


def run_threshold_ablation(
    cases: list[BenchmarkCase], minimum_deltas: list[float]
) -> list[ThresholdAblationResult]:
    """Evaluate the same matched cases across unique routing thresholds."""

    _validate_thresholds(minimum_deltas)
    return [
        ThresholdAblationResult(
            minimum_delta=minimum_delta,
            benchmark_result=run_benchmark(
                cases,
                CounterfactualRouter(minimum_delta=minimum_delta),
            ),
        )
        for minimum_delta in minimum_deltas
    ]


def save_threshold_ablation_evidence(
    cases: list[BenchmarkCase],
    minimum_deltas: list[float],
    output_path: str | Path,
) -> Path:
    """Persist exact ablation inputs and measured outputs as deterministic JSON."""

    results = run_threshold_ablation(cases, minimum_deltas)
    payload = {
        "ablation": "counterfactual_router_minimum_delta",
        "cases": [asdict(case) for case in cases],
        "minimum_deltas": minimum_deltas,
        "results": [
            {
                "minimum_delta": result.minimum_delta,
                "benchmark_result": {
                    **asdict(result.benchmark_result),
                    "mean_routing_regret": result.benchmark_result.mean_routing_regret,
                    "memory_induced_negative_transfer_rate": (
                        result.benchmark_result.memory_induced_negative_transfer_rate
                    ),
                    "negative_transfer_avoidance_rate": (
                        result.benchmark_result.negative_transfer_avoidance_rate
                    ),
                    "negative_transfer_rate": result.benchmark_result.negative_transfer_rate,
                },
            }
            for result in results
        ],
    }
    return atomic_write_json(output_path, payload)


def _validate_thresholds(minimum_deltas: list[float]) -> None:
    """Reject ambiguous or invalid threshold sweeps."""

    if not minimum_deltas:
        raise ValueError("minimum_deltas must not be empty")
    if any(not math.isfinite(value) for value in minimum_deltas):
        raise ValueError("minimum_deltas must contain only finite values")
    if len(minimum_deltas) != len(set(minimum_deltas)):
        raise ValueError("minimum_deltas must be unique")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for evidence-producing threshold sweeps."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", required=True, type=Path, help="JSON array of matched benchmark cases"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="path for deterministic JSON evidence"
    )
    parser.add_argument(
        "--minimum-delta",
        required=True,
        type=float,
        action="append",
        dest="minimum_deltas",
        help="routing threshold to evaluate; repeat for each threshold",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a configured threshold sweep over one auditable matched case set."""

    arguments = _build_argument_parser().parse_args(argv)
    cases = load_benchmark_cases(arguments.cases)
    save_threshold_ablation_evidence(cases, arguments.minimum_deltas, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
