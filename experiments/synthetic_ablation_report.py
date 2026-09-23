"""Build deterministic reports from verified synthetic threshold-ablation evidence.

This module deliberately accepts persisted evidence only through the replay-verifying
loader. Reporting therefore cannot accidentally summarize an unverified JSON aggregate.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.report_io import atomic_write_json
from experiments.synthetic_ablation_evidence import (
    VerifiedThresholdAblationEvidence,
    load_verified_threshold_ablation_evidence,
)
from experiments.synthetic_negative_transfer import BenchmarkResult


@dataclass(frozen=True, slots=True)
class ThresholdMetricRecord:
    """Reportable measured metrics for one verified routing threshold."""

    minimum_delta: float
    memory_selected: int
    self_reasoning_selected: int
    selected_negative_transfer_cases: int
    avoided_negative_transfer_cases: int
    routing_regret: float
    mean_routing_regret: float
    memory_induced_negative_transfer_rate: float
    negative_transfer_avoidance_rate: float


@dataclass(frozen=True, slots=True)
class VerifiedThresholdAblationReport:
    """Deterministic report derived exclusively from replay-verified evidence."""

    ablation: str
    case_count: int
    negative_transfer_cases: int
    thresholds: tuple[ThresholdMetricRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation suitable for research artifacts."""

        return asdict(self)


def build_verified_threshold_ablation_report(
    evidence: VerifiedThresholdAblationEvidence,
) -> VerifiedThresholdAblationReport:
    """Summarize verified results without introducing inferential claims."""

    if not evidence.results:
        raise ValueError("verified threshold ablation evidence must contain results")

    first_result = evidence.results[0].benchmark_result
    case_count = len(evidence.cases)
    if first_result.total_cases != case_count:
        raise ValueError("verified result case count does not match embedded cases")

    negative_transfer_cases = first_result.negative_transfer_cases
    records: list[ThresholdMetricRecord] = []
    for result in evidence.results:
        benchmark_result = result.benchmark_result
        _validate_comparable_result(
            benchmark_result,
            case_count=case_count,
            negative_transfer_cases=negative_transfer_cases,
        )
        records.append(
            ThresholdMetricRecord(
                minimum_delta=result.minimum_delta,
                memory_selected=benchmark_result.memory_selected,
                self_reasoning_selected=benchmark_result.self_reasoning_selected,
                selected_negative_transfer_cases=(
                    benchmark_result.selected_negative_transfer_cases
                ),
                avoided_negative_transfer_cases=benchmark_result.avoided_negative_transfer_cases,
                routing_regret=benchmark_result.routing_regret,
                mean_routing_regret=benchmark_result.mean_routing_regret,
                memory_induced_negative_transfer_rate=(
                    benchmark_result.memory_induced_negative_transfer_rate
                ),
                negative_transfer_avoidance_rate=(
                    benchmark_result.negative_transfer_avoidance_rate
                ),
            )
        )

    return VerifiedThresholdAblationReport(
        ablation="counterfactual_router_minimum_delta",
        case_count=case_count,
        negative_transfer_cases=negative_transfer_cases,
        thresholds=tuple(records),
    )


def _validate_comparable_result(
    result: BenchmarkResult,
    *,
    case_count: int,
    negative_transfer_cases: int,
) -> None:
    """Reject internally incomparable threshold results before reporting."""

    if result.total_cases != case_count:
        raise ValueError("threshold results must use the same case count")
    if result.negative_transfer_cases != negative_transfer_cases:
        raise ValueError("threshold results must use the same negative-transfer cases")
    if result.memory_selected + result.self_reasoning_selected != case_count:
        raise ValueError("threshold routing counts must partition all benchmark cases")
    if (
        result.selected_negative_transfer_cases + result.avoided_negative_transfer_cases
        != negative_transfer_cases
    ):
        raise ValueError("negative-transfer routing counts must partition negative-transfer cases")


def write_verified_threshold_ablation_report(
    evidence_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Verify persisted evidence, derive its report, and write deterministic JSON."""

    evidence = load_verified_threshold_ablation_evidence(evidence_path)
    report = build_verified_threshold_ablation_report(evidence)
    return atomic_write_json(output_path, report.to_dict())


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for replay-verified threshold reporting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path, help="threshold evidence JSON")
    parser.add_argument("--output", required=True, type=Path, help="verified report JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a report only after deterministic evidence verification succeeds."""

    arguments = _build_argument_parser().parse_args(argv)
    write_verified_threshold_ablation_report(arguments.evidence, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ThresholdMetricRecord",
    "VerifiedThresholdAblationReport",
    "build_verified_threshold_ablation_report",
    "write_verified_threshold_ablation_report",
]
