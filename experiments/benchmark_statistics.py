"""Statistical summaries for independent benchmark seed reports.

This module computes descriptive statistics only. It does not pool or invent
benchmark observations and does not change routing, training, or evaluation
semantics. Each run contributes one seed-level observation to the mean and
sample standard deviation, preserving independence across repetitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Sequence

from remem.benchmark import BenchmarkRunReport

Z_95 = 1.96


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive summary of one metric across independent seed runs."""

    mean: float
    sample_stddev: float
    standard_error: float
    confidence_interval_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BenchmarkSeedStatistics:
    """Measured seed-level metrics and descriptive aggregate statistics."""

    benchmark_name: str
    seeds: tuple[int | None, ...]
    success_rate: MetricSummary
    mean_reward: MetricSummary
    transfer_success_rate: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the statistics."""

        return asdict(self)


def summarize_benchmark_reports(
    reports: Sequence[BenchmarkRunReport],
) -> BenchmarkSeedStatistics:
    """Summarize independent benchmark reports without pooling their episodes.

    The function requires a non-empty collection with unique seeds and one
    benchmark name. Seed-level metrics are summarized using the arithmetic
    mean, sample standard deviation, and a normal-approximation 95% confidence
    interval. With one seed, the interval collapses to the observed value and
    the sample standard deviation and standard error are zero.
    """

    selected_reports = tuple(reports)
    if not selected_reports:
        raise ValueError("reports must contain at least one benchmark report")

    seeds = tuple(report.seed for report in selected_reports)
    if len(seeds) != len(set(seeds)):
        raise ValueError("benchmark report seeds must be unique")

    benchmark_names = {report.benchmark_name for report in selected_reports}
    if len(benchmark_names) != 1:
        raise ValueError("benchmark reports must use one benchmark name")

    return BenchmarkSeedStatistics(
        benchmark_name=selected_reports[0].benchmark_name,
        seeds=seeds,
        success_rate=_summarize(
            tuple(report.success_rate for report in selected_reports)
        ),
        mean_reward=_summarize(
            tuple(report.mean_reward for report in selected_reports)
        ),
        transfer_success_rate=_summarize(
            tuple(report.transfer_success_rate for report in selected_reports)
        ),
    )


def _summarize(values: tuple[float, ...]) -> MetricSummary:
    """Compute descriptive seed-level statistics for one metric."""

    mean = sum(values) / len(values)
    if len(values) == 1:
        sample_stddev = 0.0
    else:
        squared_deviations = sum((value - mean) ** 2 for value in values)
        sample_stddev = sqrt(squared_deviations / (len(values) - 1))
    standard_error = sample_stddev / sqrt(len(values))
    margin = Z_95 * standard_error
    return MetricSummary(
        mean=mean,
        sample_stddev=sample_stddev,
        standard_error=standard_error,
        confidence_interval_95=(mean - margin, mean + margin),
    )


__all__ = ["BenchmarkSeedStatistics", "MetricSummary", "summarize_benchmark_reports"]
