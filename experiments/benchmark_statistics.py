"""Descriptive statistics for independent benchmark seed reports.

The statistics layer operates on one aggregate observation per seed. It never
pools episode-level observations across seeds and never performs significance
testing. This keeps uncertainty estimates aligned with the independent-run
experimental design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Sequence

from remem.benchmark import BenchmarkRunReport

CONFIDENCE_Z_95 = 1.96


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive summary for one metric across independent seed runs."""

    mean: float
    sample_stddev: float
    standard_error: float
    confidence_interval_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BenchmarkSeedStatistics:
    """Seed-level benchmark statistics suitable for JSON serialization."""

    benchmark_name: str
    seeds: tuple[int | None, ...]
    success_rate: MetricSummary
    mean_reward: MetricSummary
    transfer_success_rate: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the summary."""

        return asdict(self)


def summarize_benchmark_reports(
    reports: Sequence[BenchmarkRunReport],
) -> BenchmarkSeedStatistics:
    """Summarize independent benchmark runs without pooling episodes.

    A single run has zero estimated sampling variability because there is no
    second independent observation from which to estimate it. The resulting
    interval therefore collapses to the observed metric value.
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
        success_rate=_summarize(tuple(report.success_rate for report in selected_reports)),
        mean_reward=_summarize(tuple(report.mean_reward for report in selected_reports)),
        transfer_success_rate=_summarize(
            tuple(report.transfer_success_rate for report in selected_reports)
        ),
    )


def _summarize(values: tuple[float, ...]) -> MetricSummary:
    """Compute mean, sample deviation, standard error, and 95% interval."""

    mean = sum(values) / len(values)
    if len(values) == 1:
        sample_stddev = 0.0
    else:
        sample_stddev = sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
    standard_error = sample_stddev / sqrt(len(values))
    margin = CONFIDENCE_Z_95 * standard_error
    return MetricSummary(
        mean=mean,
        sample_stddev=sample_stddev,
        standard_error=standard_error,
        confidence_interval_95=(mean - margin, mean + margin),
    )


__all__ = ["BenchmarkSeedStatistics", "MetricSummary", "summarize_benchmark_reports"]
