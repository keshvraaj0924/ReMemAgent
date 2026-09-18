"""Statistical summaries for independent benchmark seed reports.

This module computes descriptive statistics only. It does not pool or invent
benchmark observations and does not change routing, training, or evaluation
semantics. Each run contributes one seed-level observation to the mean and
sample standard deviation, preserving independence across repetitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any

from remem.benchmark import BenchmarkRunReport

Z_95 = 1.96
T_95_TWO_SIDED = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive summary of one metric across independent seed runs."""

    sample_size: int
    mean: float
    sample_stddev: float
    standard_error: float
    confidence_interval_95: tuple[float, float]

    def __post_init__(self) -> None:
        """Reject malformed summaries before they can enter research reports."""

        if self.sample_size < 1:
            raise ValueError("sample_size must be at least one")
        scalar_values = (self.mean, self.sample_stddev, self.standard_error)
        if any(not isfinite(value) for value in scalar_values):
            raise ValueError("metric summary values must be finite")
        if self.sample_stddev < 0.0:
            raise ValueError("sample_stddev must be non-negative")
        if self.standard_error < 0.0:
            raise ValueError("standard_error must be non-negative")
        lower_bound, upper_bound = self.confidence_interval_95
        if not isfinite(lower_bound) or not isfinite(upper_bound):
            raise ValueError("confidence interval bounds must be finite")
        if lower_bound > upper_bound:
            raise ValueError("confidence interval lower bound must not exceed upper bound")
        if not lower_bound <= self.mean <= upper_bound:
            raise ValueError("confidence interval must contain the metric mean")


@dataclass(frozen=True, slots=True)
class BenchmarkSeedStatistics:
    """Measured seed-level metrics and descriptive aggregate statistics."""

    benchmark_name: str
    seeds: tuple[int, ...]
    success_rate: MetricSummary
    mean_reward: MetricSummary
    transfer_success_rate: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the statistics."""

        return asdict(self)


def summarize_benchmark_reports(
    reports: Sequence[BenchmarkRunReport],
) -> BenchmarkSeedStatistics:
    """Summarize independent, comparable benchmark seed reports.

    The function requires a non-empty collection with explicit unique seeds,
    one benchmark name, and the same number of episodes in every repetition.
    Seed-level metrics are summarized using the arithmetic mean, sample
    standard deviation, and a two-sided 95% Student t confidence interval.
    Student t intervals are used because research experiments commonly have a
    small number of independent seeds and the population variance is unknown.
    For more than 31 seed runs the critical value falls back to the asymptotic
    normal value. Each metric summary records its seed-level sample size
    explicitly so a serialized report cannot hide how many independent
    repetitions support an uncertainty estimate. With one seed, the interval
    collapses to the observed value and the sample standard deviation and
    standard error are zero.

    Requiring explicit seeds prevents unseeded runs from being presented as
    reproducible independent repetitions. Requiring equal episode counts
    prevents a partially completed run from being silently treated as a full
    benchmark repetition. The statistics remain seed-level rather than pooling
    episodes across repetitions.
    """

    selected_reports = tuple(reports)
    if not selected_reports:
        raise ValueError("reports must contain at least one benchmark report")

    optional_seeds = tuple(report.seed for report in selected_reports)
    if any(seed is None for seed in optional_seeds):
        raise ValueError("benchmark report seeds must be explicit")
    seeds = tuple(seed for seed in optional_seeds if seed is not None)
    if len(seeds) != len(set(seeds)):
        raise ValueError("benchmark report seeds must be unique")

    benchmark_names = {report.benchmark_name for report in selected_reports}
    if len(benchmark_names) != 1:
        raise ValueError("benchmark reports must use one benchmark name")

    episode_counts = {len(report.episodes) for report in selected_reports}
    if len(episode_counts) != 1:
        raise ValueError("benchmark reports must contain the same number of episodes")

    return BenchmarkSeedStatistics(
        benchmark_name=selected_reports[0].benchmark_name,
        seeds=seeds,
        success_rate=_summarize(
            tuple(report.success_rate for report in selected_reports),
            metric_name="success_rate",
        ),
        mean_reward=_summarize(
            tuple(report.mean_reward for report in selected_reports),
            metric_name="mean_reward",
        ),
        transfer_success_rate=_summarize(
            tuple(report.transfer_success_rate for report in selected_reports),
            metric_name="transfer_success_rate",
        ),
    )


def _critical_value_95(sample_size: int) -> float:
    """Return a two-sided 95% critical value for a seed-level mean."""

    if sample_size < 2:
        raise ValueError("sample_size must be at least two for a critical value")
    degrees_of_freedom = sample_size - 1
    if degrees_of_freedom <= len(T_95_TWO_SIDED):
        return T_95_TWO_SIDED[degrees_of_freedom - 1]
    return Z_95


def _summarize(values: tuple[float, ...], *, metric_name: str) -> MetricSummary:
    """Compute descriptive seed-level statistics for one finite metric."""

    if not values:
        raise ValueError(f"{metric_name} observations must not be empty")
    if any(not isfinite(value) for value in values):
        raise ValueError(f"{metric_name} observations must be finite")

    sample_size = len(values)
    mean = sum(values) / sample_size
    if sample_size == 1:
        sample_stddev = 0.0
        standard_error = 0.0
        confidence_interval_95 = (mean, mean)
    else:
        squared_deviations = sum((value - mean) ** 2 for value in values)
        sample_stddev = sqrt(squared_deviations / (sample_size - 1))
        standard_error = sample_stddev / sqrt(sample_size)
        margin = _critical_value_95(sample_size) * standard_error
        confidence_interval_95 = (mean - margin, mean + margin)
    return MetricSummary(
        sample_size=sample_size,
        mean=mean,
        sample_stddev=sample_stddev,
        standard_error=standard_error,
        confidence_interval_95=confidence_interval_95,
    )


__all__ = ["BenchmarkSeedStatistics", "MetricSummary", "summarize_benchmark_reports"]
