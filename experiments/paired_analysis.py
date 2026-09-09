"""Inferential analysis for paired benchmark conditions.

This module composes the repository's descriptive comparison, exact paired
sign-flip test, Holm correction, and paired effect-size utilities. Statistical
analysis remains downstream of the benchmark and routing layers: it cannot
change policy behavior or memory selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

from experiments.benchmark_statistics import (
    compare_benchmark_reports,
    exact_paired_sign_flip_test,
    holm_bonferroni_adjust,
)
from experiments.effect_sizes import paired_cohens_dz
from remem.benchmark import BenchmarkRunReport


METRIC_NAMES = ("success_rate", "mean_reward", "transfer_success_rate")
MetricGetter = Callable[[BenchmarkRunReport], float]
_METRIC_GETTERS: dict[str, MetricGetter] = {
    "success_rate": lambda report: report.success_rate,
    "mean_reward": lambda report: report.mean_reward,
    "transfer_success_rate": lambda report: report.transfer_success_rate,
}


@dataclass(frozen=True, slots=True)
class PairedMetricAnalysis:
    """Inferential statistics for one paired benchmark metric."""

    observed_mean_delta: float
    p_value: float
    adjusted_p_value: float
    effect_size_dz: float | None
    sample_size: int

    def __post_init__(self) -> None:
        """Reject malformed statistical values before serialization."""

        if not isfinite(self.observed_mean_delta):
            raise ValueError("observed_mean_delta must be finite")
        _validate_probability(self.p_value, "p_value")
        _validate_probability(self.adjusted_p_value, "adjusted_p_value")
        if self.effect_size_dz is not None and not isfinite(self.effect_size_dz):
            raise ValueError("effect_size_dz must be finite when provided")
        if not isinstance(self.sample_size, int) or isinstance(self.sample_size, bool):
            raise TypeError("sample_size must be an integer")
        if self.sample_size < 1:
            raise ValueError("sample_size must be at least one")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedBenchmarkAnalysis:
    """Complete paired analysis across the benchmark's primary metrics."""

    baseline_label: str
    treatment_label: str
    seeds: tuple[int, ...]
    metrics: dict[str, PairedMetricAnalysis]

    def __post_init__(self) -> None:
        """Enforce the immutable schema expected by research artifacts."""

        normalized_baseline = _validate_condition_label(self.baseline_label, "baseline_label")
        normalized_treatment = _validate_condition_label(self.treatment_label, "treatment_label")
        if normalized_baseline.casefold() == normalized_treatment.casefold():
            raise ValueError("baseline_label and treatment_label must identify distinct conditions")
        if not self.seeds:
            raise ValueError("seeds must contain at least one paired seed")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        if tuple(sorted(self.seeds)) != self.seeds:
            raise ValueError("seeds must be sorted in ascending order")
        if set(self.metrics) != set(METRIC_NAMES):
            raise ValueError("metrics must contain exactly the primary benchmark metrics")
        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, PairedMetricAnalysis):
                raise TypeError(f"metrics[{metric_name!r}] must be PairedMetricAnalysis")
            if metric.sample_size != len(self.seeds):
                raise ValueError(f"metrics[{metric_name!r}].sample_size must match seeds")

        object.__setattr__(self, "baseline_label", normalized_baseline)
        object.__setattr__(self, "treatment_label", normalized_treatment)
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "baseline_label": self.baseline_label,
            "treatment_label": self.treatment_label,
            "seeds": list(self.seeds),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
        }


def analyze_paired_benchmark_reports(
    baseline_reports: Sequence[BenchmarkRunReport],
    treatment_reports: Sequence[BenchmarkRunReport],
    *,
    baseline_label: str = "baseline",
    treatment_label: str = "treatment",
) -> PairedBenchmarkAnalysis:
    """Analyze matched seed-level treatment effects without pooling episodes.

    The function first applies the existing paired-configuration validation.
    It then computes one exact sign-flip test and one paired Cohen's ``d_z`` per
    primary metric, followed by Holm-Bonferroni correction across those three
    hypotheses. No significance threshold is applied and no scientific claim
    is inferred from the resulting p-values.
    """

    comparison = compare_benchmark_reports(
        baseline_reports,
        treatment_reports,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
    )
    baseline_by_seed = _index_reports_by_seed(baseline_reports)
    treatment_by_seed = _index_reports_by_seed(treatment_reports)
    seeds = comparison.seeds

    metric_deltas = {
        metric_name: _paired_metric_deltas(
            baseline_by_seed,
            treatment_by_seed,
            seeds,
            getter,
        )
        for metric_name, getter in _METRIC_GETTERS.items()
    }

    raw_results = {
        metric_name: exact_paired_sign_flip_test(deltas)
        for metric_name, deltas in metric_deltas.items()
    }
    adjusted_p_values = holm_bonferroni_adjust(
        {metric_name: result.p_value for metric_name, result in raw_results.items()}
    )

    metrics = {
        metric_name: PairedMetricAnalysis(
            observed_mean_delta=raw_results[metric_name].observed_mean_delta,
            p_value=raw_results[metric_name].p_value,
            adjusted_p_value=adjusted_p_values[metric_name],
            effect_size_dz=paired_cohens_dz(deltas),
            sample_size=len(deltas),
        )
        for metric_name, deltas in metric_deltas.items()
    }
    return PairedBenchmarkAnalysis(
        baseline_label=comparison.baseline_label,
        treatment_label=comparison.treatment_label,
        seeds=seeds,
        metrics=metrics,
    )


def _index_reports_by_seed(
    reports: Sequence[BenchmarkRunReport],
) -> dict[int, BenchmarkRunReport]:
    """Index explicitly seeded reports after the paired validation boundary."""

    return {_require_seed(report): report for report in reports}


def _require_seed(report: BenchmarkRunReport) -> int:
    """Return a report seed as a non-optional integer."""

    if report.seed is None:
        raise ValueError("paired benchmark reports must provide explicit seeds for pairing")
    return report.seed


def _paired_metric_deltas(
    baseline_by_seed: Mapping[int, BenchmarkRunReport],
    treatment_by_seed: Mapping[int, BenchmarkRunReport],
    seeds: tuple[int, ...],
    getter: MetricGetter,
) -> tuple[float, ...]:
    """Return treatment-minus-baseline deltas for one metric and seed set."""

    return tuple(
        getter(treatment_by_seed[seed]) - getter(baseline_by_seed[seed]) for seed in seeds
    )


def _validate_condition_label(label: str, field_name: str) -> str:
    """Normalize and validate a human-readable condition label."""

    if not isinstance(label, str):
        raise TypeError(f"{field_name} must be a string")
    normalized_label = label.strip()
    if not normalized_label:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_label


def _validate_probability(value: float, field_name: str) -> None:
    """Validate a finite probability in the closed unit interval."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real numeric value")
    if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be a finite value in [0, 1]")


__all__ = [
    "METRIC_NAMES",
    "PairedBenchmarkAnalysis",
    "PairedMetricAnalysis",
    "analyze_paired_benchmark_reports",
]
