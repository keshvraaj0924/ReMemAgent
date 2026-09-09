"""Inferential analysis for paired benchmark conditions.

This module composes the repository's descriptive comparison, exact paired
sign-flip test, Holm correction, and paired effect-size utilities. Statistical
analysis remains downstream of the benchmark and routing layers: it cannot
change policy behavior or memory selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from experiments.benchmark_statistics import (
    compare_benchmark_reports,
    exact_paired_sign_flip_test,
    holm_bonferroni_adjust,
)
from experiments.effect_sizes import paired_cohens_dz
from remem.benchmark import BenchmarkRunReport


METRIC_NAMES = ("success_rate", "mean_reward", "transfer_success_rate")


@dataclass(frozen=True, slots=True)
class PairedMetricAnalysis:
    """Inferential statistics for one paired benchmark metric."""

    observed_mean_delta: float
    p_value: float
    adjusted_p_value: float
    effect_size_dz: float | None
    sample_size: int

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
    baseline_by_seed = {report.seed: report for report in baseline_reports}
    treatment_by_seed = {report.seed: report for report in treatment_reports}
    seeds = tuple(comparison.seeds)

    metric_deltas = {
        "success_rate": tuple(
            treatment_by_seed[seed].success_rate - baseline_by_seed[seed].success_rate
            for seed in seeds
        ),
        "mean_reward": tuple(
            treatment_by_seed[seed].mean_reward - baseline_by_seed[seed].mean_reward
            for seed in seeds
        ),
        "transfer_success_rate": tuple(
            treatment_by_seed[seed].transfer_success_rate
            - baseline_by_seed[seed].transfer_success_rate
            for seed in seeds
        ),
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
            effect_size_dz=paired_cohens_dz(metric_deltas[metric_name]),
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


__all__ = [
    "METRIC_NAMES",
    "PairedBenchmarkAnalysis",
    "PairedMetricAnalysis",
    "analyze_paired_benchmark_reports",
]
