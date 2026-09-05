"""Descriptive and inferential statistics for benchmark seed reports.

The descriptive layer operates on one aggregate observation per seed. It never
pools episode-level observations across seeds. The paired sign-flip test is a
separate, explicitly opt-in inferential primitive for matched seed deltas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

from experiments.benchmark_report import benchmark_configuration_fingerprint
from remem.benchmark import BenchmarkRunReport
from remem.benchmark_validation import validate_benchmark_run_report

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


@dataclass(frozen=True, slots=True)
class BenchmarkConditionComparison:
    """Paired descriptive deltas between two conditions sharing independent seeds."""

    baseline_label: str
    treatment_label: str
    seeds: tuple[int | None, ...]
    success_rate_delta: MetricSummary
    mean_reward_delta: MetricSummary
    transfer_success_rate_delta: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the paired comparison."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedSignFlipResult:
    """Exact two-sided paired sign-flip test result for one metric."""

    observed_mean_delta: float
    p_value: float
    sample_size: int
    nonzero_delta_count: int
    evaluated_permutations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the test result."""

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
    _validate_report_collection(selected_reports)

    return BenchmarkSeedStatistics(
        benchmark_name=selected_reports[0].benchmark_name,
        seeds=tuple(report.seed for report in selected_reports),
        success_rate=_summarize(tuple(report.success_rate for report in selected_reports)),
        mean_reward=_summarize(tuple(report.mean_reward for report in selected_reports)),
        transfer_success_rate=_summarize(
            tuple(report.transfer_success_rate for report in selected_reports)
        ),
    )


def compare_benchmark_reports(
    baseline_reports: Sequence[BenchmarkRunReport],
    treatment_reports: Sequence[BenchmarkRunReport],
    *,
    baseline_label: str = "baseline",
    treatment_label: str = "treatment",
) -> BenchmarkConditionComparison:
    """Compute paired treatment-minus-baseline deltas by independent seed.

    Both conditions must contain the same unique, explicitly recorded seed set,
    benchmark name, and seed-independent evaluation configuration. The policy
    implementation itself is allowed to differ because it is the treatment
    variable in this comparison. Each seed contributes exactly one paired
    observation to the descriptive delta statistics. The function does not
    pool episodes and does not perform a hypothesis test or claim statistical
    significance.
    """

    baseline = tuple(baseline_reports)
    treatment = tuple(treatment_reports)
    _validate_report_collection(baseline)
    _validate_report_collection(treatment)

    if baseline[0].benchmark_name != treatment[0].benchmark_name:
        raise ValueError("baseline and treatment reports must use one benchmark name")

    _validate_paired_configuration(baseline, treatment)
    _validate_explicit_seeds(baseline, "baseline")
    _validate_explicit_seeds(treatment, "treatment")

    baseline_by_seed = {report.seed: report for report in baseline}
    treatment_by_seed = {report.seed: report for report in treatment}
    if set(baseline_by_seed) != set(treatment_by_seed):
        raise ValueError("baseline and treatment reports must use the same seed set")

    seeds = tuple(sorted(baseline_by_seed))
    success_deltas = tuple(
        treatment_by_seed[seed].success_rate - baseline_by_seed[seed].success_rate
        for seed in seeds
    )
    reward_deltas = tuple(
        treatment_by_seed[seed].mean_reward - baseline_by_seed[seed].mean_reward
        for seed in seeds
    )
    transfer_deltas = tuple(
        treatment_by_seed[seed].transfer_success_rate
        - baseline_by_seed[seed].transfer_success_rate
        for seed in seeds
    )

    return BenchmarkConditionComparison(
        baseline_label=_validate_label(baseline_label, "baseline_label"),
        treatment_label=_validate_label(treatment_label, "treatment_label"),
        seeds=seeds,
        success_rate_delta=_summarize(success_deltas),
        mean_reward_delta=_summarize(reward_deltas),
        transfer_success_rate_delta=_summarize(transfer_deltas),
    )


def exact_paired_sign_flip_test(
    deltas: Sequence[float],
) -> PairedSignFlipResult:
    """Run an exact two-sided paired sign-flip test on seed-level deltas.

    Under the null hypothesis that treatment has no systematic paired effect,
    each non-zero paired difference is assigned either sign with equal
    probability. All ``2**n`` sign assignments are enumerated for up to 20
    non-zero differences. The two-sided p-value counts assignments whose
    absolute mean delta is at least as large as the observed absolute mean.

    Zero differences do not contribute sign choices. This test is intentionally
    separate from descriptive benchmark summaries and makes no adjustment for
    multiple metrics or multiple comparisons.
    """

    normalized_deltas = tuple(deltas)
    _validate_deltas(normalized_deltas)
    nonzero_deltas = tuple(delta for delta in normalized_deltas if delta != 0.0)
    nonzero_count = len(nonzero_deltas)
    if nonzero_count > 20:
        raise ValueError("exact sign-flip test supports at most 20 non-zero deltas")

    observed_mean = sum(normalized_deltas) / len(normalized_deltas)
    if nonzero_count == 0:
        return PairedSignFlipResult(
            observed_mean_delta=observed_mean,
            p_value=1.0,
            sample_size=len(normalized_deltas),
            nonzero_delta_count=0,
            evaluated_permutations=1,
        )

    observed_statistic = abs(observed_mean)
    total_assignments = 1 << nonzero_count
    extreme_assignments = 0
    scale = len(normalized_deltas)

    for signs in product((-1.0, 1.0), repeat=nonzero_count):
        permuted_mean = sum(delta * sign for delta, sign in zip(nonzero_deltas, signs)) / scale
        if abs(permuted_mean) >= observed_statistic:
            extreme_assignments += 1

    return PairedSignFlipResult(
        observed_mean_delta=observed_mean,
        p_value=extreme_assignments / total_assignments,
        sample_size=len(normalized_deltas),
        nonzero_delta_count=nonzero_count,
        evaluated_permutations=total_assignments,
    )


def holm_bonferroni_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-Bonferroni adjusted p-values in the original key order.

    The correction controls the family-wise error rate for a finite family of
    hypotheses. Input values must be finite probabilities in ``[0, 1]``. The
    returned values are monotone with respect to the ordered raw p-values and
    are capped at one. This utility does not choose an alpha threshold or claim
    that any hypothesis is significant.
    """

    if not p_values:
        raise ValueError("p_values must contain at least one hypothesis")

    normalized: list[tuple[str, float]] = []
    for name, value in p_values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("p_value hypothesis names must be non-empty strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("p_values must contain real numeric values")
        numeric_value = float(value)
        if not isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise ValueError("p_values must contain finite values in [0, 1]")
        normalized.append((name, numeric_value))

    ordered = sorted(normalized, key=lambda item: (item[1], item[0]))
    hypothesis_count = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        corrected_value = min(1.0, (hypothesis_count - rank) * p_value)
        running_max = max(running_max, corrected_value)
        adjusted[name] = running_max

    return {name: adjusted[name] for name, _ in normalized}


def _validate_report_collection(reports: tuple[BenchmarkRunReport, ...]) -> None:
    """Validate report structure before deriving any aggregate statistics."""

    if not reports:
        raise ValueError("reports must contain at least one benchmark report")

    for report in reports:
        validate_benchmark_run_report(report)

    seeds = tuple(report.seed for report in reports)
    if len(seeds) != len(set(seeds)):
        raise ValueError("benchmark report seeds must be unique")

    benchmark_names = {report.benchmark_name for report in reports}
    if len(benchmark_names) != 1:
        raise ValueError("benchmark reports must use one benchmark name")


def _validate_explicit_seeds(
    reports: tuple[BenchmarkRunReport, ...],
    condition_label: str,
) -> None:
    """Require explicit seeds before pairing independent experimental runs."""

    if any(report.seed is None for report in reports):
        raise ValueError(f"{condition_label} reports must provide explicit seeds for pairing")


def _validate_paired_configuration(
    baseline: tuple[BenchmarkRunReport, ...],
    treatment: tuple[BenchmarkRunReport, ...],
) -> None:
    """Ensure paired conditions share evaluation configuration apart from policy."""

    if any(report.configuration is None for report in baseline + treatment):
        raise ValueError("paired benchmark reports must include explicit configuration")

    baseline_fingerprints = {_paired_configuration_fingerprint(report) for report in baseline}
    treatment_fingerprints = {_paired_configuration_fingerprint(report) for report in treatment}
    if baseline_fingerprints != treatment_fingerprints:
        raise ValueError(
            "baseline and treatment reports must share configuration apart from the seed and policy"
        )


def _paired_configuration_fingerprint(report: BenchmarkRunReport) -> str:
    """Return a fingerprint excluding independent seed and policy identity."""

    if report.configuration is None:
        raise ValueError("paired benchmark reports must include explicit configuration")
    configuration = replace(report.configuration, seed=None, policy_factory=None)
    return benchmark_configuration_fingerprint(configuration)


def _validate_label(label: str, field_name: str) -> str:
    """Normalize and validate a human-readable condition label."""

    if not isinstance(label, str):
        raise TypeError(f"{field_name} must be a string")
    normalized_label = label.strip()
    if not normalized_label:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_label


def _validate_deltas(deltas: tuple[float, ...]) -> None:
    """Validate paired deltas before inferential analysis."""

    if not deltas:
        raise ValueError("deltas must contain at least one paired observation")
    if any(not isinstance(delta, (int, float)) or isinstance(delta, bool) for delta in deltas):
        raise TypeError("deltas must contain real numeric values")
    if any(not isfinite(float(delta)) for delta in deltas):
        raise ValueError("deltas must contain only finite values")


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


__all__ = [
    "BenchmarkConditionComparison",
    "BenchmarkSeedStatistics",
    "MetricSummary",
    "PairedSignFlipResult",
    "compare_benchmark_reports",
    "exact_paired_sign_flip_test",
    "holm_bonferroni_adjust",
    "summarize_benchmark_reports",
]
