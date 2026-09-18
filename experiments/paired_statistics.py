"""Paired statistical comparisons for benchmark policy evaluations.

Comparisons are computed from matched deterministic seeds. This preserves the
within-seed pairing expected when two policies are evaluated on the same
benchmark repetitions and avoids treating correlated runs as independent
samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.benchmark_statistics import MetricSummary, _summarize
from remem.benchmark import BenchmarkRunReport


@dataclass(frozen=True, slots=True)
class PairedBenchmarkStatistics:
    """Seed-matched metric deltas between a candidate and baseline policy."""

    benchmark_name: str
    seeds: tuple[int, ...]
    success_rate_delta: MetricSummary
    mean_reward_delta: MetricSummary
    transfer_success_rate_delta: MetricSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the paired comparison."""

        return asdict(self)


def compare_paired_benchmark_reports(
    baseline_reports: tuple[BenchmarkRunReport, ...],
    candidate_reports: tuple[BenchmarkRunReport, ...],
) -> PairedBenchmarkStatistics:
    """Compare candidate and baseline reports using matched seed-level deltas.

    Both policy collections must contain the same benchmark, explicit unique
    seeds, the same seed set, and equal episode counts for every matched seed.
    Candidate-minus-baseline deltas are summarized across seeds, so positive
    values consistently mean that the candidate metric is larger.
    """

    baseline_by_seed = _index_comparable_reports(baseline_reports, label="baseline")
    candidate_by_seed = _index_comparable_reports(candidate_reports, label="candidate")

    baseline_seeds = set(baseline_by_seed)
    candidate_seeds = set(candidate_by_seed)
    if baseline_seeds != candidate_seeds:
        raise ValueError("baseline and candidate reports must contain the same seed set")

    ordered_seeds = tuple(sorted(baseline_seeds))
    baseline_benchmark = next(iter(baseline_by_seed.values())).benchmark_name
    candidate_benchmark = next(iter(candidate_by_seed.values())).benchmark_name
    if baseline_benchmark != candidate_benchmark:
        raise ValueError("baseline and candidate reports must use the same benchmark")

    for seed in ordered_seeds:
        baseline_count = len(baseline_by_seed[seed].episodes)
        candidate_count = len(candidate_by_seed[seed].episodes)
        if baseline_count != candidate_count:
            raise ValueError("matched seed reports must contain the same number of episodes")

    return PairedBenchmarkStatistics(
        benchmark_name=baseline_benchmark,
        seeds=ordered_seeds,
        success_rate_delta=_summarize(
            tuple(
                candidate_by_seed[seed].success_rate - baseline_by_seed[seed].success_rate
                for seed in ordered_seeds
            ),
            metric_name="success_rate_delta",
        ),
        mean_reward_delta=_summarize(
            tuple(
                candidate_by_seed[seed].mean_reward - baseline_by_seed[seed].mean_reward
                for seed in ordered_seeds
            ),
            metric_name="mean_reward_delta",
        ),
        transfer_success_rate_delta=_summarize(
            tuple(
                candidate_by_seed[seed].transfer_success_rate
                - baseline_by_seed[seed].transfer_success_rate
                for seed in ordered_seeds
            ),
            metric_name="transfer_success_rate_delta",
        ),
    )


def _index_comparable_reports(
    reports: tuple[BenchmarkRunReport, ...],
    *,
    label: str,
) -> dict[int, BenchmarkRunReport]:
    """Validate one policy's repetitions and index them by deterministic seed."""

    if not reports:
        raise ValueError(f"{label} reports must contain at least one benchmark report")

    indexed: dict[int, BenchmarkRunReport] = {}
    benchmark_names: set[str] = set()
    for report in reports:
        if report.seed is None:
            raise ValueError(f"{label} report seeds must be explicit")
        if report.seed in indexed:
            raise ValueError(f"{label} report seeds must be unique")
        indexed[report.seed] = report
        benchmark_names.add(report.benchmark_name)

    if len(benchmark_names) != 1:
        raise ValueError(f"{label} reports must use one benchmark name")
    return indexed


__all__ = ["PairedBenchmarkStatistics", "compare_paired_benchmark_reports"]
