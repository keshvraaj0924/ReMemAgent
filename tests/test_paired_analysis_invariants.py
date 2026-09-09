"""Regression tests for paired-analysis artifact invariants."""

from __future__ import annotations

import pytest

from experiments.paired_analysis import (
    METRIC_NAMES,
    PairedBenchmarkAnalysis,
    PairedMetricAnalysis,
)


def _metric(sample_size: int = 2) -> PairedMetricAnalysis:
    """Build a valid metric result for schema tests."""

    return PairedMetricAnalysis(
        observed_mean_delta=0.1,
        p_value=0.25,
        adjusted_p_value=0.75,
        effect_size_dz=0.5,
        sample_size=sample_size,
    )


def _analysis(*, seeds: tuple[int, ...] = (1, 2)) -> PairedBenchmarkAnalysis:
    """Build a valid paired-analysis artifact."""

    return PairedBenchmarkAnalysis(
        baseline_label="baseline",
        treatment_label="memory",
        seeds=seeds,
        metrics={name: _metric(len(seeds)) for name in METRIC_NAMES},
    )


def test_paired_analysis_normalizes_condition_labels() -> None:
    """Condition labels are stripped before they become artifact metadata."""

    analysis = PairedBenchmarkAnalysis(
        baseline_label=" baseline ",
        treatment_label=" memory ",
        seeds=(2, 5),
        metrics={name: _metric() for name in METRIC_NAMES},
    )

    assert analysis.baseline_label == "baseline"
    assert analysis.treatment_label == "memory"


def test_paired_analysis_rejects_duplicate_or_unsorted_seeds() -> None:
    """Seed order is canonical and duplicates cannot represent independent runs."""

    with pytest.raises(ValueError, match="unique"):
        _analysis(seeds=(1, 1))

    with pytest.raises(ValueError, match="sorted"):
        _analysis(seeds=(2, 1))


def test_paired_analysis_requires_exact_primary_metric_set() -> None:
    """Persisted analyses cannot silently omit or invent primary metrics."""

    metrics = {name: _metric() for name in METRIC_NAMES[:-1]}
    with pytest.raises(ValueError, match="exactly the primary benchmark metrics"):
        PairedBenchmarkAnalysis(
            baseline_label="baseline",
            treatment_label="memory",
            seeds=(1, 2),
            metrics=metrics,
        )


def test_paired_analysis_requires_metric_sample_size_to_match_seeds() -> None:
    """Every metric must describe every paired seed."""

    metrics = {name: _metric() for name in METRIC_NAMES}
    metrics["mean_reward"] = _metric(sample_size=1)

    with pytest.raises(ValueError, match="sample_size must match seeds"):
        PairedBenchmarkAnalysis(
            baseline_label="baseline",
            treatment_label="memory",
            seeds=(1, 2),
            metrics=metrics,
        )


def test_paired_metric_rejects_invalid_probabilities_and_sample_size() -> None:
    """Statistical records reject impossible p-values and sample sizes."""

    with pytest.raises(ValueError, match="p_value"):
        PairedMetricAnalysis(0.1, 1.1, 0.5, None, 2)

    with pytest.raises(TypeError, match="sample_size"):
        PairedMetricAnalysis(0.1, 0.5, 0.5, None, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one"):
        PairedMetricAnalysis(0.1, 0.5, 0.5, None, 0)
