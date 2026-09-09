"""Tests for paired benchmark inferential analysis."""

from __future__ import annotations

import pytest

from experiments.paired_analysis import (
    METRIC_NAMES,
    analyze_paired_benchmark_reports,
)
from remem.benchmark import BenchmarkRunReport


def _report(seed: int, *, success_rate: float, reward: float, transfer: float) -> BenchmarkRunReport:
    """Build a minimal valid report for paired-analysis tests."""

    return BenchmarkRunReport(
        benchmark_name="synthetic",
        environment="test",
        seed=seed,
        episode_count=10,
        successful_episodes=round(success_rate * 10),
        mean_reward=reward,
        transfer_success_rate=transfer,
    )


def test_analyze_paired_reports_computes_effects_and_adjusted_p_values() -> None:
    """All primary metrics receive inferential and effect-size statistics."""

    baseline = [
        _report(1, success_rate=0.4, reward=1.0, transfer=0.3),
        _report(2, success_rate=0.5, reward=1.5, transfer=0.4),
        _report(3, success_rate=0.6, reward=2.0, transfer=0.5),
    ]
    treatment = [
        _report(1, success_rate=0.5, reward=1.5, transfer=0.4),
        _report(2, success_rate=0.6, reward=2.0, transfer=0.5),
        _report(3, success_rate=0.7, reward=2.5, transfer=0.6),
    ]

    analysis = analyze_paired_benchmark_reports(baseline, treatment)

    assert analysis.seeds == (1, 2, 3)
    assert set(analysis.metrics) == set(METRIC_NAMES)
    assert analysis.metrics["success_rate"].observed_mean_delta == pytest.approx(0.1)
    assert analysis.metrics["success_rate"].effect_size_dz is not None
    assert analysis.metrics["success_rate"].p_value == pytest.approx(0.25)
    assert analysis.metrics["success_rate"].adjusted_p_value == pytest.approx(0.75)
    assert analysis.metrics["mean_reward"].sample_size == 3


def test_analyze_paired_reports_preserves_zero_variance_effect_size_as_none() -> None:
    """A constant paired effect has no finite Cohen's d_z denominator."""

    baseline = [_report(seed, success_rate=0.5, reward=1.0, transfer=0.5) for seed in (1, 2, 3)]
    treatment = [_report(seed, success_rate=0.6, reward=1.5, transfer=0.6) for seed in (1, 2, 3)]

    analysis = analyze_paired_benchmark_reports(baseline, treatment)

    assert analysis.metrics["success_rate"].effect_size_dz is None
    assert analysis.metrics["mean_reward"].effect_size_dz is None
    assert analysis.metrics["transfer_success_rate"].effect_size_dz is None


def test_analyze_paired_reports_rejects_unpaired_seed_sets() -> None:
    """Inferential analysis must retain the strict paired-seed contract."""

    baseline = [_report(1, success_rate=0.5, reward=1.0, transfer=0.5)]
    treatment = [_report(2, success_rate=0.6, reward=1.5, transfer=0.6)]

    with pytest.raises(ValueError, match="same seed set"):
        analyze_paired_benchmark_reports(baseline, treatment)
