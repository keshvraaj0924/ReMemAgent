"""Cross-sidecar consistency tests for benchmark verification evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.verify_benchmark_artifact import verify_report_artifact
from remem.observability import ObservationSnapshot, write_observation_snapshot
from remem.observability_distribution_artifacts import (
    DistributionObservationSnapshot,
    write_distribution_observation_snapshot,
)
from remem.observability_distributions import ObservationHistogramSnapshot

_DURATION_METRIC = "benchmark.episode.duration_seconds"
_COMPLETED_METRIC = "benchmark.episodes.completed"


def _write_report(path: Path) -> Path:
    """Persist the minimal valid report and return its integrity manifest path."""

    path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    return save_benchmark_artifact_manifest(path)


def _write_observability(path: Path, *, episode_count: float, total: float) -> None:
    """Persist aggregate benchmark observability evidence."""

    write_observation_snapshot(
        path,
        ObservationSnapshot(
            counters={_COMPLETED_METRIC: episode_count},
            durations_seconds={_DURATION_METRIC: total},
        ),
        overwrite=False,
    )


def _write_distribution(path: Path, *, bucket_counts: tuple[int, ...], total: float) -> None:
    """Persist fixed-bucket benchmark duration evidence."""

    write_distribution_observation_snapshot(
        path,
        DistributionObservationSnapshot(
            duration_histograms={
                _DURATION_METRIC: ObservationHistogramSnapshot(
                    upper_bounds=(0.25, 1.0),
                    bucket_counts=bucket_counts,
                    total=total,
                )
            }
        ),
    )


def test_verify_report_accepts_consistent_observability_sidecars(tmp_path: Path) -> None:
    """Evidence from the same measured episodes should verify successfully."""

    report_path = tmp_path / "benchmark.json"
    manifest_path = _write_report(report_path)
    observability_path = tmp_path / "benchmark.observability.json"
    distribution_path = tmp_path / "benchmark.distributions.json"
    _write_observability(observability_path, episode_count=2.0, total=0.75)
    _write_distribution(distribution_path, bucket_counts=(1, 1, 0), total=0.75)

    result = verify_report_artifact(
        report_path,
        manifest_path,
        distribution_sidecar_path=distribution_path,
        observability_sidecar_path=observability_path,
    )

    assert result.bundle_sha256 is not None


def test_verify_report_rejects_sidecars_with_different_episode_counts(tmp_path: Path) -> None:
    """Individually valid sidecars must not be bundled when their sample counts differ."""

    report_path = tmp_path / "benchmark.json"
    manifest_path = _write_report(report_path)
    observability_path = tmp_path / "benchmark.observability.json"
    distribution_path = tmp_path / "benchmark.distributions.json"
    _write_observability(observability_path, episode_count=3.0, total=0.75)
    _write_distribution(distribution_path, bucket_counts=(1, 1, 0), total=0.75)

    with pytest.raises(ValueError, match="completed episode count"):
        verify_report_artifact(
            report_path,
            manifest_path,
            distribution_sidecar_path=distribution_path,
            observability_sidecar_path=observability_path,
        )


def test_verify_report_rejects_sidecars_with_different_duration_totals(tmp_path: Path) -> None:
    """A bundle must not associate telemetry collected from different timing samples."""

    report_path = tmp_path / "benchmark.json"
    manifest_path = _write_report(report_path)
    observability_path = tmp_path / "benchmark.observability.json"
    distribution_path = tmp_path / "benchmark.distributions.json"
    _write_observability(observability_path, episode_count=2.0, total=0.80)
    _write_distribution(distribution_path, bucket_counts=(1, 1, 0), total=0.75)

    with pytest.raises(ValueError, match="duration total"):
        verify_report_artifact(
            report_path,
            manifest_path,
            distribution_sidecar_path=distribution_path,
            observability_sidecar_path=observability_path,
        )


def test_verify_report_requires_aggregate_duration_when_distribution_is_bound(
    tmp_path: Path,
) -> None:
    """Combined evidence must expose the aggregate duration measured by the same collector."""

    report_path = tmp_path / "benchmark.json"
    manifest_path = _write_report(report_path)
    observability_path = tmp_path / "benchmark.observability.json"
    distribution_path = tmp_path / "benchmark.distributions.json"
    write_observation_snapshot(
        observability_path,
        ObservationSnapshot(counters={_COMPLETED_METRIC: 2.0}, durations_seconds={}),
        overwrite=False,
    )
    _write_distribution(distribution_path, bucket_counts=(1, 1, 0), total=0.75)

    with pytest.raises(ValueError, match="duration aggregate"):
        verify_report_artifact(
            report_path,
            manifest_path,
            distribution_sidecar_path=distribution_path,
            observability_sidecar_path=observability_path,
        )


def test_verify_report_requires_completed_count_when_distribution_is_bound(tmp_path: Path) -> None:
    """Combined evidence must retain the completed-episode count used by the benchmark runner."""

    report_path = tmp_path / "benchmark.json"
    manifest_path = _write_report(report_path)
    observability_path = tmp_path / "benchmark.observability.json"
    distribution_path = tmp_path / "benchmark.distributions.json"
    write_observation_snapshot(
        observability_path,
        ObservationSnapshot(counters={}, durations_seconds={_DURATION_METRIC: 0.75}),
        overwrite=False,
    )
    _write_distribution(distribution_path, bucket_counts=(1, 1, 0), total=0.75)

    with pytest.raises(ValueError, match="episodes completed counter"):
        verify_report_artifact(
            report_path,
            manifest_path,
            distribution_sidecar_path=distribution_path,
            observability_sidecar_path=observability_path,
        )
