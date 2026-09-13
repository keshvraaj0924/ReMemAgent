from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.research_evidence_record import (
    build_research_evidence_record,
    write_research_evidence_record,
)
from experiments.research_sidecar_binding import verify_research_sidecar_binding
from remem.observability import ObservationSnapshot, write_observation_snapshot
from remem.observability_distribution_artifacts import (
    DistributionObservationSnapshot,
    write_distribution_observation_snapshot,
)
from remem.observability_distributions import ObservationHistogramSnapshot

REVISION = "1" * 40


def _write_bound_record(
    root: Path,
    *,
    observability_sha256: str | None = None,
    distribution_byte_count: int | None = None,
    bundle_sha256: str = "a" * 64,
) -> tuple[Path, str, str]:
    observability_path = root / "benchmark.observability.json"
    write_observation_snapshot(
        observability_path,
        ObservationSnapshot(
            counters={"benchmark.episodes.completed": 2.0},
            durations_seconds={"benchmark.episode.duration_seconds": 0.75},
        ),
        overwrite=False,
    )

    distribution_path = root / "benchmark.duration-distribution.json"
    write_distribution_observation_snapshot(
        distribution_path,
        DistributionObservationSnapshot(
            duration_histograms={
                "benchmark.episode.duration_seconds": ObservationHistogramSnapshot(
                    upper_bounds=(0.25, 0.5, 1.0),
                    bucket_counts=(1, 0, 1, 0),
                    total=0.75,
                )
            }
        ),
    )

    observability_bytes = observability_path.read_bytes()
    distribution_bytes = distribution_path.read_bytes()
    actual_observability_sha256 = hashlib.sha256(observability_bytes).hexdigest()
    actual_distribution_sha256 = hashlib.sha256(distribution_bytes).hexdigest()

    verification_path = root / "benchmark.verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "bundle_sha256": bundle_sha256,
                "distribution_byte_count": (
                    len(distribution_bytes)
                    if distribution_byte_count is None
                    else distribution_byte_count
                ),
                "distribution_schema_version": 1,
                "distribution_sha256": actual_distribution_sha256,
                "observability_byte_count": len(observability_bytes),
                "observability_schema_version": 1,
                "observability_sha256": observability_sha256 or actual_observability_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    record_path = root / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
        evidence_level="E3",
        remem_revision=REVISION,
        artifacts={
            "distribution_sidecar": distribution_path,
            "observability_sidecar": observability_path,
            "verification": verification_path,
        },
        record_directory=root,
    )
    write_research_evidence_record(record_path, record)
    return record_path, actual_observability_sha256, actual_distribution_sha256


def test_verify_research_sidecar_binding_proves_exact_retained_sidecars(tmp_path: Path) -> None:
    record_path, observability_sha256, distribution_sha256 = _write_bound_record(tmp_path)

    binding = verify_research_sidecar_binding(record_path)

    assert binding.observability_sha256 == observability_sha256
    assert binding.distribution_sha256 == distribution_sha256
    assert binding.bundle_sha256 == "a" * 64


def test_verify_research_sidecar_binding_rejects_unrelated_observability_attestation(
    tmp_path: Path,
) -> None:
    record_path, _, _ = _write_bound_record(
        tmp_path,
        observability_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="observability SHA-256 mismatch"):
        verify_research_sidecar_binding(record_path)


def test_verify_research_sidecar_binding_rejects_distribution_byte_count_drift(
    tmp_path: Path,
) -> None:
    record_path, _, _ = _write_bound_record(
        tmp_path,
        distribution_byte_count=1,
    )

    with pytest.raises(ValueError, match="distribution byte-count mismatch"):
        verify_research_sidecar_binding(record_path)


def test_verify_research_sidecar_binding_requires_canonical_roles(tmp_path: Path) -> None:
    verification_path = tmp_path / "verification.json"
    verification_path.write_text("{}\n", encoding="utf-8")
    record_path = tmp_path / "research-evidence.json"
    record = build_research_evidence_record(
        experiment_name="webshop-seed-study",
        evidence_level="E2",
        remem_revision=REVISION,
        artifacts={"verification": verification_path},
        record_directory=tmp_path,
    )
    write_research_evidence_record(record_path, record)

    with pytest.raises(ValueError, match="sidecar-binding verification requires evidence roles"):
        verify_research_sidecar_binding(record_path)


def test_verify_research_sidecar_binding_rejects_invalid_bundle_digest(tmp_path: Path) -> None:
    record_path, _, _ = _write_bound_record(tmp_path, bundle_sha256="not-a-sha256")

    with pytest.raises(ValueError, match="missing valid bundle_sha256"):
        verify_research_sidecar_binding(record_path)
