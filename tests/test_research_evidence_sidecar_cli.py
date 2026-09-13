from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from experiments.research_evidence_cli import main
from experiments.research_evidence_record import (
    build_research_evidence_record,
    write_research_evidence_record,
)
from remem.observability import ObservationSnapshot, write_observation_snapshot
from remem.observability_distribution_artifacts import (
    DistributionObservationSnapshot,
    write_distribution_observation_snapshot,
)
from remem.observability_distributions import ObservationHistogramSnapshot

REVISION = "1" * 40


def _write_sidecar_bound_record(root: Path) -> tuple[Path, str, str, str]:
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
    observability_sha256 = hashlib.sha256(observability_bytes).hexdigest()
    distribution_sha256 = hashlib.sha256(distribution_bytes).hexdigest()
    bundle_sha256 = "a" * 64

    verification_path = root / "benchmark.verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "bundle_sha256": bundle_sha256,
                "distribution_byte_count": len(distribution_bytes),
                "distribution_schema_version": 1,
                "distribution_sha256": distribution_sha256,
                "observability_byte_count": len(observability_bytes),
                "observability_schema_version": 1,
                "observability_sha256": observability_sha256,
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
    return record_path, observability_sha256, distribution_sha256, bundle_sha256


def test_cli_verifies_retained_sidecar_binding_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    record_path, observability_sha256, distribution_sha256, bundle_sha256 = (
        _write_sidecar_bound_record(tmp_path)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-sidecar-binding",
            "--json",
        ],
    )

    assert main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["sidecar_binding_verified"] is True
    assert payload["observability_sidecar_sha256"] == observability_sha256
    assert payload["distribution_sidecar_sha256"] == distribution_sha256
    assert payload["sidecar_bundle_sha256"] == bundle_sha256


def test_cli_sidecar_binding_fails_closed_when_required_role_is_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-research-evidence",
            "verify",
            str(record_path),
            "--require-sidecar-binding",
            "--json",
        ],
    )

    assert main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "sidecar-binding verification requires evidence roles" in captured.err
    assert "observability_sidecar" in captured.err
    assert "distribution_sidecar" in captured.err
