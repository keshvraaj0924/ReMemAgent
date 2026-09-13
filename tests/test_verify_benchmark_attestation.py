"""Tests for machine-readable benchmark artifact verification attestations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.preflight_evidence import PREFLIGHT_EVIDENCE_PROVENANCE_KEY
from experiments.verify_benchmark_artifact import main, verify_report_artifact
from remem.observability_distribution_artifacts import (
    DistributionObservationSnapshot,
    write_distribution_observation_snapshot,
)
from remem.observability_distributions import ObservationHistogramSnapshot


def _readiness_evidence() -> dict[str, object]:
    """Build valid readiness evidence with its canonical fingerprint."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "runtime_provenance": {"code_revision": "a" * 40},
        "source_checkout_requirements": {},
        "source_checkout_requirements_sha256": "b" * 64,
        "source_checkout_provenance": {},
        "source_checkout_provenance_sha256": "c" * 64,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _write_distribution_sidecar(path: Path) -> None:
    """Persist one valid deterministic duration-distribution sidecar."""

    snapshot = DistributionObservationSnapshot(
        duration_histograms={
            "benchmark.episode.duration_seconds": ObservationHistogramSnapshot(
                upper_bounds=(0.25, 1.0),
                bucket_counts=(1, 1, 0),
                total=0.75,
            )
        }
    )
    write_distribution_observation_snapshot(path, snapshot)


def _expected_bundle_digest(report_sha256: str, distribution_sha256: str) -> str:
    """Recompute the public bundle-digest contract used by verification attestations."""

    digest = hashlib.sha256()
    digest.update(b"remem-benchmark-bundle-v1\0")
    digest.update(bytes.fromhex(report_sha256))
    digest.update(bytes.fromhex(distribution_sha256))
    return digest.hexdigest()


def test_verify_report_artifact_returns_exact_manifest_attestation(
    tmp_path: Path,
) -> None:
    """Successful verification should expose the exact byte-level manifest evidence."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    manifest_path = save_benchmark_artifact_manifest(report_path)

    result = verify_report_artifact(report_path, manifest_path)

    expected_bytes = report_path.read_bytes()
    assert result.schema_version == 1
    assert result.byte_count == len(expected_bytes)
    assert result.sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert result.benchmark_name is None
    assert result.configuration_fingerprint is None
    assert result.experiment_identity is None
    assert result.preflight_evidence_sha256 is None
    assert result.distribution_schema_version is None
    assert result.distribution_byte_count is None
    assert result.distribution_sha256 is None
    assert result.bundle_sha256 is None


def test_verify_report_artifact_attests_validated_experiment_identity(
    tmp_path: Path,
) -> None:
    """Verification should expose validated report identity fields for CI correlation."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [],
                "benchmark_name": "webshop",
                "configuration_fingerprint": "b" * 64,
                "experiment_identity": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = save_benchmark_artifact_manifest(report_path)

    result = verify_report_artifact(report_path, manifest_path)

    assert result.benchmark_name == "webshop"
    assert result.configuration_fingerprint == "b" * 64
    assert result.experiment_identity == "c" * 64


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("benchmark_name", 3),
        ("configuration_fingerprint", ""),
        ("experiment_identity", []),
    ],
)
def test_verify_report_artifact_rejects_invalid_attestation_identity_field(
    tmp_path: Path,
    key: str,
    invalid_value: object,
) -> None:
    """Attestation metadata should never silently coerce malformed report identity fields."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text(
        json.dumps({"schema_version": 1, "episodes": [], key: invalid_value}),
        encoding="utf-8",
    )
    manifest_path = save_benchmark_artifact_manifest(report_path)

    with pytest.raises(ValueError, match=key):
        verify_report_artifact(report_path, manifest_path)


def test_verify_report_artifact_attests_matching_readiness_digest(
    tmp_path: Path,
) -> None:
    """Evidence-bound verification should return the authenticated readiness digest."""

    evidence = _readiness_evidence()
    evidence_digest = str(evidence["evidence_sha256"])
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    report_path = tmp_path / "paired.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [],
                "runtime_provenance": {
                    PREFLIGHT_EVIDENCE_PROVENANCE_KEY: evidence_digest,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = save_benchmark_artifact_manifest(report_path)

    result = verify_report_artifact(report_path, manifest_path, evidence_path)

    assert result.preflight_evidence_sha256 == evidence_digest


def test_verify_report_artifact_binds_valid_distribution_sidecar(
    tmp_path: Path,
) -> None:
    """A validated sidecar should be bound to the exact report bytes in one digest."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    manifest_path = save_benchmark_artifact_manifest(report_path)
    distribution_path = tmp_path / "benchmark.distributions.json"
    _write_distribution_sidecar(distribution_path)

    result = verify_report_artifact(
        report_path,
        manifest_path,
        distribution_sidecar_path=distribution_path,
    )

    distribution_bytes = distribution_path.read_bytes()
    distribution_sha256 = hashlib.sha256(distribution_bytes).hexdigest()
    assert result.distribution_schema_version == 1
    assert result.distribution_byte_count == len(distribution_bytes)
    assert result.distribution_sha256 == distribution_sha256
    assert result.bundle_sha256 == _expected_bundle_digest(
        result.sha256,
        distribution_sha256,
    )


def test_verify_report_artifact_rejects_tampered_distribution_sidecar(
    tmp_path: Path,
) -> None:
    """Sidecars must pass their structural and derived-field validation before attestation."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    manifest_path = save_benchmark_artifact_manifest(report_path)
    distribution_path = tmp_path / "benchmark.distributions.json"
    _write_distribution_sidecar(distribution_path)
    payload = json.loads(distribution_path.read_text(encoding="utf-8"))
    histogram = payload["duration_histograms"]["benchmark.episode.duration_seconds"]
    histogram["count"] = 999
    distribution_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="count"):
        verify_report_artifact(
            report_path,
            manifest_path,
            distribution_sidecar_path=distribution_path,
        )


def test_main_json_output_is_canonical_and_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should expose one stable JSON object without human text around it."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    save_benchmark_artifact_manifest(report_path)
    monkeypatch.setattr(
        "sys.argv",
        ["remem-verify-benchmark", str(report_path), "--json"],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    parsed = json.loads(captured.out)
    assert parsed == {
        "benchmark_name": None,
        "bundle_sha256": None,
        "byte_count": len(report_path.read_bytes()),
        "configuration_fingerprint": None,
        "distribution_byte_count": None,
        "distribution_schema_version": None,
        "distribution_sha256": None,
        "experiment_identity": None,
        "preflight_evidence_sha256": None,
        "schema_version": 1,
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    assert captured.out == (
        json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def test_main_json_output_attests_distribution_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI JSON should include exact distribution bytes and the joint bundle digest."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    save_benchmark_artifact_manifest(report_path)
    distribution_path = tmp_path / "benchmark.distributions.json"
    _write_distribution_sidecar(distribution_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-verify-benchmark",
            str(report_path),
            "--distribution-sidecar",
            str(distribution_path),
            "--json",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    distribution_sha256 = hashlib.sha256(distribution_path.read_bytes()).hexdigest()
    assert exit_code == 0
    assert captured.err == ""
    assert parsed["distribution_sha256"] == distribution_sha256
    assert parsed["distribution_byte_count"] == len(distribution_path.read_bytes())
    assert parsed["distribution_schema_version"] == 1
    assert parsed["bundle_sha256"] == _expected_bundle_digest(
        report_sha256,
        distribution_sha256,
    )
