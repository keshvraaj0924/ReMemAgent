"""Tests for machine-readable benchmark artifact verification attestations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.preflight_evidence import PREFLIGHT_EVIDENCE_PROVENANCE_KEY
from experiments.verify_benchmark_artifact import main, verify_report_artifact


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
        "byte_count": len(report_path.read_bytes()),
        "configuration_fingerprint": None,
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
