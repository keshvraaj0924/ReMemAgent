"""Regression tests for the benchmark artifact verification CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.paired_artifacts import PAIRED_EXECUTION_ORDER_PROVENANCE_KEY
from experiments.preflight_evidence import PREFLIGHT_EVIDENCE_PROVENANCE_KEY
from experiments.verify_benchmark_artifact import main, verify_report_artifact


def _write_report(path: Path) -> None:
    """Write the minimum JSON document accepted by the artifact manifest layer."""

    path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")


def _readiness_evidence(*, runtime_revision: str = "a" * 40) -> dict[str, object]:
    """Build a structurally valid readiness object with a canonical fingerprint."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "runtime_provenance": {"code_revision": runtime_revision},
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


def _write_bound_report(path: Path, evidence_sha256: str) -> None:
    """Write a minimal benchmark artifact bound to readiness evidence."""

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [],
                "runtime_provenance": {
                    PREFLIGHT_EVIDENCE_PROVENANCE_KEY: evidence_sha256,
                },
            }
        ),
        encoding="utf-8",
    )


def test_verify_report_artifact_uses_default_sidecar(tmp_path: Path) -> None:
    """The verifier should resolve the conventional sidecar path."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    manifest_path = save_benchmark_artifact_manifest(report_path)

    verify_report_artifact(report_path)

    assert manifest_path == tmp_path / "benchmark.json.manifest.json"


def test_verify_report_artifact_rejects_modified_report(tmp_path: Path) -> None:
    """The CLI helper should fail closed after report bytes change."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    manifest_path = save_benchmark_artifact_manifest(report_path)
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity verification failed"):
        verify_report_artifact(report_path, manifest_path)


def test_verify_report_artifact_rejects_missing_manifest(tmp_path: Path) -> None:
    """A missing sidecar must not be interpreted as an unverified success."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)

    with pytest.raises(ValueError, match="invalid benchmark artifact manifest"):
        verify_report_artifact(report_path)


def test_verify_report_artifact_rejects_invalid_paired_execution_provenance(
    tmp_path: Path,
) -> None:
    """Byte-valid paired artifacts must still satisfy temporal provenance semantics."""

    report_path = tmp_path / "paired.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [],
                "seeds": [11],
                "execution_order": [
                    {
                        "seed": 11,
                        "first_condition": "treatment",
                        "second_condition": "baseline",
                    }
                ],
                "runtime_provenance": {
                    PAIRED_EXECUTION_ORDER_PROVENANCE_KEY: "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = save_benchmark_artifact_manifest(report_path)

    with pytest.raises(ValueError, match="deterministic counterbalanced"):
        verify_report_artifact(report_path, manifest_path)


def test_verify_report_artifact_requires_readiness_json_for_bound_artifact(
    tmp_path: Path,
) -> None:
    """A stored readiness digest is not independently verifiable without its JSON."""

    evidence = _readiness_evidence()
    report_path = tmp_path / "paired.json"
    _write_bound_report(report_path, str(evidence["evidence_sha256"]))
    manifest_path = save_benchmark_artifact_manifest(report_path)

    with pytest.raises(ValueError, match="requires --preflight-evidence"):
        verify_report_artifact(report_path, manifest_path)


def test_verify_report_artifact_accepts_matching_readiness_evidence(tmp_path: Path) -> None:
    """The verifier should authenticate the exact retained readiness object."""

    evidence = _readiness_evidence()
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    report_path = tmp_path / "paired.json"
    _write_bound_report(report_path, str(evidence["evidence_sha256"]))
    manifest_path = save_benchmark_artifact_manifest(report_path)

    verify_report_artifact(report_path, manifest_path, evidence_path)


def test_verify_report_artifact_rejects_different_readiness_evidence(tmp_path: Path) -> None:
    """A valid but unrelated readiness object must not satisfy the artifact binding."""

    admitted_evidence = _readiness_evidence()
    unrelated_evidence = _readiness_evidence(runtime_revision="d" * 40)
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(unrelated_evidence), encoding="utf-8")
    report_path = tmp_path / "paired.json"
    _write_bound_report(report_path, str(admitted_evidence["evidence_sha256"]))
    manifest_path = save_benchmark_artifact_manifest(report_path)

    with pytest.raises(ValueError, match="readiness digest does not match"):
        verify_report_artifact(report_path, manifest_path, evidence_path)


def test_verify_report_artifact_rejects_evidence_for_unbound_artifact(tmp_path: Path) -> None:
    """Supplying unrelated evidence must not retroactively imply a benchmark binding."""

    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(_readiness_evidence()), encoding="utf-8")
    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    manifest_path = save_benchmark_artifact_manifest(report_path)

    with pytest.raises(ValueError, match="is not bound to preflight evidence"):
        verify_report_artifact(report_path, manifest_path, evidence_path)


def test_main_returns_nonzero_and_writes_diagnostic_for_invalid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The executable CLI should report verification failures without a traceback."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    manifest_path = save_benchmark_artifact_manifest(report_path)
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["remem-verify-benchmark", str(report_path), "--manifest", str(manifest_path)],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "benchmark artifact verification failed:" in captured.err
    assert "Traceback" not in captured.err


def test_main_returns_zero_for_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The executable CLI should return zero and emit a concise success message."""

    report_path = tmp_path / "benchmark.json"
    _write_report(report_path)
    save_benchmark_artifact_manifest(report_path)
    monkeypatch.setattr("sys.argv", ["remem-verify-benchmark", str(report_path)])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "benchmark artifact integrity verified:" in captured.out
    assert captured.err == ""


def test_main_verifies_bound_artifact_with_preflight_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The installed verifier should expose readiness binding verification."""

    evidence = _readiness_evidence()
    evidence_path = tmp_path / "readiness.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    report_path = tmp_path / "paired.json"
    _write_bound_report(report_path, str(evidence["evidence_sha256"]))
    save_benchmark_artifact_manifest(report_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-verify-benchmark",
            str(report_path),
            "--preflight-evidence",
            str(evidence_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "benchmark artifact integrity verified:" in captured.out
    assert captured.err == ""