"""Persistence tests for benchmark verification attestations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.verify_benchmark_artifact import (
    BenchmarkVerificationResult,
    main,
    write_verification_attestation,
)


def _verification_result() -> BenchmarkVerificationResult:
    """Return one representative immutable verification result."""

    return BenchmarkVerificationResult(
        schema_version=1,
        byte_count=17,
        sha256="a" * 64,
        benchmark_name="webshop",
        configuration_fingerprint="b" * 64,
        experiment_identity="c" * 64,
        bundle_sha256="d" * 64,
    )


def test_write_verification_attestation_is_canonical(tmp_path: Path) -> None:
    """Persisted attestations should use one deterministic canonical JSON encoding."""

    destination = tmp_path / "nested" / "verification.json"
    result = _verification_result()

    write_verification_attestation(destination, result)

    expected = (
        json.dumps(
            result.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert destination.read_text(encoding="utf-8") == expected


def test_write_verification_attestation_refuses_overwrite(tmp_path: Path) -> None:
    """Existing evidence must not be silently replaced by a later verification run."""

    destination = tmp_path / "verification.json"
    destination.write_text("retained-evidence\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_verification_attestation(destination, _verification_result())

    assert destination.read_text(encoding="utf-8") == "retained-evidence\n"
    assert list(tmp_path.glob(".verification.json.*.tmp")) == []


def test_main_persists_same_canonical_attestation_as_json_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI persistence and --json output should be byte-for-byte identical evidence."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    save_benchmark_artifact_manifest(report_path)
    attestation_path = tmp_path / "benchmark.verification.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-verify-benchmark",
            str(report_path),
            "--attestation-output",
            str(attestation_path),
            "--json",
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert attestation_path.read_text(encoding="utf-8") == captured.out
    parsed = json.loads(captured.out)
    assert parsed["sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()


def test_main_refuses_to_replace_existing_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should fail closed when a requested evidence destination already exists."""

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"schema_version":1,"episodes":[]}', encoding="utf-8")
    save_benchmark_artifact_manifest(report_path)
    attestation_path = tmp_path / "benchmark.verification.json"
    attestation_path.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-verify-benchmark",
            str(report_path),
            "--attestation-output",
            str(attestation_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "already exists" in captured.err
    assert attestation_path.read_text(encoding="utf-8") == "existing\n"
