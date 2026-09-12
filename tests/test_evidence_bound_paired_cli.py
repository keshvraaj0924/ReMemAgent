"""Regression coverage for evidence-bound paired CLI delegation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import experiments.evidence_bound_paired_cli as evidence_cli


def test_main_delegates_unchanged_without_evidence_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_argv: list[str] = []

    def delegated_main() -> int:
        observed_argv.extend(sys.argv)
        return 17

    monkeypatch.setattr(evidence_cli.paired_cli, "main", delegated_main)
    monkeypatch.setattr(sys, "argv", ["remem-paired-benchmark", "--benchmark", "webshop"])

    assert evidence_cli.main() == 17
    assert observed_argv == ["remem-paired-benchmark", "--benchmark", "webshop"]


def test_main_binds_controlled_measurement_to_loaded_readiness_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_payload = {"schema_version": 1, "evidence_fingerprint": "abc"}
    readiness_path.write_text(json.dumps(readiness_payload), encoding="utf-8")
    original_runner = evidence_cli.paired_cli.run_controlled_paired_external_benchmarks
    observed: dict[str, object] = {}
    source_path = tmp_path / "webshop"
    source_revision = "c" * 40

    def evidence_bound_runner(readiness_evidence, *args, **kwargs):
        observed["evidence"] = readiness_evidence
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "controlled-result"

    def delegated_main() -> int:
        observed["delegated_argv"] = list(sys.argv)
        observed["result"] = evidence_cli.paired_cli.run_controlled_paired_external_benchmarks(
            "baseline",
            "treatment",
            (11, 17),
            runtime_requirements="runtime",
            source_checkout_paths={"webshop": tmp_path},
            source_checkout_requirements={"webshop": "source"},
        )
        return 0

    monkeypatch.setattr(
        evidence_cli,
        "run_evidence_bound_paired_external_benchmarks",
        evidence_bound_runner,
    )
    monkeypatch.setattr(evidence_cli.paired_cli, "main", delegated_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--source-checkout",
            f"webshop={source_path}",
            "--require-source-revision",
            f"webshop={source_revision}",
            "--benchmark",
            "webshop",
        ],
    )

    assert evidence_cli.main() == 0
    assert observed["evidence"] == readiness_payload
    assert observed["result"] == "controlled-result"
    assert observed["delegated_argv"] == [
        "remem-paired-benchmark",
        "--source-checkout",
        f"webshop={source_path}",
        "--require-source-revision",
        f"webshop={source_revision}",
        "--benchmark",
        "webshop",
    ]
    assert evidence_cli.paired_cli.run_controlled_paired_external_benchmarks is original_runner


def test_main_rejects_evidence_binding_for_preflight_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            f"--require-preflight-evidence={readiness_path}",
            "--preflight-only",
        ],
    )

    with pytest.raises(SystemExit, match="for measured execution"):
        evidence_cli.main()


def test_main_rejects_evidence_binding_without_controlled_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--benchmark",
            "webshop",
        ],
    )

    with pytest.raises(SystemExit, match="requires declared source checkouts"):
        evidence_cli.main()


def test_main_rejects_non_object_readiness_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--source-checkout=webshop=/tmp/webshop",
            f"--require-source-revision=webshop={'c' * 40}",
        ],
    )

    with pytest.raises(SystemExit, match="root must be a JSON object"):
        evidence_cli.main()


def test_extract_readiness_evidence_rejects_duplicates() -> None:
    with pytest.raises(SystemExit, match="may be specified only once"):
        evidence_cli._extract_readiness_evidence_path(
            [
                "remem-paired-benchmark",
                "--require-preflight-evidence=first.json",
                "--require-preflight-evidence",
                "second.json",
            ]
        )
