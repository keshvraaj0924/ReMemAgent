"""Regression coverage for identity-bound controlled benchmark evidence."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

import experiments.controlled_benchmark_artifacts as artifacts
from experiments.controlled_external_benchmark import (
    ControlledExternalBenchmarkResult,
    ControlledRepeatedExternalBenchmarkResult,
)
from experiments.paired_artifacts import (
    RUNTIME_REQUIREMENTS_PROVENANCE_KEY,
    SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY,
    SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY,
)
from experiments.runtime_provenance import (
    CLEAN_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
)
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement
from remem.benchmark import BenchmarkRunReport


def _runtime_provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision="remem-revision",
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.0.0-test",
        dependency_fingerprint="0" * 64,
        dependency_versions={"transformers": "1.0.0"},
    )


def _source_provenance() -> dict[str, SourceCheckoutProvenance]:
    return {
        "webshop": SourceCheckoutProvenance(
            revision="webshop-revision",
            working_tree_state=CLEAN_STATE,
        )
    }


def _source_requirements() -> dict[str, SourceCheckoutRequirement]:
    return {
        "WebShop": SourceCheckoutRequirement(
            expected_revision="webshop-revision",
            require_clean_working_tree=True,
        )
    }


def _report() -> BenchmarkRunReport:
    return cast(BenchmarkRunReport, cast(Any, object()))


def test_save_controlled_single_binds_exact_admitted_evidence(tmp_path, monkeypatch) -> None:
    runtime = _runtime_provenance()
    requirements = RuntimeRequirements(expected_code_revision="remem-revision")
    source_requirements = _source_requirements()
    result = ControlledExternalBenchmarkResult(
        report=_report(),
        runtime_provenance=runtime,
        source_checkout_provenance=_source_provenance(),
    )
    captured_provenance: dict[str, object] = {}

    def fake_save(report, output_path, *, runtime_provenance):
        del report
        captured_provenance.update(runtime_provenance)
        output_path.write_text(
            json.dumps({"runtime_provenance": runtime_provenance}),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(artifacts, "save_benchmark_report", fake_save)

    output_path = artifacts.save_controlled_benchmark_result(
        result,
        tmp_path / "controlled.json",
        runtime_requirements=requirements,
        source_checkout_requirements=source_requirements,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert captured_provenance["code_revision"] == "remem-revision"
    assert captured_provenance[RUNTIME_REQUIREMENTS_PROVENANCE_KEY] == requirements.sha256
    assert SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY in captured_provenance
    assert SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY in captured_provenance
    assert payload["runtime_requirements"] == requirements.to_dict()
    assert (
        payload["source_checkout_requirements"]["repositories"]["webshop"]["expected_revision"]
        == "webshop-revision"
    )
    artifacts.validate_persisted_controlled_benchmark_artifact(payload)


def test_save_controlled_repeated_forwards_statistics_and_shared_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    result = ControlledRepeatedExternalBenchmarkResult(
        reports=(_report(), _report()),
        runtime_provenance=_runtime_provenance(),
        source_checkout_provenance={},
    )
    requirements = RuntimeRequirements(expected_code_revision="remem-revision")
    captured: dict[str, object] = {}

    def fake_save(reports, output_path, *, runtime_provenance, statistics):
        captured["reports"] = reports
        captured["runtime_provenance"] = runtime_provenance
        captured["statistics"] = statistics
        output_path.write_text(
            json.dumps({"runtime_provenance": runtime_provenance}),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(artifacts, "save_repeated_benchmark_reports", fake_save)

    output_path = artifacts.save_controlled_repeated_benchmark_result(
        result,
        tmp_path / "controlled-repeated.json",
        runtime_requirements=requirements,
        statistics={"success_rate": 0.5},
    )

    assert captured["reports"] is result.reports
    assert captured["statistics"] == {"success_rate": 0.5}
    provenance = captured["runtime_provenance"]
    assert isinstance(provenance, dict)
    assert provenance[RUNTIME_REQUIREMENTS_PROVENANCE_KEY] == requirements.sha256
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    artifacts.validate_persisted_controlled_benchmark_artifact(payload)


def test_controlled_artifact_rejects_source_state_that_violates_contract(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "save_benchmark_report",
        lambda *args, **kwargs: pytest.fail("writer must not run after admission mismatch"),
    )
    result = ControlledExternalBenchmarkResult(
        report=_report(),
        runtime_provenance=_runtime_provenance(),
        source_checkout_provenance={
            "webshop": SourceCheckoutProvenance("actual-revision", CLEAN_STATE)
        },
    )

    with pytest.raises(ValueError, match="source checkout revision mismatch"):
        artifacts.save_controlled_benchmark_result(
            result,
            tmp_path / "invalid.json",
            source_checkout_requirements=_source_requirements(),
        )


def test_controlled_artifact_rejects_incomplete_source_evidence(tmp_path) -> None:
    result = ControlledExternalBenchmarkResult(
        report=_report(),
        runtime_provenance=_runtime_provenance(),
        source_checkout_provenance=_source_provenance(),
    )

    with pytest.raises(ValueError, match="must be provided together"):
        artifacts.save_controlled_benchmark_result(result, tmp_path / "incomplete.json")


def test_controlled_artifact_validation_rejects_tampered_source_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    result = ControlledExternalBenchmarkResult(
        report=_report(),
        runtime_provenance=_runtime_provenance(),
        source_checkout_provenance=_source_provenance(),
    )

    def fake_save(report, output_path, *, runtime_provenance):
        del report
        output_path.write_text(
            json.dumps({"runtime_provenance": runtime_provenance}),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(artifacts, "save_benchmark_report", fake_save)
    output_path = artifacts.save_controlled_benchmark_result(
        result,
        tmp_path / "tampered.json",
        source_checkout_requirements=_source_requirements(),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["source_checkout_provenance"]["repositories"]["webshop"]["revision"] = "tampered"

    with pytest.raises(ValueError, match="invalid persisted metadata"):
        artifacts.validate_persisted_controlled_benchmark_artifact(payload)
