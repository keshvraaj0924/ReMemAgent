"""Regression coverage for source-checkout evidence in paired artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.paired_artifacts import (
    SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY,
    SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY,
    save_paired_execution_result,
    validate_persisted_source_checkouts,
)
from experiments.paired_benchmark import PairedBenchmarkResult, PairedSeedExecution
from experiments.runtime_provenance import CLEAN_STATE
from experiments.source_checkouts import (
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    source_checkout_provenance_sha256,
    source_checkout_provenance_to_dict,
    source_checkout_requirements_sha256,
    source_checkout_requirements_to_dict,
)


def _result() -> PairedBenchmarkResult:
    return PairedBenchmarkResult(
        baseline_reports=(),
        treatment_reports=(),
        execution_order=(
            PairedSeedExecution(11, "baseline", "treatment"),
            PairedSeedExecution(17, "treatment", "baseline"),
        ),
        comparison=SimpleNamespace(seeds=(11, 17)),  # type: ignore[arg-type]
    )


def _source_requirements() -> dict[str, SourceCheckoutRequirement]:
    return {
        "WebShop": SourceCheckoutRequirement(
            expected_revision="webshop-revision",
            require_clean_working_tree=True,
        )
    }


def _source_provenance() -> dict[str, SourceCheckoutProvenance]:
    return {
        "webshop": SourceCheckoutProvenance(
            revision="webshop-revision",
            working_tree_state=CLEAN_STATE,
        )
    }


def _persisted_source_payload() -> dict[str, object]:
    requirements = _source_requirements()
    provenance = _source_provenance()
    return {
        "source_checkout_requirements": source_checkout_requirements_to_dict(requirements),
        "source_checkout_provenance": source_checkout_provenance_to_dict(provenance),
        "runtime_provenance": {
            SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY: source_checkout_requirements_sha256(
                requirements
            ),
            SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY: source_checkout_provenance_sha256(provenance),
        },
    }


def test_save_paired_execution_result_binds_source_contract_and_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements = _source_requirements()
    provenance = _source_provenance()

    def fake_save(
        baseline_reports,
        treatment_reports,
        comparison,
        output_path,
        *,
        runtime_provenance,
    ):
        del baseline_reports, treatment_reports, comparison
        output_path.write_text(
            json.dumps({"runtime_provenance": runtime_provenance}),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr("experiments.paired_artifacts.save_paired_benchmark_result", fake_save)

    output_path = save_paired_execution_result(
        _result(),
        tmp_path / "controlled-source.json",
        runtime_provenance={"code_revision": "abc123"},
        source_checkout_provenance=provenance,
        source_checkout_requirements=requirements,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["source_checkout_requirements"] == source_checkout_requirements_to_dict(
        requirements
    )
    assert persisted["source_checkout_provenance"] == source_checkout_provenance_to_dict(provenance)
    assert (
        persisted["runtime_provenance"][SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY]
        == source_checkout_requirements_sha256(requirements)
    )
    assert (
        persisted["runtime_provenance"][SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY]
        == source_checkout_provenance_sha256(provenance)
    )
    validate_persisted_source_checkouts(persisted)


def test_save_paired_execution_result_rejects_source_state_that_violates_contract(tmp_path) -> None:
    with pytest.raises(ValueError, match="source checkout revision mismatch"):
        save_paired_execution_result(
            _result(),
            tmp_path / "invalid-source.json",
            source_checkout_provenance={
                "webshop": SourceCheckoutProvenance("actual", CLEAN_STATE)
            },
            source_checkout_requirements={
                "webshop": SourceCheckoutRequirement("required")
            },
        )


def test_save_paired_execution_result_requires_complete_source_evidence(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        save_paired_execution_result(
            _result(),
            tmp_path / "incomplete-source.json",
            source_checkout_provenance=_source_provenance(),
        )


def test_validate_persisted_source_checkouts_rejects_tampered_snapshot() -> None:
    payload = _persisted_source_payload()
    snapshot = payload["source_checkout_provenance"]
    assert isinstance(snapshot, dict)
    repositories = snapshot["repositories"]
    assert isinstance(repositories, dict)
    webshop = repositories["webshop"]
    assert isinstance(webshop, dict)
    webshop["revision"] = "tampered"

    with pytest.raises(ValueError, match="invalid persisted metadata"):
        validate_persisted_source_checkouts(payload)


def test_validate_persisted_source_checkouts_rejects_stale_requirement_digest() -> None:
    payload = _persisted_source_payload()
    runtime_provenance = payload["runtime_provenance"]
    assert isinstance(runtime_provenance, dict)
    runtime_provenance[SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY] = "0" * 64

    with pytest.raises(ValueError, match="requirement digest does not match"):
        validate_persisted_source_checkouts(payload)


def test_validate_persisted_source_checkouts_rejects_orphaned_source_digest() -> None:
    payload = {
        "runtime_provenance": {
            SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY: "0" * 64,
        }
    }

    with pytest.raises(ValueError, match="requires persisted requirements"):
        validate_persisted_source_checkouts(payload)
