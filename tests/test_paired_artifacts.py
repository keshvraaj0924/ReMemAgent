from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.paired_artifacts import (
    PAIRED_EXECUTION_ORDER_PROVENANCE_KEY,
    RUNTIME_REQUIREMENTS_PROVENANCE_KEY,
    save_paired_execution_result,
    validate_persisted_paired_execution_provenance,
    validate_persisted_runtime_requirements,
)
from experiments.paired_benchmark import PairedBenchmarkResult, PairedSeedExecution
from experiments.runtime_requirements import RuntimeRequirements


def _result(
    execution_order: tuple[PairedSeedExecution, ...] | None = None,
) -> PairedBenchmarkResult:
    return PairedBenchmarkResult(
        baseline_reports=(),
        treatment_reports=(),
        execution_order=execution_order
        or (
            PairedSeedExecution(
                seed=11,
                first_condition="baseline",
                second_condition="treatment",
            ),
            PairedSeedExecution(
                seed=17,
                first_condition="treatment",
                second_condition="baseline",
            ),
        ),
        comparison=SimpleNamespace(seeds=(11, 17)),  # type: ignore[arg-type]
    )


def _persisted_execution_payload() -> dict[str, object]:
    execution_order = [
        {
            "seed": 11,
            "first_condition": "baseline",
            "second_condition": "treatment",
        },
        {
            "seed": 17,
            "first_condition": "treatment",
            "second_condition": "baseline",
        },
    ]
    canonical_payload = json.dumps(
        execution_order,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_payload).hexdigest()
    return {
        "seeds": [11, 17],
        "execution_order": execution_order,
        "runtime_provenance": {PAIRED_EXECUTION_ORDER_PROVENANCE_KEY: digest},
    }


def test_save_paired_execution_result_persists_temporal_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_provenance: dict[str, object] = {}

    def fake_save(
        baseline_reports,
        treatment_reports,
        comparison,
        output_path,
        *,
        runtime_provenance,
    ):
        del baseline_reports, treatment_reports, comparison
        captured_provenance.update(runtime_provenance)
        output_path.write_text(
            json.dumps(
                {
                    "experiment_identity": "paired-test-identity",
                    "runtime_provenance": runtime_provenance,
                }
            ),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr("experiments.paired_artifacts.save_paired_benchmark_result", fake_save)

    output_path = save_paired_execution_result(
        _result(),
        tmp_path / "paired.json",
        runtime_provenance={"code_revision": "abc123"},
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["execution_order"] == [
        {
            "seed": 11,
            "first_condition": "baseline",
            "second_condition": "treatment",
        },
        {
            "seed": 17,
            "first_condition": "treatment",
            "second_condition": "baseline",
        },
    ]
    digest = captured_provenance[PAIRED_EXECUTION_ORDER_PROVENANCE_KEY]
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert persisted["runtime_provenance"][PAIRED_EXECUTION_ORDER_PROVENANCE_KEY] == digest
    assert persisted["runtime_provenance"]["code_revision"] == "abc123"


def test_save_paired_execution_result_binds_runtime_requirement_contract(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements = RuntimeRequirements(
        expected_code_revision="abc123",
        require_clean_working_tree=True,
        dependency_versions={"ALFWorld": "0.4.2"},
    )

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
        tmp_path / "controlled.json",
        runtime_provenance={"code_revision": "abc123"},
        runtime_requirements=requirements,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["runtime_requirements"] == requirements.to_dict()
    assert (
        persisted["runtime_provenance"][RUNTIME_REQUIREMENTS_PROVENANCE_KEY] == requirements.sha256
    )
    validate_persisted_runtime_requirements(persisted)


def test_validate_persisted_runtime_requirements_rejects_tampered_contract() -> None:
    requirements = RuntimeRequirements(
        expected_code_revision="abc123",
        dependency_versions={"ALFWorld": "0.4.2"},
    )
    payload = {
        "runtime_requirements": requirements.to_dict(),
        "runtime_provenance": {RUNTIME_REQUIREMENTS_PROVENANCE_KEY: requirements.sha256},
    }
    persisted_requirements = payload["runtime_requirements"]
    assert isinstance(persisted_requirements, dict)
    dependencies = persisted_requirements["dependency_versions"]
    assert isinstance(dependencies, dict)
    dependencies["ALFWorld"] = "0.4.3"

    with pytest.raises(ValueError, match="digest does not match"):
        validate_persisted_runtime_requirements(payload)


def test_validate_persisted_runtime_requirements_rejects_orphaned_digest() -> None:
    payload = {"runtime_provenance": {RUNTIME_REQUIREMENTS_PROVENANCE_KEY: "0" * 64}}

    with pytest.raises(ValueError, match="requires persisted runtime_requirements"):
        validate_persisted_runtime_requirements(payload)


def test_save_paired_execution_result_rejects_seed_mismatch_before_persistence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _result(
        (
            PairedSeedExecution(11, "baseline", "treatment"),
            PairedSeedExecution(19, "treatment", "baseline"),
        )
    )
    monkeypatch.setattr(
        "experiments.paired_artifacts.save_paired_benchmark_result",
        lambda *args, **kwargs: pytest.fail("persistence must not start"),
    )

    with pytest.raises(ValueError, match="seeds must exactly match"):
        save_paired_execution_result(invalid, tmp_path / "paired.json")


def test_save_paired_execution_result_rejects_non_counterbalanced_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _result(
        (
            PairedSeedExecution(11, "baseline", "treatment"),
            PairedSeedExecution(17, "baseline", "treatment"),
        )
    )
    monkeypatch.setattr(
        "experiments.paired_artifacts.save_paired_benchmark_result",
        lambda *args, **kwargs: pytest.fail("persistence must not start"),
    )

    with pytest.raises(ValueError, match="deterministic counterbalanced"):
        save_paired_execution_result(invalid, tmp_path / "paired.json")


def test_save_paired_execution_result_rejects_reserved_provenance_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="reserves"):
        save_paired_execution_result(
            _result(),
            tmp_path / "paired.json",
            runtime_provenance={PAIRED_EXECUTION_ORDER_PROVENANCE_KEY: "caller-value"},
        )


def test_validate_persisted_paired_execution_provenance_accepts_valid_trace() -> None:
    validate_persisted_paired_execution_provenance(_persisted_execution_payload())


def test_validate_persisted_paired_execution_provenance_rejects_tampered_trace() -> None:
    payload = _persisted_execution_payload()
    execution_order = payload["execution_order"]
    assert isinstance(execution_order, list)
    execution_order[1] = {
        "seed": 17,
        "first_condition": "baseline",
        "second_condition": "treatment",
    }

    with pytest.raises(ValueError, match="deterministic counterbalanced"):
        validate_persisted_paired_execution_provenance(payload)


def test_validate_persisted_paired_execution_provenance_rejects_stale_digest() -> None:
    payload = _persisted_execution_payload()
    runtime_provenance = payload["runtime_provenance"]
    assert isinstance(runtime_provenance, dict)
    runtime_provenance[PAIRED_EXECUTION_ORDER_PROVENANCE_KEY] = "0" * 64

    with pytest.raises(ValueError, match="digest does not match"):
        validate_persisted_paired_execution_provenance(payload)


def test_validate_persisted_paired_execution_provenance_allows_legacy_artifact() -> None:
    validate_persisted_paired_execution_provenance({"seeds": [11, 17]})
