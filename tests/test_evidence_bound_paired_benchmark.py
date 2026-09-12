"""Regression coverage for readiness-evidence-bound paired measurement."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import experiments.evidence_bound_paired_benchmark as evidence_bound
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_source_preflight import (
    ControlledPairedBenchmarkResult,
    ControlledPairedPreflightResult,
)
from experiments.preflight_evidence import build_controlled_paired_preflight_evidence
from experiments.runtime_provenance import RUNTIME_PROVENANCE_SCHEMA_VERSION, RuntimeProvenance
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop",
        episode_count=1,
        max_steps=3,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def _preflight_result(*, revision: str = "a" * 40) -> ControlledPairedPreflightResult:
    runtime = RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision=revision,
        working_tree_state="clean",
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint="b" * 64,
        dependency_versions={"pytest": "8.0.0"},
    )
    sources = MappingProxyType(
        {
            "webshop": SourceCheckoutProvenance(
                revision="c" * 40,
                working_tree_state="clean",
            )
        }
    )
    return ControlledPairedPreflightResult(runtime, sources)


def _requirements() -> dict[str, SourceCheckoutRequirement]:
    return {
        "webshop": SourceCheckoutRequirement(
            expected_revision="c" * 40,
            require_clean_working_tree=True,
        )
    }


def test_evidence_bound_run_validates_before_measurement(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    admitted = _preflight_result()
    requirements = _requirements()
    evidence = build_controlled_paired_preflight_evidence(admitted, requirements)
    events: list[str] = []
    expected_result = ControlledPairedBenchmarkResult(
        paired_result=SimpleNamespace(runtime_provenance=admitted.runtime_provenance),
        runtime_provenance=admitted.runtime_provenance,
        source_checkout_provenance=admitted.source_checkout_provenance,
    )

    def preflight(*args, **kwargs):
        events.append("preflight")
        return admitted

    def validate(payload, result, source_requirements):
        assert payload is evidence
        assert result is admitted
        assert source_requirements is requirements
        events.append("evidence")

    def measure(*args, **kwargs):
        assert args[0] is admitted
        events.append("measurement")
        return expected_result

    monkeypatch.setattr(evidence_bound, "preflight_controlled_paired_external_benchmarks", preflight)
    monkeypatch.setattr(evidence_bound, "validate_preflight_evidence_matches_admission", validate)
    monkeypatch.setattr(evidence_bound, "run_admitted_paired_external_benchmarks", measure)

    result = evidence_bound.run_evidence_bound_paired_external_benchmarks(
        evidence,
        baseline,
        treatment,
        (11, 17),
        runtime_requirements=RuntimeRequirements(expected_code_revision="a" * 40),
        source_checkout_paths={"webshop": Path("/tmp/webshop")},
        source_checkout_requirements=requirements,
        probe_action="search",
    )

    assert events == ["preflight", "evidence", "measurement"]
    assert result is expected_result


def test_stale_readiness_evidence_blocks_measurement(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    original = _preflight_result()
    drifted = _preflight_result(revision="d" * 40)
    requirements = _requirements()
    evidence = build_controlled_paired_preflight_evidence(original, requirements)

    monkeypatch.setattr(
        evidence_bound,
        "preflight_controlled_paired_external_benchmarks",
        lambda *args, **kwargs: drifted,
    )
    monkeypatch.setattr(
        evidence_bound,
        "run_admitted_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("stale readiness evidence must block measurement"),
    )

    with pytest.raises(ValueError, match="does not match the current admitted state"):
        evidence_bound.run_evidence_bound_paired_external_benchmarks(
            evidence,
            baseline,
            treatment,
            (11, 17),
            runtime_requirements=RuntimeRequirements(expected_code_revision="d" * 40),
            source_checkout_paths={"webshop": Path("/tmp/webshop")},
            source_checkout_requirements=requirements,
        )


def test_evidence_bound_run_rejects_non_mapping_evidence() -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    with pytest.raises(TypeError, match="readiness_evidence must be a mapping"):
        evidence_bound.run_evidence_bound_paired_external_benchmarks(
            [],  # type: ignore[arg-type]
            baseline,
            treatment,
            (11,),
            runtime_requirements=RuntimeRequirements(),
            source_checkout_paths={"webshop": Path("/tmp/webshop")},
            source_checkout_requirements=_requirements(),
        )
