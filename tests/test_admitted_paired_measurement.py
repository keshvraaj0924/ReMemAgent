"""Regression coverage for measuring from an already-admitted paired snapshot."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

import experiments.paired_source_preflight as controlled_paired
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.runtime_provenance import RUNTIME_PROVENANCE_SCHEMA_VERSION, RuntimeProvenance
from experiments.source_checkouts import SourceCheckoutProvenance


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


def _preflight_result() -> controlled_paired.ControlledPairedPreflightResult:
    runtime = RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision="a" * 40,
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
    return controlled_paired.ControlledPairedPreflightResult(runtime, sources)


def test_admitted_measurement_reuses_exact_snapshots_without_readmission(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    admitted = _preflight_result()
    measured = SimpleNamespace(runtime_provenance=None)

    monkeypatch.setattr(
        controlled_paired,
        "collect_runtime_provenance",
        lambda: pytest.fail("admitted measurement must not recollect runtime state"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "collect_source_checkout_provenance",
        lambda repositories: pytest.fail("admitted measurement must not recollect source state"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "preflight_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("admitted measurement must not rerun preflight"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "run_paired_external_benchmarks",
        lambda *args, **kwargs: measured,
    )
    monkeypatch.setattr(
        controlled_paired,
        "replace",
        lambda result, **changes: SimpleNamespace(runtime_provenance=changes["runtime_provenance"]),
    )

    result = controlled_paired.run_admitted_paired_external_benchmarks(
        admitted,
        baseline,
        treatment,
        (11, 17),
    )

    assert result.runtime_provenance is admitted.runtime_provenance
    assert result.source_checkout_provenance is admitted.source_checkout_provenance
    assert result.paired_result.runtime_provenance is admitted.runtime_provenance


def test_admitted_measurement_rejects_unknown_preflight_type() -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    with pytest.raises(TypeError, match="ControlledPairedPreflightResult"):
        controlled_paired.run_admitted_paired_external_benchmarks(
            object(),  # type: ignore[arg-type]
            baseline,
            treatment,
            (11,),
        )
