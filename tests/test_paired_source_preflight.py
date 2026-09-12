"""Regression coverage for controlled paired source-checkout admission."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import experiments.paired_source_preflight as controlled_paired
from experiments.external_benchmark import ExternalBenchmarkSpec
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


def _runtime_provenance(*, revision: str = "actual") -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision=revision,
        working_tree_state="clean",
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint="0" * 64,
        dependency_versions={"pytest": "8.0.0"},
    )


def _source_provenance(*, revision: str = "source-revision"):
    return MappingProxyType(
        {
            "webshop": SourceCheckoutProvenance(
                revision=revision,
                working_tree_state="clean",
            )
        }
    )


def test_runtime_mismatch_blocks_source_collection_and_all_paired_side_effects(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(
        controlled_paired,
        "collect_runtime_provenance",
        lambda: _runtime_provenance(revision="actual"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "collect_source_checkout_provenance",
        lambda repositories: pytest.fail(
            "source checkout collection must not run after runtime drift"
        ),
    )
    monkeypatch.setattr(
        controlled_paired,
        "preflight_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("paired preflight must not run after runtime drift"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "run_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("measurement must not run after runtime drift"),
    )

    with pytest.raises(ValueError, match="runtime code revision"):
        controlled_paired.run_controlled_paired_external_benchmarks(
            baseline,
            treatment,
            (11,),
            runtime_requirements=RuntimeRequirements(expected_code_revision="required"),
            source_checkout_paths={"webshop": Path("/tmp/webshop")},
            source_checkout_requirements={
                "webshop": SourceCheckoutRequirement(expected_revision="source-revision")
            },
        )


def test_source_revision_mismatch_blocks_paired_preflight_and_measurement(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(
        controlled_paired,
        "collect_runtime_provenance",
        _runtime_provenance,
    )
    monkeypatch.setattr(
        controlled_paired,
        "collect_source_checkout_provenance",
        lambda repositories: _source_provenance(revision="actual-source"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "preflight_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("paired preflight must not run after source drift"),
    )
    monkeypatch.setattr(
        controlled_paired,
        "run_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("measurement must not run after source drift"),
    )

    with pytest.raises(ValueError, match="source checkout revision mismatch"):
        controlled_paired.run_controlled_paired_external_benchmarks(
            baseline,
            treatment,
            (11,),
            runtime_requirements=RuntimeRequirements(expected_code_revision="actual"),
            source_checkout_paths={"webshop": Path("/tmp/webshop")},
            source_checkout_requirements={
                "webshop": SourceCheckoutRequirement(expected_revision="required-source")
            },
        )


def test_successful_controlled_run_carries_exact_admitted_snapshots(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    expected_runtime = _runtime_provenance()
    expected_source = _source_provenance()
    events: list[str] = []

    def collect_runtime() -> RuntimeProvenance:
        events.append("runtime")
        return expected_runtime

    def collect_source(repositories):
        events.append("source")
        return expected_source

    def paired_preflight(*args, **kwargs):
        assert kwargs["runtime_requirements"] is None
        events.append("preflight")
        return None

    measured_result = SimpleNamespace(runtime_provenance=None)

    def paired_measurement(*args, **kwargs):
        events.append("measurement")
        return measured_result

    def attach_runtime(result, **changes):
        assert result is measured_result
        assert changes == {"runtime_provenance": expected_runtime}
        return SimpleNamespace(runtime_provenance=expected_runtime)

    monkeypatch.setattr(controlled_paired, "collect_runtime_provenance", collect_runtime)
    monkeypatch.setattr(controlled_paired, "collect_source_checkout_provenance", collect_source)
    monkeypatch.setattr(controlled_paired, "preflight_paired_external_benchmarks", paired_preflight)
    monkeypatch.setattr(controlled_paired, "run_paired_external_benchmarks", paired_measurement)
    monkeypatch.setattr(controlled_paired, "replace", attach_runtime)

    result = controlled_paired.run_controlled_paired_external_benchmarks(
        baseline,
        treatment,
        (11, 17),
        runtime_requirements=RuntimeRequirements(expected_code_revision="actual"),
        source_checkout_paths={"webshop": Path("/tmp/webshop")},
        source_checkout_requirements={
            "webshop": SourceCheckoutRequirement(expected_revision="source-revision")
        },
        probe_action="search",
    )

    assert events == ["runtime", "source", "preflight", "measurement"]
    assert result.runtime_provenance is expected_runtime
    assert result.source_checkout_provenance is expected_source
    assert result.paired_result.runtime_provenance is expected_runtime


def test_controlled_paired_source_requirements_must_not_be_empty(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(controlled_paired, "collect_runtime_provenance", _runtime_provenance)

    with pytest.raises(ValueError, match="source_checkout_requirements must not be empty"):
        controlled_paired.run_controlled_paired_external_benchmarks(
            baseline,
            treatment,
            (11,),
            runtime_requirements=RuntimeRequirements(expected_code_revision="actual"),
            source_checkout_paths={"webshop": Path("/tmp/webshop")},
            source_checkout_requirements={},
        )


def test_controlled_preflight_returns_exact_snapshots_without_measurement(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    expected_runtime = _runtime_provenance()
    expected_source = _source_provenance()
    events: list[str] = []

    monkeypatch.setattr(
        controlled_paired,
        "collect_runtime_provenance",
        lambda: events.append("runtime") or expected_runtime,
    )
    monkeypatch.setattr(
        controlled_paired,
        "collect_source_checkout_provenance",
        lambda repositories: events.append("source") or expected_source,
    )

    def paired_preflight(*args, **kwargs):
        assert kwargs["runtime_requirements"] is None
        assert kwargs["probe_action"] == "search"
        events.append("preflight")

    monkeypatch.setattr(controlled_paired, "preflight_paired_external_benchmarks", paired_preflight)
    monkeypatch.setattr(
        controlled_paired,
        "run_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("preflight-only admission must not measure"),
    )

    result = controlled_paired.preflight_controlled_paired_external_benchmarks(
        baseline,
        treatment,
        (11, 17),
        runtime_requirements=RuntimeRequirements(expected_code_revision="actual"),
        source_checkout_paths={"webshop": Path("/tmp/webshop")},
        source_checkout_requirements={
            "webshop": SourceCheckoutRequirement(expected_revision="source-revision")
        },
        probe_action="search",
    )

    assert events == ["runtime", "source", "preflight"]
    assert result.runtime_provenance is expected_runtime
    assert result.source_checkout_provenance is expected_source


def test_controlled_run_reuses_preflight_snapshots_for_measurement(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    expected_runtime = _runtime_provenance()
    expected_source = _source_provenance()
    preflight_result = controlled_paired.ControlledPairedPreflightResult(
        runtime_provenance=expected_runtime,
        source_checkout_provenance=expected_source,
    )
    measured_result = SimpleNamespace(runtime_provenance=None)

    monkeypatch.setattr(
        controlled_paired,
        "preflight_controlled_paired_external_benchmarks",
        lambda *args, **kwargs: preflight_result,
    )
    monkeypatch.setattr(
        controlled_paired,
        "run_paired_external_benchmarks",
        lambda *args, **kwargs: measured_result,
    )
    monkeypatch.setattr(
        controlled_paired,
        "replace",
        lambda result, **changes: SimpleNamespace(
            runtime_provenance=changes["runtime_provenance"]
        ),
    )

    result = controlled_paired.run_controlled_paired_external_benchmarks(
        baseline,
        treatment,
        (11, 17),
        runtime_requirements=RuntimeRequirements(expected_code_revision="actual"),
        source_checkout_paths={"webshop": Path("/tmp/webshop")},
        source_checkout_requirements={
            "webshop": SourceCheckoutRequirement(expected_revision="source-revision")
        },
    )

    assert result.runtime_provenance is expected_runtime
    assert result.source_checkout_provenance is expected_source
    assert result.paired_result.runtime_provenance is expected_runtime
