from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

import experiments.controlled_external_benchmark as controlled
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.runtime_provenance import (
    CLEAN_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
)
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement


def _build_spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop",
        episode_count=1,
        max_steps=2,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


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


def _source_provenance() -> MappingProxyType[str, SourceCheckoutProvenance]:
    return MappingProxyType(
        {
            "WebShop": SourceCheckoutProvenance(
                revision="webshop-revision",
                working_tree_state=CLEAN_STATE,
            )
        }
    )


def _source_requirements() -> dict[str, SourceCheckoutRequirement]:
    return {
        "WebShop": SourceCheckoutRequirement(
            expected_revision="webshop-revision",
            require_clean_working_tree=True,
        )
    }


def test_controlled_single_preserves_exact_admitted_snapshots(monkeypatch) -> None:
    events: list[str] = []
    runtime = _runtime_provenance()
    source = _source_provenance()
    report = object()

    monkeypatch.setattr(
        controlled,
        "collect_runtime_provenance",
        lambda: events.append("runtime_collect") or runtime,
    )
    monkeypatch.setattr(
        controlled,
        "validate_runtime_requirements",
        lambda provenance, requirements: events.append("runtime_validate"),
    )
    monkeypatch.setattr(
        controlled,
        "collect_source_checkout_provenance",
        lambda repositories: events.append("source_collect") or source,
    )
    monkeypatch.setattr(
        controlled,
        "validate_source_checkout_requirements",
        lambda provenance, requirements: events.append("source_validate"),
    )
    monkeypatch.setattr(
        controlled,
        "validate_external_benchmark_runtime",
        lambda spec, probe_action=None: events.append("preflight"),
    )
    monkeypatch.setattr(
        controlled,
        "run_external_benchmark",
        lambda spec, runner=None: events.append("measure") or report,
    )

    result = controlled.run_controlled_external_benchmark(
        _build_spec(),
        runtime_requirements=RuntimeRequirements(),
        source_checkout_paths={"WebShop": Path("/tmp/webshop")},
        source_checkout_requirements=_source_requirements(),
        probe_action="look",
    )

    assert result.report is report
    assert result.runtime_provenance is runtime
    assert result.source_checkout_provenance is source
    assert events == [
        "runtime_collect",
        "runtime_validate",
        "source_collect",
        "source_validate",
        "preflight",
        "measure",
    ]


def test_controlled_single_admission_failure_prevents_preflight_and_measurement(
    monkeypatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(controlled, "collect_runtime_provenance", _runtime_provenance)

    def reject_runtime(provenance, requirements) -> None:
        events.append("runtime_validate")
        raise ValueError("runtime drift")

    monkeypatch.setattr(controlled, "validate_runtime_requirements", reject_runtime)
    monkeypatch.setattr(
        controlled,
        "collect_source_checkout_provenance",
        lambda repositories: pytest.fail("source collection must not follow runtime failure"),
    )
    monkeypatch.setattr(
        controlled,
        "validate_external_benchmark_runtime",
        lambda *args, **kwargs: pytest.fail("preflight must not follow admission failure"),
    )
    monkeypatch.setattr(
        controlled,
        "run_external_benchmark",
        lambda *args, **kwargs: pytest.fail("measurement must not follow admission failure"),
    )

    with pytest.raises(ValueError, match="runtime drift"):
        controlled.run_controlled_external_benchmark(
            _build_spec(),
            runtime_requirements=RuntimeRequirements(),
            source_checkout_paths={},
            source_checkout_requirements=_source_requirements(),
        )

    assert events == ["runtime_validate"]


def test_controlled_repeated_validates_shared_state_once_before_seed_preflight(
    monkeypatch,
) -> None:
    events: list[str] = []
    runtime = _runtime_provenance()
    source = _source_provenance()
    reports = (object(), object())

    monkeypatch.setattr(
        controlled,
        "collect_runtime_provenance",
        lambda: events.append("runtime_collect") or runtime,
    )
    monkeypatch.setattr(
        controlled,
        "validate_runtime_requirements",
        lambda provenance, requirements: events.append("runtime_validate"),
    )
    monkeypatch.setattr(
        controlled,
        "collect_source_checkout_provenance",
        lambda repositories: events.append("source_collect") or source,
    )
    monkeypatch.setattr(
        controlled,
        "validate_source_checkout_requirements",
        lambda provenance, requirements: events.append("source_validate"),
    )
    monkeypatch.setattr(
        controlled,
        "validate_repeated_external_benchmark_runtime",
        lambda spec, seeds, probe_action=None: events.append(f"preflight:{tuple(seeds)}"),
    )
    monkeypatch.setattr(
        controlled,
        "run_repeated_external_benchmarks",
        lambda spec, seeds, runner=None: events.append(f"measure:{tuple(seeds)}") or reports,
    )

    result = controlled.run_controlled_repeated_external_benchmarks(
        _build_spec(),
        (11, 17),
        runtime_requirements=RuntimeRequirements(),
        source_checkout_paths={"WebShop": Path("/tmp/webshop")},
        source_checkout_requirements=_source_requirements(),
    )

    assert result.reports is reports
    assert result.runtime_provenance is runtime
    assert result.source_checkout_provenance is source
    assert events == [
        "runtime_collect",
        "runtime_validate",
        "source_collect",
        "source_validate",
        "preflight:(11, 17)",
        "measure:(11, 17)",
    ]


def test_controlled_repeated_rejects_source_drift_before_preflight(monkeypatch) -> None:
    monkeypatch.setattr(controlled, "collect_runtime_provenance", _runtime_provenance)
    monkeypatch.setattr(
        controlled,
        "validate_runtime_requirements",
        lambda provenance, requirements: None,
    )
    monkeypatch.setattr(
        controlled,
        "collect_source_checkout_provenance",
        lambda repositories: _source_provenance(),
    )

    def reject_source(provenance, requirements) -> None:
        raise ValueError("source drift")

    monkeypatch.setattr(controlled, "validate_source_checkout_requirements", reject_source)
    monkeypatch.setattr(
        controlled,
        "validate_repeated_external_benchmark_runtime",
        lambda *args, **kwargs: pytest.fail("preflight must not follow source drift"),
    )
    monkeypatch.setattr(
        controlled,
        "run_repeated_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("measurement must not follow source drift"),
    )

    with pytest.raises(ValueError, match="source drift"):
        controlled.run_controlled_repeated_external_benchmarks(
            _build_spec(),
            (11, 17),
            runtime_requirements=RuntimeRequirements(),
            source_checkout_paths={"WebShop": Path("/tmp/webshop")},
            source_checkout_requirements=_source_requirements(),
        )
