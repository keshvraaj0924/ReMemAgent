"""Regression coverage for independent runtime and source admission controls."""

from __future__ import annotations

from pathlib import Path

import experiments.controlled_external_benchmark as controlled
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.runtime_provenance import (
    CLEAN_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
)
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement


def _spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop",
        episode_count=1,
        max_steps=2,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def _runtime() -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision="remem-revision",
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.0.0-test",
        dependency_fingerprint="0" * 64,
        dependency_versions={},
    )


def test_runtime_only_controlled_run_preserves_admitted_runtime(monkeypatch) -> None:
    runtime = _runtime()
    events: list[str] = []
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
        lambda repositories: events.append("source_collect"),
    )
    monkeypatch.setattr(
        controlled,
        "validate_external_benchmark_runtime",
        lambda spec, probe_action=None: events.append("preflight"),
    )
    monkeypatch.setattr(
        controlled,
        "run_external_benchmark",
        lambda spec, runner=None: events.append("measure") or object(),
    )

    result = controlled.run_controlled_external_benchmark(
        _spec(),
        runtime_requirements=RuntimeRequirements(expected_code_revision="remem-revision"),
    )

    assert result.runtime_provenance is runtime
    assert result.source_checkout_provenance == {}
    assert events == ["runtime_collect", "runtime_validate", "preflight", "measure"]


def test_source_only_controlled_run_collects_runtime_without_runtime_validation(monkeypatch) -> None:
    runtime = _runtime()
    source = {
        "webshop": SourceCheckoutProvenance(
            revision="webshop-revision",
            working_tree_state=CLEAN_STATE,
        )
    }
    events: list[str] = []
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
        lambda spec, runner=None: events.append("measure") or object(),
    )

    result = controlled.run_controlled_external_benchmark(
        _spec(),
        source_checkout_paths={"WebShop": Path("/tmp/webshop")},
        source_checkout_requirements={
            "WebShop": SourceCheckoutRequirement("webshop-revision")
        },
    )

    assert result.runtime_provenance is runtime
    assert result.source_checkout_provenance is source
    assert events == [
        "runtime_collect",
        "source_collect",
        "source_validate",
        "preflight",
        "measure",
    ]
