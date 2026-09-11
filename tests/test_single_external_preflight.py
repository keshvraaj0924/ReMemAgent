"""Regression coverage for controlled single-run external preflight."""

from __future__ import annotations

import pytest

from experiments import external_preflight
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.runtime_requirements import RuntimeRequirements
from remem.environments import EnvironmentContractReport


def _spec() -> ExternalBenchmarkSpec:
    """Build a dependency-free benchmark specification for orchestration tests."""

    return ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
        seed=11,
    )


def test_controlled_single_preflight_validates_runtime_before_probe(monkeypatch) -> None:
    """Runtime requirements are enforced before external runtime validation."""

    events: list[str] = []
    requirements = RuntimeRequirements(expected_code_revision="required-sha")

    def collect_runtime_provenance():
        events.append("collect-runtime")
        return object()

    def validate_runtime_requirements(provenance, selected_requirements):
        assert selected_requirements is requirements
        events.append("validate-runtime")

    def validate_external_runtime(spec, *, probe_action=None):
        assert spec.seed == 11
        assert probe_action == "look"
        events.append("probe")
        return EnvironmentContractReport(initial_observation="ready")

    monkeypatch.setattr(
        external_preflight,
        "collect_runtime_provenance",
        collect_runtime_provenance,
    )
    monkeypatch.setattr(
        external_preflight,
        "validate_runtime_requirements",
        validate_runtime_requirements,
    )
    monkeypatch.setattr(
        external_preflight,
        "validate_external_benchmark_runtime",
        validate_external_runtime,
    )

    report = external_preflight.validate_controlled_external_benchmark_runtime(
        _spec(),
        probe_action="look",
        runtime_requirements=requirements,
    )

    assert report.initial_observation == "ready"
    assert events == ["collect-runtime", "validate-runtime", "probe"]


def test_controlled_single_preflight_stops_before_probe_on_runtime_failure(monkeypatch) -> None:
    """A failed runtime gate cannot construct or probe an external environment."""

    probe_called = False
    requirements = RuntimeRequirements(expected_code_revision="required-sha")

    monkeypatch.setattr(external_preflight, "collect_runtime_provenance", lambda: object())

    def reject_runtime(provenance, selected_requirements):
        assert selected_requirements is requirements
        raise ValueError("code revision mismatch")

    def validate_external_runtime(spec, *, probe_action=None):
        nonlocal probe_called
        probe_called = True
        return EnvironmentContractReport(initial_observation="ready")

    monkeypatch.setattr(external_preflight, "validate_runtime_requirements", reject_runtime)
    monkeypatch.setattr(
        external_preflight,
        "validate_external_benchmark_runtime",
        validate_external_runtime,
    )

    with pytest.raises(ValueError, match="code revision mismatch"):
        external_preflight.validate_controlled_external_benchmark_runtime(
            _spec(),
            runtime_requirements=requirements,
        )

    assert not probe_called


def test_single_run_executes_only_after_controlled_preflight(monkeypatch) -> None:
    """Measured execution begins only after the controlled preflight succeeds."""

    events: list[str] = []
    requirements = RuntimeRequirements()

    def controlled_preflight(
        spec,
        *,
        probe_action=None,
        runtime_requirements=None,
        source_checkout_paths=None,
        source_checkout_requirements=None,
    ):
        assert runtime_requirements is requirements
        assert source_checkout_paths is None
        assert source_checkout_requirements is None
        events.append("preflight")
        return EnvironmentContractReport(initial_observation="ready")

    def run_external(spec, *, runner=None):
        events.append("run")
        return object()

    monkeypatch.setattr(
        external_preflight,
        "validate_controlled_external_benchmark_runtime",
        controlled_preflight,
    )
    monkeypatch.setattr(external_preflight, "run_external_benchmark", run_external)

    result = external_preflight.run_external_benchmark_with_preflight(
        _spec(),
        runtime_requirements=requirements,
    )

    assert result is not None
    assert events == ["preflight", "run"]


def test_controlled_single_preflight_rejects_invalid_runtime_requirements(monkeypatch) -> None:
    """Invalid runtime gate objects fail before external runtime validation."""

    probe_called = False

    def validate_external_runtime(spec, *, probe_action=None):
        nonlocal probe_called
        probe_called = True
        return EnvironmentContractReport(initial_observation="ready")

    monkeypatch.setattr(
        external_preflight,
        "validate_external_benchmark_runtime",
        validate_external_runtime,
    )

    with pytest.raises(TypeError, match="RuntimeRequirements"):
        external_preflight.validate_controlled_external_benchmark_runtime(
            _spec(),
            runtime_requirements=object(),  # type: ignore[arg-type]
        )

    assert not probe_called
