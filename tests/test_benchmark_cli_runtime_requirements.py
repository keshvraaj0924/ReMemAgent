"""Regression coverage for controlled runtime requirements in the benchmark CLI."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import experiments.benchmark_cli as benchmark_cli
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.runtime_requirements import RuntimeRequirements
from remem.environments import EnvironmentContractReport


def _arguments(tmp_path: Path) -> Namespace:
    """Build a minimal measured benchmark CLI namespace."""

    return Namespace(
        benchmark="alfworld-eval",
        episodes=1,
        max_steps=3,
        seed=17,
        seeds=None,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        action_policy_factory=None,
        minimum_trust=0.0,
        success_evaluator="example:is_success",
        transfer_success_evaluator=None,
        output=tmp_path / "report.json",
        manifest=None,
        observability_output=None,
        overwrite=False,
        preflight=False,
        runtime_preflight=False,
        repeated_runtime_preflight=False,
        preflight_before_run=False,
        probe_action=None,
        require_code_revision=None,
        require_clean_working_tree=False,
        require_dependency_version=None,
    )


def test_build_runtime_requirements_parses_exact_cli_contract(tmp_path: Path) -> None:
    """Revision, cleanliness, and dependency pins are preserved exactly."""

    arguments = _arguments(tmp_path)
    arguments.require_code_revision = "abc123"
    arguments.require_clean_working_tree = True
    arguments.require_dependency_version = ["alfworld==0.4.2", "Transformers==4.45.0"]

    requirements = benchmark_cli._build_runtime_requirements(arguments)

    assert requirements == RuntimeRequirements(
        expected_code_revision="abc123",
        require_clean_working_tree=True,
        dependency_versions={"alfworld": "0.4.2", "Transformers": "4.45.0"},
    )


def test_parse_dependency_requirements_rejects_duplicate_normalized_names() -> None:
    """Case-only duplicate dependency pins fail before measurement."""

    with pytest.raises(ValueError, match="unique ignoring case"):
        benchmark_cli._parse_dependency_requirements(
            ["transformers==4.45.0", "Transformers==4.46.0"]
        )


def test_runtime_preflight_uses_controlled_gate_when_requirements_are_declared(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Runtime-only CLI probes enforce requirements before external probing."""

    arguments = _arguments(tmp_path)
    arguments.runtime_preflight = True
    arguments.probe_action = "look"
    arguments.require_code_revision = "abc123"
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def controlled_preflight(
        spec: ExternalBenchmarkSpec,
        *,
        probe_action: str | None,
        runtime_requirements: RuntimeRequirements | None,
    ) -> EnvironmentContractReport:
        captured["spec"] = spec
        captured["probe_action"] = probe_action
        captured["requirements"] = runtime_requirements
        return EnvironmentContractReport(initial_observation="ready")

    def fail_legacy_preflight(*args, **kwargs) -> None:
        pytest.fail("legacy preflight must not bypass runtime requirements")

    monkeypatch.setattr(
        benchmark_cli,
        "validate_controlled_external_benchmark_runtime",
        controlled_preflight,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "validate_external_benchmark_runtime",
        fail_legacy_preflight,
    )

    assert benchmark_cli.main() == 0
    assert captured["probe_action"] == "look"
    requirements = captured["requirements"]
    assert isinstance(requirements, RuntimeRequirements)
    assert requirements.expected_code_revision == "abc123"


def test_measured_single_run_enforces_requirements_without_extra_preflight_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Declaring a runtime contract automatically gates measured execution."""

    arguments = _arguments(tmp_path)
    arguments.require_clean_working_tree = True
    report = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda **kwargs: type("Provenance", (), {"to_dict": lambda self: {}})(),
    )

    def controlled_run(
        spec: ExternalBenchmarkSpec,
        *,
        probe_action: str | None,
        runner: object | None,
        runtime_requirements: RuntimeRequirements | None,
    ) -> object:
        captured["requirements"] = runtime_requirements
        return report

    def fail_direct_measurement(*args, **kwargs) -> None:
        pytest.fail("direct measurement must not bypass runtime requirements")

    monkeypatch.setattr(
        benchmark_cli,
        "run_external_benchmark_with_preflight",
        controlled_run,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "run_external_benchmark",
        fail_direct_measurement,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "_persist_benchmark_bundle",
        lambda output_path, **kwargs: output_path,
    )

    assert benchmark_cli.main() == 0
    requirements = captured["requirements"]
    assert isinstance(requirements, RuntimeRequirements)
    assert requirements.require_clean_working_tree is True


def test_callable_only_preflight_rejects_runtime_requirements(monkeypatch, tmp_path: Path) -> None:
    """A callable-only preflight cannot pretend to enforce runtime requirements."""

    arguments = _arguments(tmp_path)
    arguments.preflight = True
    arguments.require_code_revision = "abc123"
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "validate_external_benchmark",
        lambda spec: pytest.fail("callable validation must not run after contract rejection"),
    )

    with pytest.raises(ValueError, match="runtime requirements require --runtime-preflight"):
        benchmark_cli.main()
