"""Regression coverage for controlled runtime gates on paired experiments."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.paired_benchmark as paired_benchmark
import experiments.paired_benchmark_cli as paired_cli
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.runtime_provenance import RUNTIME_PROVENANCE_SCHEMA_VERSION, RuntimeProvenance
from experiments.runtime_requirements import RuntimeRequirements


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=1,
        max_steps=3,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def _provenance(*, revision: str = "actual") -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision=revision,
        working_tree_state="clean",
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint="0" * 64,
        dependency_versions={"alfworld": "0.4.2"},
    )


def _cli_arguments(tmp_path: Path) -> Namespace:
    return Namespace(
        benchmark="alfworld",
        episodes=1,
        max_steps=3,
        seeds="11,17",
        environment_factory="tests.test_external_benchmark:make_environment",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        transfer_success_evaluator=None,
        baseline_policy_factory="tests.test_external_benchmark:make_policy",
        baseline_action_policy_factory=None,
        treatment_policy_factory="tests.test_external_benchmark:make_memory_policy",
        treatment_action_policy_factory=None,
        minimum_trust=0.0,
        baseline_label="baseline",
        treatment_label="treatment",
        output=tmp_path / "paired.json",
        manifest=None,
        overwrite=False,
        probe_action=None,
        require_code_revision="abc123",
        require_clean_working_tree=True,
        require_dependency_version=["alfworld==0.4.2"],
    )


def test_paired_runtime_mismatch_fails_before_callable_or_environment_preflight(
    monkeypatch,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    requirements = RuntimeRequirements(expected_code_revision="required")

    monkeypatch.setattr(paired_benchmark, "collect_runtime_provenance", lambda: _provenance())
    monkeypatch.setattr(
        paired_benchmark,
        "validate_external_benchmark",
        lambda spec: pytest.fail("callable preflight must not run after runtime mismatch"),
    )
    monkeypatch.setattr(
        paired_benchmark,
        "validate_repeated_external_benchmark_runtime",
        lambda *args, **kwargs: pytest.fail(
            "environment preflight must not run after runtime mismatch"
        ),
    )

    with pytest.raises(ValueError, match="runtime code revision"):
        paired_benchmark.preflight_paired_external_benchmarks(
            baseline,
            treatment,
            (11, 17),
            runtime_requirements=requirements,
        )


def test_paired_runtime_requirements_are_validated_once_before_seed_probes(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    requirements = RuntimeRequirements(
        expected_code_revision="actual",
        dependency_versions={"alfworld": "0.4.2"},
    )
    provenance_calls = 0
    probe_calls = 0
    expected_provenance = _provenance()

    def collect_provenance() -> RuntimeProvenance:
        nonlocal provenance_calls
        provenance_calls += 1
        return expected_provenance

    def validate_probe(spec, seeds, *, probe_action):
        nonlocal probe_calls
        probe_calls += 1

    monkeypatch.setattr(paired_benchmark, "collect_runtime_provenance", collect_provenance)
    monkeypatch.setattr(
        paired_benchmark, "validate_repeated_external_benchmark_runtime", validate_probe
    )

    actual_provenance = paired_benchmark.preflight_paired_external_benchmarks(
        baseline,
        treatment,
        (11, 17),
        runtime_requirements=requirements,
    )

    assert actual_provenance is expected_provenance
    assert provenance_calls == 1
    assert probe_calls == 4


def test_controlled_paired_run_carries_exact_preflight_runtime_snapshot(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    requirements = RuntimeRequirements(expected_code_revision="actual")
    expected_provenance = _provenance()

    monkeypatch.setattr(
        paired_benchmark,
        "collect_runtime_provenance",
        lambda: expected_provenance,
    )
    monkeypatch.setattr(
        paired_benchmark,
        "validate_repeated_external_benchmark_runtime",
        lambda spec, seeds, *, probe_action: None,
    )

    result = paired_benchmark.run_paired_external_benchmarks_with_preflight(
        baseline,
        treatment,
        (11,),
        runtime_requirements=requirements,
    )

    assert result.runtime_provenance is expected_provenance


def test_paired_cli_builds_exact_runtime_requirements() -> None:
    arguments = Namespace(
        require_code_revision="abc123",
        require_clean_working_tree=True,
        require_dependency_version=["alfworld==0.4.2", "Transformers==4.45.0"],
    )

    requirements = paired_cli._build_runtime_requirements(arguments)

    assert requirements == RuntimeRequirements(
        expected_code_revision="abc123",
        require_clean_working_tree=True,
        dependency_versions={"alfworld": "0.4.2", "Transformers": "4.45.0"},
    )


def test_paired_cli_forwards_runtime_requirements_to_controlled_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    arguments = _cli_arguments(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)

    def controlled_run(
        baseline_spec,
        treatment_spec,
        seeds,
        *,
        baseline_label,
        treatment_label,
        probe_action,
        runtime_requirements,
    ):
        captured["requirements"] = runtime_requirements
        raise RuntimeError("stop after runtime requirement forwarding assertion")

    monkeypatch.setattr(paired_cli, "run_paired_external_benchmarks_with_preflight", controlled_run)

    with pytest.raises(RuntimeError, match="stop after"):
        paired_cli.main()

    requirements = captured["requirements"]
    assert isinstance(requirements, RuntimeRequirements)
    assert requirements.expected_code_revision == "abc123"
    assert requirements.require_clean_working_tree is True
    assert requirements.dependency_versions == {"alfworld": "0.4.2"}


def test_paired_cli_persists_exact_validated_runtime_snapshot_and_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    arguments = _cli_arguments(tmp_path)
    expected_provenance = _provenance(revision="abc123")
    captured: dict[str, object] = {}

    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        paired_cli,
        "run_paired_external_benchmarks_with_preflight",
        lambda *args, **kwargs: SimpleNamespace(runtime_provenance=expected_provenance),
    )
    monkeypatch.setattr(
        paired_cli,
        "collect_runtime_provenance",
        lambda **kwargs: pytest.fail("controlled CLI must not recollect runtime provenance"),
    )

    def save_result(
        result,
        output_path,
        *,
        runtime_provenance,
        runtime_requirements,
        overwrite,
    ):
        captured["runtime_provenance"] = runtime_provenance
        captured["runtime_requirements"] = runtime_requirements
        return output_path

    monkeypatch.setattr(paired_cli, "save_paired_execution_result", save_result)

    assert paired_cli.main() == 0

    persisted_provenance = captured["runtime_provenance"]
    assert isinstance(persisted_provenance, dict)
    assert persisted_provenance["code_revision"] == "abc123"
    assert persisted_provenance["working_tree_state"] == "clean"
    assert persisted_provenance[paired_cli.PAIRED_PREFLIGHT_STATUS_KEY] == "completed"

    persisted_requirements = captured["runtime_requirements"]
    assert isinstance(persisted_requirements, RuntimeRequirements)
    assert persisted_requirements.expected_code_revision == "abc123"
    assert persisted_requirements.require_clean_working_tree is True
    assert persisted_requirements.dependency_versions == {"alfworld": "0.4.2"}


def test_paired_cli_rejects_duplicate_dependency_names_ignoring_case() -> None:
    with pytest.raises(ValueError, match="unique ignoring case"):
        paired_cli._parse_dependency_requirements(["transformers==4.45.0", "Transformers==4.46.0"])
