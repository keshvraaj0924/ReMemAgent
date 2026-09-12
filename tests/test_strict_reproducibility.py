from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import experiments.paired_benchmark_cli as paired_benchmark_cli
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement
from experiments.strict_reproducibility import validate_strict_paired_reproducibility


def _runtime_requirements(**overrides: object) -> RuntimeRequirements:
    values: dict[str, object] = {
        "expected_code_revision": "abc123",
        "require_clean_working_tree": True,
        "dependency_versions": {"torch": "2.6.0"},
    }
    values.update(overrides)
    return RuntimeRequirements(**values)  # type: ignore[arg-type]


def _source_requirements(*, require_clean: bool = True) -> dict[str, SourceCheckoutRequirement]:
    return {
        "webshop": SourceCheckoutRequirement(
            expected_revision="def456",
            require_clean_working_tree=require_clean,
        )
    }


def test_strict_reproducibility_accepts_complete_contract(tmp_path: Path) -> None:
    validate_strict_paired_reproducibility(
        seeds=(11, 17, 29),
        runtime_requirements=_runtime_requirements(),
        source_checkout_requirements=_source_requirements(),
        manifest_path=tmp_path / "paired.json.manifest.json",
    )


@pytest.mark.parametrize("seeds", [(), (11,)])
def test_strict_reproducibility_requires_multiple_independent_seeds(
    seeds: tuple[int, ...], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="at least 2 independent seeds"):
        validate_strict_paired_reproducibility(
            seeds=seeds,
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_runtime_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires runtime requirements"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=None,
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_exact_code_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact code revision"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(expected_code_revision=None),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_clean_framework_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clean ReMemAgent working tree"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(require_clean_working_tree=False),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_exact_dependency_pin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one exact dependency version"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(dependency_versions={}),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_source_checkout_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one source checkout requirement"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=None,
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_rejects_dirty_source_exception(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dirty checkout exceptions.*webshop"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=_source_requirements(require_clean=False),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_integrity_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integrity manifest destination"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=_source_requirements(),
            manifest_path=None,
        )


def test_paired_cli_applies_strict_gate_before_measured_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = Namespace(
        benchmark="webshop",
        episodes=2,
        max_steps=4,
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
        manifest=tmp_path / "paired.json.manifest.json",
        overwrite=False,
        probe_action=None,
        strict_reproducibility=True,
        require_code_revision=None,
        require_clean_working_tree=False,
        require_dependency_version=None,
        source_checkout=None,
        require_source_revision=None,
        allow_dirty_source_checkout=None,
    )
    monkeypatch.setattr(paired_benchmark_cli, "parse_args", lambda: arguments)

    def fail_if_executed(*args: object, **kwargs: object) -> None:
        pytest.fail("measured execution must not start before strict admission succeeds")

    monkeypatch.setattr(
        paired_benchmark_cli,
        "run_paired_external_benchmarks_with_preflight",
        fail_if_executed,
    )
    monkeypatch.setattr(
        paired_benchmark_cli,
        "run_controlled_paired_external_benchmarks",
        fail_if_executed,
    )

    with pytest.raises(SystemExit, match="strict reproducibility requires runtime requirements"):
        paired_benchmark_cli.main()
