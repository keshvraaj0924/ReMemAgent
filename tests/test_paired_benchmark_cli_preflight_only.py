"""Regression coverage for paired CLI readiness-only execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import experiments.paired_benchmark_cli as paired_cli
from experiments.runtime_requirements import RuntimeRequirements


def _arguments(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "benchmark": "webshop",
        "episodes": 2,
        "max_steps": 4,
        "seeds": "11,17",
        "environment_factory": "tests.test_external_benchmark:make_environment",
        "success_evaluator": "tests.test_external_benchmark:evaluate_success",
        "transfer_success_evaluator": None,
        "baseline_policy_factory": "tests.test_external_benchmark:make_policy",
        "baseline_action_policy_factory": None,
        "treatment_policy_factory": "tests.test_external_benchmark:make_memory_policy",
        "treatment_action_policy_factory": None,
        "minimum_trust": 0.0,
        "baseline_label": "baseline",
        "treatment_label": "treatment",
        "output": tmp_path / "paired.json",
        "manifest": None,
        "overwrite": False,
        "probe_action": "search",
        "preflight_only": True,
        "strict_reproducibility": False,
        "require_code_revision": None,
        "require_clean_working_tree": False,
        "require_dependency_version": None,
        "source_checkout": None,
        "require_source_revision": None,
        "allow_dirty_source_checkout": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _forbid_measurement_and_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("preflight-only execution must not measure or persist artifacts")

    monkeypatch.setattr(paired_cli, "run_controlled_paired_external_benchmarks", fail)
    monkeypatch.setattr(paired_cli, "run_paired_external_benchmarks_with_preflight", fail)
    monkeypatch.setattr(paired_cli, "save_paired_execution_result", fail)
    monkeypatch.setattr(paired_cli, "save_benchmark_artifact_manifest", fail)


def test_preflight_only_routes_runtime_contract_without_artifact_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "paired.json"
    output_path.write_text("existing evidence", encoding="utf-8")
    arguments = _arguments(
        tmp_path,
        require_code_revision="remem-revision",
        require_clean_working_tree=True,
    )
    observed: dict[str, object] = {}

    def preflight(baseline_spec, treatment_spec, seeds, **kwargs) -> None:
        observed["seeds"] = seeds
        observed["probe_action"] = kwargs["probe_action"]
        observed["runtime_requirements"] = kwargs["runtime_requirements"]

    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(paired_cli, "preflight_paired_external_benchmarks", preflight)
    _forbid_measurement_and_persistence(monkeypatch)

    assert paired_cli.main() == 0
    assert observed["seeds"] == (11, 17)
    assert observed["probe_action"] == "search"
    runtime_requirements = observed["runtime_requirements"]
    assert isinstance(runtime_requirements, RuntimeRequirements)
    assert runtime_requirements.expected_code_revision == "remem-revision"
    assert runtime_requirements.require_clean_working_tree is True
    assert output_path.read_text(encoding="utf-8") == "existing evidence"


def test_preflight_only_routes_exact_source_contract_without_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "webshop"
    arguments = _arguments(
        tmp_path,
        source_checkout=[f"webshop={source_path}"],
        require_source_revision=["webshop=source-revision"],
    )
    observed: dict[str, object] = {}

    def controlled_preflight(baseline_spec, treatment_spec, seeds, **kwargs) -> None:
        observed["seeds"] = seeds
        observed.update(kwargs)

    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        paired_cli,
        "preflight_controlled_paired_external_benchmarks",
        controlled_preflight,
    )
    monkeypatch.setattr(
        paired_cli,
        "preflight_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("source-controlled preflight must use controlled path"),
    )
    _forbid_measurement_and_persistence(monkeypatch)

    assert paired_cli.main() == 0
    assert observed["seeds"] == (11, 17)
    assert observed["probe_action"] == "search"
    assert observed["source_checkout_paths"] == {"webshop": source_path}
    requirements = observed["source_checkout_requirements"]
    assert requirements["webshop"].expected_revision == "source-revision"
    assert requirements["webshop"].require_clean_working_tree is True
    assert isinstance(observed["runtime_requirements"], RuntimeRequirements)
