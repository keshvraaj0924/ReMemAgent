"""Regression coverage for controlled benchmark CLI artifact handoff."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.benchmark_cli as benchmark_cli
from experiments.runtime_requirements import RuntimeRequirements


def _arguments(tmp_path: Path, **overrides: object) -> Namespace:
    """Build a measured benchmark namespace with optional control contracts."""

    values: dict[str, object] = {
        "benchmark": "webshop",
        "episodes": 2,
        "max_steps": 4,
        "seed": 11,
        "seeds": None,
        "environment_factory": "example:make_environment",
        "policy_factory": "example:make_policy",
        "action_policy_factory": None,
        "minimum_trust": 0.0,
        "success_evaluator": "example:is_success",
        "transfer_success_evaluator": None,
        "output": tmp_path / "benchmark.json",
        "manifest": None,
        "observability_output": None,
        "overwrite": False,
        "preflight": False,
        "runtime_preflight": False,
        "repeated_runtime_preflight": False,
        "preflight_before_run": False,
        "probe_action": None,
        "require_code_revision": None,
        "require_clean_working_tree": False,
        "require_dependency_version": None,
        "source_checkout": None,
        "require_source_revision": None,
        "allow_dirty_source_checkout": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _execute_writer_bundle(
    output_path: Path,
    *,
    writer: object,
    **kwargs: object,
) -> Path:
    """Execute the staged report writer without exercising bundle publication mechanics."""

    assert callable(writer)
    writer(output_path)
    return output_path


def test_runtime_controlled_cli_persists_exact_admitted_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime-controlled persistence uses the admission result without recollection."""

    arguments = _arguments(tmp_path, require_code_revision="abc123")
    controlled_result = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "run_controlled_external_benchmark",
        lambda *args, **kwargs: controlled_result,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda **kwargs: pytest.fail(
            "controlled persistence must not recollect runtime provenance"
        ),
    )

    def fake_save(
        result: object,
        path: Path,
        *,
        runtime_requirements: RuntimeRequirements | None,
        source_checkout_requirements: object | None,
    ) -> Path:
        captured["result"] = result
        captured["requirements"] = runtime_requirements
        captured["source_requirements"] = source_checkout_requirements
        return path

    monkeypatch.setattr(benchmark_cli, "save_controlled_benchmark_result", fake_save)
    monkeypatch.setattr(benchmark_cli, "_persist_benchmark_bundle", _execute_writer_bundle)

    assert benchmark_cli.main() == 0
    assert captured["result"] is controlled_result
    requirements = captured["requirements"]
    assert isinstance(requirements, RuntimeRequirements)
    assert requirements.expected_code_revision == "abc123"
    assert captured["source_requirements"] is None


def test_source_controlled_cli_persists_exact_source_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-controlled persistence receives the same declared checkout contract."""

    arguments = _arguments(
        tmp_path,
        source_checkout=["WebShop=/opt/webshop"],
        require_source_revision=["webshop=deadbeef"],
    )
    controlled_result = object()
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        captured["run_requirements"] = kwargs["source_checkout_requirements"]
        return controlled_result

    def fake_save(
        result: object,
        path: Path,
        *,
        runtime_requirements: RuntimeRequirements | None,
        source_checkout_requirements: object | None,
    ) -> Path:
        captured["saved_result"] = result
        captured["saved_requirements"] = source_checkout_requirements
        captured["runtime_requirements"] = runtime_requirements
        return path

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(benchmark_cli, "run_controlled_external_benchmark", fake_run)
    monkeypatch.setattr(benchmark_cli, "save_controlled_benchmark_result", fake_save)
    monkeypatch.setattr(benchmark_cli, "_persist_benchmark_bundle", _execute_writer_bundle)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda **kwargs: pytest.fail(
            "controlled persistence must not recollect runtime provenance"
        ),
    )

    assert benchmark_cli.main() == 0
    assert captured["saved_result"] is controlled_result
    assert captured["runtime_requirements"] is None
    assert captured["saved_requirements"] is captured["run_requirements"]


def test_repeated_controlled_cli_persists_exact_result_and_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repeated controlled runs bind admitted snapshots and their measured statistics."""

    arguments = _arguments(
        tmp_path,
        seed=None,
        seeds="11,17",
        require_clean_working_tree=True,
    )
    reports = (object(), object())
    controlled_result = SimpleNamespace(reports=reports)
    statistics = {"run_count": 2}
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "run_controlled_repeated_external_benchmarks",
        lambda *args, **kwargs: controlled_result,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "summarize_benchmark_reports",
        lambda measured_reports: SimpleNamespace(
            to_dict=lambda: (
                statistics
                if measured_reports is reports
                else pytest.fail("statistics must use the controlled result reports")
            )
        ),
    )

    def fake_save(
        result: object,
        path: Path,
        *,
        runtime_requirements: RuntimeRequirements | None,
        source_checkout_requirements: object | None,
        statistics: object | None,
    ) -> Path:
        captured["result"] = result
        captured["runtime_requirements"] = runtime_requirements
        captured["source_checkout_requirements"] = source_checkout_requirements
        captured["statistics"] = statistics
        return path

    monkeypatch.setattr(
        benchmark_cli,
        "save_controlled_repeated_benchmark_result",
        fake_save,
    )
    monkeypatch.setattr(benchmark_cli, "_persist_benchmark_bundle", _execute_writer_bundle)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda **kwargs: pytest.fail(
            "controlled persistence must not recollect runtime provenance"
        ),
    )

    assert benchmark_cli.main() == 0
    assert captured["result"] is controlled_result
    requirements = captured["runtime_requirements"]
    assert isinstance(requirements, RuntimeRequirements)
    assert requirements.require_clean_working_tree is True
    assert captured["source_checkout_requirements"] is None
    assert captured["statistics"] is statistics
