from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli
from experiments.external_benchmark import ExternalBenchmarkSpec
from remem.environments import EnvironmentContractReport


def _base_arguments(tmp_path: Path) -> Namespace:
    return Namespace(
        benchmark="webshop-eval",
        episodes=3,
        max_steps=7,
        seed=41,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        action_policy_factory=None,
        minimum_trust=0.25,
        success_evaluator="example:is_success",
        transfer_success_evaluator="example:is_transfer_success",
        output=tmp_path / "report.json",
        manifest=None,
        preflight=False,
        runtime_preflight=False,
        probe_action=None,
        seeds=None,
    )


def test_main_builds_external_spec_and_persists_report(monkeypatch, tmp_path: Path) -> None:
    report = object()
    arguments = _base_arguments(tmp_path)
    captured: dict[str, ExternalBenchmarkSpec] = {}
    captured_provenance: dict[str, str] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def run_external_benchmark(spec: ExternalBenchmarkSpec) -> object:
        captured["spec"] = spec
        return report

    def save_report(value: object, path: Path, *, runtime_provenance: dict[str, str]) -> Path:
        assert value is report
        captured_provenance.update(runtime_provenance)
        return path

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", run_external_benchmark)
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", save_report)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda: benchmark_cli.collect_runtime_provenance(environment={"REMEM_GIT_COMMIT": "abc123"}),
    )

    assert benchmark_cli.main() == 0
    spec = captured["spec"]
    assert spec.benchmark_name == "webshop-eval"
    assert spec.episode_count == 3
    assert spec.max_steps == 7
    assert spec.seed == 41
    assert spec.environment_factory == "example:make_environment"
    assert spec.policy_factory == "example:make_policy"
    assert spec.action_policy_factory is None
    assert spec.minimum_trust == 0.25
    assert spec.success_evaluator == "example:is_success"
    assert spec.transfer_success_evaluator == "example:is_transfer_success"
    assert captured_provenance["code_revision"] == "abc123"


def test_main_builds_memory_guided_spec(monkeypatch, tmp_path: Path) -> None:
    report = object()
    arguments = _base_arguments(tmp_path)
    arguments.benchmark = "alfworld-eval"
    arguments.episodes = 2
    arguments.max_steps = 5
    arguments.seed = 9
    arguments.policy_factory = None
    arguments.action_policy_factory = "model_policy:make_action_policy"
    arguments.minimum_trust = 0.8
    captured: dict[str, ExternalBenchmarkSpec] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "run_external_benchmark",
        lambda spec: captured.setdefault("spec", spec) or report,
    )
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", lambda value, path, **_: path)

    assert benchmark_cli.main() == 0
    spec = captured["spec"]
    assert spec.policy_factory is None
    assert spec.action_policy_factory == "model_policy:make_action_policy"
    assert spec.minimum_trust == 0.8


def test_main_runtime_preflight_uses_configured_seed_and_probe_action(monkeypatch) -> None:
    arguments = _base_arguments(Path("unused"))
    arguments.episodes = 1
    arguments.max_steps = 3
    arguments.seed = 17
    arguments.runtime_preflight = True
    arguments.probe_action = "look"
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def runtime_preflight(spec: ExternalBenchmarkSpec, *, probe_action: str | None):
        captured["spec"] = spec
        captured["probe_action"] = probe_action
        return EnvironmentContractReport(initial_observation="ready")

    monkeypatch.setattr(benchmark_cli, "validate_external_benchmark_runtime", runtime_preflight)

    assert benchmark_cli.main() == 0
    assert captured["probe_action"] == "look"
    assert isinstance(captured["spec"], ExternalBenchmarkSpec)
    assert captured["spec"].seed == 17


def test_main_rejects_probe_action_without_runtime_preflight(monkeypatch) -> None:
    arguments = _base_arguments(Path("unused"))
    arguments.probe_action = "look"
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    import pytest

    with pytest.raises(ValueError, match="--probe-action requires --runtime-preflight"):
        benchmark_cli.main()


def test_main_persists_requested_benchmark_manifest(monkeypatch, tmp_path: Path) -> None:
    arguments = _base_arguments(tmp_path)
    manifest_path = tmp_path / "report.manifest.json"
    arguments.manifest = manifest_path
    report = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", lambda spec: report)
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", lambda value, path, **_: path)

    def save_manifest(report_path: Path, requested_path: Path) -> Path:
        captured["report_path"] = report_path
        captured["manifest_path"] = requested_path
        return requested_path

    monkeypatch.setattr(benchmark_cli, "save_benchmark_artifact_manifest", save_manifest)

    assert benchmark_cli.main() == 0
    assert captured == {
        "report_path": arguments.output,
        "manifest_path": manifest_path,
    }


def test_main_rejects_manifest_during_preflight(monkeypatch) -> None:
    arguments = _base_arguments(Path("unused"))
    arguments.preflight = True
    arguments.manifest = Path("manifest.json")
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    import pytest

    with pytest.raises(ValueError, match="--manifest requires a measured benchmark run"):
        benchmark_cli.main()
